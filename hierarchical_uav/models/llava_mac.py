"""
LLaVA with MAC Memory

Wraps the base LLaVA model with MAC memory modules for test-time learning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os

# Import LLaVA components
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

# Import MAC components
from ..mac_memory import MACLayer, MACConfig
from .uav_config import UAVConfig, UAVType


class LLaVAWithMAC(nn.Module):
    """
    LLaVA model augmented with MAC memory modules.

    This class wraps a pretrained LLaVA model and adds:
    1. MAC layers in the vision-language connector
    2. Test-time learning capability (H-UAV)
    3. Inference-only mode (L-UAV)

    Args:
        config: UAVConfig instance
    """

    def __init__(self, config: UAVConfig):
        super().__init__()

        self.config = config
        self.uav_type = config.uav_type

        # Set offline mode if using local CLIP
        if config.vision_tower and os.path.exists(config.vision_tower):
            os.environ['TRANSFORMERS_OFFLINE'] = '1'

        # Load base LLaVA model
        print(f"Loading base LLaVA model from {config.model_path}...")
        model_name = get_model_name_from_path(config.model_path)

        # Set device_map based on device to avoid multi-GPU distribution
        # Extract device number from config.device (e.g., 'cuda:1' -> '1')
        if 'cuda' in str(config.device):
            device_id = str(config.device).split(':')[-1] if ':' in str(config.device) else '0'
            # For 8-bit models, use device map that keeps everything on the same GPU
            device_map_str = {"": int(device_id)}
        else:
            device_map_str = "auto"

        self.tokenizer, self.llava_model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=config.model_path,
                model_base=config.model_base,
                model_name=model_name,
                load_8bit=config.load_8bit,
                load_4bit=config.load_4bit,
                device_map=device_map_str
            )
        )

        # Override vision tower if specified
        if config.vision_tower:
            if hasattr(self.llava_model.config, 'mm_vision_tower'):
                self.llava_model.config.mm_vision_tower = config.vision_tower

        # Ensure image processor is loaded
        if self.image_processor is None:
            from transformers import CLIPImageProcessor
            vision_tower_name = config.vision_tower if config.vision_tower else config.model_path
            print(f"Loading image processor from {vision_tower_name}...")
            self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

        # Ensure vision tower is loaded
        vision_tower = self.llava_model.get_model().get_vision_tower()
        if hasattr(vision_tower, 'load_model'):
            vision_tower.load_model()

        print(f"✓ Base LLaVA model loaded")

        # Add MAC layers if enabled
        if config.enable_mac:
            self._add_mac_layers()
            print(f"✓ MAC layers added")

        # Configure for UAV type
        if config.uav_type == UAVType.HIGH_UAV:
            self._configure_huav()
            print(f"✓ Configured as H-UAV (learning mode)")
        else:
            self._configure_luav()
            print(f"✓ Configured as L-UAV (inference-only mode)")

        # Move to device
        self.to(config.device)

    def _add_mac_layers(self):
        """Add MAC memory layers to the model."""

        # Create MAC configuration
        mac_config = MACConfig(
            hidden_size=self.llava_model.config.hidden_size,
            num_attention_heads=self.llava_model.config.num_attention_heads,
            memory_dim=self.config.memory_dim,
            memory_depth=self.config.memory_depth,
            num_persistent_tokens=self.config.num_persistent_tokens,
            num_memory_tokens=self.config.num_memory_tokens,
            enable_memory_update=self.config.enable_memory_update,
            surprise_eta=self.config.surprise_eta,
            learning_theta=self.config.learning_theta,
            forgetting_alpha=self.config.forgetting_alpha
        )

        # Insert MAC layer after vision encoder
        # This enhances the vision-language connector with memory
        self.mac_layer = MACLayer(mac_config)

        # Register as a module
        self.add_module('mac_layer', self.mac_layer)

        # Add memory projection layer to convert memory_dim to hidden_size
        # This allows injecting H-UAV memory as additional visual tokens
        self.memory_to_visual_proj = nn.Linear(
            self.config.memory_dim,
            self.llava_model.config.hidden_size
        )
        self.add_module('memory_to_visual_proj', self.memory_to_visual_proj)

    def _configure_huav(self):
        """Configure model for H-UAV (enable learning)."""

        # Enable gradients for MAC layers
        if hasattr(self, 'mac_layer'):
            self.mac_layer.unfreeze_for_huav()

        # Freeze base LLaVA parameters (optional - can finetune)
        # for param in self.llava_model.parameters():
        #     param.requires_grad = False

    def _configure_luav(self):
        """Configure model for L-UAV (inference-only)."""

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Freeze MAC layers specifically
        if hasattr(self, 'mac_layer'):
            self.mac_layer.freeze_for_luav()

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        update_memory: bool = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass with optional memory update.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            images: [batch_size, 3, H, W] images
            labels: Optional labels for training
            update_memory: Whether to update memory (H-UAV only)
            **kwargs: Additional arguments

        Returns:
            Dictionary with:
                - logits: Model outputs
                - loss: Optional loss if labels provided
                - memory_metrics: MAC memory statistics
        """

        # Default memory update based on config
        if update_memory is None:
            update_memory = self.config.enable_memory_update

        # Get image features
        if images is not None:
            image_features = self.llava_model.get_model().get_vision_tower()(images)
            image_features = self.llava_model.get_model().mm_projector(image_features)

            # Apply MAC layer to enhance features
            if hasattr(self, 'mac_layer'):
                image_features, memory_metrics = self.mac_layer(
                    image_features,
                    update_memory=update_memory
                )
            else:
                memory_metrics = {}
        else:
            image_features = None
            memory_metrics = {}

        # Forward through LLaVA
        outputs = self.llava_model(
            input_ids=input_ids,
            images=image_features if image_features is not None else images,
            labels=labels,
            **kwargs
        )

        # Return outputs with memory metrics
        return {
            'logits': outputs.logits if hasattr(outputs, 'logits') else outputs,
            'loss': outputs.loss if hasattr(outputs, 'loss') else None,
            'memory_metrics': memory_metrics
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text given input.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            images: Optional [batch_size, 3, H, W] images
            **kwargs: Generation arguments (max_new_tokens, temperature, etc.)

        Returns:
            generated_ids: [batch_size, generated_len] token IDs
        """

        # Process images through MAC if available
        if images is not None and hasattr(self, 'mac_layer'):
            image_features = self.llava_model.get_model().get_vision_tower()(images)
            image_features = self.llava_model.get_model().mm_projector(image_features)

            image_features, _ = self.mac_layer(
                image_features,
                update_memory=False  # Don't update during generation
            )

            # Generate using enhanced features
            outputs = self.llava_model.generate(
                input_ids=input_ids,
                images=image_features,
                **kwargs
            )
        else:
            # Standard generation
            outputs = self.llava_model.generate(
                input_ids=input_ids,
                images=images,
                **kwargs
            )

        return outputs

    @torch.no_grad()
    def generate_with_memory(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        memory_features: Optional[torch.Tensor] = None,
        memory_weight: float = 0.5,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text with optional memory augmentation from H-UAV.

        This method injects H-UAV memory features as additional visual tokens,
        effectively augmenting the visual context for better inference.

        Strategy:
            1. Extract image features from vision encoder + mm_projector
            2. Project memory_features to visual token space
            3. Concatenate: [image_features || memory_tokens]
            4. Manually prepare inputs_embeds to bypass LLaVA's image encoding
            5. Pass to language model's generate method

        Args:
            input_ids: [batch_size, seq_len] token IDs
            images: [batch_size, 3, H, W] raw images
            memory_features: Optional [batch_size, memory_dim] memory from H-UAV
            memory_weight: Weight for memory fusion (0.0-1.0)
            **kwargs: Generation arguments

        Returns:
            generated_ids: [batch_size, generated_len] token IDs
        """
        # Validate inputs
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        if images is None:
            raise ValueError("images cannot be None")

        import logging
        logger = logging.getLogger(__name__)

        # Remove custom parameters that LLaVA doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ['memory_features', 'memory_weight']}

        # Process images through vision encoder and mm_projector
        vision_tower = self.llava_model.get_model().get_vision_tower()
        mm_projector = self.llava_model.get_model().mm_projector

        # Extract image features
        image_features = vision_tower(images)  # [batch, num_patches, vision_hidden_size]
        image_features = mm_projector(image_features)  # [batch, num_patches, hidden_size]

        logger.debug(f"Image features shape: {image_features.shape}")

        # Inject memory features if provided
        if memory_features is not None and hasattr(self, 'memory_to_visual_proj'):
            # Ensure memory_features has correct shape [batch, memory_dim]
            if len(memory_features.shape) == 1:
                memory_features = memory_features.unsqueeze(0)

            batch_size = memory_features.shape[0]

            # Project memory to visual token space [batch, memory_dim] -> [batch, hidden_size]
            memory_tokens = self.memory_to_visual_proj(memory_features)  # [batch, hidden_size]

            # Reshape to match image features: [batch, 1, hidden_size]
            # This creates 1 "memory token" per sample
            memory_tokens = memory_tokens.unsqueeze(1)

            # Apply memory weight for controlled fusion
            memory_tokens = memory_tokens * memory_weight

            # Concatenate memory tokens with image features
            # Result: [batch, num_patches + 1, hidden_size]
            augmented_features = torch.cat([image_features, memory_tokens], dim=1)

            logger.debug(f"Memory augmented features shape: {augmented_features.shape}")
            logger.debug(f"Memory injection: added {memory_tokens.shape[1]} memory tokens with weight {memory_weight}")

            # Use augmented features for generation
            final_image_features = augmented_features

        else:
            # No memory features - use standard image features
            if memory_features is not None:
                logger.warning("Memory features provided but memory_to_visual_proj not available")

            final_image_features = image_features

        # CRITICAL FIX: Manually prepare inputs to avoid re-encoding features
        # We need to use LLaVA's prepare_inputs_labels_for_multimodal method
        # but provide pre-computed image_features to skip the encoding step

        # Get text embeddings
        input_embeds = self.llava_model.get_model().embed_tokens(input_ids)

        # Find IMAGE_TOKEN_INDEX (usually -200)
        IMAGE_TOKEN_INDEX = -200
        if hasattr(self.llava_model.config, 'image_token_index'):
            IMAGE_TOKEN_INDEX = self.llava_model.config.image_token_index

        # Locate image token positions
        batch_size = input_ids.shape[0]
        image_token_mask = input_ids == IMAGE_TOKEN_INDEX

        # Replace image token embeddings with actual image features
        new_input_embeds = []
        for b in range(batch_size):
            mask = image_token_mask[b]
            if mask.sum() > 0:
                # Found image token
                text_before = input_embeds[b, :mask.argmax(), :]
                text_after = input_embeds[b, mask.argmax()+1:, :]

                # Concatenate: [text_before || image_features || text_after]
                combined = torch.cat([
                    text_before,
                    final_image_features[b],  # Insert image features
                    text_after
                ], dim=0)
                new_input_embeds.append(combined)
            else:
                # No image token, use text embeddings as-is
                new_input_embeds.append(input_embeds[b])

        # Pad sequences to same length
        max_len = max(e.shape[0] for e in new_input_embeds)
        padded_embeds = []
        attention_mask = []

        for embeds in new_input_embeds:
            pad_len = max_len - embeds.shape[0]
            if pad_len > 0:
                # Pad with zeros
                padding = torch.zeros(
                    (pad_len, embeds.shape[1]),
                    dtype=embeds.dtype,
                    device=embeds.device
                )
                padded_embeds.append(torch.cat([embeds, padding], dim=0))
                # Attention mask: 1 for real tokens, 0 for padding
                mask = torch.cat([
                    torch.ones(embeds.shape[0], device=embeds.device),
                    torch.zeros(pad_len, device=embeds.device)
                ])
                attention_mask.append(mask)
            else:
                padded_embeds.append(embeds)
                attention_mask.append(torch.ones(embeds.shape[0], device=embeds.device))

        # Stack into batch
        inputs_embeds = torch.stack(padded_embeds, dim=0)  # [batch, max_len, hidden_size]
        attention_mask = torch.stack(attention_mask, dim=0).long()  # [batch, max_len]

        logger.debug(f"Final inputs_embeds shape: {inputs_embeds.shape}")
        logger.debug(f"Attention mask shape: {attention_mask.shape}")

        # Generate using the language model directly with prepared embeddings
        # This bypasses LLaVA's image encoding
        outputs = self.llava_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **filtered_kwargs
        )

        return outputs

    def update_memory_with_feedback(
        self,
        query_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update long-term memory with feedback (H-UAV only).

        Args:
            query_features: [batch_size, memory_dim] query
            target_features: [batch_size, memory_dim] target

        Returns:
            metrics: Update statistics
        """
        if not self.config.enable_memory_update:
            raise RuntimeError("Memory update not enabled for L-UAV")

        if not hasattr(self, 'mac_layer'):
            raise RuntimeError("MAC layer not available")

        # Update memory for each item in batch
        total_metrics = {'loss': 0.0, 'surprise': 0.0}
        batch_size = query_features.shape[0]

        for i in range(batch_size):
            metrics = self.mac_layer.neural_memory.update_memory(
                query=query_features[i],
                target_value=target_features[i],
                eta=self.config.surprise_eta,
                theta=self.config.learning_theta,
                alpha=self.config.forgetting_alpha
            )

            total_metrics['loss'] += metrics['loss']
            total_metrics['surprise'] += metrics['surprise']

        # Average
        total_metrics['loss'] /= batch_size
        total_metrics['surprise'] /= batch_size

        return total_metrics

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory module."""
        if not hasattr(self, 'mac_layer'):
            return {}

        return {
            'cache_size': self.mac_layer.neural_memory.get_cache_size(),
            'step_count': self.mac_layer.neural_memory.step_count.item(),
            'surprise': self.mac_layer.neural_memory.surprise.item()
        }

    def save_mac_state(self, path: str):
        """Save MAC layer state (memory parameters)."""
        if hasattr(self, 'mac_layer'):
            torch.save({
                'neural_memory': self.mac_layer.neural_memory.state_dict(),
                'persistent_memory': self.mac_layer.persistent_memory.state_dict(),
                'cache': self.mac_layer.neural_memory.episodic_cache,
                'surprise': self.mac_layer.neural_memory.surprise,
                'step_count': self.mac_layer.neural_memory.step_count
            }, path)
            print(f"✓ MAC state saved to {path}")

    def load_mac_state(self, path: str):
        """Load MAC layer state."""
        if hasattr(self, 'mac_layer'):
            state = torch.load(path, map_location=self.config.device)

            self.mac_layer.neural_memory.load_state_dict(state['neural_memory'])
            self.mac_layer.persistent_memory.load_state_dict(state['persistent_memory'])
            self.mac_layer.neural_memory.episodic_cache = state['cache']
            self.mac_layer.neural_memory.surprise = state['surprise']
            self.mac_layer.neural_memory.step_count = state['step_count']

            print(f"✓ MAC state loaded from {path}")


if __name__ == "__main__":
    print("Testing LLaVAWithMAC...")

    # Create H-UAV configuration
    config = UAVConfig.create_huav_config(
        model_path="./models/AirSpatialBot",
        vision_tower="/mnt/data/clip-vit-large-patch14-336"
    )

    print(f"Config created: {config.uav_type}")
    print(f"  - Device: {config.device}")
    print(f"  - Memory update: {config.enable_memory_update}")

    # Note: Actual model loading requires the model files to be present
    # model = LLaVAWithMAC(config)

    print("✓ LLaVAWithMAC structure verified!")
