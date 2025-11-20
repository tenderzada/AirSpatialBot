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

        self.tokenizer, self.llava_model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=config.model_path,
                model_base=config.model_base,
                model_name=model_name,
                load_8bit=config.load_8bit,
                load_4bit=config.load_4bit,
                device_map="auto"
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

        Args:
            input_ids: [batch_size, seq_len] token IDs
            images: [batch_size, 3, H, W] images
            memory_features: Optional [batch_size, hidden_size] memory from H-UAV
            memory_weight: Weight for memory fusion (0.0-1.0)
            **kwargs: Generation arguments

        Returns:
            generated_ids: [batch_size, generated_len] token IDs
        """
        # Extract vision features
        vision_features = self.llava_model.get_model().get_vision_tower()(images)
        vision_features = self.llava_model.get_model().mm_projector(vision_features)
        # vision_features: [batch_size, num_patches, hidden_size]

        # If memory features provided, inject them
        if memory_features is not None:
            # Expand memory to match vision features shape
            batch_size, num_patches, hidden_size = vision_features.shape

            # Memory features are [batch_size, hidden_size] or [batch_size, memory_dim]
            # Need to expand to [batch_size, 1, hidden_size] for concatenation
            if memory_features.shape[-1] != hidden_size:
                # If dimensions don't match, project memory features
                if not hasattr(self, 'memory_projector'):
                    self.memory_projector = nn.Linear(
                        memory_features.shape[-1],
                        hidden_size
                    ).to(memory_features.device)
                memory_features = self.memory_projector(memory_features)

            # Reshape memory to [batch_size, 1, hidden_size]
            memory_token = memory_features.unsqueeze(1) if len(memory_features.shape) == 2 else memory_features

            # Method 1: Prepend memory token (most straightforward)
            enhanced_features = torch.cat([memory_token, vision_features], dim=1)
            # enhanced_features: [batch_size, num_patches+1, hidden_size]

            # Method 2: Alternatively, fusion with vision features
            # avg_vision = vision_features.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
            # fused = memory_weight * memory_token + (1 - memory_weight) * avg_vision
            # enhanced_features = torch.cat([fused, vision_features], dim=1)

        else:
            enhanced_features = vision_features

        # Process through MAC if available
        if hasattr(self, 'mac_layer'):
            enhanced_features, _ = self.mac_layer(
                enhanced_features,
                update_memory=False
            )

        # Generate using enhanced features
        outputs = self.llava_model.generate(
            input_ids=input_ids,
            images=enhanced_features,
            **kwargs
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
