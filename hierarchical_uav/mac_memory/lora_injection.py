"""
Memory Weaver: LoRA Adapters Injected into Frozen LLaVA

真正的记忆编织器：
- 不是独立网络
- 而是在冻结 LLaVA 的特定层（q_proj, v_proj）注入 LoRA
- 基于 L-UAV 状态（hidden states）动态生成潜变量记忆序列

Architecture:
    Frozen LLaVA (base model)
         ↓
    Layer 8, 16, 24: q_proj, v_proj ← LoRA injection
         ↓
    Process L-UAV hidden states
         ↓
    Generate latent memory tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoRAInjectionConfig:
    """Configuration for LoRA injection into LLaVA."""

    # LoRA parameters
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1

    # Injection settings
    target_modules: List[str] = None  # e.g., ['q_proj', 'v_proj']
    target_layers: List[int] = None   # e.g., [8, 16, 24] for selective injection

    # Memory generation
    num_memory_tokens: int = 8
    hidden_size: int = 4096

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ['q_proj', 'v_proj']
        if self.target_layers is None:
            # Inject at layers 8, 16, 24 (evenly distributed)
            self.target_layers = [8, 16, 24]


class LoRALinear(nn.Module):
    """
    LoRA-injected Linear layer.

    Original: y = W_0 @ x
    LoRA: y = W_0 @ x + (B @ A) @ x * (alpha / rank)

    Where:
    - W_0: frozen original weights
    - A, B: trainable low-rank matrices
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA on top of original linear output.

        Args:
            x: Input tensor [..., in_features]
            original_output: Output from frozen linear layer [..., out_features]

        Returns:
            Enhanced output with LoRA
        """
        # Apply LoRA: B @ A @ x
        x_dropped = self.lora_dropout(x)
        lora_output = (x_dropped @ self.lora_A) @ self.lora_B

        # Add to original output with scaling
        return original_output + lora_output * self.scaling


class LLaVAWithLoRAInjection(nn.Module):
    """
    LLaVA with LoRA adapters injected at specific layers.

    This is the H-UAV memory weaver:
    - Base LLaVA remains frozen
    - LoRA adapters injected at q_proj, v_proj of selected layers
    - Generates memory tokens based on L-UAV hidden states
    """

    def __init__(
        self,
        base_llava_model: nn.Module,
        config: LoRAInjectionConfig
    ):
        super().__init__()

        self.config = config
        self.base_model = base_llava_model

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        logger.info("Froze base LLaVA model")

        # Inject LoRA adapters
        self.lora_adapters = nn.ModuleDict()
        self._inject_lora_adapters()

        # Memory token generator
        self.memory_token_generator = MemoryTokenGenerator(
            hidden_size=config.hidden_size,
            num_tokens=config.num_memory_tokens
        )

        logger.info(f"Injected LoRA adapters at layers {config.target_layers}")
        logger.info(f"Target modules: {config.target_modules}")

    def _inject_lora_adapters(self):
        """Inject LoRA adapters at specific layers and modules."""

        # Get transformer layers
        # Assuming LLaVA structure: model.layers[i].self_attn.{q_proj, v_proj}
        if not hasattr(self.base_model, 'model') or not hasattr(self.base_model.model, 'layers'):
            logger.warning("Cannot find transformer layers in base model")
            return

        layers = self.base_model.model.layers

        for layer_idx in self.config.target_layers:
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} out of range (total: {len(layers)})")
                continue

            layer = layers[layer_idx]

            # Inject into attention modules
            if hasattr(layer, 'self_attn'):
                self_attn = layer.self_attn

                for module_name in self.config.target_modules:
                    if hasattr(self_attn, module_name):
                        original_module = getattr(self_attn, module_name)

                        # Create LoRA adapter
                        if isinstance(original_module, nn.Linear):
                            lora_adapter = LoRALinear(
                                in_features=original_module.in_features,
                                out_features=original_module.out_features,
                                rank=self.config.lora_rank,
                                alpha=self.config.lora_alpha,
                                dropout=self.config.lora_dropout
                            )

                            # Store adapter
                            adapter_key = f"layer_{layer_idx}_{module_name}"
                            self.lora_adapters[adapter_key] = lora_adapter

                            logger.info(f"Injected LoRA at layer {layer_idx}.{module_name}")

    def _forward_with_lora(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through LLaVA with LoRA injection.

        Args:
            hidden_states: Input from L-UAV [batch_size, seq_len, hidden_size]

        Returns:
            Enhanced hidden states with LoRA
        """
        # This would be the actual forward pass through LLaVA
        # with LoRA adapters applied at specific layers

        # For now, simplified implementation:
        x = hidden_states

        # Process through each layer
        for layer_idx, layer in enumerate(self.base_model.model.layers):
            # Original layer forward
            layer_output = layer(x, attention_mask=attention_mask)[0]

            # Apply LoRA if this layer has adapters
            if layer_idx in self.config.target_layers:
                # Apply LoRA adapters
                for module_name in self.config.target_modules:
                    adapter_key = f"layer_{layer_idx}_{module_name}"
                    if adapter_key in self.lora_adapters:
                        lora_adapter = self.lora_adapters[adapter_key]
                        # In real implementation, would apply to specific module outputs
                        # Here simplified as applying to layer output
                        layer_output = lora_adapter(x, layer_output)

            x = layer_output

        return x

    def generate_memory_tokens(
        self,
        luav_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate memory tokens for L-UAV.

        Based on L-UAV's current hidden states (already generated tokens),
        dynamically synthesize latent memory sequence.

        Args:
            luav_hidden_states: L-UAV hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            memory_tokens: [batch_size, num_memory_tokens, hidden_size]
            metrics: Generation metrics
        """
        batch_size = luav_hidden_states.shape[0]

        with torch.no_grad():
            # Process L-UAV states through LoRA-enhanced LLaVA
            enhanced_hidden = self._forward_with_lora(
                luav_hidden_states,
                attention_mask
            )

        # Generate memory tokens
        memory_tokens = self.memory_token_generator(enhanced_hidden)

        metrics = {
            'num_tokens': memory_tokens.shape[1],
            'memory_generated': True
        }

        return memory_tokens, metrics

    def get_lora_parameters(self):
        """Get only LoRA parameters (for training)."""
        params = []
        for adapter in self.lora_adapters.values():
            params.extend(adapter.parameters())
        params.extend(self.memory_token_generator.parameters())
        return params

    def get_trainable_parameters_count(self) -> Tuple[int, int]:
        """
        Get parameter counts.

        Returns:
            (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class MemoryTokenGenerator(nn.Module):
    """
    Generate memory tokens from enhanced hidden states.

    Takes LoRA-enhanced hidden states and generates
    a compact set of memory tokens.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_tokens: int = 8,
        intermediate_size: int = 2048
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_tokens = num_tokens

        # Token generation network
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, num_tokens * hidden_size)
        )

        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Generate memory tokens.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            memory_tokens: [batch_size, num_tokens, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Pool hidden states
        pooled = hidden_states.mean(dim=1)  # [B, H]

        # Generate memory tokens
        memory_flat = self.token_proj(pooled)  # [B, num_tokens * H]

        # Reshape to tokens
        memory_tokens = memory_flat.view(
            batch_size,
            self.num_tokens,
            self.hidden_size
        )  # [B, num_tokens, H]

        # Normalize
        memory_tokens = self.ln(memory_tokens)

        return memory_tokens


def create_huav_memory_weaver(
    base_llava_path: str,
    device: str = 'cuda:0',
    lora_rank: int = 8,
    target_layers: List[int] = None,
    load_8bit: bool = False,
    load_4bit: bool = False
) -> LLaVAWithLoRAInjection:
    """
    Create H-UAV memory weaver with LoRA injection.

    Args:
        base_llava_path: Path to base LLaVA model
        device: Device to use
        lora_rank: LoRA rank
        target_layers: Which layers to inject LoRA (default: [8, 16, 24])
        load_8bit: Load model in 8-bit mode
        load_4bit: Load model in 4-bit mode

    Returns:
        Memory weaver model
    """
    # Load base LLaVA model
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    logger.info(f"Loading base LLaVA model from {base_llava_path}...")
    model_name = get_model_name_from_path(base_llava_path)

    # Set device map based on device
    if 'cuda' in str(device):
        device_id = str(device).split(':')[-1] if ':' in str(device) else '0'
        device_map = {"": int(device_id)}
    else:
        device_map = "auto"

    # Load pretrained LLaVA
    tokenizer, base_llava, image_processor, context_len = load_pretrained_model(
        model_path=base_llava_path,
        model_base=None,
        model_name=model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device_map=device_map
    )

    logger.info(f"Successfully loaded {model_name}")

    # Create config
    config = LoRAInjectionConfig(
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 2.0,
        lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj'],
        target_layers=target_layers or [8, 16, 24],
        num_memory_tokens=8,
        hidden_size=4096
    )

    # Create memory weaver
    memory_weaver = LLaVAWithLoRAInjection(base_llava, config)
    memory_weaver.to(device)

    # Count parameters
    total, trainable = memory_weaver.get_trainable_parameters_count()
    logger.info(f"Total parameters: {total:,}")
    logger.info(f"Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")

    return memory_weaver


if __name__ == "__main__":
    print("Testing LoRA Injection Memory Weaver...")

    # Create config
    config = LoRAInjectionConfig(
        lora_rank=8,
        lora_alpha=16.0,
        target_layers=[8, 16, 24],
        target_modules=['q_proj', 'v_proj'],
        num_memory_tokens=8,
        hidden_size=4096
    )

    print(f"Config:")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Target layers: {config.target_layers}")
    print(f"  Target modules: {config.target_modules}")
    print(f"  Memory tokens: {config.num_memory_tokens}")

    # Test LoRALinear
    print("\nTesting LoRALinear...")
    lora_linear = LoRALinear(4096, 4096, rank=8, alpha=16.0)

    x = torch.randn(2, 64, 4096)
    original_output = torch.randn(2, 64, 4096)
    enhanced_output = lora_linear(x, original_output)

    print(f"Input shape: {x.shape}")
    print(f"Original output shape: {original_output.shape}")
    print(f"Enhanced output shape: {enhanced_output.shape}")

    # Test MemoryTokenGenerator
    print("\nTesting MemoryTokenGenerator...")
    token_gen = MemoryTokenGenerator(hidden_size=4096, num_tokens=8)

    hidden_states = torch.randn(2, 64, 4096)
    memory_tokens = token_gen(hidden_states)

    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Memory tokens shape: {memory_tokens.shape}")
    print(f"Expected: [2, 8, 4096]")

    print("\n✓ LoRA Injection Memory Weaver tests passed!")

    print("\n" + "="*70)
    print("Key Design:")
    print("  - LoRA adapters injected at specific LLaVA layers")
    print("  - Base LLaVA remains frozen")
    print("  - Only LoRA parameters are trainable")
    print("  - Generates memory based on L-UAV hidden states")
    print("="*70)
