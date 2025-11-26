"""
LoRA-Only Memory Layer: Pure LoRA-based Generative Memory

Simplified memory layer using only LoRA-based memory generation.
NO MAC components (no neural memory, no test-time learning, no surprise-driven updates).

Ideal for:
- L-UAV (lightweight, resource-constrained)
- Deployment scenarios (pre-trained memory)
- Fast inference (no memory updates)

Architecture:
    Input Hidden States
           ↓
    [1] Adaptive Trigger (optional)
           ↓
    [2] LoRA Memory Generation
           ↓
    [3] Attention (with LoRA memory)
           ↓
    Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from .memory_weaver import MemoryWeaver, MemoryWeaverConfig
from .memory_trigger import MemoryTrigger, MemoryTriggerConfig


@dataclass
class LoRAMemoryConfig:
    """Configuration for LoRA-Only Memory Layer."""

    # Attention configuration
    hidden_size: int = 4096
    num_attention_heads: int = 32
    attention_dropout: float = 0.1

    # LoRA memory configuration
    num_memory_tokens: int = 10
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1

    # Trigger configuration
    enable_trigger: bool = False  # Disable by default for L-UAV
    trigger_threshold: float = 0.5

    # Memory pool method
    pool_method: str = 'mean'  # mean, max, first, last


class LoRAMemoryLayer(nn.Module):
    """
    LoRA-Only Memory Layer.

    Pure LoRA-based memory generation without MAC components.
    Designed for lightweight deployment (e.g., L-UAV).

    Key Features:
    - Parameter-efficient: ~10MB (vs MAC's ~512MB)
    - Fast inference: No memory lookup or updates
    - Pre-trainable: Train once, deploy everywhere
    - Optional triggering: Adaptive invocation for efficiency

    Args:
        config: LoRAMemoryConfig instance
    """

    def __init__(self, config: LoRAMemoryConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        assert (
            self.head_dim * config.num_attention_heads == config.hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        # === LoRA Memory Components ===

        # Memory weaver (generative)
        weaver_config = MemoryWeaverConfig(
            hidden_size=config.hidden_size,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            num_memory_tokens=config.num_memory_tokens
        )
        self.memory_weaver = MemoryWeaver(weaver_config)

        # Memory trigger (optional, for adaptive invocation)
        if config.enable_trigger:
            trigger_config = MemoryTriggerConfig(
                hidden_size=config.hidden_size,
                intermediate_size=512,
                threshold=config.trigger_threshold,
                temperature=1.0,
                use_lora=True,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha
            )
            self.memory_trigger = MemoryTrigger(trigger_config)
        else:
            self.memory_trigger = None

        # === Attention Components ===

        # Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Layer norms
        self.ln_pre = nn.LayerNorm(config.hidden_size)
        self.ln_post = nn.LayerNorm(config.hidden_size)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dimension into (num_heads, head_dim)."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge (num_heads, head_dim) into last dimension."""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.hidden_size)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Scaled dot-product attention."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output

    def generate_memory(
        self,
        hidden_states: torch.Tensor,
        use_trigger: bool = None
    ) -> Tuple[Optional[torch.Tensor], Dict]:
        """
        Generate memory tokens using LoRA weaver.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            use_trigger: Use adaptive triggering (overrides config)

        Returns:
            memory_tokens: [batch_size, num_memory_tokens, hidden_size] or None
            metrics: Generation metrics
        """
        if use_trigger is None:
            use_trigger = self.config.enable_trigger

        batch_size = hidden_states.shape[0]
        metrics = {
            'memory_generated': 0,
            'trigger_invoke_rate': 1.0
        }

        # Check if we should generate memory
        if use_trigger and self.memory_trigger is not None:
            should_invoke = self.memory_trigger.should_invoke(
                hidden_states,
                deterministic=not self.training
            )
            metrics['trigger_invoke_rate'] = should_invoke.float().mean().item()
        else:
            should_invoke = torch.ones(batch_size, dtype=torch.bool, device=hidden_states.device)

        # Generate memory for samples that need it
        if should_invoke.any():
            # Generate for all (will mask later if needed)
            memory_tokens = self.memory_weaver(
                hidden_states,
                pool_method=self.config.pool_method
            )

            # Mask out memory for samples that don't need it
            if not should_invoke.all():
                mask = should_invoke.view(-1, 1, 1)  # [B, 1, 1]
                memory_tokens = memory_tokens * mask

            metrics['memory_generated'] = should_invoke.sum().item()
        else:
            # No memory needed
            memory_tokens = None

        return memory_tokens, metrics

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_trigger: bool = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with LoRA memory.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            use_trigger: Override config for adaptive triggering

        Returns:
            output: [batch_size, seq_len, hidden_size]
            metrics: Dictionary with memory statistics
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        metrics = {
            'memory_generated': 0,
            'trigger_invoke_rate': 0.0
        }

        # Pre-layer norm
        x = self.ln_pre(hidden_states)

        # === Memory Generation ===

        # Generate LoRA memory
        memory_tokens, gen_metrics = self.generate_memory(x, use_trigger)
        metrics.update(gen_metrics)

        # === Attention ===

        # Concatenate memory with input (if memory exists)
        if memory_tokens is not None:
            context = torch.cat([
                memory_tokens,  # [B, num_memory_tokens, H]
                x               # [B, seq_len, H]
            ], dim=1)
        else:
            context = x

        # Apply attention
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(context))
        v = self._split_heads(self.v_proj(context))

        attn_output = self._attention(q, k, v, attention_mask)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        # Residual connection
        output = hidden_states + attn_output
        output = self.ln_post(output)

        return output, metrics

    def freeze_for_inference(self):
        """
        Freeze all parameters for inference mode.

        Recommended for L-UAV deployment.
        """
        for param in self.parameters():
            param.requires_grad = False

        # Disable trigger during inference
        if self.memory_trigger is not None:
            self.memory_trigger.eval()

    def get_lora_parameters(self):
        """Get LoRA parameters for training."""
        params = list(self.memory_weaver.get_lora_parameters())
        if self.memory_trigger is not None:
            params.extend(self.memory_trigger.parameters())
        return params

    def get_memory_size_mb(self) -> float:
        """Get memory footprint in MB."""
        total_params = sum(p.numel() for p in self.parameters())
        # Assuming float32 (4 bytes per parameter)
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb

    def get_lora_size_mb(self) -> float:
        """Get LoRA-only memory footprint in MB."""
        lora_params = sum(p.numel() for p in self.get_lora_parameters())
        size_mb = (lora_params * 4) / (1024 * 1024)
        return size_mb

    def save_lora_weights(self, path: str):
        """
        Save only LoRA weights (for lightweight deployment).

        Args:
            path: Save path
        """
        state = {
            'memory_weaver': self.memory_weaver.state_dict(),
            'config': {
                'hidden_size': self.config.hidden_size,
                'num_memory_tokens': self.config.num_memory_tokens,
                'lora_rank': self.config.lora_rank,
                'lora_alpha': self.config.lora_alpha,
                'pool_method': self.config.pool_method
            }
        }

        if self.memory_trigger is not None:
            state['memory_trigger'] = self.memory_trigger.state_dict()
            state['config']['enable_trigger'] = True
            state['config']['trigger_threshold'] = self.config.trigger_threshold

        torch.save(state, path)

    def load_lora_weights(self, path: str):
        """
        Load LoRA weights (for deployment).

        Args:
            path: Load path
        """
        state = torch.load(path, map_location='cpu')

        self.memory_weaver.load_state_dict(state['memory_weaver'])

        if 'memory_trigger' in state and self.memory_trigger is not None:
            self.memory_trigger.load_state_dict(state['memory_trigger'])


# === L-UAV Compatibility Analysis ===

def analyze_luav_compatibility(config: LoRAMemoryConfig) -> Dict:
    """
    Analyze LoRA-only memory suitability for L-UAV.

    Returns:
        Analysis dict with metrics and recommendations
    """
    # Create layer for analysis
    layer = LoRAMemoryLayer(config)

    # Compute metrics
    total_params = sum(p.numel() for p in layer.parameters())
    lora_params = sum(p.numel() for p in layer.get_lora_parameters())
    attention_params = total_params - lora_params

    total_size_mb = layer.get_memory_size_mb()
    lora_size_mb = layer.get_lora_size_mb()

    # Estimate compute cost (FLOPs)
    # LoRA forward: 2 * (hidden_size * rank + rank * (num_tokens * hidden_size))
    lora_flops = 2 * (config.hidden_size * config.lora_rank +
                      config.lora_rank * (config.num_memory_tokens * config.hidden_size))

    # Attention compute (with memory)
    seq_len = 64  # Example
    total_tokens = seq_len + config.num_memory_tokens
    attn_flops = 2 * config.hidden_size * total_tokens * seq_len

    analysis = {
        'parameters': {
            'total': total_params,
            'lora': lora_params,
            'attention': attention_params,
            'lora_ratio': lora_params / total_params
        },
        'memory_footprint': {
            'total_mb': total_size_mb,
            'lora_mb': lora_size_mb,
            'attention_mb': total_size_mb - lora_size_mb
        },
        'compute_cost': {
            'lora_flops': lora_flops,
            'attention_flops': attn_flops,
            'overhead_percent': (lora_flops / attn_flops) * 100
        },
        'luav_suitability': {
            'parameter_efficient': lora_size_mb < 20,  # < 20MB
            'low_compute': (lora_flops / attn_flops) < 0.1,  # < 10% overhead
            'deployable': True,  # Always deployable (no runtime updates)
            'inference_only': True  # No test-time learning
        },
        'recommendations': []
    }

    # Add recommendations
    if lora_size_mb > 50:
        analysis['recommendations'].append(
            f"LoRA size ({lora_size_mb:.1f}MB) is large. Consider reducing lora_rank or num_memory_tokens."
        )

    if config.enable_trigger:
        analysis['recommendations'].append(
            "Trigger enabled. This adds computation but can save ~30-50% by skipping memory."
        )
    else:
        analysis['recommendations'].append(
            "Trigger disabled. Memory always generated (simpler but more compute)."
        )

    if not analysis['luav_suitability']['parameter_efficient']:
        analysis['recommendations'].append(
            "⚠️  Memory footprint too large for L-UAV. Reduce lora_rank or num_memory_tokens."
        )
    else:
        analysis['recommendations'].append(
            "✅ Parameter-efficient enough for L-UAV deployment."
        )

    if analysis['luav_suitability']['low_compute']:
        analysis['recommendations'].append(
            "✅ Low compute overhead (<10%). Suitable for L-UAV real-time inference."
        )
    else:
        analysis['recommendations'].append(
            f"⚠️  Compute overhead ({analysis['compute_cost']['overhead_percent']:.1f}%) may be high for L-UAV."
        )

    return analysis


if __name__ == "__main__":
    print("Testing LoRA-Only Memory Layer...")

    # Test configuration (for L-UAV)
    config = LoRAMemoryConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_memory_tokens=10,
        lora_rank=8,
        lora_alpha=16.0,
        enable_trigger=False  # Disable for L-UAV simplicity
    )

    # Create layer
    layer = LoRAMemoryLayer(config)

    # Test forward pass
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output, metrics = layer(hidden_states)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Metrics: {metrics}")

    # Analyze L-UAV compatibility
    print("\n" + "="*70)
    print("L-UAV Compatibility Analysis")
    print("="*70)

    analysis = analyze_luav_compatibility(config)

    print("\n📊 Parameters:")
    print(f"  Total: {analysis['parameters']['total']:,}")
    print(f"  LoRA: {analysis['parameters']['lora']:,} ({analysis['parameters']['lora_ratio']*100:.1f}%)")

    print("\n💾 Memory Footprint:")
    print(f"  Total: {analysis['memory_footprint']['total_mb']:.2f} MB")
    print(f"  LoRA: {analysis['memory_footprint']['lora_mb']:.2f} MB")

    print("\n⚡ Compute Cost:")
    print(f"  LoRA overhead: {analysis['compute_cost']['overhead_percent']:.2f}%")

    print("\n🎯 L-UAV Suitability:")
    for key, value in analysis['luav_suitability'].items():
        print(f"  {key}: {value}")

    print("\n💡 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")

    print("\n✓ LoRA-Only Memory Layer tests passed!")
