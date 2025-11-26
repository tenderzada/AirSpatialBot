"""
Hybrid MAC-LoRA Layer: Combines MAC Test-Time Learning with MemGen LoRA

Integrates:
- MAC: Test-time surprise-driven memory updates
- MemGen: LoRA-based generative memory synthesis
- Adaptive triggering for efficient memory invocation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from .neural_memory import NeuralMemory
from .persistent_memory import PersistentMemory
from .memory_weaver import MemoryWeaver, MemoryWeaverConfig
from .memory_trigger import MemoryTrigger, MemoryTriggerConfig


@dataclass
class HybridMACLoRAConfig:
    """Configuration for Hybrid MAC-LoRA Layer."""

    # Attention configuration
    hidden_size: int = 4096
    num_attention_heads: int = 32
    attention_dropout: float = 0.1

    # Memory configuration
    memory_dim: int = 4096
    memory_depth: int = 2
    num_persistent_tokens: int = 64

    # LoRA memory configuration
    num_lora_memory_tokens: int = 10
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1

    # MAC memory configuration
    num_mac_memory_tokens: int = 32
    enable_mac_memory: bool = True  # Use MAC retrieval-based memory

    # Trigger configuration
    enable_trigger: bool = True  # Use adaptive triggering
    trigger_threshold: float = 0.5
    trigger_temperature: float = 1.0

    # Learning configuration
    enable_memory_update: bool = True
    surprise_eta: float = 0.9
    learning_theta: float = 0.1
    forgetting_alpha: float = 0.01

    # Hybrid mode
    fusion_mode: str = 'concat'  # 'concat', 'add', 'learned', 'adaptive'


class HybridMACLoRALayer(nn.Module):
    """
    Hybrid MAC-LoRA Layer.

    Architecture:
        1. Adaptive Triggering: Decide whether to invoke memory
        2. LoRA Memory Generation: Generate latent memory tokens
        3. MAC Memory Retrieval: Retrieve from neural memory (optional)
        4. Memory Fusion: Combine LoRA and MAC memories
        5. Attention: Process fused memory with input
        6. Memory Update: Update MAC neural memory (test-time learning)

    Args:
        config: HybridMACLoRAConfig instance
    """

    def __init__(self, config: HybridMACLoRAConfig):
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
            num_memory_tokens=config.num_lora_memory_tokens
        )
        self.memory_weaver = MemoryWeaver(weaver_config)

        # Memory trigger (adaptive invocation)
        if config.enable_trigger:
            trigger_config = MemoryTriggerConfig(
                hidden_size=config.hidden_size,
                intermediate_size=512,
                threshold=config.trigger_threshold,
                temperature=config.trigger_temperature,
                use_lora=True,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha
            )
            self.memory_trigger = MemoryTrigger(trigger_config)
        else:
            self.memory_trigger = None

        # === MAC Memory Components ===

        if config.enable_mac_memory:
            # Neural memory (retrieval-based with test-time learning)
            self.neural_memory = NeuralMemory(
                input_dim=config.memory_dim,
                output_dim=config.memory_dim,
                hidden_dim=config.memory_dim,
                num_layers=config.memory_depth
            )

            # Memory query/value projections
            self.memory_query_proj = nn.Linear(config.hidden_size, config.memory_dim)
            self.memory_value_proj = nn.Linear(
                config.memory_dim,
                config.hidden_size * config.num_mac_memory_tokens
            )
        else:
            self.neural_memory = None

        # Persistent memory (learnable context)
        self.persistent_memory = PersistentMemory(
            num_tokens=config.num_persistent_tokens,
            token_dim=config.hidden_size
        )

        # === Memory Fusion ===

        if config.fusion_mode == 'learned':
            # Learned fusion weights
            total_memory_tokens = config.num_lora_memory_tokens
            if config.enable_mac_memory:
                total_memory_tokens += config.num_mac_memory_tokens

            self.fusion_layer = nn.Linear(
                config.hidden_size * 2,  # LoRA + MAC
                config.hidden_size
            )
        elif config.fusion_mode == 'adaptive':
            # Adaptive fusion based on confidence
            self.fusion_gate = nn.Linear(config.hidden_size, 1)
        else:
            # No learnable fusion
            self.fusion_layer = None
            self.fusion_gate = None

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

    def generate_lora_memory(
        self,
        hidden_states: torch.Tensor,
        use_trigger: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate memory tokens using LoRA weaver.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            use_trigger: Use adaptive triggering

        Returns:
            lora_memory: [batch_size, num_lora_tokens, hidden_size] or None
            metrics: Trigger metrics
        """
        batch_size = hidden_states.shape[0]
        metrics = {}

        # Check if we should invoke memory
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
            lora_memory = self.memory_weaver(hidden_states)  # [B, num_lora_tokens, H]

            # Mask out memory for samples that don't need it
            if not should_invoke.all():
                mask = should_invoke.view(-1, 1, 1)  # [B, 1, 1]
                lora_memory = lora_memory * mask
        else:
            # No memory needed
            lora_memory = None

        return lora_memory, metrics

    def retrieve_mac_memory(
        self,
        query_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve memory tokens from MAC neural memory.

        Args:
            query_features: [batch_size, hidden_size]

        Returns:
            mac_memory: [batch_size, num_mac_tokens, hidden_size]
        """
        if not self.config.enable_mac_memory:
            return None

        batch_size = query_features.shape[0]

        # Project to memory query space
        memory_query = self.memory_query_proj(query_features)  # [B, memory_dim]

        # Retrieve from neural memory
        memory_values = []
        for i in range(batch_size):
            value, _ = self.neural_memory.retrieve_with_cache(memory_query[i])
            memory_values.append(value)

        memory_values = torch.stack(memory_values, dim=0)  # [B, memory_dim]

        # Project to token space
        memory_tokens_flat = self.memory_value_proj(memory_values)  # [B, H * num_tokens]

        # Reshape to tokens
        mac_memory = memory_tokens_flat.view(
            batch_size,
            self.config.num_mac_memory_tokens,
            self.hidden_size
        )

        return mac_memory

    def fuse_memories(
        self,
        lora_memory: Optional[torch.Tensor],
        mac_memory: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse LoRA and MAC memories.

        Args:
            lora_memory: [batch_size, num_lora_tokens, hidden_size] or None
            mac_memory: [batch_size, num_mac_tokens, hidden_size] or None

        Returns:
            fused_memory: [batch_size, num_total_tokens, hidden_size]
        """
        if lora_memory is None and mac_memory is None:
            return None

        if self.config.fusion_mode == 'concat':
            # Simple concatenation
            memories = []
            if lora_memory is not None:
                memories.append(lora_memory)
            if mac_memory is not None:
                memories.append(mac_memory)
            fused = torch.cat(memories, dim=1)

        elif self.config.fusion_mode == 'add':
            # Element-wise addition (requires same size)
            if lora_memory is not None and mac_memory is not None:
                # Average pool to same size
                fused = (lora_memory + mac_memory) / 2
            else:
                fused = lora_memory if lora_memory is not None else mac_memory

        elif self.config.fusion_mode == 'learned':
            # Learned fusion
            if lora_memory is not None and mac_memory is not None:
                # Concatenate along feature dim
                concat_features = torch.cat([lora_memory, mac_memory], dim=-1)
                fused = self.fusion_layer(concat_features)
            else:
                fused = lora_memory if lora_memory is not None else mac_memory

        elif self.config.fusion_mode == 'adaptive':
            # Adaptive gating
            if lora_memory is not None and mac_memory is not None:
                # Compute gate weight
                avg_lora = lora_memory.mean(dim=1)  # [B, H]
                gate = torch.sigmoid(self.fusion_gate(avg_lora))  # [B, 1]
                gate = gate.unsqueeze(1)  # [B, 1, 1]

                # Weighted combination
                fused = gate * lora_memory + (1 - gate) * mac_memory
            else:
                fused = lora_memory if lora_memory is not None else mac_memory

        else:
            raise ValueError(f"Unknown fusion_mode: {self.config.fusion_mode}")

        return fused

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = None,
        use_trigger: bool = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with hybrid MAC-LoRA memory.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            update_memory: Override config for memory update
            use_trigger: Override config for adaptive triggering

        Returns:
            output: [batch_size, seq_len, hidden_size]
            metrics: Dictionary with memory statistics
        """
        if update_memory is None:
            update_memory = self.config.enable_memory_update

        if use_trigger is None:
            use_trigger = self.config.enable_trigger

        batch_size, seq_len, hidden_size = hidden_states.shape

        metrics = {
            'memory_loss': 0.0,
            'surprise': 0.0,
            'lora_active': 0,
            'mac_active': 0
        }

        # Pre-layer norm
        x = self.ln_pre(hidden_states)

        # === Memory Generation ===

        # 1. Generate LoRA memory
        lora_memory, trigger_metrics = self.generate_lora_memory(x, use_trigger)
        metrics.update(trigger_metrics)
        if lora_memory is not None:
            metrics['lora_active'] = 1

        # 2. Retrieve MAC memory
        query_features = x.mean(dim=1)  # [B, H]
        mac_memory = self.retrieve_mac_memory(query_features)
        if mac_memory is not None:
            metrics['mac_active'] = 1

        # 3. Fuse memories
        fused_memory = self.fuse_memories(lora_memory, mac_memory)

        # 4. Get persistent memory
        persistent_tokens = self.persistent_memory(batch_size)

        # === Attention ===

        # Concatenate: persistent || fused_memory || input
        if fused_memory is not None:
            context = torch.cat([
                persistent_tokens,  # [B, Np, H]
                fused_memory,       # [B, Nm, H]
                x                   # [B, seq_len, H]
            ], dim=1)
        else:
            context = torch.cat([persistent_tokens, x], dim=1)

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

        # === Memory Update (MAC only) ===

        if update_memory and self.config.enable_mac_memory:
            # Update MAC neural memory
            target_features = output.mean(dim=1)  # [B, H]
            target_memory = self.memory_query_proj(target_features)  # [B, memory_dim]

            for i in range(batch_size):
                update_metrics = self.neural_memory.update_memory(
                    query=self.memory_query_proj(query_features[i]),
                    target_value=target_memory[i],
                    eta=self.config.surprise_eta,
                    theta=self.config.learning_theta,
                    alpha=self.config.forgetting_alpha
                )

                metrics['memory_loss'] += update_metrics['loss']
                metrics['surprise'] += update_metrics['surprise']

            # Average metrics
            metrics['memory_loss'] /= batch_size
            metrics['surprise'] /= batch_size

        return output, metrics

    def freeze_for_luav(self):
        """Freeze all parameters for L-UAV (inference-only mode)."""
        for param in self.parameters():
            param.requires_grad = False

        self.config.enable_memory_update = False

    def unfreeze_for_huav(self):
        """Unfreeze parameters for H-UAV (learning mode)."""
        for param in self.parameters():
            param.requires_grad = True

        self.config.enable_memory_update = True

    def get_lora_parameters(self):
        """Get LoRA parameters for separate optimization."""
        params = []
        params.extend(self.memory_weaver.get_lora_parameters())
        if self.memory_trigger is not None:
            params.extend(self.memory_trigger.parameters())
        return params

    def freeze_except_lora(self):
        """Freeze all parameters except LoRA (for efficient fine-tuning)."""
        # Freeze all
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze LoRA
        for param in self.get_lora_parameters():
            param.requires_grad = True


if __name__ == "__main__":
    print("Testing HybridMACLoRALayer...")

    # Test configuration
    config = HybridMACLoRAConfig(
        hidden_size=512,
        num_attention_heads=8,
        memory_dim=512,
        num_persistent_tokens=16,
        num_lora_memory_tokens=10,
        num_mac_memory_tokens=32,
        lora_rank=8,
        enable_mac_memory=True,
        enable_trigger=True,
        fusion_mode='concat'
    )

    # Create layer
    hybrid_layer = HybridMACLoRALayer(config)

    # Test forward pass
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output, metrics = hybrid_layer(hidden_states, update_memory=True)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Metrics: {metrics}")

    # Test parameter counts
    total_params = sum(p.numel() for p in hybrid_layer.parameters())
    lora_params = sum(p.numel() for p in hybrid_layer.get_lora_parameters())

    print(f"\nParameter count:")
    print(f"  Total: {total_params:,}")
    print(f"  LoRA: {lora_params:,}")
    print(f"  LoRA ratio: {lora_params / total_params * 100:.1f}%")

    # Test freeze operations
    hybrid_layer.freeze_except_lora()
    trainable = sum(p.numel() for p in hybrid_layer.parameters() if p.requires_grad)
    print(f"\nAfter freeze_except_lora:")
    print(f"  Trainable: {trainable:,} ({trainable / total_params * 100:.1f}%)")

    print("\n✓ HybridMACLoRALayer tests passed!")
