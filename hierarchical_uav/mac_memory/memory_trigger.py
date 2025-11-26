"""
Memory Trigger: Adaptive Memory Invocation Module

Inspired by MemGen's trigger mechanism.
Learns when to invoke memory weaver based on input characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class MemoryTriggerConfig:
    """Configuration for Memory Trigger."""

    # Model configuration
    hidden_size: int = 4096
    intermediate_size: int = 512  # Bottleneck dimension

    # Trigger configuration
    threshold: float = 0.5  # Decision threshold
    temperature: float = 1.0  # Temperature for soft trigger

    # LoRA configuration (optional)
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0

    # Training
    use_gumbel: bool = True  # Use Gumbel-Softmax for differentiable sampling


class MemoryTrigger(nn.Module):
    """
    Memory Trigger: Learns when to invoke memory.

    Architecture:
        1. Pool input hidden states
        2. MLP classifier: hidden -> intermediate -> 2 (invoke/skip)
        3. Output: probability of invoking memory

    Can be trained with:
        - Supervised learning (labels for when to use memory)
        - Reinforcement learning (reward based on task performance)

    Args:
        config: MemoryTriggerConfig instance
    """

    def __init__(self, config: MemoryTriggerConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.threshold = config.threshold

        # MLP classifier
        if config.use_lora:
            # Use LoRA for parameter efficiency
            from .memory_weaver import LoRALayer

            self.down_proj = LoRALayer(
                in_features=config.hidden_size,
                out_features=config.intermediate_size,
                rank=config.lora_rank,
                alpha=config.lora_alpha
            )
        else:
            self.down_proj = nn.Linear(
                config.hidden_size,
                config.intermediate_size
            )

        self.activation = nn.GELU()
        self.up_proj = nn.Linear(config.intermediate_size, 2)  # [skip, invoke]

        # Layer norm
        self.ln = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_logits: bool = False,
        use_gumbel: bool = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute memory invocation probability.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            return_logits: Return raw logits instead of probabilities
            use_gumbel: Use Gumbel-Softmax for sampling (overrides config)

        Returns:
            decision: [batch_size, 2] probabilities or logits
            metrics: Dictionary with trigger statistics
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        if use_gumbel is None:
            use_gumbel = self.config.use_gumbel and self.training

        # 1. Pool hidden states
        pooled = hidden_states.mean(dim=1)  # [B, H]
        pooled = self.ln(pooled)

        # 2. MLP classifier
        hidden = self.down_proj(pooled)  # [B, intermediate]
        hidden = self.activation(hidden)
        logits = self.up_proj(hidden)  # [B, 2]

        # 3. Compute probabilities
        if use_gumbel:
            # Gumbel-Softmax for differentiable sampling
            probs = F.gumbel_softmax(
                logits,
                tau=self.config.temperature,
                hard=False,
                dim=-1
            )
        else:
            # Standard softmax
            probs = F.softmax(logits / self.config.temperature, dim=-1)

        # Extract invoke probability
        invoke_prob = probs[:, 1]  # [B]

        # 4. Collect metrics
        metrics = {
            'invoke_prob': invoke_prob.mean().item(),
            'invoke_count': (invoke_prob > self.threshold).sum().item(),
            'total_samples': batch_size
        }

        if return_logits:
            return logits, metrics
        else:
            return probs, metrics

    def should_invoke(
        self,
        hidden_states: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Make binary decision: invoke memory or not.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            deterministic: Use threshold (True) or sample (False)

        Returns:
            decision: [batch_size] boolean tensor
        """
        with torch.no_grad():
            probs, _ = self.forward(hidden_states, return_logits=False, use_gumbel=False)
            invoke_prob = probs[:, 1]  # [B]

            if deterministic:
                # Threshold-based decision
                decision = invoke_prob > self.threshold
            else:
                # Sample from Bernoulli
                decision = torch.bernoulli(invoke_prob).bool()

        return decision

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute supervised loss for trigger training.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            labels: [batch_size] binary labels (0=skip, 1=invoke)
            reduction: Loss reduction method

        Returns:
            loss: Scalar loss
            metrics: Training metrics
        """
        logits, metrics = self.forward(hidden_states, return_logits=True)

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits,
            labels.long(),
            reduction=reduction
        )

        # Accuracy
        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()

        metrics['loss'] = loss.item()
        metrics['accuracy'] = accuracy

        return loss, metrics


class RewardBasedTrigger(nn.Module):
    """
    Reward-Based Memory Trigger for Reinforcement Learning.

    Learns to invoke memory based on downstream task rewards.
    Uses policy gradient methods (REINFORCE).

    Args:
        config: MemoryTriggerConfig instance
    """

    def __init__(self, config: MemoryTriggerConfig):
        super().__init__()

        self.config = config
        self.trigger = MemoryTrigger(config)

        # Baseline for variance reduction
        self.baseline = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample trigger decision and compute log probabilities.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            sample: Sample decision (True) or use max probability (False)

        Returns:
            decision: [batch_size] binary decision
            log_prob: [batch_size] log probability of decision
            baseline_value: [batch_size] baseline estimate
        """
        batch_size = hidden_states.shape[0]

        # Get probabilities
        probs, _ = self.trigger(hidden_states, return_logits=False, use_gumbel=False)
        invoke_prob = probs[:, 1]  # [B]

        # Make decision
        if sample:
            decision = torch.bernoulli(invoke_prob)
        else:
            decision = (invoke_prob > 0.5).float()

        # Compute log probability
        log_prob = torch.log(invoke_prob * decision + (1 - invoke_prob) * (1 - decision) + 1e-10)

        # Compute baseline
        pooled = hidden_states.mean(dim=1)
        baseline_value = self.baseline(pooled).squeeze(-1)  # [B]

        return decision.bool(), log_prob, baseline_value

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        baseline_values: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute REINFORCE policy gradient loss.

        Args:
            log_probs: [batch_size] log probabilities of actions
            rewards: [batch_size] rewards for each sample
            baseline_values: [batch_size] optional baseline for variance reduction

        Returns:
            loss: Scalar policy loss
            metrics: Training metrics
        """
        # Compute advantage
        if baseline_values is not None:
            advantage = rewards - baseline_values.detach()
            # Baseline loss (MSE)
            baseline_loss = F.mse_loss(baseline_values, rewards)
        else:
            advantage = rewards
            baseline_loss = 0.0

        # Policy gradient loss: -log_prob * advantage
        policy_loss = -(log_probs * advantage).mean()

        # Total loss
        total_loss = policy_loss + 0.5 * baseline_loss

        metrics = {
            'policy_loss': policy_loss.item(),
            'baseline_loss': baseline_loss if isinstance(baseline_loss, float) else baseline_loss.item(),
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantage.mean().item()
        }

        return total_loss, metrics


if __name__ == "__main__":
    print("Testing MemoryTrigger...")

    # Test configuration
    config = MemoryTriggerConfig(
        hidden_size=512,
        intermediate_size=128,
        threshold=0.5
    )

    # Create trigger
    trigger = MemoryTrigger(config)

    # Test input
    batch_size = 4
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Test forward pass
    probs, metrics = trigger(hidden_states)
    print(f"Input shape: {hidden_states.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Metrics: {metrics}")

    # Test decision making
    decisions = trigger.should_invoke(hidden_states, deterministic=True)
    print(f"\nDecisions: {decisions}")
    print(f"Invoke count: {decisions.sum().item()} / {batch_size}")

    # Test supervised training
    labels = torch.randint(0, 2, (batch_size,))
    loss, train_metrics = trigger.compute_loss(hidden_states, labels)
    print(f"\nSupervised training:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.2%}")

    # Test reward-based trigger
    print("\nTesting RewardBasedTrigger...")
    rl_trigger = RewardBasedTrigger(config)

    decision, log_prob, baseline = rl_trigger(hidden_states, sample=True)
    print(f"RL Decision: {decision}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Baseline shape: {baseline.shape}")

    # Test policy loss
    rewards = torch.randn(batch_size)
    policy_loss, rl_metrics = rl_trigger.compute_policy_loss(log_prob, rewards, baseline)
    print(f"\nPolicy gradient:")
    print(f"  Loss: {policy_loss.item():.4f}")
    print(f"  Metrics: {rl_metrics}")

    print("\n✓ MemoryTrigger tests passed!")
