"""
Dynamic Memory Weight Adjustment

Instead of using fixed memory_weight=0.5, adjust it based on:
1. Self-matching score (relevance of retrieved memory)
2. H-UAV confidence (how certain H-UAV is)
3. L-UAV confidence (how certain L-UAV would be without memory)

This allows graceful degradation when memory quality is uncertain.
"""

import torch
import numpy as np
from typing import Dict, Any


class DynamicMemoryWeightAdjuster:
    """
    Dynamically adjusts memory weight based on multiple confidence signals.
    """

    def __init__(
        self,
        base_weight: float = 0.5,
        min_weight: float = 0.0,
        max_weight: float = 0.9
    ):
        """
        Args:
            base_weight: Default weight when all signals are neutral
            min_weight: Minimum weight (when confidence is very low)
            max_weight: Maximum weight (when confidence is very high)
        """
        self.base_weight = base_weight
        self.min_weight = min_weight
        self.max_weight = max_weight

    def compute_weight(
        self,
        self_match_score: float,
        huav_confidence: float = None,
        luav_confidence: float = None
    ) -> Dict[str, Any]:
        """
        Compute memory weight based on multiple confidence signals.

        Args:
            self_match_score: Relevance score from self-matching [0, 1]
            huav_confidence: H-UAV's confidence in its answer [0, 1]
            luav_confidence: L-UAV's confidence without memory [0, 1]

        Returns:
            Dictionary containing:
                - weight: Computed memory weight
                - strategy: Which strategy was used
                - signals: Individual confidence signals
        """

        # Strategy 1: Self-matching score only (current baseline)
        if huav_confidence is None and luav_confidence is None:
            weight = self._weight_from_self_match(self_match_score)
            strategy = "self_match_only"

        # Strategy 2: Self-match + H-UAV confidence
        elif luav_confidence is None:
            weight = self._weight_from_huav_relevance(
                self_match_score,
                huav_confidence
            )
            strategy = "huav_relevance"

        # Strategy 3: Full adaptive (all signals available)
        else:
            weight = self._weight_from_all_signals(
                self_match_score,
                huav_confidence,
                luav_confidence
            )
            strategy = "full_adaptive"

        # Clamp to valid range
        weight = np.clip(weight, self.min_weight, self.max_weight)

        return {
            'weight': weight,
            'strategy': strategy,
            'signals': {
                'self_match_score': self_match_score,
                'huav_confidence': huav_confidence,
                'luav_confidence': luav_confidence
            }
        }

    def _weight_from_self_match(self, score: float) -> float:
        """
        Compute weight from self-matching score alone.

        Logic:
        - High score (>0.7): High relevance → higher weight
        - Medium score (0.5-0.7): Uncertain → moderate weight
        - Low score (<0.5): Low relevance → very low weight
        """
        if score > 0.8:
            return 0.8  # High confidence in relevance
        elif score > 0.7:
            return 0.6  # Good relevance
        elif score > 0.5:
            return 0.3  # Moderate relevance
        else:
            return 0.1  # Low relevance, use memory sparingly

    def _weight_from_huav_relevance(
        self,
        self_match_score: float,
        huav_confidence: float
    ) -> float:
        """
        Compute weight from self-match score and H-UAV confidence.

        Logic:
        - Both high → trust memory strongly
        - Self-match high but H-UAV unsure → moderate weight
        - Self-match low → low weight regardless of H-UAV confidence
        """
        # Relevance gate: if memory is not relevant, don't use it
        if self_match_score < 0.5:
            return 0.1

        # Combined signal: geometric mean
        # This ensures both signals must be high for high weight
        combined = np.sqrt(self_match_score * huav_confidence)

        # Map to weight range
        if combined > 0.8:
            return 0.9  # Both very confident
        elif combined > 0.6:
            return 0.7  # Both moderately confident
        elif combined > 0.4:
            return 0.4  # One is confident, one is not
        else:
            return 0.2  # Both uncertain

    def _weight_from_all_signals(
        self,
        self_match_score: float,
        huav_confidence: float,
        luav_confidence: float
    ) -> float:
        """
        Adaptive weight based on all available signals.

        Strategy:
        - If L-UAV is already confident, use less memory
        - If L-UAV is uncertain AND memory is relevant AND H-UAV is confident,
          use more memory
        - Balance between local knowledge and external help
        """

        # Relevance gate
        if self_match_score < 0.5:
            return 0.1

        # Case 1: L-UAV is confident (>0.8)
        # → Trust local inference, use memory only as refinement
        if luav_confidence > 0.8:
            return 0.2 * self_match_score

        # Case 2: L-UAV is uncertain (<0.5) AND memory is good
        # → Rely more on H-UAV memory
        elif luav_confidence < 0.5:
            if huav_confidence > 0.7 and self_match_score > 0.7:
                return 0.8  # Trust H-UAV's help
            else:
                return 0.5 * (huav_confidence + self_match_score) / 2

        # Case 3: L-UAV is moderately confident (0.5-0.8)
        # → Balance between local and memory
        else:
            # Weighted combination favoring the more confident source
            alpha = 1 - luav_confidence  # More uncertain L-UAV → higher alpha
            memory_signal = (huav_confidence + self_match_score) / 2

            return alpha * memory_signal + (1 - alpha) * 0.3

    def get_decision_summary(self, weight: float) -> str:
        """Get human-readable summary of the weighting decision."""
        if weight > 0.7:
            return "High trust in H-UAV memory - will strongly influence answer"
        elif weight > 0.5:
            return "Moderate trust - memory will influence but not dominate"
        elif weight > 0.3:
            return "Low trust - memory used as weak signal only"
        else:
            return "Very low trust - memory barely used, rely on local"


# Integration with eval_task1.py
def apply_dynamic_weight_in_eval():
    """
    How to integrate dynamic weight adjustment into eval_task1.py

    Replace the fixed weight with dynamic computation.
    """

    example_code = '''
# In eval_task1.py, around line 580

# OLD CODE (fixed weight):
# output_ids = luav_model.generate_with_memory(
#     input_ids=input_ids,
#     images=image_tensor,
#     memory_features=memory_value,
#     memory_weight=0.5,  # ← Fixed weight
#     ...
# )

# NEW CODE (dynamic weight):
from hierarchical_uav.dynamic_memory_weight import DynamicMemoryWeightAdjuster

# Initialize adjuster (once, outside the loop)
weight_adjuster = DynamicMemoryWeightAdjuster(
    base_weight=0.5,
    min_weight=0.1,
    max_weight=0.9
)

# Inside the loop, when using memory:
if memory_value is not None:
    # Compute dynamic weight based on self-match score
    weight_result = weight_adjuster.compute_weight(
        self_match_score=score[0].item(),
        huav_confidence=None,  # TODO: Get from H-UAV if available
        luav_confidence=None   # TODO: Extract from L-UAV logits
    )

    memory_weight = weight_result['weight']

    logger.info(
        f"Sample {i}: self_match={score[0].item():.3f}, "
        f"memory_weight={memory_weight:.3f} "
        f"({weight_adjuster.get_decision_summary(memory_weight)})"
    )

    output_ids = luav_model.generate_with_memory(
        input_ids=input_ids,
        images=image_tensor,
        memory_features=memory_value,
        memory_weight=memory_weight,  # ← Dynamic weight!
        ...
    )
'''

    print(example_code)


if __name__ == "__main__":
    print("=" * 70)
    print("Dynamic Memory Weight Adjustment - Examples")
    print("=" * 70)

    adjuster = DynamicMemoryWeightAdjuster()

    # Test different scenarios
    scenarios = [
        {
            'name': "High relevance, good H-UAV confidence",
            'self_match': 0.85,
            'huav_conf': 0.9,
            'luav_conf': None
        },
        {
            'name': "Low relevance (should barely use memory)",
            'self_match': 0.45,
            'huav_conf': 0.9,
            'luav_conf': None
        },
        {
            'name': "L-UAV confident, don't need much help",
            'self_match': 0.75,
            'huav_conf': 0.8,
            'luav_conf': 0.9
        },
        {
            'name': "L-UAV uncertain, H-UAV confident, good match",
            'self_match': 0.80,
            'huav_conf': 0.85,
            'luav_conf': 0.4
        },
        {
            'name': "L-UAV uncertain, but memory not very relevant",
            'self_match': 0.55,
            'huav_conf': 0.7,
            'luav_conf': 0.3
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Inputs: self_match={scenario['self_match']:.2f}, "
              f"huav_conf={scenario['huav_conf']}, "
              f"luav_conf={scenario['luav_conf']}")

        result = adjuster.compute_weight(
            scenario['self_match'],
            scenario['huav_conf'],
            scenario['luav_conf']
        )

        print(f"  → Weight: {result['weight']:.2f}")
        print(f"  → Strategy: {result['strategy']}")
        print(f"  → Decision: {adjuster.get_decision_summary(result['weight'])}")

    print("\n" + "=" * 70)
    print("Integration guide:")
    print("=" * 70)
    apply_dynamic_weight_in_eval()
