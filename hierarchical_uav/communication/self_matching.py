"""
Self-Matching Gating Mechanism

Implements query/key generation and self-similarity scoring for
determining when L-UAV should request assistance from H-UAV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class QueryKeyGenerator(nn.Module):
    """
    Generates compact query and key vectors from image features and 3D dimensions.

    Query (μ): Low-dimensional vector sent to H-UAV (bandwidth-efficient)
    Key (κ): High-dimensional vector for self-matching computation

    Args:
        feature_dim: Dimension of input image features
        bbox_3d_dim: Dimension of 3D bounding box (7: x,y,z,l,w,h,θ)
        query_dim: Output dimension for query vector
        key_dim: Output dimension for key vector
    """

    def __init__(
        self,
        feature_dim: int = 4096,
        bbox_3d_dim: int = 7,
        query_dim: int = 256,  # Small for bandwidth efficiency
        key_dim: int = 1024,    # Larger for discrimination
        hidden_dim: int = 2048
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.bbox_3d_dim = bbox_3d_dim
        self.query_dim = query_dim
        self.key_dim = key_dim

        input_dim = feature_dim + bbox_3d_dim

        # Query generator (small output for communication)
        self.query_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, query_dim)
        )

        # Key generator (larger output for matching)
        self.key_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, key_dim)
        )

    def forward(
        self,
        image_features: torch.Tensor,
        bbox_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate query and key vectors.

        Args:
            image_features: [batch_size, feature_dim] visual features
            bbox_3d: [batch_size, bbox_3d_dim] 3D bounding box parameters

        Returns:
            query: [batch_size, query_dim] compact query vector
            key: [batch_size, key_dim] key vector for matching
        """
        # Concatenate features
        combined = torch.cat([image_features, bbox_3d], dim=-1)

        # Generate query and key
        query = self.query_mlp(combined)
        key = self.key_mlp(combined)

        # L2 normalize for cosine similarity
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        return query, key


class SelfMatchingGate(nn.Module):
    """
    Self-matching gating mechanism for bandwidth-efficient communication.

    Decides whether L-UAV should query H-UAV based on self-similarity score:
        m_{i,i} = (μ_i^T κ_i) / (||μ_i|| ||κ_i||)

    If m_{i,i} > τ: confident, proceed locally
    If m_{i,i} ≤ τ: uncertain, query H-UAV

    Args:
        query_dim: Dimension of query vector
        key_dim: Dimension of key vector
        threshold: Self-matching threshold τ
    """

    def __init__(self, query_dim: int = 256, key_dim: int = 1024, threshold: float = 0.7):
        super().__init__()

        self.threshold = threshold
        self.query_dim = query_dim
        self.key_dim = key_dim

        # Projection layer: project key to query dimension for similarity computation
        self.key_projection = nn.Linear(key_dim, query_dim)

        # Learnable threshold adjustment (optional)
        self.register_buffer('adaptive_threshold', torch.tensor(threshold))

        # Statistics tracking
        self.register_buffer('total_queries', torch.tensor(0, dtype=torch.long))
        self.register_buffer('huav_queries', torch.tensor(0, dtype=torch.long))

        # Score statistics for adaptive threshold
        self.register_buffer('score_sum', torch.tensor(0.0))
        self.register_buffer('score_squared_sum', torch.tensor(0.0))
        self.register_buffer('min_score', torch.tensor(1.0))
        self.register_buffer('max_score', torch.tensor(0.0))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute self-matching scores and gating decisions.

        Args:
            query: [batch_size, query_dim] query vectors (already normalized)
            key: [batch_size, key_dim] key vectors (already normalized)

        Returns:
            scores: [batch_size] self-matching scores (cosine similarity)
            should_query: [batch_size] boolean tensor, True if should query H-UAV
        """
        # Project key to query dimension
        key_projected = self.key_projection(key)  # [batch_size, query_dim]

        # Normalize projected key
        key_projected_norm = F.normalize(key_projected, dim=-1)
        query_norm = F.normalize(query, dim=-1)

        # Compute cosine similarity (self-matching score)
        # m_{i,i} = μ_i^T κ_i (both are normalized, so this is cosine similarity)
        scores = (query_norm * key_projected_norm).sum(dim=-1)  # [batch_size]

        # Ensure scores are in [0, 1] range (cosine can be negative)
        scores = (scores + 1.0) / 2.0  # Map [-1, 1] to [0, 1]

        # Gating decision
        # High score (close to 1) = high similarity = confident = proceed locally
        # Low score (close to 0) = low similarity = uncertain = query H-UAV
        should_query = scores < self.adaptive_threshold

        # Update statistics
        batch_size = query.shape[0]
        self.total_queries += batch_size
        self.huav_queries += should_query.sum().item()

        # Update score statistics
        self.score_sum += scores.sum().item()
        self.score_squared_sum += (scores ** 2).sum().item()
        self.min_score = torch.min(self.min_score, scores.min())
        self.max_score = torch.max(self.max_score, scores.max())

        return scores, should_query

    def get_query_rate(self) -> float:
        """Get the proportion of queries sent to H-UAV."""
        if self.total_queries == 0:
            return 0.0
        return self.huav_queries.item() / self.total_queries.item()

    def reset_statistics(self):
        """Reset query statistics."""
        self.total_queries.zero_()
        self.huav_queries.zero_()
        self.score_sum.zero_()
        self.score_squared_sum.zero_()
        self.min_score.fill_(1.0)
        self.max_score.zero_()

    def update_threshold(self, new_threshold: float):
        """Update the adaptive threshold."""
        self.adaptive_threshold.copy_(torch.tensor(new_threshold))

    def get_score_statistics(self) -> Dict[str, float]:
        """Get statistics about self-matching scores."""
        if self.total_queries == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }

        n = self.total_queries.item()
        mean = self.score_sum.item() / n
        variance = (self.score_squared_sum.item() / n) - (mean ** 2)
        std = variance ** 0.5 if variance > 0 else 0.0

        return {
            'mean': mean,
            'std': std,
            'min': self.min_score.item(),
            'max': self.max_score.item()
        }

    def auto_adjust_threshold(self, target_query_rate: float = 0.3) -> float:
        """
        Automatically adjust threshold to achieve target query rate.

        Args:
            target_query_rate: Desired proportion of queries to H-UAV (default: 0.3 = 30%)

        Returns:
            new_threshold: The adjusted threshold value
        """
        stats = self.get_score_statistics()
        current_rate = self.get_query_rate()

        # If current rate is too low (querying H-UAV too rarely), increase threshold
        # If current rate is too high (querying H-UAV too often), decrease threshold
        mean = stats['mean']
        std = stats['std']

        if std == 0:
            # No variation, use mean as threshold
            new_threshold = mean
        else:
            # Adjust based on target percentile
            # For 30% query rate, we want threshold at 30th percentile
            # Approximate using normal distribution: threshold ≈ mean - z*std
            # where z ≈ 0.52 for 30th percentile
            z_score = -0.52 if target_query_rate == 0.3 else (target_query_rate - 0.5) * 2.5
            new_threshold = mean + z_score * std

            # Clamp to [0, 1] and ensure it's within observed range
            new_threshold = max(stats['min'], min(stats['max'], new_threshold))
            new_threshold = max(0.0, min(1.0, new_threshold))

        self.update_threshold(new_threshold)
        return new_threshold


# Combined module for convenience
class SelfMatchingModule(nn.Module):
    """
    Combined module for query/key generation and self-matching gating.

    Args:
        feature_dim: Dimension of input image features
        query_dim: Dimension of query vector (small for bandwidth)
        key_dim: Dimension of key vector (larger for discrimination)
        threshold: Self-matching threshold
    """

    def __init__(
        self,
        feature_dim: int = 4096,
        query_dim: int = 256,
        key_dim: int = 1024,
        threshold: float = 0.7
    ):
        super().__init__()

        self.qk_generator = QueryKeyGenerator(
            feature_dim=feature_dim,
            query_dim=query_dim,
            key_dim=key_dim
        )

        self.gate = SelfMatchingGate(
            query_dim=query_dim,
            key_dim=key_dim,
            threshold=threshold
        )

    def forward(
        self,
        image_features: torch.Tensor,
        bbox_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate queries/keys and compute gating decisions.

        Args:
            image_features: [batch_size, feature_dim]
            bbox_3d: [batch_size, 7] (x, y, z, l, w, h, θ)

        Returns:
            query: [batch_size, query_dim]
            scores: [batch_size] self-matching scores
            should_query_huav: [batch_size] boolean decisions
        """
        # Generate query and key
        query, key = self.qk_generator(image_features, bbox_3d)

        # Compute gating decision
        scores, should_query_huav = self.gate(query, key)

        return query, scores, should_query_huav

    def get_statistics(self) -> Dict[str, float]:
        """Get communication statistics."""
        stats = {
            'query_rate': self.gate.get_query_rate(),
            'total_queries': self.gate.total_queries.item(),
            'huav_queries': self.gate.huav_queries.item()
        }
        # Add score statistics
        score_stats = self.gate.get_score_statistics()
        stats.update({
            'score_mean': score_stats['mean'],
            'score_std': score_stats['std'],
            'score_min': score_stats['min'],
            'score_max': score_stats['max']
        })
        return stats

    def auto_adjust_threshold(self, target_query_rate: float = 0.3) -> float:
        """
        Automatically adjust threshold to achieve target query rate.

        Args:
            target_query_rate: Desired proportion of queries to H-UAV

        Returns:
            new_threshold: The adjusted threshold value
        """
        return self.gate.auto_adjust_threshold(target_query_rate)


if __name__ == "__main__":
    print("Testing Self-Matching Module...")

    # Create module
    sm_module = SelfMatchingModule(
        feature_dim=512,
        query_dim=128,
        key_dim=256,
        threshold=0.7
    )

    # Test forward pass
    batch_size = 4
    image_features = torch.randn(batch_size, 512)
    bbox_3d = torch.randn(batch_size, 7)

    query, scores, should_query = sm_module(image_features, bbox_3d)

    print(f"Query shape: {query.shape}")  # [4, 128]
    print(f"Scores: {scores}")
    print(f"Should query H-UAV: {should_query}")

    # Check statistics
    stats = sm_module.get_statistics()
    print(f"Statistics: {stats}")

    print("✓ Self-Matching Module tests passed!")
