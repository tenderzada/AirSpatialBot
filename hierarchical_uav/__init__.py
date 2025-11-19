"""
Hierarchical Multi-Agent UAV System with Memory-Augmented Continual Learning

This package implements a two-tier UAV network for fine-grained vehicle recognition:
- High-UAV (H-UAV): Resource-rich agent with MAC memory module supporting test-time learning
- Low-UAV (L-UAV): Lightweight agent with inference-only mode and selective querying

Key components:
- MAC Memory Module: Neural long-term memory, persistent memory, episodic cache
- Communication Protocol: Self-matching gating mechanism for bandwidth efficiency
- Distributed Deployment: Multi-GPU support for concurrent UAV execution
"""

__version__ = "1.0.0"
__author__ = "AirSpatial Research Team"
