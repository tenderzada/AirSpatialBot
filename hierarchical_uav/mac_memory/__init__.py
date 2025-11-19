"""
MAC (Memory-Augmented Continual Learning) Memory Module

Implements the core memory components for test-time learning:
1. Neural Long-term Memory (NeuralMemory)
2. Persistent Memory (learnable context tokens)
3. MAC Layer (integrates memory with attention mechanism)
"""

from .neural_memory import NeuralMemory
from .mac_layer import MACLayer, MACConfig
from .persistent_memory import PersistentMemory

__all__ = [
    'NeuralMemory',
    'MACLayer',
    'MACConfig',
    'PersistentMemory'
]
