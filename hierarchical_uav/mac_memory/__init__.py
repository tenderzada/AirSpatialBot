"""
MAC (Memory-Augmented Continual Learning) Memory Module

Implements the core memory components for test-time learning:
1. Neural Long-term Memory (NeuralMemory)
2. Persistent Memory (learnable context tokens)
3. MAC Layer (integrates memory with attention mechanism)
4. Memory Weaver (LoRA-based generative memory)
5. Memory Trigger (adaptive memory invocation)
6. Hybrid MAC-LoRA Layer (combines MAC and MemGen approaches)
"""

from .neural_memory import NeuralMemory
from .mac_layer import MACLayer, MACConfig
from .persistent_memory import PersistentMemory
from .memory_weaver import MemoryWeaver, MemoryWeaverConfig, LoRALayer, AdaptiveMemoryWeaver
from .memory_trigger import MemoryTrigger, MemoryTriggerConfig, RewardBasedTrigger
from .hybrid_mac_lora import HybridMACLoRALayer, HybridMACLoRAConfig

__all__ = [
    # Original MAC components
    'NeuralMemory',
    'MACLayer',
    'MACConfig',
    'PersistentMemory',
    # LoRA-based components
    'MemoryWeaver',
    'MemoryWeaverConfig',
    'LoRALayer',
    'AdaptiveMemoryWeaver',
    'MemoryTrigger',
    'MemoryTriggerConfig',
    'RewardBasedTrigger',
    # Hybrid architecture
    'HybridMACLoRALayer',
    'HybridMACLoRAConfig',
]
