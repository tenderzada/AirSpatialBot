#!/usr/bin/env python3
"""
Verify MAC memory optimization by loading the actual configuration.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hierarchical_uav.models.uav_config import UAVConfig
from hierarchical_uav.mac_memory.mac_layer import MACConfig

print("=" * 60)
print("Verifying MAC Memory Optimization")
print("=" * 60)

# Load UAV config
config = UAVConfig()

print(f"\n✅ Current UAVConfig:")
print(f"   memory_dim: {config.memory_dim}")
print(f"   memory_depth: {config.memory_depth}")
print(f"   num_persistent_tokens: {config.num_persistent_tokens}")
print(f"   num_memory_tokens: {config.num_memory_tokens}")

# Load MAC config
mac_config = MACConfig()

print(f"\n✅ Current MACConfig:")
print(f"   memory_dim: {mac_config.memory_dim}")
print(f"   memory_depth: {mac_config.memory_depth}")
print(f"   num_persistent_tokens: {mac_config.num_persistent_tokens}")
print(f"   num_memory_tokens: {mac_config.num_memory_tokens}")

# Calculate expected memory usage
def calculate_memory_size(num_memory_tokens, memory_dim=4096, hidden_size=4096):
    """Calculate MAC memory size in GB."""
    # Main bottleneck: memory_value_proj
    mem_value_params = memory_dim * (hidden_size * num_memory_tokens)

    # Other components
    mlp_params = 33_562_624
    persistent_params = 262_144
    qkvo_params = 67_108_864
    mem_query_params = 16_777_216
    ln_params = 16_384

    total_params = (mlp_params + persistent_params + qkvo_params +
                   mem_query_params + mem_value_params + ln_params)

    # float32 = 4 bytes per parameter
    total_gb = (total_params * 4) / (1024 ** 3)

    return total_gb, mem_value_params

old_size, old_bottleneck = calculate_memory_size(128)
new_size, new_bottleneck = calculate_memory_size(config.num_memory_tokens)

print("\n" + "=" * 60)
print("Memory Comparison")
print("=" * 60)
print(f"Before (num_memory_tokens=128):")
print(f"  Total: {old_size:.2f} GB")
print(f"  memory_value_proj: {old_bottleneck:,} params ({old_bottleneck * 4 / 1024**3:.2f} GB)")

print(f"\nAfter (num_memory_tokens={config.num_memory_tokens}):")
print(f"  Total: {new_size:.2f} GB")
print(f"  memory_value_proj: {new_bottleneck:,} params ({new_bottleneck * 4 / 1024**3:.2f} GB)")

savings = ((old_size - new_size) / old_size) * 100
print(f"\n💰 Memory Saved: {old_size - new_size:.2f} GB ({savings:.1f}%)")

print("\n" + "=" * 60)

# Recommendations for RTX 4090
rtx4090_total = 24  # GB
llava_8bit_estimate = 8  # GB (LLaVA-7B in 8-bit)
available_for_mac = rtx4090_total - llava_8bit_estimate

print("RTX 4090 Memory Budget:")
print(f"  Total: {rtx4090_total} GB")
print(f"  LLaVA (8-bit): ~{llava_8bit_estimate} GB")
print(f"  Available for MAC: ~{available_for_mac} GB")
print(f"  Current MAC size: {new_size:.2f} GB")

if new_size < available_for_mac:
    print(f"\n✅ SUCCESS: MAC memory ({new_size:.2f} GB) fits within budget!")
    remaining = available_for_mac - new_size
    print(f"   Remaining: {remaining:.2f} GB for activations and gradients")
else:
    print(f"\n⚠️  WARNING: MAC memory ({new_size:.2f} GB) exceeds budget!")
    print(f"   Consider reducing num_memory_tokens further")

print("=" * 60)
