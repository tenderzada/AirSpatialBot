#!/usr/bin/env python3
"""
Analyze MAC memory size and provide optimization recommendations.
"""

def calculate_mac_memory_size(
    memory_dim=4096,
    memory_depth=2,
    num_persistent_tokens=64,
    num_memory_tokens=128,
    hidden_size=4096,
    num_attention_heads=32,
    cache_size=1000
):
    """Calculate total MAC layer memory footprint."""

    print("=" * 60)
    print("MAC Memory Size Analysis")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  memory_dim: {memory_dim}")
    print(f"  memory_depth: {memory_depth}")
    print(f"  num_persistent_tokens: {num_persistent_tokens}")
    print(f"  num_memory_tokens: {num_memory_tokens}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  cache_size: {cache_size}")
    print()

    total_params = 0
    bytes_per_param = 4  # float32

    # 1. NeuralMemory MLP
    print("1. NeuralMemory MLP:")
    mlp_params = 0
    # First layer
    layer1 = memory_dim * memory_dim  # Linear
    layer1_ln = memory_dim * 2  # LayerNorm (weight + bias)
    # Second layer (output)
    layer2 = memory_dim * memory_dim  # Linear

    mlp_params = layer1 + layer1_ln + layer2
    mlp_mb = (mlp_params * bytes_per_param) / (1024 ** 2)
    print(f"   Parameters: {mlp_params:,} ({mlp_mb:.1f} MB)")
    total_params += mlp_params

    # 2. Persistent Memory
    print("2. Persistent Memory:")
    persistent_params = num_persistent_tokens * hidden_size
    persistent_mb = (persistent_params * bytes_per_param) / (1024 ** 2)
    print(f"   Parameters: {persistent_params:,} ({persistent_mb:.1f} MB)")
    total_params += persistent_params

    # 3. MAC Layer Projections
    print("3. MAC Layer Projections:")

    # Q, K, V, O projections
    qkvo_params = 4 * (hidden_size * hidden_size)
    qkvo_mb = (qkvo_params * bytes_per_param) / (1024 ** 2)
    print(f"   Q/K/V/O: {qkvo_params:,} ({qkvo_mb:.1f} MB)")
    total_params += qkvo_params

    # Memory query projection
    mem_query_params = hidden_size * memory_dim
    mem_query_mb = (mem_query_params * bytes_per_param) / (1024 ** 2)
    print(f"   memory_query_proj: {mem_query_params:,} ({mem_query_mb:.1f} MB)")
    total_params += mem_query_params

    # Memory value projection - THIS IS THE BOTTLENECK!
    mem_value_params = memory_dim * (hidden_size * num_memory_tokens)
    mem_value_mb = (mem_value_params * bytes_per_param) / (1024 ** 2)
    mem_value_gb = mem_value_mb / 1024
    print(f"   memory_value_proj: {mem_value_params:,} ({mem_value_mb:.1f} MB = {mem_value_gb:.2f} GB) ⚠️")
    total_params += mem_value_params

    # 4. Layer Norms
    print("4. Layer Norms:")
    ln_params = 2 * hidden_size * 2  # 2 LayerNorms, each has weight + bias
    ln_mb = (ln_params * bytes_per_param) / (1024 ** 2)
    print(f"   Parameters: {ln_params:,} ({ln_mb:.3f} MB)")
    total_params += ln_params

    # 5. Episodic Cache (runtime)
    print("5. Episodic Cache (runtime):")
    cache_keys = cache_size * memory_dim * bytes_per_param
    cache_values = cache_size * memory_dim * bytes_per_param
    cache_total = (cache_keys + cache_values) / (1024 ** 2)
    print(f"   Cache entries: {cache_size}")
    print(f"   Cache size: {cache_total:.1f} MB")

    # Total
    print()
    print("=" * 60)
    total_mb = (total_params * bytes_per_param) / (1024 ** 2)
    total_gb = total_mb / 1024
    total_with_cache_mb = total_mb + cache_total
    total_with_cache_gb = total_with_cache_mb / 1024

    print(f"Total MAC Parameters: {total_params:,}")
    print(f"Total Size (without cache): {total_mb:.1f} MB = {total_gb:.2f} GB")
    print(f"Total Size (with cache): {total_with_cache_mb:.1f} MB = {total_with_cache_gb:.2f} GB")
    print("=" * 60)

    return {
        'total_params': total_params,
        'total_gb': total_gb,
        'total_with_cache_gb': total_with_cache_gb,
        'bottleneck': 'memory_value_proj',
        'bottleneck_params': mem_value_params,
        'bottleneck_gb': mem_value_gb
    }


def suggest_optimizations():
    """Suggest parameter configurations for different memory budgets."""

    print("\n" + "=" * 60)
    print("Optimization Recommendations")
    print("=" * 60)

    configs = [
        {
            'name': 'Current (9.1GB)',
            'memory_dim': 4096,
            'num_memory_tokens': 128,
            'cache_size': 1000
        },
        {
            'name': 'Optimized - Large (2.3GB)',
            'memory_dim': 4096,
            'num_memory_tokens': 32,
            'cache_size': 500
        },
        {
            'name': 'Optimized - Medium (600MB)',
            'memory_dim': 4096,
            'num_memory_tokens': 8,
            'cache_size': 200
        },
        {
            'name': 'Optimized - Small (300MB)',
            'memory_dim': 2048,
            'num_memory_tokens': 8,
            'cache_size': 100
        }
    ]

    for config in configs:
        print(f"\n📌 {config['name']}:")
        print(f"   memory_dim: {config['memory_dim']}")
        print(f"   num_memory_tokens: {config['num_memory_tokens']}")
        print(f"   cache_size: {config['cache_size']}")

        result = calculate_mac_memory_size(
            memory_dim=config['memory_dim'],
            num_memory_tokens=config['num_memory_tokens'],
            cache_size=config['cache_size']
        )
        print()


if __name__ == "__main__":
    # Analyze current configuration
    print("\n🔍 Analyzing CURRENT configuration...")
    current = calculate_mac_memory_size()

    print(f"\n⚠️  BOTTLENECK IDENTIFIED:")
    print(f"   {current['bottleneck']}: {current['bottleneck_gb']:.2f} GB")
    print(f"   This is {current['bottleneck_params']:,} parameters!")

    # Suggest optimizations
    suggest_optimizations()

    print("\n" + "=" * 60)
    print("Recommendation:")
    print("=" * 60)
    print("For RTX 4090 (24GB), recommended configuration:")
    print("  - num_memory_tokens: 32 (减少到 1/4)")
    print("  - cache_size: 500 (减少缓存)")
    print("Expected size: ~2.3GB (节省 75% 内存)")
    print("=" * 60)
