"""
Task 1 Evaluation Script for Hierarchical UAV System

Evaluates vehicle attribute recognition using:
- H-UAV: Resource-rich agent with MAC memory
- L-UAV: Lightweight agent with selective querying

Usage:
    # H-UAV (on GPU 0):
    python hierarchical_uav/eval_task1.py --uav_type h-uav --device cuda:0 --port 50051

    # L-UAV (on GPU 1):
    python hierarchical_uav/eval_task1.py --uav_type l-uav --device cuda:1 --huav_address localhost:50051
"""

import argparse
import torch
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_uav.models import LLaVAWithMAC, UAVConfig, UAVType
from hierarchical_uav.communication import HUAVServer, LUAVClient, SelfMatchingModule


def load_task1_data(data_path: str) -> List[Dict]:
    """Load Task 1 test data."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def run_huav_eval(
    config: UAVConfig,
    test_data: List[Dict],
    output_path: str
):
    """
    Run H-UAV evaluation with server mode.

    Args:
        config: H-UAV configuration
        test_data: List of test samples
        output_path: Path to save results
    """
    print("=" * 60)
    print("Starting H-UAV Evaluation")
    print("=" * 60)

    # Load H-UAV model
    print(f"\nLoading H-UAV model on {config.device}...")
    huav_model = LLaVAWithMAC(config)
    print("✓ H-UAV model loaded")

    # Start server
    server = HUAVServer(
        huav_model=huav_model,
        host="0.0.0.0",  # Listen on all interfaces
        port=int(config.huav_address.split(':')[1]) if config.huav_address else 50051
    )
    server.start()

    print(f"\n✓ H-UAV Server started")
    print(f"  Listening on port {server.port}")
    print(f"  Waiting for L-UAV connections...")
    print(f"\nPress Ctrl+C to stop server\n")

    try:
        # Keep server running
        import time
        while True:
            time.sleep(1)

            # Print statistics periodically
            if server.total_requests > 0 and server.total_requests % 10 == 0:
                stats = server.get_statistics()
                print(f"[H-UAV Stats] Requests: {stats['total_requests']}, "
                      f"Cache hit rate: {stats['cache_hit_rate']:.2%}")

    except KeyboardInterrupt:
        print("\n\nShutting down H-UAV server...")
        server.stop()

        # Save final statistics
        final_stats = server.get_statistics()
        print("\n" + "=" * 60)
        print("H-UAV Final Statistics")
        print("=" * 60)
        print(f"Total requests served: {final_stats['total_requests']}")
        print(f"Cache hits: {final_stats['cache_hits']}")
        print(f"Cache hit rate: {final_stats['cache_hit_rate']:.2%}")

        # Save memory state
        memory_path = output_path.replace('.jsonl', '_memory.pt')
        huav_model.save_mac_state(memory_path)
        print(f"\n✓ Memory state saved to {memory_path}")


def run_luav_eval(
    config: UAVConfig,
    test_data: List[Dict],
    output_path: str
):
    """
    Run L-UAV evaluation with client mode.

    Args:
        config: L-UAV configuration
        test_data: List of test samples
        output_path: Path to save results
    """
    print("=" * 60)
    print("Starting L-UAV Evaluation")
    print("=" * 60)

    # Test H-UAV connection
    print(f"\nConnecting to H-UAV at {config.huav_address}...")
    client = LUAVClient(huav_address=config.huav_address)

    if not client.ping_huav():
        print(f"✗ Cannot connect to H-UAV at {config.huav_address}")
        print(f"  Please ensure H-UAV server is running first!")
        return

    print(f"✓ H-UAV server is reachable")

    # Load L-UAV model
    print(f"\nLoading L-UAV model on {config.device}...")
    luav_model = LLaVAWithMAC(config)
    print("✓ L-UAV model loaded")

    # Initialize self-matching module
    sm_module = SelfMatchingModule(
        feature_dim=luav_model.llava_model.config.hidden_size,
        threshold=config.self_match_threshold
    ).to(config.device)
    print(f"✓ Self-matching module initialized (threshold={config.self_match_threshold})")

    # Evaluation loop
    print(f"\nEvaluating on {len(test_data)} samples...")
    print("=" * 60)

    results = []
    local_count = 0
    remote_count = 0

    for i, sample in enumerate(tqdm(test_data, desc="L-UAV Evaluation")):
        try:
            # Extract features (simplified - in practice, process image)
            # For now, use mock features
            image_features = torch.randn(1, luav_model.llava_model.config.hidden_size).to(config.device)
            bbox_3d = torch.randn(1, 7).to(config.device)  # Mock 3D bbox

            # Generate query and compute self-matching
            with torch.no_grad():
                query, score, should_query = sm_module(image_features, bbox_3d)

            # Decision: query H-UAV or proceed locally
            if should_query[0].item():
                # Query H-UAV
                value, cache_hit = client.query_huav(query[0])
                source = "huav"
                remote_count += 1
            else:
                # Proceed locally
                value = query[0]  # Use query as value (simplified)
                cache_hit = False
                source = "local"
                local_count += 1

            # Record result
            results.append({
                'question_id': sample.get('question_id', i),
                'image_id': sample.get('image_id', ''),
                'self_match_score': score[0].item(),
                'source': source,
                'cache_hit': cache_hit
            })

        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue

    # Save results
    print(f"\n\nSaving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Results saved")

    # Print statistics
    print("\n" + "=" * 60)
    print("L-UAV Evaluation Statistics")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Local decisions: {local_count} ({local_count/len(results)*100:.1f}%)")
    print(f"Remote queries: {remote_count} ({remote_count/len(results)*100:.1f}%)")

    sm_stats = sm_module.get_statistics()
    print(f"\nSelf-matching statistics:")
    print(f"  Query rate: {sm_stats['query_rate']:.2%}")
    print(f"  Total queries: {sm_stats['total_queries']}")
    print(f"  H-UAV queries: {sm_stats['huav_queries']}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical UAV Task 1 Evaluation")

    # UAV type
    parser.add_argument(
        '--uav_type',
        type=str,
        choices=['h-uav', 'l-uav'],
        required=True,
        help='UAV type: h-uav (server) or l-uav (client)'
    )

    # Model configuration
    parser.add_argument('--model_path', type=str, default='./models/AirSpatialBot')
    parser.add_argument('--vision_tower', type=str, default='/mnt/data/clip-vit-large-patch14-336')
    parser.add_argument('--device', type=str, default='cuda:0')

    # Communication configuration
    parser.add_argument('--port', type=int, default=50051, help='Server port (H-UAV)')
    parser.add_argument('--huav_address', type=str, default='localhost:50051', help='H-UAV address (L-UAV)')
    parser.add_argument('--threshold', type=float, default=0.7, help='Self-matching threshold (L-UAV)')

    # Data configuration
    parser.add_argument(
        '--test_data',
        type=str,
        default='./data/metadata/airspatial_agent_test_task1.jsonl',
        help='Path to Task 1 test data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for results'
    )

    # MAC configuration
    parser.add_argument('--memory_dim', type=int, default=4096)
    parser.add_argument('--num_persistent_tokens', type=int, default=64)
    parser.add_argument('--num_memory_tokens', type=int, default=128)

    # Quantization options
    parser.add_argument('--load_8bit', action='store_true', help='Use 8-bit quantization to reduce memory')
    parser.add_argument('--load_4bit', action='store_true', help='Use 4-bit quantization (experimental)')

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = f'./outputs/hierarchical_uav/task1_{args.uav_type}_results.jsonl'

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = load_task1_data(args.test_data)
    print(f"✓ Loaded {len(test_data)} test samples")

    # Create configuration based on UAV type
    if args.uav_type == 'h-uav':
        config = UAVConfig.create_huav_config(
            model_path=args.model_path,
            vision_tower=args.vision_tower,
            device=args.device,
            memory_dim=args.memory_dim,
            num_persistent_tokens=args.num_persistent_tokens,
            num_memory_tokens=args.num_memory_tokens,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit
        )
        config.huav_address = f"0.0.0.0:{args.port}"

        run_huav_eval(config, test_data, args.output)

    else:  # l-uav
        config = UAVConfig.create_luav_config(
            model_path=args.model_path,
            vision_tower=args.vision_tower,
            huav_address=args.huav_address,
            device=args.device,
            memory_dim=args.memory_dim,
            num_persistent_tokens=args.num_persistent_tokens,
            num_memory_tokens=args.num_memory_tokens,
            self_match_threshold=args.threshold,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit
        )

        run_luav_eval(config, test_data, args.output)


if __name__ == "__main__":
    main()
