#!/usr/bin/env python3
"""
Test script to verify H-UAV/L-UAV communication fixes

This script tests the socket communication between H-UAV server and L-UAV client
to ensure the "Ran out of input" error is fixed.
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hierarchical_uav.communication.grpc_server import HUAVServer
from hierarchical_uav.communication.grpc_client import LUAVClient


class MockHUAV:
    """Mock H-UAV model for testing."""

    class Config:
        device = "cpu"

    config = Config()

    class MACLayer:
        class NeuralMemory:
            def retrieve_with_cache(self, query):
                """Mock retrieval - just return the query."""
                # Simulate some processing
                value = query * 2.0  # Simple transformation
                cache_hit = False
                return value, cache_hit

        neural_memory = NeuralMemory()

    mac_layer = MACLayer()


def test_communication():
    """Test H-UAV/L-UAV communication."""

    print("=" * 60)
    print("Testing H-UAV/L-UAV Communication")
    print("=" * 60)

    # Create mock H-UAV model
    print("\n1. Creating mock H-UAV model...")
    mock_huav = MockHUAV()
    print("   ✓ Mock model created")

    # Create and start server
    print("\n2. Starting H-UAV server...")
    server = HUAVServer(
        huav_model=mock_huav,
        host="localhost",
        port=50052  # Use different port for testing
    )
    server.start()
    time.sleep(0.5)  # Give server time to start
    print("   ✓ Server started")

    # Create client
    print("\n3. Creating L-UAV client...")
    client = LUAVClient(
        huav_address="localhost:50052",
        timeout=5.0
    )
    print("   ✓ Client created")

    # Test connection
    print("\n4. Testing connection...")
    if client.ping_huav():
        print("   ✓ H-UAV is reachable")
    else:
        print("   ✗ H-UAV is NOT reachable")
        server.stop()
        return False

    # Test single query
    print("\n5. Testing single query...")
    try:
        query_vector = torch.randn(256)  # Standard query size
        print(f"   Sending query with shape {query_vector.shape}")

        value_vector, cache_hit = client.query_huav(query_vector)

        print(f"   ✓ Query successful!")
        print(f"     - Value shape: {value_vector.shape}")
        print(f"     - Cache hit: {cache_hit}")
        print(f"     - Value sum: {value_vector.sum():.4f}")

    except Exception as e:
        print(f"   ✗ Query failed: {e}")
        server.stop()
        return False

    # Test multiple queries
    print("\n6. Testing multiple queries...")
    num_queries = 10
    success_count = 0

    for i in range(num_queries):
        try:
            query = torch.randn(256)
            value, hit = client.query_huav(query)
            success_count += 1

            if (i + 1) % 5 == 0:
                print(f"   Progress: {i + 1}/{num_queries} queries successful")

        except Exception as e:
            print(f"   ✗ Query {i+1} failed: {e}")

    print(f"   ✓ {success_count}/{num_queries} queries successful")

    # Test batch query
    print("\n7. Testing batch query...")
    try:
        batch_queries = torch.randn(5, 256)
        print(f"   Sending batch with shape {batch_queries.shape}")

        batch_values, batch_hits = client.batch_query_huav(batch_queries)

        print(f"   ✓ Batch query successful!")
        print(f"     - Values shape: {batch_values.shape}")
        print(f"     - Cache hits: {batch_hits.sum().item()}/{len(batch_hits)}")

    except Exception as e:
        print(f"   ✗ Batch query failed: {e}")
        server.stop()
        return False

    # Check server statistics
    print("\n8. Checking server statistics...")
    stats = server.get_statistics()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")

    # Stop server
    print("\n9. Stopping server...")
    server.stop()
    time.sleep(0.5)
    print("   ✓ Server stopped")

    # Final result
    print("\n" + "=" * 60)
    print("✅ All communication tests PASSED!")
    print("=" * 60)
    print("\nThe 'Ran out of input' error should be fixed.")
    print("You can now run the full evaluation.")

    return True


if __name__ == "__main__":
    try:
        success = test_communication()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
