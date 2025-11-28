#!/usr/bin/env python3
"""
Quick test to verify H-UAV Socket server is working correctly.

Usage:
    1. Start H-UAV server: ./start_huav_lora_socket.sh
    2. Run this test: python test_socket_communication.py
"""

import socket
import pickle
import numpy as np
from PIL import Image
import io
import sys

def test_huav_socket(host='localhost', port=50051):
    """Test H-UAV Socket server with a dummy request."""

    print("=" * 70)
    print("H-UAV Socket Communication Test")
    print("=" * 70)

    try:
        # Create a dummy image (random RGB)
        print("\n1. Creating test image (336x336 RGB)...")
        dummy_image = Image.new('RGB', (336, 336), color=(128, 128, 128))

        # Convert to bytes
        buf = io.BytesIO()
        dummy_image.save(buf, format='PNG')
        image_bytes = buf.getvalue()
        print(f"   ✓ Image size: {len(image_bytes)} bytes")

        # Create request
        print("\n2. Creating test request...")
        request = {
            'image': image_bytes,
            'question': 'What is the depth of this vehicle?',
            'request_id': 1
        }
        print(f"   ✓ Request created")

        # Connect to H-UAV server
        print(f"\n3. Connecting to H-UAV at {host}:{port}...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(30.0)
        client_socket.connect((host, port))
        print(f"   ✓ Connected")

        # Send request
        print("\n4. Sending request...")
        request_data = pickle.dumps(request)
        client_socket.sendall(request_data + b'<END>')
        print(f"   ✓ Sent {len(request_data)} bytes")

        # Receive response
        print("\n5. Receiving response...")
        data = b''
        while True:
            chunk = client_socket.recv(4096)
            data += chunk

            if b'<END>' in data:
                data = data.replace(b'<END>', b'')
                break

            if not chunk and len(data) == 0:
                break

        print(f"   ✓ Received {len(data)} bytes")

        # Deserialize response
        print("\n6. Parsing response...")
        response = pickle.loads(data)

        # Check response
        success = response.get('success', False)
        error = response.get('error', None)
        memory_tokens = response.get('memory_tokens', None)
        memory_shape = response.get('memory_shape', None)

        print(f"   Success: {success}")
        if error:
            print(f"   Error: {error}")

        if memory_tokens is not None:
            print(f"   Memory tokens shape: {memory_tokens.shape}")
            print(f"   Memory tokens dtype: {memory_tokens.dtype}")
            print(f"   Memory tokens range: [{memory_tokens.min():.3f}, {memory_tokens.max():.3f}]")

            # Validate shape
            expected_shape = (8, 4096)
            if memory_tokens.shape == expected_shape:
                print(f"   ✓ Shape matches expected {expected_shape}")
            else:
                print(f"   ⚠️  Shape mismatch: expected {expected_shape}, got {memory_tokens.shape}")
        else:
            print(f"   ⚠️  No memory tokens received")

        # Close socket
        client_socket.close()

        print("\n" + "=" * 70)
        if success and memory_tokens is not None and memory_tokens.shape == (8, 4096):
            print("✅ Test PASSED - H-UAV Socket server is working correctly!")
        else:
            print("❌ Test FAILED - Please check H-UAV server")
        print("=" * 70)

        return success

    except ConnectionRefusedError:
        print("\n" + "=" * 70)
        print("❌ Connection Refused")
        print("=" * 70)
        print("\nH-UAV server is not running or not reachable.")
        print("\nPlease start H-UAV server first:")
        print("  ./start_huav_lora_socket.sh")
        print("=" * 70)
        return False

    except socket.timeout:
        print("\n" + "=" * 70)
        print("❌ Timeout")
        print("=" * 70)
        print("\nRequest timed out. H-UAV server may be overloaded or stuck.")
        print("=" * 70)
        return False

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Error: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_huav_socket()
    sys.exit(0 if success else 1)
