"""
L-UAV Client

Implements a simple socket-based client for L-UAV to send queries to H-UAV
and receive memory retrievals.
"""

import torch
import socket
import pickle
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LUAVClient:
    """
    L-UAV Client for querying H-UAV server.

    Sends compact query vectors to H-UAV and receives value vectors.

    Args:
        huav_address: H-UAV server address (host:port)
        timeout: Socket timeout in seconds
    """

    def __init__(
        self,
        huav_address: str = "localhost:50051",
        timeout: float = 10.0
    ):
        # Parse address
        if ':' in huav_address:
            self.host, port_str = huav_address.split(':')
            self.port = int(port_str)
        else:
            self.host = huav_address
            self.port = 50051

        self.timeout = timeout
        self.request_counter = 0

        logger.info(f"✓ L-UAV Client initialized for H-UAV at {self.host}:{self.port}")

    def query_huav(
        self,
        query_vector: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """
        Send query to H-UAV and receive response.

        Args:
            query_vector: [query_dim] compact query vector

        Returns:
            value_vector: [memory_dim] retrieved value
            cache_hit: Whether H-UAV used cached result
        """
        self.request_counter += 1

        try:
            # Create socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(self.timeout)

            # Connect to H-UAV
            client_socket.connect((self.host, self.port))

            # Prepare request
            request = {
                'query': query_vector.cpu().numpy(),
                'request_id': self.request_counter
            }

            # Serialize and send
            request_data = pickle.dumps(request)
            client_socket.sendall(request_data + b'<END>')

            logger.debug(f"Sent query #{self.request_counter} to H-UAV")

            # Receive response
            data = b''
            while True:
                chunk = client_socket.recv(4096)
                data += chunk

                # Check for end marker
                if b'<END>' in data:
                    data = data.replace(b'<END>', b'')
                    break

                # Only break on empty chunk if we haven't received anything
                if not chunk and len(data) == 0:
                    break

            # Deserialize response
            if not data:
                raise RuntimeError("Received empty response from H-UAV")

            response = pickle.loads(data)

            # Check if response indicates success
            if not response.get('success', True):
                error_msg = response.get('error', 'Unknown error')
                raise RuntimeError(f"H-UAV returned error: {error_msg}")

            value_vector = torch.tensor(response['value'])
            cache_hit = response['cache_hit']

            logger.debug(f"Received response #{self.request_counter} (cache_hit={cache_hit})")

            client_socket.close()

            return value_vector, cache_hit

        except socket.timeout:
            logger.error(f"Query #{self.request_counter} timed out")
            raise TimeoutError(f"H-UAV query timed out after {self.timeout}s")

        except pickle.UnpicklingError as e:
            logger.error(f"Query #{self.request_counter} failed: Pickle deserialization error - {e}")
            logger.error(f"  Received data length: {len(data)} bytes")
            logger.error(f"  Data preview: {data[:100] if len(data) > 0 else 'empty'}")
            raise RuntimeError(f"Failed to deserialize H-UAV response: {e}")

        except Exception as e:
            logger.error(f"Query #{self.request_counter} failed: {e}")
            logger.error(f"  Exception type: {type(e).__name__}")
            raise RuntimeError(f"Failed to query H-UAV: {e}")

    def batch_query_huav(
        self,
        query_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Send multiple queries to H-UAV (one at a time).

        Args:
            query_vectors: [batch_size, query_dim]

        Returns:
            value_vectors: [batch_size, memory_dim]
            cache_hits: [batch_size] boolean tensor
        """
        batch_size = query_vectors.shape[0]
        values = []
        hits = []

        for i in range(batch_size):
            value, hit = self.query_huav(query_vectors[i])
            values.append(value)
            hits.append(hit)

        value_vectors = torch.stack(values, dim=0)
        cache_hits = torch.tensor(hits, dtype=torch.bool)

        return value_vectors, cache_hits

    def ping_huav(self) -> bool:
        """
        Check if H-UAV server is reachable.

        Returns:
            True if server is reachable, False otherwise
        """
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(2.0)
            client_socket.connect((self.host, self.port))
            client_socket.close()
            return True
        except:
            return False


if __name__ == "__main__":
    print("Testing L-UAV Client...")

    # Test client initialization
    client = LUAVClient(huav_address="localhost:50052", timeout=5.0)

    print(f"Client configured for {client.host}:{client.port}")

    # Test ping (will fail if server not running)
    reachable = client.ping_huav()
    print(f"H-UAV reachable: {reachable}")

    if reachable:
        # Test query
        query = torch.randn(256)
        try:
            value, hit = client.query_huav(query)
            print(f"Query successful! Value shape: {value.shape}, Cache hit: {hit}")
        except Exception as e:
            print(f"Query failed (expected if server not running): {e}")

    print("✓ L-UAV Client test completed!")
