"""
H-UAV Server

Implements a simple socket-based server for H-UAV to receive queries from L-UAVs
and return memory retrievals.
"""

import torch
import torch.nn as nn
import socket
import pickle
import threading
import numpy as np
from typing import Optional, Callable, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HUAVServer:
    """
    H-UAV Server for handling L-UAV queries.

    Receives compact query vectors from L-UAVs and returns
    memory-retrieved value vectors.

    Args:
        huav_model: LLaVAWithMAC model instance (H-UAV)
        host: Server host address
        port: Server port
        max_connections: Maximum concurrent connections
        query_dim: Dimension of query vectors from L-UAV (default: 256)
    """

    def __init__(
        self,
        huav_model,
        host: str = "localhost",
        port: int = 50051,
        max_connections: int = 10,
        query_dim: int = 256
    ):
        self.huav_model = huav_model
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.query_dim = query_dim

        self.server_socket = None
        self.is_running = False
        self.server_thread = None

        # Statistics
        self.total_requests = 0
        self.cache_hits = 0

        # Query projection layer (256D query -> memory_dim)
        # This expands compact L-UAV queries to H-UAV memory space
        if hasattr(huav_model, 'mac_layer'):
            memory_dim = huav_model.mac_layer.neural_memory.input_dim
            self.query_projection = nn.Linear(query_dim, memory_dim).to(huav_model.config.device)
            logger.info(f"Created query projection: {query_dim}D -> {memory_dim}D")
        else:
            self.query_projection = None

    def start(self):
        """Start the H-UAV server in a background thread."""
        if self.is_running:
            logger.warning("Server is already running")
            return

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_connections)

        self.is_running = True

        self.server_thread = threading.Thread(target=self._serve, daemon=True)
        self.server_thread.start()

        logger.info(f"✓ H-UAV Server started at {self.host}:{self.port}")

    def stop(self):
        """Stop the H-UAV server."""
        if not self.is_running:
            return

        self.is_running = False

        if self.server_socket:
            self.server_socket.close()

        logger.info("✓ H-UAV Server stopped")

    def _serve(self):
        """Main server loop (runs in background thread)."""
        logger.info("H-UAV Server listening for connections...")

        while self.is_running:
            try:
                client_socket, address = self.server_socket.accept()
                logger.info(f"Connection from L-UAV at {address}")

                # Handle request in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                )
                client_thread.start()

            except Exception as e:
                if self.is_running:
                    logger.error(f"Server error: {e}")
                break

    def _handle_client(self, client_socket: socket.socket):
        """Handle a single L-UAV request."""
        response_sent = False
        request = None

        try:
            # Receive query data
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

            if not data:
                logger.warning("Received empty request")
                return

            # Deserialize query
            request = pickle.loads(data)
            query_vector = torch.tensor(request['query']).to(self.huav_model.config.device)

            logger.debug(f"Received query with shape {query_vector.shape}")

            # Retrieve from memory
            cache_hit = False  # Initialize cache_hit
            with torch.no_grad():
                if hasattr(self.huav_model, 'mac_layer') and self.query_projection is not None:
                    # Project compact query to memory dimension
                    expanded_query = self.query_projection(query_vector)
                    logger.debug(f"Expanded query from {query_vector.shape} to {expanded_query.shape}")

                    # Retrieve from MAC memory
                    value_vector, cache_hit = self.huav_model.mac_layer.neural_memory.retrieve_with_cache(
                        expanded_query
                    )

                    if cache_hit:
                        self.cache_hits += 1

                else:
                    # Fallback if no MAC layer: return query as-is
                    value_vector = query_vector

            self.total_requests += 1

            # Prepare response
            response = {
                'value': value_vector.cpu().numpy(),
                'cache_hit': cache_hit,
                'request_id': request.get('request_id', 0),
                'success': True
            }

            # Serialize and send
            response_data = pickle.dumps(response)
            client_socket.sendall(response_data + b'<END>')
            response_sent = True

            logger.debug(f"Sent response (cache_hit={response['cache_hit']})")

        except pickle.UnpicklingError as e:
            logger.error(f"Error deserializing client request: {e}")
            logger.error(f"  Received data length: {len(data) if 'data' in locals() else 0} bytes")
            logger.error(f"  Data preview: {data[:100] if 'data' in locals() and len(data) > 0 else 'empty'}")

            # Send error response
            self._send_error_response(client_socket, str(e), request)
            response_sent = True

        except Exception as e:
            logger.error(f"Error handling client: {e}")
            logger.error(f"  Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")

            # Send error response
            if not response_sent:
                self._send_error_response(client_socket, str(e), request)
                response_sent = True

        finally:
            client_socket.close()

    def _send_error_response(self, client_socket: socket.socket, error_msg: str, request: Optional[Dict] = None):
        """Send an error response to the client."""
        try:
            # Determine appropriate value dimension
            if hasattr(self.huav_model, 'mac_layer'):
                value_dim = self.huav_model.mac_layer.neural_memory.output_dim
            else:
                value_dim = self.query_dim

            # Create error response with dummy value
            error_response = {
                'value': np.zeros(value_dim),
                'cache_hit': False,
                'request_id': request.get('request_id', 0) if request else 0,
                'success': False,
                'error': error_msg
            }

            response_data = pickle.dumps(error_response)
            client_socket.sendall(response_data + b'<END>')
            logger.debug(f"Sent error response: {error_msg}")

        except Exception as e:
            logger.error(f"Failed to send error response: {e}")

    def get_statistics(self) -> Dict:
        """Get server statistics."""
        cache_hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0

        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate
        }


if __name__ == "__main__":
    print("Testing H-UAV Server...")

    # Create a mock H-UAV model
    class MockHUAV:
        class Config:
            device = "cpu"

        config = Config()

        class MACLayer:
            class NeuralMemory:
                def retrieve_with_cache(self, query):
                    # Mock retrieval
                    return query * 2.0, False

            neural_memory = NeuralMemory()

        mac_layer = MACLayer()

    mock_huav = MockHUAV()

    # Create server
    server = HUAVServer(mock_huav, host="localhost", port=50052)

    # Start server
    server.start()

    import time
    time.sleep(1)

    print(f"Server running: {server.is_running}")
    print(f"Statistics: {server.get_statistics()}")

    # Stop server
    server.stop()

    print("✓ H-UAV Server test completed!")
