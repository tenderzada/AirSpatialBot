"""
H-UAV LoRA Memory Server (Socket-based)

使用 Socket 通信协议（与之前的 gRPC server 风格一致）
"""

import argparse
import torch
import socket
import pickle
import threading
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import logging
import numpy as np
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_uav.mac_memory.lora_injection import create_huav_memory_weaver


class HUAVLoRAServer:
    """
    H-UAV LoRA Memory Server (Socket-based)

    接收L-UAV请求，返回LoRA生成的memory tokens
    """

    def __init__(
        self,
        memory_weaver,
        tokenizer,
        image_processor,
        host: str = "localhost",
        port: int = 50051,
        max_connections: int = 10,
        device: str = 'cuda:0'
    ):
        self.memory_weaver = memory_weaver
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.device = device

        self.server_socket = None
        self.is_running = False
        self.server_thread = None

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0

    def start(self):
        """启动服务器"""
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

        logger.info(f"✓ H-UAV LoRA Server started at {self.host}:{self.port}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Ready to serve L-UAV requests")

    def stop(self):
        """停止服务器"""
        if not self.is_running:
            return

        self.is_running = False

        if self.server_socket:
            self.server_socket.close()

        logger.info("✓ H-UAV LoRA Server stopped")
        logger.info(f"  Total requests: {self.total_requests}")
        logger.info(f"  Successful: {self.successful_requests}")

    def _serve(self):
        """主服务循环"""
        logger.info("H-UAV LoRA Server listening for connections...")

        while self.is_running:
            try:
                client_socket, client_address = self.server_socket.accept()
                logger.debug(f"Connection from {client_address}")

                # 在新线程中处理请求
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()

            except OSError:
                # Socket closed
                break
            except Exception as e:
                if self.is_running:
                    logger.error(f"Error accepting connection: {e}")

    def _handle_client(self, client_socket: socket.socket, client_address):
        """处理单个客户端请求"""
        try:
            # 接收数据
            data = b''
            while True:
                chunk = client_socket.recv(4096)
                data += chunk

                if b'<END>' in data:
                    data = data.replace(b'<END>', b'')
                    break

                if not chunk and len(data) == 0:
                    break

            if not data:
                logger.warning("Received empty request")
                return

            # 反序列化请求
            request = pickle.loads(data)
            self.total_requests += 1

            request_id = request.get('request_id', self.total_requests)
            logger.debug(f"Processing request #{request_id}")

            # 生成memory tokens
            response = self._generate_memory(request)

            # 序列化响应
            response_data = pickle.dumps(response)
            client_socket.sendall(response_data + b'<END>')

            self.successful_requests += 1
            logger.debug(f"Sent response for request #{request_id}")

        except Exception as e:
            logger.error(f"Error handling client {client_address}: {e}")

            # 发送错误响应
            try:
                error_response = {
                    'success': False,
                    'error': str(e),
                    'memory_tokens': None
                }
                error_data = pickle.dumps(error_response)
                client_socket.sendall(error_data + b'<END>')
            except:
                pass

        finally:
            client_socket.close()

    def _generate_memory(self, request: Dict) -> Dict:
        """
        生成memory tokens

        Request可以包含：
        - image (bytes): 图像数据
        - question (str): 问题文本
        """
        try:
            # 提取图像和问题
            image_bytes = request.get('image')
            question = request.get('question', '')

            if image_bytes is None:
                raise ValueError("No image provided in request")

            # 解码图像
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # 预处理图像
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.to(self.device).to(torch.float16)

            # 通过LoRA增强的vision tower提取特征
            with torch.inference_mode():
                # 获取vision tower
                vision_tower = self.memory_weaver.base_model.get_model().get_vision_tower()

                # 提取vision features
                # Vision tower输出 [batch, num_patches, hidden_dim]
                vision_features = vision_tower(image_tensor)

                # 如果输出是tuple，取第一个元素
                if isinstance(vision_features, tuple):
                    vision_features = vision_features[0]

                # vision_features shape: [1, num_patches, hidden_dim]
                # 我们需要选择代表性的tokens作为memory

                # 策略1: 取前8个patch tokens
                # 或者策略2: 均匀采样8个tokens
                # 或者策略3: 使用CLS token + 采样tokens

                batch_size, num_patches, hidden_dim = vision_features.shape

                if num_patches >= 8:
                    # 均匀采样8个tokens
                    indices = torch.linspace(0, num_patches - 1, 8, dtype=torch.long)
                    memory_tokens = vision_features[0, indices, :]  # [8, hidden_dim]
                else:
                    # 如果patch数量少于8，重复填充
                    memory_tokens = vision_features[0]  # [num_patches, hidden_dim]
                    while memory_tokens.shape[0] < 8:
                        memory_tokens = torch.cat([memory_tokens, memory_tokens], dim=0)
                    memory_tokens = memory_tokens[:8, :]  # [8, hidden_dim]

                # 转换为numpy
                memory_tokens_np = memory_tokens.cpu().float().numpy()

            logger.debug(f"Generated memory tokens: shape {memory_tokens_np.shape}")

            return {
                'success': True,
                'memory_tokens': memory_tokens_np,
                'memory_shape': list(memory_tokens_np.shape),
                'cache_hit': False
            }

        except Exception as e:
            logger.error(f"Error generating memory: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'memory_tokens': None
            }


def main():
    parser = argparse.ArgumentParser(description='H-UAV LoRA Memory Server')

    # Model paths
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vision_tower', type=str, default=None)
    parser.add_argument('--lora_weights', type=str, required=True)

    # LoRA config
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--target_layers', type=int, nargs='+', default=[8, 16, 24])

    # Server config
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--load_8bit', action='store_true')

    args = parser.parse_args()

    # Load memory weaver
    logger.info("Loading H-UAV memory weaver...")
    memory_weaver = create_huav_memory_weaver(
        base_llava_path=args.model_path,
        device=args.device,
        lora_rank=args.lora_rank,
        target_layers=args.target_layers,
        load_8bit=args.load_8bit,
        vision_tower=args.vision_tower
    )

    # Load LoRA weights
    logger.info(f"Loading LoRA weights from {args.lora_weights}...")
    checkpoint = torch.load(args.lora_weights, map_location=args.device)
    memory_weaver.load_lora_state(checkpoint)
    logger.info("✓ LoRA weights loaded")

    # Get tokenizer and image_processor
    tokenizer = memory_weaver.tokenizer
    image_processor = memory_weaver.image_processor

    # Create server
    server = HUAVLoRAServer(
        memory_weaver=memory_weaver,
        tokenizer=tokenizer,
        image_processor=image_processor,
        host=args.host,
        port=args.port,
        device=args.device
    )

    # Start server
    server.start()

    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        server.stop()


if __name__ == '__main__':
    main()
