"""
H-UAV LoRA Answer Server V3 (Socket-based)

使用10-epoch LoRA生成增强的答案，直接返回给L-UAV
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
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_uav.mac_memory.lora_injection import create_huav_memory_weaver
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN


class HUAVLoRAAnswerServerV3:
    """
    H-UAV LoRA Answer Server V3

    使用10-epoch LoRA生成增强的答案并返回给L-UAV
    """

    def __init__(
        self,
        memory_weaver,
        tokenizer,
        image_processor,
        host: str = "localhost",
        port: int = 50052,  # Different port for V3
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

        logger.info(f"✓ H-UAV LoRA Answer Server V3 started at {self.host}:{self.port}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  LoRA: 10-epoch enhanced answers")
        logger.info(f"  Strategy: Return answers directly to L-UAV")
        logger.info(f"  Ready to serve L-UAV requests")

    def stop(self):
        """停止服务器"""
        if not self.is_running:
            return

        self.is_running = False

        if self.server_socket:
            self.server_socket.close()

        logger.info("✓ H-UAV LoRA Answer Server V3 stopped")
        logger.info(f"  Total requests: {self.total_requests}")
        logger.info(f"  Successful: {self.successful_requests}")

    def _serve(self):
        """主服务循环"""
        logger.info("H-UAV LoRA Answer Server V3 listening for connections...")

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
            question = request.get('question', 'N/A')[:50]  # First 50 chars
            logger.info(f"📥 Received request #{request_id} | Question: {question}...")

            # 生成LoRA增强的答案
            response = self._generate_lora_enhanced_answer(request)

            # 序列化响应
            response_data = pickle.dumps(response)
            client_socket.sendall(response_data + b'<END>')

            self.successful_requests += 1

            if response.get('success'):
                answer = response.get('answer', 'N/A')[:50]
                logger.info(f"✅ Request #{request_id} completed | Answer: {answer}... | LoRA-enhanced | Total: {self.successful_requests}/{self.total_requests}")
            else:
                logger.error(f"❌ Request #{request_id} failed | Error: {response.get('error', 'unknown')}")

        except Exception as e:
            logger.error(f"Error handling client {client_address}: {e}")

            # 发送错误响应
            try:
                error_response = {
                    'success': False,
                    'error': str(e),
                    'answer': None
                }
                error_data = pickle.dumps(error_response)
                client_socket.sendall(error_data + b'<END>')
            except:
                pass

        finally:
            client_socket.close()

    def _generate_lora_enhanced_answer(self, request: Dict) -> Dict:
        """
        使用10-epoch LoRA生成增强的答案

        Strategy:
        1. 提取图像和问题
        2. 通过LoRA增强的H-UAV模型推理
        3. 返回答案文本
        """
        try:
            # 提取图像和问题
            image_bytes = request.get('image')
            question = request.get('question', '')

            if image_bytes is None:
                raise ValueError("No image provided in request")

            # 解码图像
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            logger.debug(f"  Image decoded: {image.size}")

            # 预处理图像
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.to(self.device).to(torch.float16)
            logger.debug(f"  Image preprocessed: {image_tensor.shape}")

            # 准备prompt
            question_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], question_with_image)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)

            logger.info(f"  🧠 Generating answer with 10-epoch LoRA...")

            # 使用LoRA增强的H-UAV模型生成答案
            with torch.inference_mode():
                output_ids = self.memory_weaver.base_model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=512,
                    use_cache=True
                )

            # 解码答案
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # 提取答案
            if "ASSISTANT:" in outputs:
                answer = outputs.split("ASSISTANT:")[-1].strip()
            else:
                answer = outputs

            logger.info(f"  ✓ LoRA-enhanced answer generated: {answer[:50]}...")

            return {
                'success': True,
                'answer': answer,
                'lora_enhanced': True,
                'lora_epochs': 10
            }

        except Exception as e:
            logger.error(f"Error generating LoRA-enhanced answer: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'answer': None
            }


def main():
    parser = argparse.ArgumentParser(description='H-UAV LoRA Answer Server V3')

    # Model paths
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vision_tower', type=str, default=None)
    parser.add_argument('--lora_weights', type=str, required=True,
                       help='Path to H-UAV 10-epoch LoRA weights')

    # LoRA config
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--target_layers', type=int, nargs='+', default=[8, 16, 24])

    # Server config
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=50052)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--load_8bit', action='store_true')

    args = parser.parse_args()

    # Load memory weaver with 10-epoch LoRA
    logger.info("Loading H-UAV memory weaver with 10-epoch LoRA...")
    memory_weaver = create_huav_memory_weaver(
        base_llava_path=args.model_path,
        device=args.device,
        lora_rank=args.lora_rank,
        target_layers=args.target_layers,
        load_8bit=args.load_8bit,
        vision_tower=args.vision_tower
    )

    # Load 10-epoch LoRA weights
    logger.info(f"Loading H-UAV LoRA weights (10 epochs) from {args.lora_weights}...")
    checkpoint = torch.load(args.lora_weights, map_location=args.device)
    memory_weaver.load_lora_state(checkpoint)
    logger.info("✓ H-UAV LoRA weights loaded (10-epoch enhanced)")

    # Get tokenizer and image_processor
    tokenizer = memory_weaver.tokenizer
    image_processor = memory_weaver.image_processor

    # Create server
    server = HUAVLoRAAnswerServerV3(
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
