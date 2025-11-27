"""
SQA Evaluation with LoRA Memory Weaver (Socket-based)

L-UAV 通过 Socket 连接到 H-UAV 获取 memory tokens
"""

import argparse
import torch
import json
import os
import sys
import re
import socket
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN


class LUAVSocketClient:
    """
    L-UAV Socket Client for H-UAV LoRA Memory Server
    """

    def __init__(
        self,
        huav_address: str = "localhost:50051",
        timeout: float = 30.0
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

        logger.info(f"✓ L-UAV Socket Client initialized for H-UAV at {self.host}:{self.port}")

    def request_memory(
        self,
        image: Image.Image,
        question: str
    ) -> Optional[np.ndarray]:
        """
        Request memory tokens from H-UAV

        Returns:
            Memory tokens [8, 4096] or None
        """
        self.request_counter += 1

        try:
            # Create socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(self.timeout)

            # Connect to H-UAV
            client_socket.connect((self.host, self.port))

            # Prepare request
            # Convert image to bytes
            import io
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            image_bytes = buf.getvalue()

            request = {
                'image': image_bytes,
                'question': question,
                'request_id': self.request_counter
            }

            # Serialize and send
            request_data = pickle.dumps(request)
            client_socket.sendall(request_data + b'<END>')

            logger.debug(f"Sent memory request #{self.request_counter} to H-UAV")

            # Receive response
            data = b''
            while True:
                chunk = client_socket.recv(4096)
                data += chunk

                if b'<END>' in data:
                    data = data.replace(b'<END>', b'')
                    break

                if not chunk and len(data) == 0:
                    break

            # Deserialize response
            if not data:
                logger.warning("Received empty response from H-UAV")
                return None

            response = pickle.loads(data)

            # Check success
            if not response.get('success', True):
                error_msg = response.get('error', 'Unknown error')
                logger.warning(f"H-UAV returned error: {error_msg}")
                return None

            memory_tokens = response.get('memory_tokens')
            if memory_tokens is not None:
                logger.debug(f"Received memory tokens: shape {memory_tokens.shape}")
                return memory_tokens

            return None

        except socket.timeout:
            logger.error(f"Request #{self.request_counter} timed out")
            return None

        except Exception as e:
            logger.error(f"Request #{self.request_counter} failed: {e}")
            return None

        finally:
            try:
                client_socket.close()
            except:
                pass


# ... (复用之前的 parse_numeric_answer 和 parse_ground_truth 函数)
def parse_numeric_answer(answer: str) -> Optional[float]:
    """Extract numeric value from answer."""
    answer = re.sub(r'(meters?|millimeters?|mm|m|cm)', '', answer, flags=re.IGNORECASE)
    match = re.search(r'-?\d+\.?\d*', answer)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def parse_ground_truth(ground_truth, qtype: str) -> Optional[float]:
    """Parse ground truth value."""
    if isinstance(ground_truth, (int, float)):
        return float(ground_truth)

    gt_str = str(ground_truth)

    if qtype == 'size':
        match = re.search(r'<size>([\d.,\s]+)</size>', gt_str)
        if not match:
            match = re.search(r'size=([\d.,\s]+)', gt_str, re.IGNORECASE)

        if match:
            numbers_str = match.group(1)
            numbers = re.findall(r'[\d.]+', numbers_str)
            if numbers:
                try:
                    return float(numbers[0])
                except ValueError:
                    pass

    try:
        return float(gt_str)
    except ValueError:
        numbers = re.findall(r'[\d.]+', gt_str)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass

    return None


class LUAVWithLoRAMemory:
    """
    L-UAV with Socket-based H-UAV collaboration
    """

    def __init__(
        self,
        model_path: str,
        vision_tower: Optional[str] = None,
        device: str = 'cuda:1',
        huav_address: Optional[str] = None,
        confidence_threshold: float = 0.5,
        load_8bit: bool = False
    ):
        self.device = device
        self.huav_address = huav_address
        self.confidence_threshold = confidence_threshold

        # Load L-UAV model
        logger.info(f"Loading L-UAV model from {model_path}...")
        model_name = get_model_name_from_path(model_path)

        if 'cuda' in str(device):
            device_id = str(device).split(':')[-1] if ':' in str(device) else '0'
            device_map = {"": int(device_id)}
        else:
            device_map = "auto"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=load_8bit,
            device_map=device_map
        )

        if vision_tower and hasattr(self.model.config, 'mm_vision_tower'):
            self.model.config.mm_vision_tower = vision_tower

        logger.info(f"✓ L-UAV model loaded on {device}")

        # Create H-UAV client if address provided
        self.huav_client = None
        if self.huav_address:
            self.huav_client = LUAVSocketClient(huav_address=self.huav_address)
            logger.info(f"✓ Will use H-UAV memory from {self.huav_address}")
            logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        else:
            logger.info("⚠️  Running in standalone mode (no H-UAV)")

        self.stats = {
            'total_queries': 0,
            'huav_requests': 0,
            'standalone_inferences': 0
        }

    def infer(
        self,
        image: Image.Image,
        question: str,
        use_memory: bool = False,
        memory_tokens: Optional[np.ndarray] = None
    ) -> Tuple[str, float]:
        """Perform inference."""
        # Prepare image
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(self.device).to(torch.float16)

        # Prepare prompt
        question_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], question_with_image)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)

        # Generate
        with torch.inference_mode():
            # TODO: Integrate memory tokens if provided
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
                use_cache=True
            )

        # Decode
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if "ASSISTANT:" in outputs:
            answer = outputs.split("ASSISTANT:")[-1].strip()
        else:
            answer = outputs

        # Simple confidence estimation
        confidence = 0.7 if not use_memory else 0.9

        return answer, confidence

    def evaluate_with_huav(
        self,
        image: Image.Image,
        question: str
    ) -> Tuple[str, float, bool]:
        """Evaluate with H-UAV collaboration."""
        self.stats['total_queries'] += 1

        # Base inference
        answer_base, confidence_base = self.infer(image, question, use_memory=False)

        # Check if H-UAV help is needed
        if self.huav_client and confidence_base < self.confidence_threshold:
            logger.debug(f"Low confidence ({confidence_base:.3f}), requesting H-UAV memory...")

            # Request memory from H-UAV
            memory_tokens = self.huav_client.request_memory(image, question)

            if memory_tokens is not None:
                # Enhanced inference with memory
                answer_enhanced, confidence_enhanced = self.infer(
                    image, question, use_memory=True, memory_tokens=memory_tokens
                )
                self.stats['huav_requests'] += 1
                return answer_enhanced, confidence_enhanced, True

        # Standalone inference
        self.stats['standalone_inferences'] += 1
        return answer_base, confidence_base, False


# ... (evaluation and main functions类似eval_sqa_lora.py)

if __name__ == '__main__':
    logger.info("Use start_luav_with_huav_socket.sh script")
