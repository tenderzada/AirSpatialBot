"""
H-UAV LoRA Memory Server (MemGen-inspired)

记忆编织器：LoRA 适配器注入冻结的 LLaVA

Architecture:
    L-UAV Client:
      1. Base inference → low confidence
      2. Send L-UAV hidden states → H-UAV

    H-UAV Server (this):
      1. Receive L-UAV hidden states
      2. Process through LoRA-injected LLaVA
         - Frozen base LLaVA
         - LoRA on layers [8, 16, 24] q_proj, v_proj
      3. Generate latent memory tokens [8, 4096]
      4. Return memory tokens → L-UAV

    L-UAV Client:
      5. Concatenate memory || input
      6. Enhanced inference → final answer

Key Design:
- LoRA adapters injected at specific LLaVA layers (NOT standalone network)
- Base LLaVA remains frozen
- Memory generated from L-UAV's hidden states (dynamic, context-aware)
- Returns latent memory sequence (machine-native tokens)

Usage:
    python hierarchical_uav/huav_lora_server_v2.py \
        --model_path /mnt/data/AirSpatialBot \
        --lora_weights ./outputs/lora_injection/lora_adapters.pt \
        --target_layers 8 16 24 \
        --port 8000
"""

import argparse
import torch
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List
import logging
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_uav.mac_memory.lora_injection import (
    LLaVAWithLoRAInjection,
    LoRAInjectionConfig,
    create_huav_memory_weaver
)

# Flask app
app = Flask(__name__)

# Global model
huav_memory_weaver = None


class HUAVMemoryWeaverServer:
    """
    H-UAV Memory Weaver Server.

    记忆编织器：基于 LoRA 注入的动态记忆生成。
    """

    def __init__(
        self,
        model_path: str,
        vision_tower: str,
        lora_weights_path: Optional[str],
        device: str = 'cuda:0',
        lora_rank: int = 8,
        target_layers: List[int] = None
    ):
        """
        Initialize H-UAV memory weaver.

        Args:
            model_path: Path to base LLaVA model
            vision_tower: Path to vision tower
            lora_weights_path: Path to trained LoRA adapters
            device: Device to use
            lora_rank: LoRA rank
            target_layers: Layers to inject LoRA (e.g., [8, 16, 24])
        """
        self.device = device

        logger.info("="*70)
        logger.info("Initializing H-UAV Memory Weaver Server")
        logger.info("="*70)
        logger.info("Architecture: LoRA injected into frozen LLaVA")
        logger.info(f"  Target layers: {target_layers}")
        logger.info(f"  LoRA rank: {lora_rank}")
        logger.info("="*70)

        # Create memory weaver with LoRA injection
        self.memory_weaver = create_huav_memory_weaver(
            base_llava_path=model_path,
            device=device,
            lora_rank=lora_rank,
            target_layers=target_layers
        )

        # Load trained LoRA weights
        if lora_weights_path and os.path.exists(lora_weights_path):
            logger.info(f"Loading LoRA weights from: {lora_weights_path}")
            state_dict = torch.load(lora_weights_path, map_location=device)
            self.memory_weaver.load_state_dict(state_dict, strict=False)
            logger.info("✓ LoRA weights loaded")
        else:
            logger.warning("No LoRA weights provided. Using untrained adapters.")

        # Freeze for inference
        self.memory_weaver.eval()

        # Statistics
        self.request_count = 0
        self.total_inference_time = 0.0

        # Get parameter counts
        total, trainable = self.memory_weaver.get_trainable_parameters_count()
        logger.info(f"\nParameter counts:")
        logger.info(f"  Total: {total:,}")
        logger.info(f"  Trainable (LoRA): {trainable:,} ({trainable/total*100:.2f}%)")
        logger.info("="*70)

    def generate_memory_from_luav_state(
        self,
        luav_hidden_states: torch.Tensor
    ) -> Dict:
        """
        Generate memory tokens based on L-UAV's hidden states.

        这是真正的记忆编织器：
        - 接收 L-UAV 已生成 token 的 hidden states
        - 通过 LoRA-injected LLaVA 处理
        - 动态合成潜变量记忆序列

        Args:
            luav_hidden_states: L-UAV hidden states [batch_size, seq_len, 4096]

        Returns:
            Response dict with memory tokens
        """
        import time
        start_time = time.time()

        self.request_count += 1

        try:
            with torch.no_grad():
                # Generate memory tokens via LoRA-injected LLaVA
                memory_tokens, metrics = self.memory_weaver.generate_memory_tokens(
                    luav_hidden_states
                )

            # memory_tokens: [batch_size, num_tokens, 4096]
            # e.g., [1, 8, 4096]

            if memory_tokens is not None:
                memory_array = memory_tokens.cpu().numpy()
                memory_shape = list(memory_array.shape)
            else:
                memory_array = None
                memory_shape = None

            inference_time = time.time() - start_time
            self.total_inference_time += inference_time

            response = {
                'memory_tokens': memory_array,
                'memory_shape': memory_shape,
                'memory_generated': memory_tokens is not None,
                'inference_time_ms': inference_time * 1000,
                'request_id': self.request_count,
                'metrics': metrics
            }

            logger.info(
                f"Request #{self.request_count}: "
                f"Generated {memory_shape[1] if memory_shape else 0} memory tokens "
                f"in {inference_time*1000:.1f}ms"
            )

            return response

        except Exception as e:
            logger.error(f"Error generating memory: {str(e)}")
            return {
                'error': str(e),
                'request_id': self.request_count
            }

    def generate_memory_from_image_question(
        self,
        image: Image.Image,
        question: str
    ) -> Dict:
        """
        Generate memory from image + question.

        For cases where L-UAV sends image+question instead of hidden states.
        H-UAV encodes to hidden states, then generates memory.

        Args:
            image: Input image
            question: Question text

        Returns:
            Response dict with memory tokens
        """
        # TODO: Encode image + question through LLaVA encoder
        # For now, create dummy hidden states
        batch_size = 1
        seq_len = 64
        luav_hidden_states = torch.randn(
            batch_size, seq_len, 4096,
            device=self.device
        )

        return self.generate_memory_from_luav_state(luav_hidden_states)

    def get_stats(self) -> Dict:
        """Get server statistics."""
        avg_time = (
            self.total_inference_time / self.request_count
            if self.request_count > 0
            else 0.0
        )

        total, trainable = self.memory_weaver.get_trainable_parameters_count()

        return {
            'total_requests': self.request_count,
            'avg_inference_time_ms': avg_time * 1000,
            'total_parameters': total,
            'trainable_parameters': trainable,
            'lora_ratio': f"{trainable/total*100:.2f}%",
            'device': str(self.device),
            'architecture': 'LoRA-injected LLaVA'
        }


# Flask routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': 'H-UAV Memory Weaver (LoRA-injected LLaVA)'
    })


@app.route('/get_memory', methods=['POST'])
def get_memory():
    """
    Generate memory tokens for L-UAV.

    Request format:
    {
        'hidden_states': list (L-UAV hidden states),
        'shape': [batch_size, seq_len, 4096],

        OR

        'image': base64-encoded image,
        'question': str
    }

    Response format:
    {
        'memory_tokens': list,
        'memory_shape': [8, 4096],
        'memory_generated': true
    }
    """
    global huav_memory_weaver

    try:
        data = request.json

        # Option 1: L-UAV sends hidden states directly
        if 'hidden_states' in data:
            hidden_list = data['hidden_states']
            shape = data.get('shape', None)

            # Convert to tensor
            luav_hidden = torch.tensor(hidden_list, dtype=torch.float32)

            if shape:
                luav_hidden = luav_hidden.view(*shape)

            luav_hidden = luav_hidden.to(huav_memory_weaver.device)

            # Generate memory
            response = huav_memory_weaver.generate_memory_from_luav_state(
                luav_hidden
            )

        # Option 2: L-UAV sends image + question (H-UAV encodes)
        elif 'image' in data and 'question' in data:
            # Decode image
            image_b64 = data['image']
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            question = data['question']

            # Generate memory
            response = huav_memory_weaver.generate_memory_from_image_question(
                image, question
            )

        else:
            return jsonify({
                'error': 'Must provide either hidden_states or (image + question)'
            }), 400

        # Serialize memory tokens
        if response.get('memory_tokens') is not None:
            response['memory_tokens'] = response['memory_tokens'].tolist()

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in /get_memory: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get server statistics."""
    global huav_memory_weaver

    if huav_memory_weaver is None:
        return jsonify({'error': 'Server not initialized'}), 500

    return jsonify(huav_memory_weaver.get_stats())


def main():
    global huav_memory_weaver

    parser = argparse.ArgumentParser(
        description='H-UAV Memory Weaver Server (LoRA-injected LLaVA)'
    )

    # Model paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to base LLaVA model')
    parser.add_argument('--vision_tower', type=str, required=True,
                       help='Path to vision tower')
    parser.add_argument('--lora_weights', type=str, default=None,
                       help='Path to trained LoRA adapters')

    # Server config
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host')
    parser.add_argument('--port', type=int, default=8000,
                       help='Server port')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')

    # LoRA config
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--target_layers', type=int, nargs='+',
                       default=[8, 16, 24],
                       help='Layers to inject LoRA (e.g., 8 16 24)')

    args = parser.parse_args()

    # Initialize H-UAV memory weaver
    huav_memory_weaver = HUAVMemoryWeaverServer(
        model_path=args.model_path,
        vision_tower=args.vision_tower,
        lora_weights_path=args.lora_weights,
        device=args.device,
        lora_rank=args.lora_rank,
        target_layers=args.target_layers
    )

    # Start Flask server
    logger.info("=" * 70)
    logger.info("Starting H-UAV Memory Weaver Server")
    logger.info("=" * 70)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Device: {args.device}")
    logger.info(f"LoRA weights: {args.lora_weights or 'None (untrained)'}")
    logger.info("=" * 70)
    logger.info("\nArchitecture:")
    logger.info("  Frozen LLaVA")
    logger.info(f"    ↓ LoRA injected at layers {args.target_layers}")
    logger.info("  q_proj, v_proj ← LoRA adapters")
    logger.info("    ↓")
    logger.info("  Process L-UAV hidden states")
    logger.info("    ↓")
    logger.info("  Generate latent memory tokens")
    logger.info("=" * 70)
    logger.info("\nFlow:")
    logger.info("  L-UAV → Send hidden states → H-UAV")
    logger.info("  H-UAV → LoRA-injected LLaVA → Memory tokens")
    logger.info("  H-UAV → Return memory tokens → L-UAV")
    logger.info("  L-UAV → Enhanced inference with memory")
    logger.info("=" * 70)
    logger.info("\nEndpoints:")
    logger.info(f"  Health: http://{args.host}:{args.port}/health")
    logger.info(f"  Get memory: http://{args.host}:{args.port}/get_memory")
    logger.info(f"  Stats: http://{args.host}:{args.port}/stats")
    logger.info("=" * 70)

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
