"""
H-UAV Server with MemGen-style Memory Weaver

This server uses true MemGen approach with learnable query latents.
"""

import os
import sys
import argparse
import base64
import io
import time
import torch
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hierarchical_uav.mac_memory.memgen_weaver import MemGenWeaver, MemGenWeaverConfig


app = Flask(__name__)

# Global variables
model = None
processor = None
weaver = None
device = None
stats = {
    'total_requests': 0,
    'total_inference_time': 0.0
}


def load_model(args):
    """Load LLaVA model and MemGen Weaver."""
    global model, processor, weaver, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load LLaVA model
    print(f"\nLoading LLaVA model from {args.model_path}...")

    # Load LLaVA model (handle both LLaVA and standard LLM)
    try:
        # Try loading as LLaVA first
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        print("Loaded as LLaVA model")
        # For LLaVA, we work with the language model part
        if hasattr(model, 'language_model'):
            model = model.language_model
            print("Using language_model component")
    except Exception as e:
        print(f"LLaVA loading failed ({e}), trying AutoModel...")
        # Fallback to AutoModel
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True
        )
        print("Loaded with AutoModel")

    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # Load MemGen Weaver
    print(f"\nLoading MemGen Weaver...")
    if args.weaver_checkpoint:
        print(f"Loading from checkpoint: {args.weaver_checkpoint}")
        weaver = MemGenWeaver.load_adapter(model, args.weaver_checkpoint)
    else:
        print("Creating new MemGen Weaver (untrained)")
        config = MemGenWeaverConfig(
            hidden_size=model.config.hidden_size,
            num_memory_tokens=args.num_memory_tokens,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"]
        )
        weaver = MemGenWeaver(model, config)

    weaver.to(device)
    weaver.eval()

    print("\n" + "="*70)
    print("H-UAV MemGen Server Ready!")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Memory Tokens: {weaver.num_memory_tokens}")
    print(f"LoRA Rank: {weaver.config.lora_rank}")
    print(f"Device: {device}")
    print("="*70)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': 'H-UAV MemGen Server',
        'memory_tokens': weaver.num_memory_tokens if weaver else 0
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get server statistics."""
    avg_time = (stats['total_inference_time'] / stats['total_requests']
                if stats['total_requests'] > 0 else 0)

    return jsonify({
        'total_requests': stats['total_requests'],
        'avg_inference_time_ms': avg_time * 1000,
        'memory_tokens': weaver.num_memory_tokens if weaver else 0,
        'device': str(device)
    })


@app.route('/get_memory', methods=['POST'])
def get_memory():
    """
    Generate memory tokens using MemGen Weaver.

    Request JSON:
        {
            "image": "base64_encoded_image",
            "question": "What is the depth of this object?"
        }

    Response JSON:
        {
            "memory_tokens": [[...], [...], ...],  # [num_tokens, hidden_size]
            "memory_shape": [num_tokens, hidden_size],
            "inference_time_ms": 12.3,
            "method": "memgen"
        }
    """
    global stats

    try:
        start_time = time.time()
        data = request.get_json()

        # Parse request
        image_b64 = data.get('image')
        question = data.get('question', '')

        if not image_b64:
            return jsonify({'error': 'No image provided'}), 400

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # === MemGen Memory Generation ===
        with torch.no_grad():
            # Step 1: Encode image + question using LLaVA processor
            # This creates the input that query latents will attend to
            inputs = processor(
                text=question,
                images=image,
                return_tensors='pt',
                padding=True
            ).to(device)

            # Step 2: Get input embeddings
            # These are what the query latents will be concatenated with
            input_ids = inputs.get('input_ids')
            if input_ids is not None:
                inputs_embeds = model.get_input_embeddings()(input_ids)
            else:
                # If processor returns embeddings directly
                inputs_embeds = inputs.get('inputs_embeds')

            attention_mask = inputs.get('attention_mask')

            # Step 3: Generate memory tokens using MemGen Weaver
            # This is the core MemGen approach:
            # - Query latents are concatenated with inputs
            # - Processed through LoRA-enhanced Transformer
            # - Enhanced query latents extracted as memory
            memory_tokens = weaver(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )

            # Move to CPU and convert to list
            memory_tokens_cpu = memory_tokens.squeeze(0).cpu().float().numpy()

        # Compute inference time
        inference_time = time.time() - start_time

        # Update stats
        stats['total_requests'] += 1
        stats['total_inference_time'] += inference_time

        # Prepare response
        response = {
            'memory_tokens': memory_tokens_cpu.tolist(),
            'memory_shape': list(memory_tokens_cpu.shape),
            'inference_time_ms': inference_time * 1000,
            'method': 'memgen',
            'num_query_latents': weaver.num_memory_tokens
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        print(f"Error in /get_memory: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/get_memory_batch', methods=['POST'])
def get_memory_batch():
    """
    Generate memory tokens for multiple inputs (batch processing).

    Request JSON:
        {
            "batch": [
                {"image": "base64_1", "question": "question_1"},
                {"image": "base64_2", "question": "question_2"},
                ...
            ]
        }
    """
    try:
        start_time = time.time()
        data = request.get_json()
        batch_data = data.get('batch', [])

        if not batch_data:
            return jsonify({'error': 'No batch data provided'}), 400

        # Process each item
        all_memory_tokens = []

        for item in batch_data:
            image_b64 = item.get('image')
            question = item.get('question', '')

            # Decode image
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Generate memory
            with torch.no_grad():
                inputs = processor(
                    text=question,
                    images=image,
                    return_tensors='pt',
                    padding=True
                ).to(device)

                input_ids = inputs.get('input_ids')
                if input_ids is not None:
                    inputs_embeds = model.get_input_embeddings()(input_ids)
                else:
                    inputs_embeds = inputs.get('inputs_embeds')

                attention_mask = inputs.get('attention_mask')

                memory_tokens = weaver(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask
                )

                memory_tokens_cpu = memory_tokens.squeeze(0).cpu().float().numpy()
                all_memory_tokens.append(memory_tokens_cpu.tolist())

        inference_time = time.time() - start_time

        stats['total_requests'] += len(batch_data)
        stats['total_inference_time'] += inference_time

        return jsonify({
            'batch_memory_tokens': all_memory_tokens,
            'batch_size': len(batch_data),
            'total_inference_time_ms': inference_time * 1000,
            'avg_time_per_item_ms': (inference_time / len(batch_data)) * 1000
        })

    except Exception as e:
        import traceback
        print(f"Error in /get_memory_batch: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description='H-UAV MemGen Server')

    # Model arguments
    parser.add_argument('--model_path', type=str, default='/mnt/data/AirSpatialBot',
                        help='Path to LLaVA model')
    parser.add_argument('--weaver_checkpoint', type=str, default=None,
                        help='Path to MemGen Weaver checkpoint (optional)')

    # Weaver arguments (if creating new)
    parser.add_argument('--num_memory_tokens', type=int, default=8,
                        help='Number of memory tokens (query latents)')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=32.0,
                        help='LoRA alpha')

    # Server arguments
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host')
    parser.add_argument('--port', type=int, default=8000,
                        help='Server port')

    args = parser.parse_args()

    # Load model
    load_model(args)

    # Start server
    print(f"\nStarting server on {args.host}:{args.port}...")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
