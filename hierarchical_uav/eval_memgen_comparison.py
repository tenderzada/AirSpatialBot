"""
Evaluation script: Compare H-UAV Standalone with/without MemGen

Evaluates performance on first 500 samples with:
1. H-UAV without MemGen (baseline)
2. H-UAV with MemGen (trained query latents)

Metrics:
- Accuracy
- BLEU score
- Inference time
"""

import os
import sys
import json
import argparse
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hierarchical_uav.mac_memory.memgen_weaver import MemGenWeaver


def compute_bleu(prediction, reference):
    """Simple BLEU-like score (word overlap)."""
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())

    if len(ref_words) == 0:
        return 0.0

    overlap = len(pred_words & ref_words)
    return overlap / len(ref_words)


def compute_exact_match(prediction, reference):
    """Compute exact match."""
    return prediction.strip().lower() == reference.strip().lower()


def evaluate_without_memgen(model, tokenizer, data, max_samples=500, device='cuda'):
    """Evaluate H-UAV without MemGen (baseline)."""
    print("\n" + "="*70)
    print("Evaluating H-UAV WITHOUT MemGen (Baseline)")
    print("="*70)

    results = []
    total_time = 0.0

    for i, sample in enumerate(tqdm(data[:max_samples], desc="Baseline")):
        question = sample.get('question', sample.get('conversations', [{}])[0].get('value', ''))
        ground_truth = sample.get('answer', sample.get('conversations', [{}])[1].get('value', ''))

        # Tokenize input
        inputs = tokenizer(
            question,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(device)

        # Generate answer
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        inference_time = time.time() - start_time
        total_time += inference_time

        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove input from prediction
        if question in prediction:
            prediction = prediction.replace(question, '').strip()

        # Compute metrics
        bleu = compute_bleu(prediction, ground_truth)
        exact_match = compute_exact_match(prediction, ground_truth)

        results.append({
            'question': question,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'bleu': bleu,
            'exact_match': exact_match,
            'time': inference_time
        })

    # Summary statistics
    avg_bleu = sum(r['bleu'] for r in results) / len(results)
    accuracy = sum(r['exact_match'] for r in results) / len(results)
    avg_time = total_time / len(results)

    print(f"\n📊 Results (WITHOUT MemGen):")
    print(f"  Samples: {len(results)}")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Average BLEU: {avg_bleu:.4f}")
    print(f"  Average Time: {avg_time * 1000:.2f} ms")

    return results, {
        'accuracy': accuracy,
        'bleu': avg_bleu,
        'avg_time': avg_time
    }


def evaluate_with_memgen(model, tokenizer, weaver, data, max_samples=500, device='cuda'):
    """Evaluate H-UAV with MemGen."""
    print("\n" + "="*70)
    print("Evaluating H-UAV WITH MemGen")
    print("="*70)

    weaver.eval()
    results = []
    total_time = 0.0

    for i, sample in enumerate(tqdm(data[:max_samples], desc="With MemGen")):
        question = sample.get('question', sample.get('conversations', [{}])[0].get('value', ''))
        ground_truth = sample.get('answer', sample.get('conversations', [{}])[1].get('value', ''))

        start_time = time.time()

        with torch.no_grad():
            # Step 1: Tokenize and get embeddings
            inputs = tokenizer(
                question,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(device)

            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Get input embeddings
            inputs_embeds = model.get_input_embeddings()(input_ids)

            # Step 2: Generate memory tokens using MemGen Weaver
            memory_tokens = weaver(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )

            # Step 3: Concatenate memory with input embeddings
            enhanced_embeds = torch.cat([memory_tokens, inputs_embeds], dim=1)

            # Update attention mask
            memory_mask = torch.ones(
                1, memory_tokens.size(1),
                dtype=attention_mask.dtype,
                device=device
            )
            enhanced_mask = torch.cat([memory_mask, attention_mask], dim=1)

            # Step 4: Generate answer with enhanced input
            outputs = model.generate(
                inputs_embeds=enhanced_embeds,
                attention_mask=enhanced_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        inference_time = time.time() - start_time
        total_time += inference_time

        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove input from prediction
        if question in prediction:
            prediction = prediction.replace(question, '').strip()

        # Compute metrics
        bleu = compute_bleu(prediction, ground_truth)
        exact_match = compute_exact_match(prediction, ground_truth)

        results.append({
            'question': question,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'bleu': bleu,
            'exact_match': exact_match,
            'time': inference_time
        })

    # Summary statistics
    avg_bleu = sum(r['bleu'] for r in results) / len(results)
    accuracy = sum(r['exact_match'] for r in results) / len(results)
    avg_time = total_time / len(results)

    print(f"\n📊 Results (WITH MemGen):")
    print(f"  Samples: {len(results)}")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Average BLEU: {avg_bleu:.4f}")
    print(f"  Average Time: {avg_time * 1000:.2f} ms")

    return results, {
        'accuracy': accuracy,
        'bleu': avg_bleu,
        'avg_time': avg_time
    }


def plot_comparison(baseline_stats, memgen_stats, output_dir):
    """Plot comparison charts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Accuracy comparison
    ax = axes[0]
    methods = ['Baseline\n(No MemGen)', 'With MemGen']
    accuracies = [baseline_stats['accuracy'] * 100, memgen_stats['accuracy'] * 100]
    colors = ['#3498db', '#2ecc71']

    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(accuracies) * 1.2])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # Improvement annotation
    improvement = memgen_stats['accuracy'] - baseline_stats['accuracy']
    if improvement != 0:
        ax.text(0.5, max(accuracies) * 1.1,
                f"{'↑' if improvement > 0 else '↓'} {abs(improvement)*100:.1f}%",
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # 2. BLEU score comparison
    ax = axes[1]
    bleu_scores = [baseline_stats['bleu'], memgen_stats['bleu']]

    bars = ax.bar(methods, bleu_scores, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('BLEU Score', fontsize=11, fontweight='bold')
    ax.set_title('BLEU Score Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(bleu_scores) * 1.2])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')

    # Improvement annotation
    improvement = memgen_stats['bleu'] - baseline_stats['bleu']
    if improvement != 0:
        ax.text(0.5, max(bleu_scores) * 1.1,
                f"{'↑' if improvement > 0 else '↓'} {abs(improvement):.3f}",
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # 3. Inference time comparison
    ax = axes[2]
    times = [baseline_stats['avg_time'] * 1000, memgen_stats['avg_time'] * 1000]

    bars = ax.bar(methods, times, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Inference Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Inference Time Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(times) * 1.2])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} ms',
                ha='center', va='bottom', fontweight='bold')

    # Slowdown annotation
    slowdown = memgen_stats['avg_time'] - baseline_stats['avg_time']
    if slowdown != 0:
        ax.text(0.5, max(times) * 1.1,
                f"{'↑' if slowdown > 0 else '↓'} {abs(slowdown)*1000:.1f} ms",
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'memgen_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Comparison chart saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare H-UAV with/without MemGen")

    parser.add_argument('--model_path', type=str, default='/mnt/data/AirSpatialBot',
                        help='Path to LLaVA model')
    parser.add_argument('--eval_data', type=str, required=True,
                        help='Path to evaluation data (JSONL)')
    parser.add_argument('--memgen_checkpoint', type=str, required=True,
                        help='Path to trained MemGen checkpoint')
    parser.add_argument('--max_samples', type=int, default=500,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--output_dir', type=str, default='./outputs/memgen_eval',
                        help='Output directory for results')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"\nLoading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation data
    print(f"\nLoading evaluation data from {args.eval_data}...")
    with open(args.eval_data, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} samples")
    print(f"Will evaluate on first {min(args.max_samples, len(data))} samples")

    # Evaluate baseline (without MemGen)
    baseline_results, baseline_stats = evaluate_without_memgen(
        model, tokenizer, data, args.max_samples, device
    )

    # Load MemGen weaver
    print(f"\nLoading MemGen Weaver from {args.memgen_checkpoint}...")
    weaver = MemGenWeaver.load_adapter(model, args.memgen_checkpoint)
    weaver.to(device)
    weaver.eval()

    # Evaluate with MemGen
    memgen_results, memgen_stats = evaluate_with_memgen(
        model, tokenizer, weaver, data, args.max_samples, device
    )

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Metric':<20} {'Baseline':<15} {'With MemGen':<15} {'Improvement':<15}")
    print("-" * 70)

    acc_imp = (memgen_stats['accuracy'] - baseline_stats['accuracy']) * 100
    print(f"{'Accuracy':<20} {baseline_stats['accuracy']*100:>7.2f}%      "
          f"{memgen_stats['accuracy']*100:>7.2f}%      {acc_imp:>+7.2f}%")

    bleu_imp = memgen_stats['bleu'] - baseline_stats['bleu']
    print(f"{'BLEU Score':<20} {baseline_stats['bleu']:>7.4f}        "
          f"{memgen_stats['bleu']:>7.4f}        {bleu_imp:>+7.4f}")

    time_diff = (memgen_stats['avg_time'] - baseline_stats['avg_time']) * 1000
    print(f"{'Avg Time (ms)':<20} {baseline_stats['avg_time']*1000:>7.1f}        "
          f"{memgen_stats['avg_time']*1000:>7.1f}        {time_diff:>+7.1f}")

    print("="*70)

    # Save results
    results = {
        'baseline': {
            'stats': baseline_stats,
            'results': baseline_results
        },
        'memgen': {
            'stats': memgen_stats,
            'results': memgen_results
        },
        'config': vars(args)
    }

    results_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")

    # Plot comparison
    plot_comparison(baseline_stats, memgen_stats, args.output_dir)

    print(f"\n✓ Evaluation complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
