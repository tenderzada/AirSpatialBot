"""
Memory Ablation Study

Run controlled experiments to isolate memory's contribution:
1. Inference with memory (different weights)
2. Inference without memory
3. Compare outputs to understand memory's role
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class MemoryAblationExperiment:
    """
    Controlled experiment to study memory's influence.

    Compares:
    - No memory (baseline)
    - With memory, weight=0.3
    - With memory, weight=0.5
    - With memory, weight=0.7
    """

    def __init__(self, luav_model, samples: List[Dict], output_dir: str = "./outputs/ablation"):
        """
        Args:
            luav_model: L-UAV model instance
            samples: List of test samples with H-UAV memory available
            output_dir: Where to save ablation results
        """
        self.model = luav_model
        self.samples = samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_ablation(self):
        """
        Run ablation study across different memory weight settings.
        """
        print("="*70)
        print("Memory Ablation Study")
        print("="*70)

        # Different configurations to test
        configs = [
            {'name': 'no_memory', 'use_memory': False, 'weight': 0.0},
            {'name': 'memory_w03', 'use_memory': True, 'weight': 0.3},
            {'name': 'memory_w05', 'use_memory': True, 'weight': 0.5},
            {'name': 'memory_w07', 'use_memory': True, 'weight': 0.7},
        ]

        results_by_config = {cfg['name']: [] for cfg in configs}

        print(f"\nRunning {len(configs)} configurations on {len(self.samples)} samples...")

        for i, sample in enumerate(self.samples):
            if i % 10 == 0:
                print(f"  Processing sample {i+1}/{len(self.samples)}...")

            # Extract sample info
            input_ids = sample['input_ids']
            image_tensor = sample['image_tensor']
            memory_features = sample.get('memory_features')
            question = sample.get('question', '')
            ground_truth = sample.get('ground_truth', '')

            if memory_features is None:
                print(f"    Warning: Sample {i} has no memory, skipping")
                continue

            # Test each configuration
            for config in configs:
                with torch.no_grad():
                    if config['use_memory']:
                        # Generate with memory
                        output_ids = self.model.generate_with_memory(
                            input_ids=input_ids,
                            images=image_tensor,
                            memory_features=memory_features,
                            memory_weight=config['weight'],
                            max_new_tokens=512,
                            do_sample=False,
                            num_beams=1
                        )
                    else:
                        # Generate without memory (baseline)
                        output_ids = self.model.llava_model.generate(
                            inputs=input_ids,
                            images=image_tensor,
                            max_new_tokens=512,
                            do_sample=False,
                            num_beams=1
                        )

                    # Decode answer
                    if output_ids.shape[1] <= input_ids.shape[1]:
                        answer = self.model.tokenizer.decode(
                            output_ids[0],
                            skip_special_tokens=True
                        ).strip()
                    else:
                        answer = self.model.tokenizer.decode(
                            output_ids[0, input_ids.shape[1]:],
                            skip_special_tokens=True
                        ).strip()

                    # Store result
                    results_by_config[config['name']].append({
                        'sample_id': i,
                        'question': question,
                        'ground_truth': ground_truth,
                        'answer': answer,
                        'config': config['name'],
                        'memory_weight': config['weight']
                    })

        # Save results
        self._save_results(results_by_config)

        # Analyze results
        self._analyze_results(results_by_config)

        return results_by_config

    def _save_results(self, results_by_config: Dict[str, List[Dict]]):
        """Save ablation results to disk."""
        for config_name, results in results_by_config.items():
            output_file = self.output_dir / f"{config_name}_results.jsonl"
            with open(output_file, 'w') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')
            print(f"\n✓ Saved {config_name} results to {output_file}")

    def _analyze_results(self, results_by_config: Dict[str, List[Dict]]):
        """Analyze and compare results across configurations."""
        print("\n" + "="*70)
        print("Ablation Analysis: How Memory Weight Affects Outputs")
        print("="*70)

        # Get baseline (no memory)
        baseline = results_by_config['no_memory']

        # Compare each memory config to baseline
        for config_name in ['memory_w03', 'memory_w05', 'memory_w07']:
            memory_results = results_by_config[config_name]

            # Count how many answers changed
            changed_count = 0
            same_count = 0

            for baseline_r, memory_r in zip(baseline, memory_results):
                if baseline_r['answer'] != memory_r['answer']:
                    changed_count += 1
                else:
                    same_count += 1

            total = changed_count + same_count
            change_rate = changed_count / total * 100 if total > 0 else 0

            print(f"\n{config_name.upper()}:")
            print(f"  Changed from baseline: {changed_count}/{total} ({change_rate:.1f}%)")
            print(f"  Same as baseline: {same_count}/{total} ({100-change_rate:.1f}%)")

            if changed_count > 0:
                print(f"  → Memory HAS INFLUENCE (changes {change_rate:.1f}% of outputs)")
            else:
                print(f"  → Memory has NO INFLUENCE (all outputs same as baseline)")

        # Show example differences
        print("\n" + "="*70)
        print("Example Output Changes (First 3)")
        print("="*70)

        baseline = results_by_config['no_memory']
        memory_w05 = results_by_config['memory_w05']

        shown = 0
        for baseline_r, memory_r in zip(baseline, memory_w05):
            if baseline_r['answer'] != memory_r['answer']:
                print(f"\nQuestion: {baseline_r['question'][:60]}...")
                print(f"  Without memory: {baseline_r['answer']}")
                print(f"  With memory (w=0.5): {memory_r['answer']}")
                print(f"  Ground truth: {baseline_r['ground_truth']}")

                shown += 1
                if shown >= 3:
                    break

    def compare_weight_sensitivity(self, results_by_config: Dict[str, List[Dict]]):
        """
        Analyze how sensitive outputs are to different memory weights.

        If outputs change a lot between w=0.3 and w=0.7, it means
        memory weight has a strong effect on inference.
        """
        print("\n" + "="*70)
        print("Memory Weight Sensitivity Analysis")
        print("="*70)

        w03_results = results_by_config['memory_w03']
        w05_results = results_by_config['memory_w05']
        w07_results = results_by_config['memory_w07']

        # Compare w=0.3 vs w=0.5
        diff_03_05 = sum(1 for r1, r2 in zip(w03_results, w05_results)
                        if r1['answer'] != r2['answer'])

        # Compare w=0.5 vs w=0.7
        diff_05_07 = sum(1 for r1, r2 in zip(w05_results, w07_results)
                        if r1['answer'] != r2['answer'])

        total = len(w03_results)

        print(f"\nWeight 0.3 → 0.5: {diff_03_05}/{total} outputs changed ({diff_03_05/total*100:.1f}%)")
        print(f"Weight 0.5 → 0.7: {diff_05_07}/{total} outputs changed ({diff_05_07/total*100:.1f}%)")

        if diff_03_05 > total * 0.2 or diff_05_07 > total * 0.2:
            print("\n→ HIGH SENSITIVITY: Memory weight strongly affects outputs")
            print("  This indicates memory is actively influencing inference")
        else:
            print("\n→ LOW SENSITIVITY: Memory weight doesn't change outputs much")
            print("  Either memory influence is stable, or memory is being ignored")


def create_ablation_dataset(eval_results_file: str, max_samples: int = 50) -> List[Dict]:
    """
    Create dataset for ablation study from evaluation results.

    Selects samples that:
    1. Have H-UAV memory available
    2. Represent diverse question types
    3. Have different self-match scores

    Args:
        eval_results_file: Path to L-UAV evaluation results
        max_samples: Maximum number of samples to include

    Returns:
        List of samples ready for ablation study
    """
    # This is a placeholder - actual implementation would need:
    # 1. Access to the model and tokenizer
    # 2. Re-process images and questions
    # 3. Get memory features from H-UAV cache

    # For now, return empty list with instructions
    print("="*70)
    print("Ablation Dataset Creation")
    print("="*70)
    print("\nTo run ablation study, you need to:")
    print("1. Modify eval_task1.py to save:")
    print("   - input_ids")
    print("   - image_tensor")
    print("   - memory_features")
    print("   for samples that query H-UAV")
    print("\n2. Then load and use this script")

    return []


if __name__ == "__main__":
    print("Memory Ablation Study Tool")
    print("\nThis tool helps understand memory's role by comparing:")
    print("  • No memory (baseline)")
    print("  • Memory with weight 0.3")
    print("  • Memory with weight 0.5")
    print("  • Memory with weight 0.7")
    print("\nKey questions answered:")
    print("  1. Does memory change outputs? (vs baseline)")
    print("  2. How much do different weights matter?")
    print("  3. When does memory help vs hurt?")
    print("\nUsage:")
    print("  python memory_ablation_study.py <eval_results.jsonl>")
