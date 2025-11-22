"""
Memory Influence Analysis Tool

Analyzes how H-UAV memory affects L-UAV's inference, regardless of accuracy.
Focus: Understanding memory's role in the inference process.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import re


class MemoryInfluenceAnalyzer:
    """Analyzes the influence of H-UAV memory on L-UAV inference."""

    def __init__(self, results_file: str):
        """Load evaluation results."""
        self.results = []
        with open(results_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.results.append(json.loads(line))

        # Separate by source
        self.memory_samples = [r for r in self.results if r.get('source') == 'huav']
        self.local_samples = [r for r in self.results if r.get('source') in ['local', 'knowledge_base']]

        print(f"Loaded {len(self.results)} total results")
        print(f"  Memory-augmented: {len(self.memory_samples)}")
        print(f"  Local-only: {len(self.local_samples)}")

    def analyze_memory_impact(self):
        """
        Analyze how memory changes the inference output.

        Key question: Does memory produce DIFFERENT outputs than local inference?
        (Different doesn't mean better/worse - just that memory HAS an effect)
        """
        print("\n" + "="*70)
        print("Memory Impact Analysis")
        print("="*70)

        if not self.memory_samples:
            print("No memory-augmented samples found!")
            return

        # Group by question type
        by_qtype = defaultdict(lambda: {'memory': [], 'local': []})

        for r in self.memory_samples:
            qtype = r.get('qtype', 'unknown')
            by_qtype[qtype]['memory'].append(r)

        for r in self.local_samples:
            qtype = r.get('qtype', 'unknown')
            by_qtype[qtype]['local'].append(r)

        # Analyze answer distribution
        print("\nAnswer Distribution Comparison:")
        print("-" * 70)

        for qtype in sorted(by_qtype.keys()):
            if qtype == 'unknown' and by_qtype[qtype]['memory']:
                print(f"\n{qtype.upper()} (memory-augmented samples):")

                memory_answers = [r.get('answer', '') for r in by_qtype[qtype]['memory']]

                # Count unique answers
                answer_counts = defaultdict(int)
                for ans in memory_answers:
                    # Extract number for numerical answers
                    num = self._extract_number(ans)
                    if num is not None:
                        answer_counts[int(num)] += 1
                    else:
                        answer_counts[ans[:20]] += 1  # First 20 chars

                print(f"  Total samples: {len(memory_answers)}")
                print(f"  Unique answers: {len(answer_counts)}")
                print(f"  Answer distribution:")
                for ans, count in sorted(answer_counts.items(), key=lambda x: -x[1]):
                    pct = count / len(memory_answers) * 100
                    print(f"    {ans}: {count} ({pct:.1f}%)")

    def analyze_memory_weight_effect(self):
        """
        Analyze how different memory weights correlate with output changes.
        """
        print("\n" + "="*70)
        print("Memory Weight Effect Analysis")
        print("="*70)

        if not self.memory_samples:
            print("No memory-augmented samples found!")
            return

        # Group by self-match score ranges
        score_groups = {
            '[0.0, 0.4)': [],
            '[0.4, 0.5)': [],
            '[0.5, 0.6)': [],
            '[0.6, 0.7)': [],
            '[0.7, 1.0]': []
        }

        for r in self.memory_samples:
            score = r.get('self_match_score', 0.5)

            if score < 0.4:
                score_groups['[0.0, 0.4)'].append(r)
            elif score < 0.5:
                score_groups['[0.4, 0.5)'].append(r)
            elif score < 0.6:
                score_groups['[0.5, 0.6)'].append(r)
            elif score < 0.7:
                score_groups['[0.6, 0.7)'].append(r)
            else:
                score_groups['[0.7, 1.0]'].append(r)

        print("\nSamples by Self-Match Score Range:")
        print("-" * 70)

        for range_key, samples in sorted(score_groups.items()):
            if samples:
                print(f"\n{range_key}: {len(samples)} samples")

                # Infer expected weight from score
                avg_score = np.mean([s.get('self_match_score', 0.5) for s in samples])
                expected_weight = self._infer_weight_from_score(avg_score)

                print(f"  Avg score: {avg_score:.4f}")
                print(f"  Expected weight: {expected_weight:.2f}")

                # Show sample answers
                answers = [s.get('answer', '') for s in samples[:3]]
                print(f"  Sample answers: {answers}")

    def analyze_memory_vs_local_comparison(self):
        """
        Compare memory-augmented vs local inference on similar questions.

        This shows if memory produces systematically different outputs.
        """
        print("\n" + "="*70)
        print("Memory vs Local Inference Comparison")
        print("="*70)

        # Find similar questions (same qtype and similar image)
        memory_by_image = defaultdict(list)
        local_by_image = defaultdict(list)

        for r in self.memory_samples:
            image_id = r.get('image_id', '')
            memory_by_image[image_id].append(r)

        for r in self.local_samples:
            image_id = r.get('image_id', '')
            local_by_image[image_id].append(r)

        # Find images that have both memory and local samples
        common_images = set(memory_by_image.keys()) & set(local_by_image.keys())

        if common_images:
            print(f"\nFound {len(common_images)} images with both memory and local samples")
            print("\nComparison examples:")
            print("-" * 70)

            for img_id in list(common_images)[:5]:  # Show first 5
                print(f"\nImage: {img_id}")

                mem_samples = memory_by_image[img_id]
                loc_samples = local_by_image[img_id]

                print(f"  Memory-augmented ({len(mem_samples)} samples):")
                for s in mem_samples[:2]:
                    print(f"    Q: {s.get('question', '')[:50]}...")
                    print(f"    A: {s.get('answer', '')}")
                    print(f"    Score: {s.get('self_match_score', 0.0):.4f}")

                print(f"  Local-only ({len(loc_samples)} samples):")
                for s in loc_samples[:2]:
                    print(f"    Q: {s.get('question', '')[:50]}...")
                    print(f"    A: {s.get('answer', '')}")
        else:
            print("\nNo common images found between memory and local samples")
            print("(This is expected if memory is only queried for specific cases)")

    def visualize_memory_distribution(self, output_dir: str = "./outputs"):
        """
        Create visualizations of memory influence.
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        print("\n" + "="*70)
        print("Generating Visualizations")
        print("="*70)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Self-match score distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Score distribution
        scores = [r.get('self_match_score', 0.5) for r in self.memory_samples]
        if scores:
            axes[0, 0].hist(scores, bins=20, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(np.mean(scores), color='red', linestyle='--',
                             label=f'Mean: {np.mean(scores):.4f}')
            axes[0, 0].set_xlabel('Self-Match Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Self-Match Score Distribution\n(Memory-Augmented Samples)')
            axes[0, 0].legend()

        # Answer distribution
        answers = [self._extract_number(r.get('answer', '')) for r in self.memory_samples]
        answers = [a for a in answers if a is not None]
        if answers:
            axes[0, 1].hist(answers, bins=range(int(min(answers)), int(max(answers))+2),
                          edgecolor='black', alpha=0.7)
            axes[0, 1].set_xlabel('Answer Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Answer Distribution\n(Memory-Augmented Samples)')

        # Score vs Answer (to see if memory creates patterns)
        if scores and answers and len(scores) == len(self.memory_samples):
            valid_pairs = [(r.get('self_match_score', 0.5),
                          self._extract_number(r.get('answer', '')))
                         for r in self.memory_samples]
            valid_pairs = [(s, a) for s, a in valid_pairs if a is not None]

            if valid_pairs:
                scores_plot, answers_plot = zip(*valid_pairs)
                axes[1, 0].scatter(scores_plot, answers_plot, alpha=0.6)
                axes[1, 0].set_xlabel('Self-Match Score')
                axes[1, 0].set_ylabel('Answer Value')
                axes[1, 0].set_title('Self-Match Score vs Answer\n(Check for patterns)')

        # Memory influence indicator
        # Compare answer variance: high variance = memory has diverse effect
        if answers:
            local_answers = [self._extract_number(r.get('answer', ''))
                           for r in self.local_samples]
            local_answers = [a for a in local_answers if a is not None]

            if local_answers:
                data_to_plot = [answers, local_answers]
                axes[1, 1].boxplot(data_to_plot, labels=['Memory', 'Local'])
                axes[1, 1].set_ylabel('Answer Value')
                axes[1, 1].set_title('Answer Distribution: Memory vs Local\n(Box Plot)')

        plt.tight_layout()
        plot_path = f"{output_dir}/memory_influence_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {plot_path}")

    def generate_report(self, output_file: str = "./outputs/memory_influence_report.txt"):
        """Generate comprehensive memory influence report."""

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MEMORY INFLUENCE ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")

            # Overall statistics
            f.write("1. OVERALL STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Total samples: {len(self.results)}\n")
            f.write(f"Memory-augmented: {len(self.memory_samples)}\n")
            f.write(f"Local-only: {len(self.local_samples)}\n")
            f.write(f"Memory usage rate: {len(self.memory_samples)/len(self.results)*100:.1f}%\n")

            # Memory characteristics
            if self.memory_samples:
                scores = [r.get('self_match_score', 0.5) for r in self.memory_samples]
                f.write(f"\n2. MEMORY CHARACTERISTICS\n")
                f.write("-"*70 + "\n")
                f.write(f"Self-match score:\n")
                f.write(f"  Mean: {np.mean(scores):.4f}\n")
                f.write(f"  Std:  {np.std(scores):.4f}\n")
                f.write(f"  Min:  {np.min(scores):.4f}\n")
                f.write(f"  Max:  {np.max(scores):.4f}\n")

                # Answer diversity
                answers = [r.get('answer', '') for r in self.memory_samples]
                unique_answers = len(set(answers))
                f.write(f"\nAnswer diversity:\n")
                f.write(f"  Total answers: {len(answers)}\n")
                f.write(f"  Unique answers: {unique_answers}\n")
                f.write(f"  Diversity ratio: {unique_answers/len(answers)*100:.1f}%\n")

            # Memory influence evidence
            f.write(f"\n3. MEMORY INFLUENCE EVIDENCE\n")
            f.write("-"*70 + "\n")
            f.write("Evidence that memory affects inference:\n")

            # Different outputs?
            if self.memory_samples and self.local_samples:
                mem_answers = set(r.get('answer', '') for r in self.memory_samples)
                loc_answers = set(r.get('answer', '') for r in self.local_samples)

                unique_to_memory = mem_answers - loc_answers
                unique_to_local = loc_answers - mem_answers

                f.write(f"  ✓ Unique answers with memory: {len(unique_to_memory)}\n")
                f.write(f"  ✓ Unique answers without memory: {len(unique_to_local)}\n")

                if unique_to_memory:
                    f.write(f"  → Memory produces DIFFERENT outputs (evidence of influence)\n")
                else:
                    f.write(f"  → Memory produces SAME outputs as local\n")

            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")

        print(f"\n✓ Report saved to: {output_file}")

    @staticmethod
    def _extract_number(text):
        """Extract numeric answer from text."""
        if isinstance(text, (int, float)):
            return float(text)
        match = re.search(r'\b(\d+)\b', str(text))
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _infer_weight_from_score(score):
        """Infer expected memory weight from self-match score."""
        if score > 0.8:
            return 0.8
        elif score > 0.7:
            return 0.7
        elif score > 0.5:
            return 0.5
        elif score > 0.4:
            return 0.4
        else:
            return 0.3


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze memory influence on inference")
    parser.add_argument('results_file', type=str,
                       help='Path to L-UAV evaluation results (JSONL)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for visualizations and reports')

    args = parser.parse_args()

    # Create analyzer
    analyzer = MemoryInfluenceAnalyzer(args.results_file)

    # Run analyses
    analyzer.analyze_memory_impact()
    analyzer.analyze_memory_weight_effect()
    analyzer.analyze_memory_vs_local_comparison()

    # Generate outputs
    analyzer.visualize_memory_distribution(args.output_dir)
    analyzer.generate_report(f"{args.output_dir}/memory_influence_report.txt")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nCheck {args.output_dir}/ for:")
    print("  • memory_influence_analysis.png (visualizations)")
    print("  • memory_influence_report.txt (detailed report)")


if __name__ == "__main__":
    main()
