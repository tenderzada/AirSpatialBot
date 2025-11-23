"""
Train Self-Matching Module

This script trains the QueryKeyGenerator MLPs to produce meaningful
self-similarity scores that correlate with L-UAV's prediction confidence.

Training Strategy:
1. Use L-UAV's actual accuracy as supervision signal
2. High accuracy samples → should have high self-match scores (confident)
3. Low accuracy samples → should have low self-match scores (uncertain)

This enables self-matching to identify when L-UAV needs H-UAV assistance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

from communication.self_matching import SelfMatchingModule
from models.llava_mac import LLaVAWithMAC
from models.uav_config import UAVConfig


class SelfMatchingDataset(Dataset):
    """
    Dataset for training self-matching module.

    Each sample contains:
    - image_features: Visual features from L-UAV
    - bbox_3d: 3D bounding box parameters
    - label: 1 if L-UAV predicted correctly, 0 otherwise
    """

    def __init__(self, features_file: str, results_file: str):
        """
        Args:
            features_file: Path to saved features (image_features, bbox_3d)
            results_file: Path to L-UAV evaluation results (to get correctness labels)
        """
        self.samples = []

        # Load results to get correctness labels
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        # Load features
        features_data = torch.load(features_file)

        # Match features with results
        for i, (result, features) in enumerate(zip(results, features_data)):
            # Only use local inference samples (not H-UAV) for training
            if result.get('source') != 'local' and result.get('source') != 'knowledge_base':
                continue

            image_features = features['image_features']
            bbox_3d = features['bbox_3d']

            # Determine if prediction was correct
            pred = self._extract_number(result.get('answer', ''))
            gt = self._extract_number(result.get('ground_truth', ''))

            if pred is not None and gt is not None:
                is_correct = abs(pred - gt) < 0.1
                label = 1.0 if is_correct else 0.0

                self.samples.append({
                    'image_features': image_features,
                    'bbox_3d': bbox_3d,
                    'label': label,
                    'question_id': result.get('question_id', i)
                })

        print(f"Loaded {len(self.samples)} training samples")

        # Statistics
        correct = sum(s['label'] for s in self.samples)
        print(f"  Correct: {correct}/{len(self.samples)} = {correct/len(self.samples)*100:.1f}%")

    @staticmethod
    def _extract_number(text):
        """Extract numeric answer from text."""
        import re
        if isinstance(text, (int, float)):
            return float(text)
        match = re.search(r'\b(\d+)\b', str(text))
        if match:
            return float(match.group(1))
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            sample['image_features'],
            sample['bbox_3d'],
            sample['label']
        )


def train_self_matching(
    sm_module: SelfMatchingModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: str = 'cuda:0'
):
    """
    Train the self-matching module.

    Loss function:
        L = BCE(self_match_score, correctness_label)

    The model learns to assign:
    - High scores to samples L-UAV handles well
    - Low scores to samples L-UAV struggles with
    """

    sm_module = sm_module.to(device)
    sm_module.train()

    # Optimizer - only train the QK generator, not the gate
    optimizer = optim.AdamW(
        sm_module.qk_generator.parameters(),
        lr=lr,
        weight_decay=1e-5
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    # Loss function
    criterion = nn.BCELoss()

    best_val_auc = 0.0

    for epoch in range(num_epochs):
        # Training
        sm_module.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for image_features, bbox_3d, labels in pbar:
            image_features = image_features.to(device)
            bbox_3d = bbox_3d.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()

            # Forward pass
            query, scores, _ = sm_module(image_features, bbox_3d)

            # Loss: scores should match correctness labels
            loss = criterion(scores, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sm_module.parameters(), 1.0)
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            predictions = (scores > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({
                'loss': loss.item(),
                'acc': train_correct / train_total
            })

        scheduler.step()

        # Validation
        val_loss, val_acc, val_auc = evaluate_self_matching(
            sm_module, val_loader, device
        )

        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_correct/train_total:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                sm_module.state_dict(),
                './outputs/hierarchical_uav/best_self_matching.pt'
            )
            print(f"  ✓ Saved best model (AUC: {val_auc:.4f})")

    print(f"\nTraining complete! Best validation AUC: {best_val_auc:.4f}")


def evaluate_self_matching(
    sm_module: SelfMatchingModule,
    data_loader: DataLoader,
    device: str
) -> Tuple[float, float, float]:
    """Evaluate self-matching module."""
    sm_module.eval()

    criterion = nn.BCELoss()
    total_loss = 0.0
    correct = 0
    total = 0

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for image_features, bbox_3d, labels in data_loader:
            image_features = image_features.to(device)
            bbox_3d = bbox_3d.to(device)
            labels = labels.to(device).float()

            # Forward pass
            query, scores, _ = sm_module(image_features, bbox_3d)

            # Loss
            loss = criterion(scores, labels)
            total_loss += loss.item()

            # Accuracy
            predictions = (scores > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Store for AUC
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_scores)

    return (
        total_loss / len(data_loader),
        correct / total,
        auc
    )


def extract_features_for_training(
    model: LLaVAWithMAC,
    dataset_path: str,
    output_path: str,
    device: str = 'cuda:0'
):
    """
    Extract image features and bbox_3d from dataset for training.

    This should be run once before training to prepare the dataset.
    """
    print("Extracting features for self-matching training...")

    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    features_list = []

    from PIL import Image
    import os

    for i, sample in enumerate(tqdm(dataset)):
        # Load image
        image_id = sample['image_id']
        image_path = f"./data/images/{image_id}"  # Adjust path as needed

        if not os.path.exists(image_path):
            continue

        # Extract image features (same as in eval_task1.py)
        from hierarchical_uav.eval_task1 import extract_image_features

        image_features = extract_image_features(model, image_path, None)

        # Get bbox_3d
        bbox_2d = sample.get('bbox', [0, 0, 100, 100])
        bbox_3d = torch.tensor([
            (bbox_2d[0] + bbox_2d[2]) / 2,  # x_center
            (bbox_2d[1] + bbox_2d[3]) / 2,  # y_center
            0.0,  # z (placeholder)
            float(sample.get('length', 4500)),
            float(sample.get('width', 1800)),
            float(sample.get('height', 1500)),
            0.0   # rotation (placeholder)
        ])

        features_list.append({
            'image_features': image_features,
            'bbox_3d': bbox_3d
        })

    # Save features
    torch.save(features_list, output_path)
    print(f"✓ Saved {len(features_list)} feature samples to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['extract', 'train'], required=True)
    parser.add_argument('--dataset', type=str, help='Path to dataset JSON')
    parser.add_argument('--features', type=str, help='Path to features file')
    parser.add_argument('--results', type=str, help='Path to results JSONL')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    if args.mode == 'extract':
        # Extract features
        config = UAVConfig.create_luav_config(
            model_path="./models/AirSpatialBot",
            device=args.device
        )
        model = LLaVAWithMAC(config)

        extract_features_for_training(
            model,
            args.dataset,
            args.features,
            args.device
        )

    elif args.mode == 'train':
        # Train self-matching
        config = UAVConfig.create_luav_config(
            model_path="./models/AirSpatialBot",
            device=args.device
        )

        # Create self-matching module
        sm_module = SelfMatchingModule(
            feature_dim=config.hidden_size,
            threshold=0.7
        )

        # Create datasets
        dataset = SelfMatchingDataset(args.features, args.results)

        # Split train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )

        # Train
        train_self_matching(
            sm_module,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            device=args.device
        )
