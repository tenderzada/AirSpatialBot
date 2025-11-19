
# Hierarchical Multi-Agent UAV System with Memory-Augmented Learning

This document describes the implementation of a novel hierarchical UAV perception framework for fine-grained vehicle attribute recognition, built upon the AirSpatialBot foundation.

## 🎯 Research Overview

### Abstract

We propose a hierarchical multi-agent perception framework consisting of:
- **H-UAV (High-UAV)**: Resource-rich agent with MAC (Memory-Augmented Continual Learning) supporting test-time learning
- **L-UAV (Low-UAV)**: Lightweight agents with inference-only mode and selective querying

Key innovations:
1. **Test-time Memorization**: H-UAV maintains a neural long-term memory that adapts during deployment
2. **Self-Gated Communication**: L-UAVs autonomously decide when to request help based on self-matching scores
3. **Bandwidth Efficiency**: Compact query vectors (256-dim) minimize communication overhead
4. **Zero-Shot Adaptation**: System learns new vehicle models without retraining

### System Architecture

```
                    ┌─────────────────┐
                    │    H-UAV        │
                    │  (GPU 0)        │
                    │                 │
                    │  ┌──────────┐   │
                    │  │ LLaVA+   │   │
                    │  │  MAC     │   │
                    │  │ Memory   │   │
                    │  └──────────┘   │
                    │                 │
                    │  Server:50051   │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ Self-Matching   │
                    │    Gating       │
                    │  (m_i,i > τ ?)  │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
            ┌───────▼──────┐   ┌─────▼────────┐
            │   L-UAV #1   │   │   L-UAV #2   │
            │   (GPU 1)    │   │   (GPU 1)    │
            │              │   │              │
            │  LLaVA+MAC   │   │  LLaVA+MAC   │
            │ (frozen)     │   │ (frozen)     │
            └──────────────┘   └──────────────┘
```

---

## 📦 Implementation Components

### 1. MAC Memory Module (`hierarchical_uav/mac_memory/`)

#### Neural Long-term Memory (`neural_memory.py`)
- MLP-based associative memory: `ℓ(M; x_t) = ||M(k_t) - v_t||²`
- Surprise-driven learning:
  ```
  S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)
  M_t = (1 - α_t) * M_{t-1} + S_t
  ```
- Episodic cache for fast retrieval

#### Persistent Memory (`persistent_memory.py`)
- Learnable context tokens prepended to input
- Provides task-specific priors

#### MAC Layer (`mac_layer.py`)
- Integrates neural memory + persistent memory + attention
- Segment-based processing
- Test-time learning capability

### 2. LLaVA Integration (`hierarchical_uav/models/`)

#### LLaVAWithMAC (`llava_mac.py`)
- Wraps base LLaVA model with MAC layers
- H-UAV mode: Enables memory updates
- L-UAV mode: Inference-only, frozen parameters

#### UAV Configuration (`uav_config.py`)
- `UAVType.HIGH_UAV`: Learning-enabled configuration
- `UAVType.LOW_UAV`: Inference-only configuration
- Automatic parameter freezing/unfreezing

### 3. Communication System (`hierarchical_uav/communication/`)

#### Self-Matching Gate (`self_matching.py`)
- Generates compact query `μ_i` (256-dim) and key `κ_i` (1024-dim)
- Computes self-matching score: `m_{i,i} = similarity(μ_i, κ_i)`
- Decision rule: Query H-UAV if `m_{i,i} < τ`

#### H-UAV Server (`grpc_server.py`)
- Socket-based server for memory retrieval
- Handles concurrent L-UAV requests
- Cache hit tracking

#### L-UAV Client (`grpc_client.py`)
- Sends compact query vectors to H-UAV
- Receives memory-retrieved values
- Timeout and error handling

---

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Install LLaVA
./install_llava.sh

# 2. Organize data
./setup_data.sh

# 3. Install additional dependencies
pip install tqdm
```

### Running the System

#### Option 1: Automated (Both UAVs)

```bash
# Runs H-UAV and L-UAV sequentially
./run_hierarchical_uav.sh both
```

#### Option 2: Manual (Recommended for Debugging)

```bash
# Terminal 1 - Start H-UAV Server (GPU 0)
./run_hierarchical_uav.sh h-uav

# Terminal 2 - Start L-UAV Client (GPU 1)
./run_hierarchical_uav.sh l-uav
```

#### Option 3: Direct Python

```bash
# H-UAV
python hierarchical_uav/eval_task1.py \
    --uav_type h-uav \
    --device cuda:0 \
    --port 50051

# L-UAV (in another terminal)
python hierarchical_uav/eval_task1.py \
    --uav_type l-uav \
    --device cuda:1 \
    --huav_address localhost:50051 \
    --threshold 0.7
```

---

## 📊 Configuration Parameters

### MAC Memory Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_dim` | 4096 | Dimension of memory vectors |
| `memory_depth` | 2 | Number of MLP layers in memory |
| `num_persistent_tokens` | 64 | Number of persistent memory tokens |
| `num_memory_tokens` | 128 | Number of retrieved memory tokens |

### Learning Configuration (H-UAV)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `surprise_eta` | 0.9 | Surprise decay (momentum) |
| `learning_theta` | 0.1 | Learning rate for memory |
| `forgetting_alpha` | 0.01 | Weight decay (forgetting) |

### Communication Configuration (L-UAV)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `self_match_threshold` | 0.7 | Decision threshold τ |
| `query_dim` | 256 | Query vector dimension |
| `key_dim` | 1024 | Key vector dimension |

---

## 📈 Expected Results

### Communication Efficiency

- **Query Rate**: 20-40% of samples require H-UAV assistance
- **Bandwidth**: ~256 floats = 1 KB per query
- **Latency**: <100ms per remote query

### Memory Statistics

- **Cache Hit Rate**: Improves over time (0% → 60%+)
- **Surprise Metric**: Decreases as memory consolidates
- **Step Count**: Total number of memory updates

### Task 1 Performance

Evaluation on AirSpatial-Bench Task 1:
- Vehicle brand/model recognition
- Fine-grained attribute extraction
- Spatial understanding via 3D bounding boxes

---

## 🔬 Testing Individual Components

### Test MAC Memory Module

```bash
cd hierarchical_uav/mac_memory
python neural_memory.py
python persistent_memory.py
python mac_layer.py
```

### Test Communication

```bash
# Terminal 1 - Start mock H-UAV server
python hierarchical_uav/communication/grpc_server.py

# Terminal 2 - Test L-UAV client
python hierarchical_uav/communication/grpc_client.py
```

### Test Self-Matching

```bash
python hierarchical_uav/communication/self_matching.py
```

---

## 📁 Project Structure

```
AirSpatialBot/
├── hierarchical_uav/
│   ├── __init__.py
│   ├── mac_memory/                  # MAC memory components
│   │   ├── neural_memory.py         # Learnable memory MLP
│   │   ├── persistent_memory.py     # Context tokens
│   │   └── mac_layer.py             # Memory-augmented attention
│   ├── models/                      # UAV models
│   │   ├── uav_config.py            # H-UAV/L-UAV configuration
│   │   └── llava_mac.py             # LLaVA + MAC integration
│   ├── communication/               # Communication system
│   │   ├── self_matching.py         # Self-matching gate
│   │   ├── grpc_server.py           # H-UAV server
│   │   └── grpc_client.py           # L-UAV client
│   └── eval_task1.py                # Task 1 evaluation script
├── run_hierarchical_uav.sh          # Convenience script
└── HIERARCHICAL_UAV_README.md       # This file
```

---

## 🛠️ Advanced Usage

### Custom Memory Configuration

```python
from hierarchical_uav.models import UAVConfig, UAVType

# Create custom H-UAV config
config = UAVConfig(
    uav_type=UAVType.HIGH_UAV,
    memory_dim=8192,          # Larger memory
    memory_depth=3,           # Deeper MLP
    num_persistent_tokens=128, # More context
    surprise_eta=0.95,        # Slower decay
    learning_theta=0.05       # Smaller learning rate
)
```

### Loading Pre-trained Memory

```python
from hierarchical_uav.models import LLaVAWithMAC

# Load model
model = LLaVAWithMAC(config)

# Load saved memory state
model.load_mac_state('./outputs/memory_checkpoint.pt')
```

### Saving Memory Checkpoints

```python
# After evaluation, save memory state
model.save_mac_state('./outputs/memory_final.pt')
```

### Adjusting Communication Threshold

```python
# Lower threshold = more L-UAV queries to H-UAV
config.self_match_threshold = 0.5  # More conservative

# Higher threshold = more local decisions
config.self_match_threshold = 0.9  # More aggressive
```

---

## 📊 Monitoring and Debugging

### H-UAV Server Logs

```bash
tail -f outputs/hierarchical_uav/huav.log
```

Expected output:
```
[H-UAV Stats] Requests: 10, Cache hit rate: 20.00%
[H-UAV Stats] Requests: 20, Cache hit rate: 35.00%
[H-UAV Stats] Requests: 30, Cache hit rate: 46.67%
```

### L-UAV Client Logs

```bash
tail -f outputs/hierarchical_uav/luav.log
```

Expected output:
```
L-UAV Evaluation: 100%|████████| 500/500 [10:23<00:00,  1.25s/it]

L-UAV Evaluation Statistics
==========================================
Total samples: 500
Local decisions: 320 (64.0%)
Remote queries: 180 (36.0%)
```

### Memory Statistics

```python
# Get real-time memory stats
stats = model.get_memory_stats()
print(f"Cache size: {stats['cache_size']}")
print(f"Steps: {stats['step_count']}")
print(f"Surprise: {stats['surprise']:.4f}")
```

---

## 🐛 Troubleshooting

### Issue 1: H-UAV Server Not Reachable

**Symptoms**: L-UAV cannot connect to H-UAV

**Solutions**:
```bash
# 1. Check if H-UAV is running
netstat -tlnp | grep 50051

# 2. Check firewall
sudo ufw allow 50051/tcp

# 3. Test connection
nc -zv localhost 50051
```

### Issue 2: CUDA Out of Memory

**Solutions**:
```bash
# 1. Use 8-bit quantization
python hierarchical_uav/eval_task1.py --load_8bit

# 2. Reduce batch size
python hierarchical_uav/eval_task1.py --batch_size 1

# 3. Reduce memory dimensions
python hierarchical_uav/eval_task1.py --memory_dim 2048
```

### Issue 3: Communication Timeout

**Symptoms**: `TimeoutError: H-UAV query timed out`

**Solutions**:
```python
# Increase timeout in L-UAV client
client = LUAVClient(huav_address="localhost:50051", timeout=30.0)
```

---

## 📚 Related Papers and Resources

### Core Papers

1. **AirSpatialBot** (Zhou et al., IEEE TGRS 2025)
   - [Paper](https://ieeexplore.ieee.org/document/11006099)
   - [Dataset](https://huggingface.co/datasets/erenzhou/AirSpatial)

2. **Titans: Learning to Memorize at Test Time** (Behrouz et al., 2024)
   - MAC memory architecture inspiration
   - [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)

3. **LLaVA: Visual Instruction Tuning** (Liu et al., NeurIPS 2023)
   - Base vision-language model
   - [GitHub](https://github.com/haotian-liu/LLaVA)

### Related Work

- V2V Communication: Collaborative perception in autonomous driving
- In-Context Learning: Few-shot adaptation for vision-language models
- Test-Time Training: Adaptation during inference

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{zhou2025airspatialbot,
  title={AirSpatialBot: A Spatially-Aware Aerial Agent for Fine-Grained Vehicle Attribute Recognition and Retrieval},
  author={Zhou, Yue and Ding, Ran and Yang, Xue and Jiang, Xue and Liu, Xingzhao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025}
}

@article{behrouz2024titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and others},
  journal={arXiv preprint arXiv:2501.00663},
  year={2024}
}
```

---

## 📧 Support and Contribution

For questions, issues, or contributions:
- Open an issue on GitHub
- Check existing documentation: `SETUP_GUIDE.md`, `LLAVA_INTEGRATION.md`
- Review test scripts in each module

---

## 🔮 Future Work

Potential extensions:
1. **Multi-L-UAV Collaboration**: L-UAVs share observations
2. **Adaptive Thresholding**: Learn optimal τ during deployment
3. **Memory Pruning**: Remove outdated entries from cache
4. **Attention-based Retrieval**: Use attention instead of nearest neighbor
5. **Real-World Deployment**: Hardware integration with actual drones

---

**Good luck with your research! 🚁✨**
