# 分布式部署指南：跨机器 H-UAV 和 L-UAV

## 🌐 架构概览

```
机器 A: 10.37.74.206              机器 B: 远程服务器
┌──────────────────────────┐    ┌──────────────────────────┐
│  H-UAV Server            │    │  L-UAV Client            │
│  ├─ GPU: cuda:0          │    │  ├─ GPU: 任意            │
│  ├─ Port: 50051          │◄───┤  ├─ 连接: 10.37.74.206   │
│  ├─ MAC Memory (trained) │gRPC│  └─ Self-matching        │
│  └─ 服务内存查询         │    │     决策是否查询 H-UAV   │
└──────────────────────────┘    └──────────────────────────┘
```

## 📋 前提条件

### 机器 A (10.37.74.206) - H-UAV Server
- [x] GPU 可用（cuda:0）
- [x] AirSpatialBot 代码仓库
- [x] 训练好的 MAC 记忆 checkpoint
- [x] Port 50051 在防火墙中开放
- [x] ping 测试成功（已验证）

### 机器 B (远程服务器) - L-UAV Client
- [ ] GPU 可用（任意编号）
- [ ] AirSpatialBot 代码仓库（相同版本）
- [ ] 网络可达 10.37.74.206:50051
- [ ] 测试数据和模型文件

---

## 🚀 部署步骤

### **步骤 1：机器 A - 检查防火墙（当前机器）**

```bash
# 检查端口 50051 是否开放
sudo firewall-cmd --list-ports 2>/dev/null || sudo ufw status | grep 50051

# 如果未开放，添加规则
# CentOS/RHEL:
sudo firewall-cmd --permanent --add-port=50051/tcp
sudo firewall-cmd --reload

# Ubuntu:
sudo ufw allow 50051/tcp
sudo ufw reload

# 验证端口监听（H-UAV 启动后）
netstat -tuln | grep 50051
```

### **步骤 2：机器 A - 启动 H-UAV Server**

```bash
# 在当前机器 (10.37.74.206) 上运行

# 选项 A: 使用训练好的 MAC 记忆（推荐）
LOAD_MEMORY=./outputs/huav_training/huav_memory_final.pt ./run_huav.sh

# 选项 B: 不加载记忆（随机初始化）
./run_huav.sh
```

**预期输出**：
```
============================================================
Starting H-UAV (High Resource UAV)
============================================================

Configuration:
  Model: /mnt/data/AirSpatialBot
  Device: cuda:0
  Port: 50051
  MAC Memory: ./outputs/huav_training/huav_memory_final.pt

Loading H-UAV model on cuda:0...
✓ H-UAV model loaded

Loading trained MAC memory from ./outputs/huav_training/huav_memory_final.pt...
✓ Trained memory loaded - H-UAV ready with learned representations

✓ H-UAV Server started
  Listening on port 50051
  Waiting for L-UAV connections...

Press Ctrl+C to stop server
```

**验证 H-UAV 运行**：
```bash
# 在另一个终端检查
ps aux | grep "h-uav"
netstat -tuln | grep 50051
```

### **步骤 3：机器 B - 准备远程 L-UAV**

#### 3.1 复制代码到远程机器

```bash
# 方式 1：如果两台机器都能访问 git
ssh user@remote-machine
cd /path/to/workspace
git clone <repository-url>
cd AirSpatialBot
git checkout claude/branch-specification-feature-01BuzandgiEkJwu655qY15xJ

# 方式 2：直接 rsync 当前代码
rsync -avz --exclude='.git' --exclude='outputs' \
    /home/user/AirSpatialBot/ \
    user@remote-machine:/path/to/AirSpatialBot/
```

#### 3.2 修改远程 L-UAV 配置

在远程机器上编辑 `run_luav_remote.sh`：

```bash
# 修改以下路径为远程机器的实际路径
MODEL_PATH="/path/to/your/AirSpatialBot"
VISION_TOWER="/path/to/your/clip-vit-large-patch14-336"
TEST_DATA="/path/to/your/airspatial_agent_test_task1.jsonl"
IMAGE_DIR="/path/to/your/images"
DEVICE="cuda:0"  # 或 cuda:1, cuda:2 等
```

#### 3.3 测试连接

```bash
# 在远程机器上测试 H-UAV 连接
ping 10.37.74.206
nc -zv 10.37.74.206 50051

# 如果 nc 不可用，使用 telnet
telnet 10.37.74.206 50051
```

### **步骤 4：机器 B - 启动 L-UAV Client**

```bash
# 在远程机器上运行
./run_luav_remote.sh
```

**预期输出**：
```
============================================================
Starting L-UAV (Remote Client Mode)
============================================================

Configuration:
  Model: /path/to/AirSpatialBot
  Device: cuda:0
  H-UAV Address: 10.37.74.206:50051
  Self-matching Threshold: 1.0
  Force Query Rate: 0.3

Testing H-UAV connection...
✓ H-UAV is reachable at 10.37.74.206:50051

Starting L-UAV remote evaluation...

INFO:hierarchical_uav.communication.grpc_client:✓ L-UAV Client initialized for H-UAV at 10.37.74.206:50051
✓ Loaded test data from /path/to/airspatial_agent_test_task1.jsonl...
✓ Loaded 971 test samples

Connecting to H-UAV at 10.37.74.206:50051...
✓ H-UAV server is reachable

Loading L-UAV model on cuda:0...
✓ L-UAV model loaded
...
```

---

## 🔍 监控和调试

### **机器 A - 监控 H-UAV 服务器**

```bash
# 查看 H-UAV 日志
tail -f outputs/hierarchical_uav/huav.log

# 查看请求统计
# H-UAV 会定期打印统计信息：
# [H-UAV Stats] Requests: 100, Cache hit rate: 25.00%
```

### **机器 B - 监控 L-UAV 客户端**

```bash
# 查看 L-UAV 日志
tail -f outputs/hierarchical_uav/luav_remote.log

# 检查处理进度
watch -n 1 "tail -20 outputs/hierarchical_uav/luav_remote.log | grep 'Sample'"
```

### **网络监控**

```bash
# 机器 A: 监控网络连接
watch -n 1 "netstat -an | grep 50051"

# 机器 B: 监控出站连接
watch -n 1 "netstat -an | grep '10.37.74.206:50051'"
```

---

## 📊 结果分析

### **收集结果**

```bash
# 机器 B: 将结果复制回机器 A 进行分析
scp outputs/hierarchical_uav/task1_l-uav_remote_results.jsonl \
    user@10.37.74.206:/home/user/AirSpatialBot/outputs/hierarchical_uav/
```

### **对比分析**

在机器 A 上运行：

```bash
# 对比 Baseline vs Remote Memory-Augmented
python compare_baseline_vs_memory.py \
    outputs/hierarchical_uav/task1_l-uav_standalone.jsonl \
    outputs/hierarchical_uav/task1_l-uav_remote_results.jsonl
```

---

## ⚠️ 常见问题

### Q1: "Cannot connect to H-UAV at 10.37.74.206:50051"

**检查清单**：
1. H-UAV 服务器是否正在运行？
   ```bash
   ps aux | grep "h-uav"
   ```

2. 端口是否监听？
   ```bash
   netstat -tuln | grep 50051
   ```

3. 防火墙是否开放？
   ```bash
   sudo firewall-cmd --list-ports | grep 50051
   # 或
   sudo ufw status | grep 50051
   ```

4. 网络是否可达？
   ```bash
   ping 10.37.74.206
   telnet 10.37.74.206 50051
   ```

### Q2: gRPC 超时错误

**症状**：
```
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
    status = StatusCode.DEADLINE_EXCEEDED
```

**解决方案**：
- 增加 gRPC 超时时间（在客户端代码中修改）
- 检查网络延迟：`ping 10.37.74.206`
- 确认 H-UAV 响应正常

### Q3: H-UAV 内存不足

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
- 确认使用 `--load_8bit` 量化
- 减少 batch size
- 使用更强大的 GPU

### Q4: 数据路径不一致

**症状**：
```
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/...'
```

**解决方案**：
- 在远程机器上修改 `run_luav_remote.sh` 中的路径
- 确保测试数据和模型在远程机器上可访问
- 或者使用 NFS/共享存储

---

## 🎯 实验配置

### **Experiment A: Untrained MAC**

```bash
# 机器 A (不加载 checkpoint)
./run_huav.sh

# 机器 B
./run_luav_remote.sh

# 保存结果
mv outputs/hierarchical_uav/task1_l-uav_remote_results.jsonl \
   outputs/hierarchical_uav/task1_remote_UNTRAINED_MAC.jsonl
```

### **Experiment B: Trained MAC**

```bash
# 机器 A (加载 trained checkpoint)
pkill -f "h-uav"  # 停止之前的服务器
LOAD_MEMORY=./outputs/huav_training/huav_memory_final.pt ./run_huav.sh

# 机器 B
./run_luav_remote.sh

# 保存结果
mv outputs/hierarchical_uav/task1_l-uav_remote_results.jsonl \
   outputs/hierarchical_uav/task1_remote_TRAINED_MAC.jsonl
```

---

## 🔐 安全注意事项

1. **防火墙规则**：只允许信任的 IP 访问 50051
   ```bash
   sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="<remote-ip>" port port="50051" protocol="tcp" accept'
   ```

2. **网络加密**：当前 gRPC 使用明文通信，生产环境建议启用 TLS

3. **访问控制**：建议添加认证机制（token 或证书）

---

## 📈 性能优化

### **网络优化**

- 确保低延迟网络连接（< 10ms 最佳）
- 使用千兆或万兆网络
- 避免跨地域部署

### **GPU 分配**

- H-UAV: 使用性能更强的 GPU（如 A100, V100）
- L-UAV: 可以使用较弱的 GPU（如 RTX 3090, 4090）

### **批处理**

- 如果需要高吞吐量，可以修改代码支持批量查询
- 当前实现是单个样本查询

---

## ✅ 检查清单

部署前确认：

**机器 A (H-UAV Server)**
- [ ] GPU 可用且空闲
- [ ] 代码最新版本
- [ ] MAC memory checkpoint 存在
- [ ] Port 50051 开放
- [ ] 磁盘空间充足（用于日志和结果）

**机器 B (L-UAV Client)**
- [ ] GPU 可用且空闲
- [ ] 代码最新版本（与机器 A 一致）
- [ ] 测试数据可访问
- [ ] 网络连接稳定
- [ ] run_luav_remote.sh 路径已修改

**网络**
- [ ] ping 测试成功
- [ ] telnet/nc 端口测试成功
- [ ] 防火墙规则正确配置

---

## 📞 需要帮助？

如果遇到问题：
1. 检查 H-UAV 和 L-UAV 日志文件
2. 使用 `debug_kb_difference.py` 分析结果差异
3. 验证网络连接和防火墙配置
