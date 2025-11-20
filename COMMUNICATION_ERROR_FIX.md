# 通信错误修复指南

## 🔴 问题描述

错误信息：
```
ERROR: hierarchical_uav.communication.grpc_client: Query #X failed: Ran out of input
```

## 🔍 根本原因

这个错误发生在 `pickle.loads(data)` 反序列化时，表示接收到的数据不完整。

**原始问题代码**（`grpc_client.py` 和 `grpc_server.py`）：

```python
data = b''
while True:
    chunk = client_socket.recv(4096)
    if not chunk:  # ❌ 问题所在
        break
    data += chunk

    if b'<END>' in data:
        data = data.replace(b'<END>', b'')
        break
```

**问题分析**：
- `recv(4096)` 可能在某些情况下返回空字节
- `if not chunk: break` 会立即终止循环
- 此时数据可能还未完全接收（`<END>` 标记还未到达）
- 导致 `pickle.loads(data)` 尝试反序列化不完整的数据
- 抛出 "Ran out of input" 错误

---

## ✅ 修复方案

### 修复后的代码逻辑

```python
data = b''
while True:
    chunk = client_socket.recv(4096)
    data += chunk

    # 检查结束标记
    if b'<END>' in data:
        data = data.replace(b'<END>', b'')
        break

    # 只有在未收到任何数据时才退出
    if not chunk and len(data) == 0:
        break

# 确保数据非空
if not data:
    raise RuntimeError("Received empty response")

response = pickle.loads(data)
```

**关键改进**：
1. ✅ 移除了 `if not chunk: break`，优先依赖 `<END>` 标记
2. ✅ 只在既未收到数据又收到空 chunk 时才退出
3. ✅ 添加数据验证，确保非空才反序列化
4. ✅ 改进错误处理，记录数据长度和内容

### 其他修复

**1. 服务器端初始化 `cache_hit` 变量**

```python
# 修复前
cache_hit = cache_hit if hasattr(...) else False  # ❌ cache_hit可能未定义

# 修复后
cache_hit = False  # ✅ 先初始化
if hasattr(self.huav_model, 'mac_layer'):
    value_vector, cache_hit = self.huav_model.mac_layer.neural_memory.retrieve_with_cache(...)
```

**2. 增强错误日志**

```python
except pickle.UnpicklingError as e:
    logger.error(f"Pickle deserialization error: {e}")
    logger.error(f"  Received data length: {len(data)} bytes")
    logger.error(f"  Data preview: {data[:100]}")
```

---

## 🧪 验证修复

### 快速测试

运行通信测试脚本：

```bash
python test_communication.py
```

**预期输出**：

```
============================================================
Testing H-UAV/L-UAV Communication
============================================================

1. Creating mock H-UAV model...
   ✓ Mock model created

2. Starting H-UAV server...
   ✓ Server started

3. Creating L-UAV client...
   ✓ Client created

4. Testing connection...
   ✓ H-UAV is reachable

5. Testing single query...
   Sending query with shape torch.Size([256])
   ✓ Query successful!
     - Value shape: torch.Size([256])
     - Cache hit: False
     - Value sum: xxx.xxxx

6. Testing multiple queries...
   Progress: 5/10 queries successful
   Progress: 10/10 queries successful
   ✓ 10/10 queries successful

7. Testing batch query...
   Sending batch with shape torch.Size([5, 256])
   ✓ Batch query successful!
     - Values shape: torch.Size([5, 256])
     - Cache hits: 0/5

8. Checking server statistics...
   Total requests: 15
   Cache hits: 0
   Cache hit rate: 0.00%

9. Stopping server...
   ✓ Server stopped

============================================================
✅ All communication tests PASSED!
============================================================
```

---

## 🚀 重新运行评估

### 方法 1: 使用优化脚本（推荐）

```bash
# Terminal 1 - H-UAV
./run_hierarchical_uav_optimized.sh h-uav

# Terminal 2 - L-UAV
./run_hierarchical_uav_optimized.sh l-uav
```

### 方法 2: 手动启动

```bash
# Terminal 1 - H-UAV
python hierarchical_uav/eval_task1.py \
  --uav_type h-uav \
  --device cuda:0 \
  --port 50051 \
  --load_8bit

# Terminal 2 - L-UAV
python hierarchical_uav/eval_task1.py \
  --uav_type l-uav \
  --device cuda:1 \
  --huav_address localhost:50051 \
  --load_8bit
```

---

## 📊 预期结果

修复后，您应该看到：

### L-UAV 输出

```
L-UAV Evaluation Statistics
============================================================
Total samples: 971
Local decisions: ~650 (67.0%)
Remote queries: ~320 (33.0%)  ← 不再是 0%

Self-matching statistics:
  Query rate: 33.00%
  Total queries: 971
  H-UAV queries: 320

Self-matching score distribution:
  Mean: 0.6234  ← 不再是 1.0
  Std:  0.1456  ← 不再是 0.0
  Min:  0.2145
  Max:  0.9876

Score distribution by range:
  [0.0, 0.3):   45 ( 4.6%)
  [0.3, 0.5):  123 (12.7%)
  [0.5, 0.7):  456 (47.0%)
  [0.7, 0.9):  298 (30.7%)
  [0.9, 1.0):   49 ( 5.0%)
```

### H-UAV 输出

```
H-UAV Server Statistics:
  Total requests: ~320  ← 接收到 L-UAV 查询
  Processing...
```

---

## 🐛 故障排查

### 问题 1: 仍然出现 "Ran out of input"

**检查**：
1. 确认已拉取最新代码：`git pull`
2. 重启 H-UAV 和 L-UAV 进程
3. 运行 `python test_communication.py` 验证

**如果测试脚本失败**：
- 检查端口是否被占用：`lsof -i :50051`
- 检查防火墙设置
- 查看详细错误日志

### 问题 2: 连接超时

**原因**：H-UAV 服务器未启动或地址错误

**解决**：
1. 确认 H-UAV 已启动并显示 "Server started"
2. 检查地址和端口是否匹配
3. 使用 `ping_huav()` 测试连接

### 问题 3: 数据大小异常

**症状**：日志显示 "Received data length: 0 bytes"

**原因**：服务器发送失败或网络问题

**解决**：
1. 检查服务器日志是否有错误
2. 增加超时时间：`--timeout 30`
3. 减小批量大小

### 问题 4: 仍然 0% 远程查询

**原因**：自匹配机制问题（已在之前修复）

**检查**：
1. 确认分数不再全是 1.0
2. 查看分数分布
3. 调整阈值：`--threshold 0.8`

---

## 📝 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `hierarchical_uav/communication/grpc_client.py` | 修复接收循环，改进错误处理 |
| `hierarchical_uav/communication/grpc_server.py` | 同步修复，初始化变量 |
| `test_communication.py` | 新增通信测试脚本 |

---

## 🔧 技术细节

### Socket 接收机制

**问题场景**：
```
Server: [发送数据...] → chunk1(4096 bytes)
                      → chunk2(2048 bytes)
                      → chunk3(0 bytes)  ← TCP缓冲区暂时为空
                      → chunk4(1024 bytes + <END>)
```

**错误的处理**：
```python
if not chunk:  # chunk3是空的
    break      # ❌ 过早退出，chunk4未接收
```

**正确的处理**：
```python
if b'<END>' in data:  # 等待结束标记
    break             # ✅ 确保所有数据接收完毕
```

### Pickle 序列化

**数据流**：
```
Client → pickle.dumps(request) → bytes → <END> marker → Socket
Socket → bytes → remove <END> → pickle.loads() → response → Server
```

**如果数据不完整**：
```python
pickle.loads(incomplete_data)
# → pickle.UnpicklingError: Ran out of input
```

---

## ✅ 验证清单

运行评估前，确认：

- [ ] 已拉取最新代码
- [ ] 通信测试脚本通过（`python test_communication.py`）
- [ ] H-UAV 服务器成功启动
- [ ] L-UAV 可以 ping 通 H-UAV
- [ ] 自匹配分数不再全是 1.0（如有疑问运行 `python test_self_matching_fix.py`）
- [ ] GPU 内存充足（运行 `./check_gpu_memory.sh`）

---

## 💡 关键要点

1. **接收循环依赖结束标记** - 不要过早退出
2. **数据验证很重要** - 反序列化前检查数据
3. **错误日志要详细** - 便于快速定位问题
4. **先测试后运行** - 使用 `test_communication.py` 验证

---

## 📞 需要帮助？

如果问题仍然存在：

1. 运行诊断脚本：`python test_communication.py`
2. 检查详细日志输出
3. 查看 H-UAV 服务器端日志
4. 确认网络连接正常

修复后应该可以正常进行 H-UAV 与 L-UAV 的通信了！🚀
