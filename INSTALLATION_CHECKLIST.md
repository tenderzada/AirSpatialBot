# ✅ 安装检查清单

使用这个清单确保所有组件都正确安装。

---

## 📋 基础环境

- [ ] **Python版本**: 3.8 或更高
  ```bash
  python3 --version
  # 应显示: Python 3.8.x 或更高
  ```

- [ ] **CUDA可用** (如果有GPU)
  ```bash
  nvidia-smi
  # 应显示GPU信息和CUDA版本
  ```

- [ ] **足够的磁盘空间**: 至少50GB
  ```bash
  df -h .
  ```

---

## 📦 依赖安装

- [ ] **PyTorch已安装**
  ```bash
  python -c "import torch; print(torch.__version__)"
  # 应显示: 2.1.2
  ```

- [ ] **CUDA在PyTorch中可用** (如果有GPU)
  ```bash
  python -c "import torch; print('CUDA:', torch.cuda.is_available())"
  # 应显示: CUDA: True
  ```

- [ ] **Transformers已安装**
  ```bash
  python -c "import transformers; print(transformers.__version__)"
  # 应显示: 4.37.2
  ```

- [ ] **所有requirements安装**
  ```bash
  pip install -r requirements.txt
  # 应无错误
  ```

---

## 🔧 LLaVA集成

- [ ] **LLaVA目录存在**
  ```bash
  ls -la LLaVA/
  # 应看到llava/目录
  ```

- [ ] **LLaVA可导入**
  ```bash
  python -c "from llava.model.builder import load_pretrained_model; print('✓ OK')"
  # 应显示: ✓ OK
  ```

- [ ] **LLaVA所有组件**
  ```bash
  python -c "
  from llava.constants import IMAGE_TOKEN_INDEX
  from llava.conversation import conv_templates
  from llava.mm_utils import tokenizer_image_token
  print('✓ All LLaVA components OK')
  "
  # 应显示: ✓ All LLaVA components OK
  ```

---

## 🤖 分层UAV模块

- [ ] **MAC记忆模块**
  ```bash
  python -c "from hierarchical_uav.mac_memory import NeuralMemory; print('✓ OK')"
  # 应显示: ✓ OK
  ```

- [ ] **模型配置**
  ```bash
  python -c "from hierarchical_uav.models import UAVConfig, LLaVAWithMAC; print('✓ OK')"
  # 应显示: ✓ OK
  ```

- [ ] **通信模块**
  ```bash
  python -c "from hierarchical_uav.communication import SelfMatchingGate; print('✓ OK')"
  # 应显示: ✓ OK
  ```

- [ ] **运行单元测试**
  ```bash
  python hierarchical_uav/mac_memory/neural_memory.py
  # 应显示: ✓ NeuralMemory tests passed!

  python hierarchical_uav/communication/self_matching.py
  # 应显示: ✓ Self-Matching Module tests passed!
  ```

---

## 💾 数据准备

- [ ] **数据目录结构**
  ```bash
  tree -L 2 data/
  # 应显示:
  # data/
  # ├── metadata/
  # │   ├── airspatial_agent_test_task1.jsonl
  # │   ├── airspatial_rec_test.jsonl
  # │   └── ...
  # └── images/
  #     └── airspatial/
  ```

- [ ] **测试数据存在**
  ```bash
  wc -l data/metadata/airspatial_agent_test_task1.jsonl
  # 应显示行数 > 0
  ```

- [ ] **图像文件存在**
  ```bash
  ls data/images/airspatial/*.jpg | head -5
  # 应显示图像文件列表
  ```

---

## 🎯 模型文件

- [ ] **AirSpatialBot模型**
  ```bash
  ls -lh models/AirSpatialBot/
  # 应看到 config.json, pytorch_model.bin 等文件
  ```

- [ ] **CLIP模型**
  ```bash
  ls -lh models/clip-vit-large-patch14-336/
  # 应看到 config.json, pytorch_model.bin 等文件
  ```

- [ ] **测试CLIP加载**
  ```bash
  python test_local_clip.py
  # 应显示: ✅ 本地 CLIP 模型加载成功！
  ```

---

## 🚀 功能测试

- [ ] **创建小测试集**
  ```bash
  head -n 10 data/metadata/airspatial_agent_test_task1.jsonl > data/metadata/test_small.jsonl
  ```

- [ ] **测试配置创建**
  ```bash
  python -c "
  from hierarchical_uav.models import UAVConfig
  config = UAVConfig.create_huav_config(
      model_path='./models/AirSpatialBot',
      vision_tower='./models/clip-vit-large-patch14-336'
  )
  print('✓ Config created:', config.uav_type)
  "
  # 应显示: ✓ Config created: UAVType.HIGH_UAV
  ```

---

## 🔌 网络和通信

- [ ] **端口50051可用**
  ```bash
  # 检查端口是否被占用
  netstat -tlnp | grep 50051
  # 应无输出（端口未被占用）
  ```

- [ ] **Socket通信测试**
  ```bash
  # 启动测试服务器（Terminal 1）
  python hierarchical_uav/communication/grpc_server.py

  # 测试客户端（Terminal 2）
  python hierarchical_uav/communication/grpc_client.py
  # 应显示连接成功
  ```

---

## 📊 完成度检查

计算完成的检查项：

```bash
# 统计完成的项目
# 手动检查上面所有 [ ] 并标记为 [x]
# 然后运行：
grep -c "\[x\]" INSTALLATION_CHECKLIST.md
# 与总数对比
grep -c "\[ \]" INSTALLATION_CHECKLIST.md
```

---

## ⚠️ 如果某些检查失败

### LLaVA导入失败
```bash
cd LLaVA
pip install -e .
cd ..
```

### MAC模块导入失败
```bash
# 确认在正确的目录
pwd  # 应该在 AirSpatialBot/ 目录

# 检查文件是否存在
ls hierarchical_uav/mac_memory/neural_memory.py
```

### 数据文件缺失
```bash
# 重新运行数据组织脚本
./setup_data.sh

# 或手动下载
huggingface-cli download erenzhou/AirSpatial --repo-type dataset --local-dir ./airspatial_data
```

### 模型文件缺失
```bash
# 下载AirSpatialBot
huggingface-cli download erenzhou/AirSpatialBot --local-dir ./models/AirSpatialBot

# 下载CLIP
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir ./models/clip-vit-large-patch14-336
```

---

## ✅ 全部通过后

如果所有检查都通过，你可以：

1. **运行快速测试**:
   ```bash
   ./run_hierarchical_uav.sh h-uav
   ```

2. **查看完整文档**:
   - `DEPLOYMENT_GUIDE.md` - 完整部署指南
   - `QUICKSTART_HIERARCHICAL.md` - 快速开始
   - `HIERARCHICAL_UAV_README.md` - 详细文档

3. **开始实验**! 🚁✨

---

**检查清单最后更新**: 2025-01-20
