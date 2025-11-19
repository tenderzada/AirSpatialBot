# LLaVA 集成指南

本文档说明如何在 AirSpatialBot 项目中使用 LLaVA。

## 📦 已完成的集成

- ✅ 克隆了 LLaVA 仓库到项目根目录
- ✅ 创建了自动安装脚本 `install_llava.sh`
- ✅ AirSpatialBot 的评估脚本已经可以使用 LLaVA

---

## 🚀 快速安装

### 方法 1: 使用自动安装脚本（推荐）

```bash
# 运行安装脚本
./install_llava.sh
```

这个脚本会：
1. 确认 LLaVA 目录存在（已克隆）
2. 以可编辑模式安装 LLaVA
3. 可选安装训练相关依赖
4. 验证安装是否成功

### 方法 2: 手动安装

```bash
# 进入 LLaVA 目录
cd LLaVA

# 安装 LLaVA（可编辑模式）
pip install -e .

# 安装训练依赖（可选）
pip install -e ".[train]"

# 返回项目根目录
cd ..

# 验证安装
python -c "from llava.model.builder import load_pretrained_model; print('Success!')"
```

---

## 📂 项目结构

集成后的目录结构：

```
AirSpatialBot/
├── LLaVA/                              # ✅ 新增：LLaVA 仓库
│   ├── llava/                          # LLaVA 核心模块
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── conversation.py
│   │   ├── mm_utils.py
│   │   ├── model/                      # 模型构建和加载
│   │   │   ├── builder.py
│   │   │   ├── language_model/
│   │   │   └── multimodal_encoder/
│   │   ├── eval/                       # 评估工具
│   │   └── train/                      # 训练脚本
│   ├── pyproject.toml
│   ├── README.md
│   └── ...
├── llava_scripts/                      # AirSpatialBot 的 LLaVA 脚本
│   ├── eval/
│   │   ├── batch_inference_3db.py      # 使用 LLaVA 进行推理
│   │   └── compute_metric_3db.py
│   └── generate_sft/
├── data/
├── models/
├── install_llava.sh                    # ✅ 新增：LLaVA 安装脚本
├── run_eval_local_clip.sh
└── ...
```

---

## 🔧 LLaVA 依赖说明

### 核心依赖

```
torch==2.1.2
torchvision==0.16.2
transformers==4.37.2
tokenizers==0.15.1
accelerate==0.21.0
peft
bitsandbytes
gradio==4.16.0
einops==0.6.1
timm==0.6.13
```

### 训练依赖（可选）

```
deepspeed==0.12.6
wandb
ninja
```

### 与 AirSpatialBot 的兼容性

- ✅ **兼容**：LLaVA 的依赖与 AirSpatialBot 兼容
- ✅ **无冲突**：两者使用相同的 transformers 和 torch 版本
- ✅ **可编辑安装**：使用 `pip install -e .` 方便开发和调试

---

## 💻 使用示例

### 1. 在评估脚本中使用 LLaVA

AirSpatialBot 的评估脚本已经集成了 LLaVA：

```bash
# 使用本地 CLIP 模型运行评估
./run_eval_local_clip.sh
```

脚本内部使用的 LLaVA 功能：

```python
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
```

### 2. 加载 AirSpatialBot 模型

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

# 模型路径
model_path = "./models/AirSpatialBot"
model_name = get_model_name_from_path(model_path)

# 加载模型
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name
)
```

### 3. 使用本地 CLIP 模型

```python
import os
from llava.model.builder import load_pretrained_model

# 设置离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 指定本地 CLIP 路径
vision_tower = "/mnt/data/clip-vit-large-patch14-336"

# 加载模型
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="./models/AirSpatialBot",
    model_base=None,
    model_name="llava-v1.5-7b"
)

# 覆盖 vision tower 路径
if hasattr(model.config, 'mm_vision_tower'):
    model.config.mm_vision_tower = vision_tower
```

### 4. 构建对话和推理

```python
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from PIL import Image

# 准备问题
question = "What color is this car?"
qs = DEFAULT_IMAGE_TOKEN + '\n' + question

# 使用对话模板
conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# 转换为 token IDs
input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
).unsqueeze(0).cuda()

# 加载和预处理图像
image = Image.open("path/to/image.jpg")
image_tensor = image_processor.preprocess(
    image, return_tensors='pt'
)['pixel_values'].half().cuda()

# 推理
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        temperature=0.2,
        max_new_tokens=256
    )

# 解码输出
output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print(output)
```

---

## 📊 验证安装

### 验证 LLaVA 模块导入

```bash
python -c "
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
print('✓ LLaVA 安装成功！')
"
```

### 验证 CLIP 模型加载

```bash
python test_local_clip.py
```

### 运行完整评估

```bash
# 小规模测试
head -n 5 data/metadata/airspatial_rec_test.jsonl > data/metadata/test_small.jsonl

# 修改 run_eval_local_clip.sh 使用小数据集
# TEST_FILE="./data/metadata/test_small.jsonl"

./run_eval_local_clip.sh
```

---

## 🔍 LLaVA 核心模块说明

### 1. `llava.model.builder`
- `load_pretrained_model()`: 加载预训练模型
- `get_model_name_from_path()`: 从路径获取模型名称

### 2. `llava.conversation`
- `conv_templates`: 对话模板字典
- 支持的模板: `llava_v1`, `llava_v1_mmtag`, `vicuna_v1`, etc.

### 3. `llava.mm_utils`
- `tokenizer_image_token()`: 将图像 token 添加到输入中
- `get_model_name_from_path()`: 路径解析
- `KeywordsStoppingCriteria`: 停止条件

### 4. `llava.constants`
- `IMAGE_TOKEN_INDEX`: 图像 token 的索引
- `DEFAULT_IMAGE_TOKEN`: 默认图像占位符 `<image>`
- `DEFAULT_IM_START_TOKEN`, `DEFAULT_IM_END_TOKEN`

---

## 🐛 常见问题

### 1. 导入错误：`ModuleNotFoundError: No module named 'llava'`

**解决方案**：
```bash
cd LLaVA
pip install -e .
cd ..
```

### 2. transformers 版本冲突

**解决方案**：
```bash
pip install transformers==4.37.2
```

### 3. CUDA 内存不足

**解决方案**：
```python
# 使用 8-bit 量化
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    load_8bit=True  # 启用 8-bit 量化
)
```

### 4. CLIP 模型仍然尝试下载

**解决方案**：
```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 然后运行脚本
./run_eval_local_clip.sh
```

---

## 📚 LLaVA 资源

- **官方仓库**: https://github.com/haotian-liu/LLaVA
- **论文**: https://arxiv.org/abs/2304.08485
- **模型库**: https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md
- **Demo**: https://llava.hliu.cc/

---

## 🔄 更新 LLaVA

如需更新到最新版本：

```bash
cd LLaVA
git pull origin main
pip install -e . --upgrade
cd ..
```

---

## 🎯 下一步

LLaVA 集成完成后，你可以：

1. **运行评估**：
   ```bash
   ./run_eval_local_clip.sh
   ```

2. **生成训练数据**：
   ```bash
   python llava_scripts/generate_sft/gen_sft_for_llava_3db_vqa.py
   ```

3. **微调 AirSpatialBot**：
   ```bash
   # 使用 LLaVA 的训练脚本
   cd LLaVA
   bash scripts/v1_5/finetune_lora.sh
   ```

4. **部署模型服务**：
   ```bash
   # 使用 vLLM 或 LLaVA 自带的服务
   python LLaVA/llava/serve/model_worker.py \
       --model-path ./models/AirSpatialBot \
       --host 0.0.0.0 \
       --port 8000
   ```

---

## ✅ 检查清单

安装完成后，确认：

- [ ] LLaVA 目录存在于项目根目录
- [ ] 可以成功导入 `llava` 模块
- [ ] `test_local_clip.py` 运行成功
- [ ] 评估脚本可以正常运行
- [ ] 模型可以正常加载

全部完成后，你就可以开始使用 AirSpatialBot 了！🎉
