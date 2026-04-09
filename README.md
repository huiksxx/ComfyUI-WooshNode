# 🎧 ComfyUI-WooshNode
Sony Woosh 音频生成自定义节点 | 文本/视频生成沉浸式音效

> 基于 Sony Research 官方 Woosh 模型，为 ComfyUI 打造的一站式音频生成节点，支持文本生成、视频匹配音效、长视频分段生成，**修复了官方模型 `description=None` 报错问题**，支持空提示词生成。

---

## ✨ 功能特性
- 🎤 **文本生成音频**：通过文字描述生成任意音效（如门关闭、环境音、特效音）
- 🎬 **视频生成音频**：自动匹配视频画面生成同步音效，支持自定义提示词增强
- 🎞️ **长视频分段生成**：自动拆分长视频为片段，生成后无缝拼接，支持重叠过渡
- ⚡ **模型缓存优化**：首次加载后缓存模型，避免重复加载，大幅提升生成速度
- 🛠️ **鲁棒性增强**：修复了官方模型 `description` 为空时的崩溃问题，兼容空提示词场景

---

## 📦 安装方法
### 方法1：ComfyUI Manager 一键安装（推荐）
1. 打开 ComfyUI → 进入 `Manager` → `Install Custom Nodes`
2. 搜索 `ComfyUI-WooshNode` 或直接粘贴仓库地址：
   `https://github.com/huiksxx/ComfyUI-WooshNode`
3. 点击 `Install` → 重启 ComfyUI 即可使用

### 方法2：手动 Git 安装
1. 进入 ComfyUI 的 `custom_nodes` 文件夹
2. 打开终端/命令行，执行：
   ```bash
   git clone https://github.com/huiksxx/ComfyUI-WooshNode.git

   📥 模型权重下载（必须）
Woosh 节点需要下载官方模型权重文件，请按以下步骤操作。

1. 下载权重文件
访问 Woosh SFX Releases 下载以下文件：

Woosh-Flow.zip：文本生成音频主模型（约 1.17 GB）

Woosh-AE.zip：音频编解码器（约 785 MB）

TextConditionerA.zip：文本条件器（约 1.21 GB）

Woosh-VFlow-8s.zip：视频生成音频模型（可选，约 1.43 GB）

TextConditionerV.zip：视频条件器（可选，约 1.21 GB）

synchformer_state_dict.pth：视频特征提取器（可选，约 2.5 GB），下载地址：hf-mirror

视频相关模型只需在需要视频生成音频时下载。

2. 放置到指定目录
解压后，将所有文件夹和 .pth 文件放入：

text

复制

下载
ComfyUI/models/woosh/checkpoints/
目录结构示例：

text

复制

下载
ComfyUI/models/woosh/checkpoints/
├── Woosh-Flow/
├── Woosh-AE/
├── TextConditionerA/
├── Woosh-VFlow-8s/      （可选）
├── TextConditionerV/     （可选）
└── synchformer_state_dict.pth
3. 安装官方 Woosh 库（必须）
在 ComfyUI 的 Python 环境中执行：

bash

复制

下载
git clone https://github.com/SonyResearch/Woosh
cd Woosh
pip install -e .
📦 依赖
Python >= 3.8

PyTorch >= 2.0

torchvision（视频功能需要）

transformers, omegaconf, huggingface_hub（Woosh 库会自动安装）

🚀 节点使用说明
安装并放置好模型后，ComfyUI 节点列表中会出现以下三个节点（位于 audio/generation 分类）：

Woosh Text to Audio：文本生成音效。输入 prompt，设置时长（1-5 秒）、引导强度（推荐 4.5）。

Woosh Video to Audio：短视频生成同步音效。输入 ≤8 秒的视频，可选 prompt 辅助。

Woosh Long Video to Audio：长视频自动分段生成。自动切分视频，生成音频后无缝拼接，支持重叠过渡。

简单工作流示例
text

复制

下载
[Load Video] → [Woosh Long Video to Audio] → [Save Audio MP3]
设置 segment_duration=8.0, overlap_duration=1.0

节点会自动分段生成并拼接，输出完整音频

⚠️ 注意事项
文本生成音频最大时长为 5 秒，超过会自动裁剪。

视频生成音频建议使用 ≤8 秒的短视频；长视频请使用 Woosh Long Video to Audio 节点。

首次运行视频节点会自动下载 synchformer_state_dict.pth（约 2.5 GB），请确保网络畅通或手动放置到 models/woosh/checkpoints/ 目录。

如果使用空提示词（不输入 prompt），节点会生成与画面匹配的默认音效（模型可能会输出随机人声，建议明确指定提示词）。

本节点修复了官方模型 description=None 的崩溃问题，无需额外操作。

模型权重版权归 Sony Research 所有，请遵守其开源协议。
