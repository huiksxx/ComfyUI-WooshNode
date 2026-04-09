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
