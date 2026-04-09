# -*- coding: utf-8 -*-
"""
ComfyUI custom nodes for Sony Woosh - 修复 description 为 None 的错误
"""

import os
import sys
import warnings
import torch
import folder_paths
from typing import Dict, Any, List, Tuple
import comfy.model_management as model_management

# ============================================================
# 定义 synchformer 权重文件的查找路径（按优先级）
# ============================================================
SYNCHFORMER_WEIGHT_CANDIDATES = [
    r"F:\ComfyUI-WorkFisher-V2\ComfyUI-WorkFisher-V2\hf_cache\models--hkchengrex--MMAudio\snapshots\default\ext_weights\synchformer_state_dict.pth",
    r"F:\ComfyUI-WorkFisher-V2\ComfyUI-WorkFisher-V2\hf_cache\models\default\synchformer_state_dict.pth",
    os.environ.get("SYNCHFORMER_WEIGHT_PATH", ""),
    os.path.join(folder_paths.models_dir, "synchformer_state_dict.pth"),
    os.path.join(os.getcwd(), "synchformer_state_dict.pth"),
]

SYNCHFORMER_WEIGHT_PATH = None
for candidate in SYNCHFORMER_WEIGHT_CANDIDATES:
    if candidate and os.path.exists(candidate):
        SYNCHFORMER_WEIGHT_PATH = candidate
        break

if SYNCHFORMER_WEIGHT_PATH is None:
    raise FileNotFoundError(
        "找不到 synchformer 权重文件 (synchformer_state_dict.pth)。\n"
        "请从以下地址下载：\n"
        "  https://hf-mirror.com/hkchengrex/MMAudio/resolve/main/ext_weights/synchformer_state_dict.pth\n"
        "然后将其放置到正确位置。"
    )

print(f"[Woosh] 使用 synchformer 权重文件: {SYNCHFORMER_WEIGHT_PATH}")

# 补丁 huggingface_hub
try:
    import huggingface_hub
    original_hf_hub_download = huggingface_hub.hf_hub_download
    def patched_hf_hub_download(repo_id, filename, **kwargs):
        if filename == "synchformer_state_dict.pth" and repo_id == "hkchengrex/MMAudio":
            return SYNCHFORMER_WEIGHT_PATH
        return original_hf_hub_download(repo_id, filename, **kwargs)
    huggingface_hub.hf_hub_download = patched_hf_hub_download
except ImportError:
    pass

# ============================================================
# 官方 Woosh 导入
# ============================================================
try:
    from woosh.model.ldm import LatentDiffusionModel
    from woosh.model.video_kontext import VideoKontext
    from woosh.components.base import LoadConfig
    from woosh.inference.flowmatching_sampler import flowmatching_integrate
    from woosh.utils.video import SynchformerProcessor
    from woosh.utils.videoio import extract_video_frames, remux_video
except ImportError as e:
    raise ImportError(
        "无法导入 Woosh 官方包。请先正确安装：\n"
        "  git clone https://github.com/SonyResearch/Woosh\n"
        "  cd Woosh\n"
        "  pip install -e .\n"
        f"原始错误: {e}"
    )

try:
    from torchvision.io import read_video
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# ============================================================
# 辅助函数
# ============================================================
def _get_device():
    try:
        return model_management.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _models_root():
    env_path = os.environ.get("WOOSH_CKPT", "").strip()
    if env_path:
        return os.path.abspath(env_path)
    return os.path.join(folder_paths.models_dir, "woosh", "checkpoints")

def _resolve_model_dir(model_key: str) -> str:
    root = _models_root()
    candidates = {
        "woosh-flow": ["Woosh-Flow", "woosh-flow", "flow"],
        "woosh-dflow": ["Woosh-DFlow", "woosh-dflow", "dflow"],
        "woosh-vflow": ["Woosh-VFlow", "Woosh-VFlow-8s", "woosh-vflow", "vflow"],
    }
    if model_key not in candidates:
        raise ValueError(f"未知的模型键: {model_key}")
    for name in candidates[model_key]:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"找不到模型目录 '{model_key}'。\n"
        f"搜索根目录: {root}\n"
        f"期望的文件夹名: {candidates[model_key]}\n"
        "请确保已将官方权重解压到上述位置。"
    )

def _make_audio_output(waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.float().cpu()
    return {"waveform": waveform, "sample_rate": sample_rate}

def _normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    max_val = waveform.abs().max()
    if max_val > 1.0 and max_val > 1e-6:
        waveform = waveform / max_val
    return waveform

def _call_flowmatching(ldm, noise, cond, cfg, device, dtype):
    result = flowmatching_integrate(
        ldm,
        noise=noise,
        cond=cond,
        cfg=cfg,
        atol=0.001,
        rtol=0.001,
        device=device,
        dtype=dtype,
    )
    if isinstance(result, tuple):
        return result[0]
    else:
        return result

# ============================================================
# 模型管理器
# ============================================================
class ModelCache:
    _models = {}
    _feature_extractors = {}

    @classmethod
    def get(cls, model_type: str):
        if model_type in cls._models:
            return cls._models[model_type]

        key_map = {
            "Woosh-Flow": "woosh-flow",
            "Woosh-DFlow": "woosh-dflow",
            "Woosh-VFlow": "woosh-vflow",
        }
        if model_type not in key_map:
            raise ValueError(f"不支持的模型类型: {model_type}")
        model_key = key_map[model_type]
        model_dir = _resolve_model_dir(model_key)
        device = _get_device()

        if model_type == "Woosh-VFlow":
            ldm = VideoKontext(LoadConfig(path=model_dir))
            ldm = ldm.eval().to(device)
            if model_type not in cls._feature_extractors:
                featuresModel = SynchformerProcessor(frame_rate=24).eval().to(device)
                cls._feature_extractors[model_type] = featuresModel
        else:
            ldm = LatentDiffusionModel(LoadConfig(path=model_dir))
            ldm = ldm.eval().to(device)
            cls._feature_extractors[model_type] = None

        cls._models[model_type] = ldm
        return ldm

    @classmethod
    def get_feature_extractor(cls, model_type: str):
        if model_type not in cls._feature_extractors:
            cls.get(model_type)
        return cls._feature_extractors.get(model_type)

# ============================================================
# 节点1: 文本生成音频
# ============================================================
class WooshTextToAudioSony:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a whoosh sound effect"}),
                "duration": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 10.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "model_type": (["Woosh-Flow", "Woosh-DFlow"], {"default": "Woosh-Flow"}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/generation"

    def generate(self, prompt, duration, guidance_scale, seed, model_type):
        device = _get_device()
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        ldm = ModelCache.get(model_type)

        steps = 501
        noise = torch.randn(1, 128, steps, device=device)

        cond = ldm.get_cond(
            {"audio": None, "description": [prompt]},
            no_dropout=True,
            device=device,
        )

        with torch.inference_mode():
            x_fake = _call_flowmatching(
                ldm,
                noise=noise,
                cond=cond,
                cfg=guidance_scale,
                device=device,
                dtype=torch.float32 if device.type == "mps" else torch.float64,
            )
            waveform_full = ldm.autoencoder.inverse(x_fake)
            if waveform_full.dim() == 2:
                waveform_full = waveform_full.unsqueeze(1)
            waveform_full = waveform_full.squeeze(0).cpu()
            if waveform_full.dim() == 1:
                waveform_full = waveform_full.unsqueeze(0)

        full_duration = waveform_full.shape[-1] / 48000
        print(f"[Woosh Text] 生成完整音频时长: {full_duration:.2f} 秒")

        target_samples = int(duration * 48000)
        if target_samples < waveform_full.shape[-1]:
            waveform = waveform_full[:, :target_samples]
            print(f"[Woosh Text] 裁剪后音频时长: {waveform.shape[-1] / 48000:.2f} 秒")
        else:
            waveform = waveform_full

        waveform = _normalize_audio(waveform)
        sample_rate = 48000
        return (_make_audio_output(waveform, sample_rate),)

# ============================================================
# 节点2: 短视频 -> 音频（修复 description 为 None 的问题）
# ============================================================
class WooshVideoToAudioSony:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 10.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/generation"

    def generate(self, video, guidance_scale, seed, prompt=""):
        device = _get_device()
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        model_type = "Woosh-VFlow"
        ldm = ModelCache.get(model_type)
        featuresModel = ModelCache.get_feature_extractor(model_type)
        if featuresModel is None:
            raise RuntimeError("视频特征提取器加载失败")

        frames, fps = self._load_video_frames_with_fps(video)
        if frames is None:
            raise RuntimeError("无法加载视频输入")
        frames = frames.to(device)

        with torch.inference_mode():
            features = featuresModel(frames, fps)
            # 修复：description 必须为列表，即使为空字符串也要用 [""] 而不是 None
            cond = ldm.get_cond(
                {
                    "audio": None,
                    "description": [prompt] if prompt else [""],
                    "synch_out": features["synch_out"],
                },
                no_dropout=True,
                device=device,
            )

        video_duration = frames.shape[0] / fps
        steps = max(1, int(round(video_duration * 100.125)))
        noise = torch.randn(1, 128, steps, device=device)

        with torch.inference_mode():
            x_fake = _call_flowmatching(
                ldm,
                noise=noise,
                cond=cond,
                cfg=guidance_scale,
                device=device,
                dtype=torch.float32 if device.type == "mps" else torch.float64,
            )
            waveform = ldm.autoencoder.inverse(x_fake)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)
            waveform = waveform.squeeze(0).cpu()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        waveform = _normalize_audio(waveform)
        sample_rate = 48000
        return (_make_audio_output(waveform, sample_rate),)

    def _load_video_frames_with_fps(self, video_input):
        if hasattr(video_input, '__class__') and video_input.__class__.__name__ == 'VideoFromFile':
            video_path = None
            if hasattr(video_input, '_VideoFromFile__file'):
                video_path = getattr(video_input, '_VideoFromFile__file')
            if video_path is None and hasattr(video_input, 'get_stream_source'):
                video_path = video_input.get_stream_source()
            if video_path is None:
                for attr in ['path', 'filename', 'file_path', 'video_path']:
                    if hasattr(video_input, attr):
                        video_path = getattr(video_input, attr)
                        if video_path and os.path.exists(video_path):
                            break
            if video_path is None:
                raise ValueError(f"无法从 VideoFromFile 对象中提取文件路径。可用属性: {dir(video_input)}")
            if not TORCHVISION_AVAILABLE:
                raise RuntimeError("处理视频需要 torchvision")
            frames, _, info = read_video(video_path, pts_unit="sec")
            fps = info.get("video_fps", 24.0)
            frames = frames.float() / 255.0
            return frames, fps
        elif isinstance(video_input, str):
            if not TORCHVISION_AVAILABLE:
                raise RuntimeError("处理视频路径需要 torchvision")
            frames, _, info = read_video(video_input, pts_unit="sec")
            fps = info.get("video_fps", 24.0)
            frames = frames.float() / 255.0
            return frames, fps
        elif isinstance(video_input, torch.Tensor):
            tensor = video_input.detach().float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            if tensor.ndim == 4:
                if tensor.shape[0] == 1:
                    tensor = tensor[0]
                if tensor.shape[-1] not in (1, 3):
                    if tensor.shape[1] in (1, 3):
                        tensor = tensor.permute(0, 2, 3, 1)
            elif tensor.ndim == 3:
                pass
            else:
                raise ValueError(f"不支持的视频张量形状: {tensor.shape}")
            return tensor, 24.0
        else:
            raise TypeError(f"不支持的视频类型: {type(video_input)}")

    def _load_video_frames(self, video_input):
        frames, _ = self._load_video_frames_with_fps(video_input)
        return frames

# ============================================================
# 节点3: 长视频 -> 音频
# ============================================================
class WooshLongVideoToAudioSony:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "segment_duration": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "overlap_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 10.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/generation"

    def generate(self, video, segment_duration, overlap_duration, guidance_scale, seed, prompt=""):
        base_node = WooshVideoToAudioSony()
        frames, fps = base_node._load_video_frames_with_fps(video)
        if frames is None:
            raise RuntimeError("无法加载视频")
        total_frames = frames.shape[0]
        total_duration = total_frames / fps

        if total_duration <= segment_duration + 0.1:
            return base_node.generate(video, guidance_scale, seed, prompt)

        step_duration = segment_duration - overlap_duration
        if step_duration <= 0:
            step_duration = segment_duration / 2
            overlap_duration = segment_duration / 2
            warnings.warn(f"重叠时长过长，已自动调整为 {overlap_duration:.1f}s")

        segments = []
        start_time = 0.0
        while start_time < total_duration:
            end_time = min(start_time + segment_duration, total_duration)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            seg_frames = frames[start_frame:end_frame]
            segments.append(seg_frames)
            start_time += step_duration
            if len(segments) > 100:
                warnings.warn("片段数超过 100，强制停止")
                break

        audio_segments = []
        for idx, seg_frames in enumerate(segments):
            print(f"生成音频片段 {idx+1}/{len(segments)}...")
            audio_dict = base_node.generate(seg_frames, guidance_scale, seed + idx, prompt)
            audio_segments.append(audio_dict[0])

        final_audio = self._stitch_audio(audio_segments, overlap_duration)
        return (final_audio,)

    def _stitch_audio(self, segments: List[Dict], overlap_sec: float) -> Dict:
        if not segments:
            raise ValueError("没有音频片段可拼接")
        sr = segments[0]["sample_rate"]
        for seg in segments:
            if seg["sample_rate"] != sr:
                raise RuntimeError("片段采样率不一致")
        overlap_samples = int(overlap_sec * sr)
        waveforms = [seg["waveform"].squeeze(0) for seg in segments]
        channels = waveforms[0].shape[0]

        total_len = 0
        for i, w in enumerate(waveforms):
            if i == 0:
                total_len += w.shape[1]
            else:
                total_len += w.shape[1] - overlap_samples
        final_wave = torch.zeros((channels, total_len), dtype=torch.float32)

        pos = 0
        for i, w in enumerate(waveforms):
            seg_len = w.shape[1]
            if i == 0:
                final_wave[:, pos:pos+seg_len] = w
                pos += seg_len
            else:
                start_overlap = pos - overlap_samples
                end_overlap = pos
                fade_out = torch.linspace(1, 0, overlap_samples, device=w.device).view(1, -1)
                final_wave[:, start_overlap:end_overlap] *= fade_out
                fade_in = torch.linspace(0, 1, overlap_samples, device=w.device).view(1, -1)
                final_wave[:, start_overlap:end_overlap] += w[:, :overlap_samples] * fade_in
                if seg_len > overlap_samples:
                    final_wave[:, pos:pos+seg_len-overlap_samples] = w[:, overlap_samples:]
                pos += seg_len - overlap_samples

        final_wave = _normalize_audio(final_wave)
        final_wave = final_wave.unsqueeze(0)
        return {"waveform": final_wave, "sample_rate": sr}

# ============================================================
# 节点注册
# ============================================================
NODE_CLASS_MAPPINGS = {
    "WooshTextToAudioSony": WooshTextToAudioSony,
    "WooshVideoToAudioSony": WooshVideoToAudioSony,
    "WooshLongVideoToAudioSony": WooshLongVideoToAudioSony,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WooshTextToAudioSony": "Woosh Text to Audio (Sony)",
    "WooshVideoToAudioSony": "Woosh Video to Audio (Sony)",
    "WooshLongVideoToAudioSony": "Woosh Long Video to Audio (Sony)",
}