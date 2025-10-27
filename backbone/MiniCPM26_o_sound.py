import os
import sys
import math
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv
from moviepy import *
import tempfile
import librosa
import torch
from transformers import AutoModel, AutoTokenizer

import soundfile as sf


# 与原脚本一致的路径注入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 复用你现有的采样策略工具
from backbone_utils import extract_frames_base64, get_max_frame_and_interval, reconstruct_video, encode_video
# 注：MiniCPM 不需要 base64 合成视频；这里只复用 get_max_frame_and_interval

load_dotenv()

def get_video_chunk_content(video_path, flatten=True):
    """
    秒驱动：每个 <unit> = [1 张图像, 1 秒音频(16000样本)]。
    - 图像：t = i + 0.5 秒；若超到尾部，复用倒数第二帧（避免坏尾帧/告警）。
    - 音频：固定 1 秒片段，长度严格 = 16000（不够就零填充）。
    """
    sr = 16000
    video = VideoFileClip(video_path)

    # 导出/生成音频（mono, 16k）
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
    try:
        if video.audio is not None:
            video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=sr, logger=None)
            audio_np, _ = librosa.load(temp_audio_file_path, sr=sr, mono=True)
        else:
            duration = float(video.duration or 0.0)
            n_samples = max(1, int(round(sr * duration)))
            audio_np = np.zeros(n_samples, dtype=np.float32)
            sf.write(temp_audio_file_path, audio_np, sr, subtype="PCM_16")

        # 时长与单位数（向上取整到整秒）
        duration = float(video.duration or 0.0)
        total_seconds = max(1, int(math.ceil(duration)))

        # 预先把音频零填充到整秒，确保每段都有 16000 个样本
        need = total_seconds * sr
        if len(audio_np) < need:
            audio_np = np.pad(audio_np.astype(np.float32, copy=False), (0, need - len(audio_np)))
        else:
            audio_np = audio_np.astype(np.float32, copy=False)

        # 计算“避开最后一帧”的安全取帧时间
        fps = (getattr(video, "fps", None)
               or (getattr(getattr(video, "reader", None), "fps", None))
               or 30.0)
        nframes = getattr(getattr(video, "reader", None), "nframes", None)

        if nframes and nframes > 1:
            # 用“倒数第二帧”的时间，避开坏尾帧
            avoid_last_t = max(0.0, (nframes - 2) / fps + 1e-3)
        else:
            # 退化：离末尾留一点余量
            avoid_last_t = max(0.0, duration - (2.0 / fps))

        # 预取一张“安全末帧”图像（供复用）
        try:
            safe_frame_arr = video.get_frame(avoid_last_t if duration > 0 else 0.0)
        except Exception:
            # 极端情况下再退回 0s
            safe_frame_arr = video.get_frame(0.0)
        safe_image = Image.fromarray(safe_frame_arr.astype(np.uint8))

        contents = []
        for i in range(total_seconds):
            # --- 图像 ---
            t_wanted = i + 0.5
            if t_wanted <= avoid_last_t:
                frame = video.get_frame(t_wanted)
                image = Image.fromarray(frame.astype(np.uint8))
            else:
                # 直接复用安全帧，避免触发坏尾帧/告警
                image = safe_image

            # --- 1 秒音频，严格 16000 样本 ---
            start = i * sr
            end = start + sr
            chunk_audio = audio_np[start:end]
            if chunk_audio.shape[0] != sr:
                # 理论上不会走到这里；双保险
                pad = sr - chunk_audio.shape[0]
                chunk_audio = np.pad(chunk_audio, (0, pad))

            if flatten:
                contents.extend(["<unit>", image, chunk_audio])
            else:
                contents.append(["<unit>", image, chunk_audio])
 
        return contents
    finally:
        try:
            os.remove(temp_audio_file_path)
        except Exception:
            pass
        video.close()

class MiniCPM26O_Sound:
    def __init__(self, model_name_or_path="openbmb/MiniCPM-o-2_6", max_tokens=36864, language="en"):

        self.model_name = model_name_or_path
        self.max_tokens = max_tokens
        self.language = language

        # 与你原始脚本保持一致的 patch / token 假设（仅用于采样策略）
        self.model_h = 448
        self.model_w = 448
        self.model_max_tokens = 576
        self.model_min_tokens = 64

        # MiniCPM 模型与tokenizer（与官方示例一致）
        torch_dtype = torch.bfloat16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
        ).eval().to(device)

        # 可选：初始化 TTS（即使默认不生成语音）
        self.model.init_tts()
        # 如遇老版本 PyTorch 的 BFloat16 TTS 问题，可改为：
        # self.model.tts.float()

    def get_completion(self, system_prompt, user_prompt, video_path):
        """
        完全沿用你 Qwen 版本的视频处理逻辑；只是把最后“构造消息并推理”的部分
        改为 MiniCPM 的 omni 输入与 chat 接口。
        """
        # ======（1）沿用原逻辑：采样参数 ======
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise IOError("Could not read first frame of video.")
        
        nframes, interval = get_max_frame_and_interval(
            self.max_tokens, cap, frame, self.model_h, self.model_w, self.model_max_tokens, self.model_min_tokens
        )

        if not original_fps or original_fps <= 0:
            original_fps = 30.0  # 默认帧率

        time_between_frames_sec = interval / original_fps
        target_fps = 1.0 / time_between_frames_sec if time_between_frames_sec > 0 else 1.0

        cap.release()

        # ======（2）沿用原逻辑：抽帧 → 重构视频 → 编码 ======
        base64_frames = extract_frames_base64(video_path, nframes=nframes, interval=interval)

        clip = VideoFileClip(video_path)
        audio = clip.audio  # 获取音频对象
        audio_path = None
        if audio is not None:
            audio_path = "extracted_audio.wav"
            # 你的环境 moviepy 不支持 verbose=；保留 logger=None 即可静默
            audio.write_audiofile(audio_path, logger=None)

        reconstruct_video(base64_frames, audio_path, target_fps, "reconstruct_video.mp4")
        # 注意：MiniCPM 不需要 base64 视频，这里只是保持原流程不变
        _ = encode_video("reconstruct_video.mp4")  # 与原流程一致地得到 base64，但后续不再使用

        # ======（3）仅在“构造消息”这里改成 MiniCPM 的格式 ======

        # 用官方示例的函数把“重构后的视频”切成 ["<unit>", image, audio, ...]
        contents = get_video_chunk_content("reconstruct_video.mp4", flatten=True)
        
        

        # 组装 MiniCPM 的系统消息（并拼接你传入的 system_prompt）
        sys_msg = self.model.get_sys_prompt(mode='omni', language=self.language)
        if isinstance(sys_msg, dict) and sys_msg.get("role") == "system":
            sys_msg["content"] = (sys_msg.get("content") or "") + ("\n" + system_prompt if system_prompt else "")

        # 把文本 user_prompt 追加到同一轮 user 的 content 尾部（MiniCPM 支持混合文本）
        msgs = [
            sys_msg,
            {"role": "user", "content": [user_prompt] + contents }
        ]

        # ======（4）MiniCPM 推理（不生成 TTS 音频）======

        try:
            res = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.5,
                max_new_tokens=4096,
                omni_input=True,         # 关键：omni 多模态
                use_tts_template=False,  # 不需要 TTS
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True
            )

            # 统一返回文本（与原函数风格一致）
            if isinstance(res, dict):
                # 常见字段：res.get("text") 或 res.get("response")
                txt = res.get("text") or res.get("response") or ""
                return txt if txt.strip() else str(res)
            return str(res)

        except Exception as e:
            return f"Error: {str(e)}"

