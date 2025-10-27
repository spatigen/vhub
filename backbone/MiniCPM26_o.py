# minicpm26_o_stream.py
# 依赖: torch, transformers>=4.44, pillow, opencv-python, numpy, soundfile(可选), python-dotenv(可选)
# 继续使用你的 backbone_utils: get_max_frame_and_interval / extract_frames_base64

import os
import io
import cv2
import base64
import numpy as np
from uuid import uuid4
from PIL import Image

import torch
from transformers import AutoModel, AutoTokenizer

try:
    import soundfile as sf
except Exception:
    sf = None  # 如果只要文本，可以不装 soundfile

from backbone_utils import (
    get_max_frame_and_interval,
    extract_frames_base64,
)

class MiniCPMO26:
    """
    用 MiniCPM-o-2.6 的流式接口跑「视频→多图→文本」生成。
    仍沿用你原有的抽帧预算与采样逻辑。
    """
    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-o-2_6",
        device: str | None = None,
        attn_impl: str = "sdpa",          # 或 "flash_attention_2"
        torch_dtype = torch.bfloat16,     # GPU 建议 bfloat16；CPU 可用 float32
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype if self.device == "cuda" else torch.float32
        )
        self.model = self.model.eval()
        if self.device == "cuda":
            self.model = self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 语音合成（若需要音频输出）
        try:
            self.model.init_tts()
            # 某些旧 PyTorch 需要浮点精度切到 fp32
            self.model.tts.float()
        except Exception:
            pass

    @staticmethod
    def _b64_to_pil_list(b64_list: list[str]) -> list[Image.Image]:
        imgs = []
        for b in b64_list:
            data = base64.b64decode(b)
            im = Image.open(io.BytesIO(data)).convert("RGB")
            imgs.append(im)
        return imgs

    @staticmethod
    def _chunk_frames(frames: list[Image.Image], chunk_size: int = 8) -> list[list[Image.Image]]:
        """把帧切成若干小块，便于多次 streaming_prefill。"""
        if chunk_size <= 0:
            return [frames]
        return [frames[i:i+chunk_size] for i in range(0, len(frames), chunk_size)]

    def get_completion(
        self,
        video_path: str,
        system_prompt: str,
        user_prompt: str,
        # 抽帧预算相关 每448*448消耗64个token，最大图像分辨率支持1344*1344，分9个切片。最多支持64帧，参考下面链接
        # https://openbmb.notion.site/MiniCPM-o-2-6-A-GPT-4o-Level-MLLM-for-Vision-Speech-and-Multimodal-Live-Streaming-on-Your-Phone-185ede1b7a558042b5d5e45e6b237da9
        
        max_tokens: int = 36864,
        model_h: int = 448,
        model_w: int = 448,
        model_max_tokens: int = 576,
        model_min_tokens: int = 64,
        # 预填充帧分块大小（每次 prefill 塞多少帧）
        prefill_chunk: int = 12,
        # 生成相关参数
        temperature: float = 0.5,
        generate_audio: bool = False, #是否开启音频输出
        out_wav: str = "output.wav", #音频输出的保存路径
    ) -> dict:
        """
        返回:
            {"text": str, "wav": np.ndarray|None, "sr": int|None}
        """
        # --- 1) 打开视频 & 估算抽帧策略 ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise IOError("Could not read first frame of video.")

        nframes, interval = get_max_frame_and_interval(
            max_tokens, cap, frame, model_h, model_w, model_max_tokens, model_min_tokens
        )

        # 真正抽帧（base64）
        base64_frames = extract_frames_base64(video_path, nframes=nframes, interval=interval)

        cap.release()

        # 转 PIL.Image，MiniCPM 支持直接传 PIL 图像对象
        pil_frames = self._b64_to_pil_list(base64_frames)

        # --- 2) 构造流式会话：reset → prefill(system) → prefill(video chunks) → generate ---
        self.model.reset_session()  # 新会话清空 KV cache
        session_id = str(uuid4())

        sys_msg = {"role": "system", "content": [system_prompt]}
        _ = self.model.streaming_prefill(
            session_id=session_id,
            msgs=[sys_msg],
            tokenizer=self.tokenizer
        )

        # 分块预填充帧；在**最后一个分块**附上文本提示
        chunks = self._chunk_frames(pil_frames, chunk_size=prefill_chunk)
        for i, ch in enumerate(chunks):
            content = list(ch)
            if i == len(chunks) - 1:
                # 最后一个分块，把用户文本提示也一起 prefill 进去
                content.append(user_prompt)
            msgs = [{"role": "user", "content": content}]
            _ = self.model.streaming_prefill(
                session_id=session_id,
                msgs=msgs,
                tokenizer=self.tokenizer
            )

        # --- 3) 开始流式生成 ---
        res_iter = self.model.streaming_generate(
            session_id=session_id,
            tokenizer=self.tokenizer,
            temperature=temperature,
            generate_audio=generate_audio
        )

        text_out = ""
        wav_chunks = []
        sr = None

        if generate_audio:
            if sf is None:
                raise RuntimeError("需要音频输出，但未安装 soundfile。请先 pip install soundfile")
            for r in res_iter:
                # 官方流式返回里含 r.text / r.audio_wav / r.sampling_rate
                if hasattr(r, "text") and r.text:
                    text_out += r.text
                if hasattr(r, "audio_wav") and r.audio_wav is not None:
                    wav_chunks.append(r.audio_wav)
                    sr = getattr(r, "sampling_rate", sr)
            wav = np.concatenate(wav_chunks) if wav_chunks else None
            if wav is not None and sr:
                sf.write(out_wav, wav, samplerate=sr)
        else:
            for r in res_iter:
                # 仅文本的情况下，常见返回是 {'text': '...'} 或对象属性 r.text
                if isinstance(r, dict):
                    text_out += r.get("text", "")
                else:
                    text_out += getattr(r, "text", "")

            wav, sr = None, None

        return {"text": text_out, "wav": (wav if generate_audio else None), "sr": sr}
