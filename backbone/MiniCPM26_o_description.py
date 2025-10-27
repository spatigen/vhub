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
from typing import Optional

class MiniCPMO26_Description:
    """
    用 MiniCPM-o-2.6 的流式接口跑「视频→多图→文本」生成。
    仍沿用你原有的抽帧预算与采样逻辑。
    """
    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-o-2_6",
        device: Optional[str] = None,
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

    def get_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.5,
        generate_audio: bool = False, #是否开启音频输出
        out_wav: str = "output.wav", #音频输出的保存路径
    ) -> dict:
        """
        返回:
            {"text": str, "wav": np.ndarray|None, "sr": int|None}
        """

        # --- 2) 构造流式会话：reset → prefill(system) → prefill(video chunks) → generate ---
        self.model.reset_session()  # 新会话清空 KV cache
        session_id = str(uuid4())

        sys_msg = {"role": "system", "content": [system_prompt]}
        _ = self.model.streaming_prefill(
            session_id=session_id,
            msgs=[sys_msg],
            tokenizer=self.tokenizer
        )

        msgs = [{"role": "user", "content": [user_prompt]}]
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

        # return {"text": text_out, "wav": (wav if generate_audio else None), "sr": sr}
        return text_out
