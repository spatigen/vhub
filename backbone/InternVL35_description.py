import os
import io
import cv2
import base64
from typing import List

from PIL import Image

from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN


class InternVL35_Description:

    def __init__(
        self,
        model: str = "OpenGVLab/InternVL3_5-8B",
        tp: int = 1,                 # 38B 用 tp=2；241B-A28B 用 tp=8（这里只是8B示例）
        session_len: int = 32768,    # 根据任务长度可调
    ):
        engine_cfg = PytorchEngineConfig(session_len=session_len, tp=tp)
        self.pipe = pipeline(model, backend_config=engine_cfg)


    #从代码里发现最大分块数为12，每个448*448为一块，占256个token
    def get_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        # 视觉预算 & 抽帧策略（与你原脚本一致的接口）
        max_tokens: int = 31000,
        model_h: int = 448,
        model_w: int = 448,
        model_max_tokens: int = 3072,
        model_min_tokens: int = 256,
    ) -> str:
        """
        返回：模型输出文本
        """
        
        prompt = (
            f"\n{system_prompt}\n"
            + f"\n{user_prompt}"
            )           
        # 4) 一次性非流式推理
        #    这里无需 streaming_* 接口，直接 pipe((prompt, images)) 即可
        resp = self.pipe(prompt)

        # 5) 返回文本
        return getattr(resp, "text", str(resp))

