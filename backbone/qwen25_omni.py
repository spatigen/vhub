import os
import sys
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from backbone_utils import extract_frames_base64, get_max_frame_and_interval

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# 加载环境变量
load_dotenv()


class Qwen25_Omni:
    def __init__(self, model_name_or_path="qwen2.5-omni-7b", max_tokens=30000):
        # 确保使用 DashScope 的 API 密钥和基础 URL
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name_or_path
        self.max_tokens = max_tokens

        # model patch height
        self.model_h = 28

        # model patch width
        self.model_w = 28

        self.model_max_tokens = 1280
        self.model_min_tokens = 4

    def get_completion(self, system_prompt, user_prompt, video_path):
        """
        从 DashScope API (Qwen-omni) 生成文本补全
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise IOError("Could not read first frame of video.")
        
            
        nframes, interval = get_max_frame_and_interval(
            self.max_tokens, cap, frame, self.model_h, self.model_w, self.model_max_tokens, self.model_min_tokens
        )

        cap.release()

        base64_frames = extract_frames_base64(video_path, nframes=nframes, interval=interval)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}} for frame in base64_frames],
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                stream=True,
                stream_options={"include_usage": True},
            )
            
            full_response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            return full_response if full_response else "No response received from API"

        except Exception as e:
            return f"Error: {str(e)}"
