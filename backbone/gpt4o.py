import os
import sys
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from backbone_utils import extract_frames_base64, get_max_frame_and_interval

# 加载环境变量
load_dotenv()

class GPT4o:
    def __init__(self, model_name_or_path="gpt-4o", max_tokens=120000):
        # 从环境变量读取 OpenAI API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model_name_or_path
        self.max_tokens = max_tokens


        # gpt-4o有一个detail参数，low是固定每个图像消耗85个token，high是每个512*512的patch消耗170个token再额外加85个token
        # patch 参数
        self.model_h = 512
        self.model_w = 512
        self.model_max_tokens = 1445
        self.model_min_tokens = 85

    def get_completion(self, system_prompt, user_prompt, video_path):
        """
        使用 GPT-4o 进行视频理解
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise IOError("Could not read first frame of video.")
        
        # 计算帧数和间隔
        nframes, interval = get_max_frame_and_interval(
            self.max_tokens, cap, frame,
            self.model_h, self.model_w,
            self.model_max_tokens, self.model_min_tokens
        )
        cap.release()

        # 抽帧并转 Base64
        base64_frames = extract_frames_base64(video_path, nframes=nframes, interval=interval)

        total_frames = len(base64_frames)
        
        if total_frames > 40:
            indices = np.linspace(0, total_frames - 1, 40, dtype=int) # 均匀取40帧
            sampled_base64Frames = []
            for ind in indices:
                sampled_base64Frames.append(base64_frames[ind])
            base64_frames = sampled_base64Frames
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}", "detail": "high"}} for frame in base64_frames], 
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"
