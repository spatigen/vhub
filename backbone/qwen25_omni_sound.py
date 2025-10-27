import os
import sys
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from backbone_utils import extract_frames_base64, get_max_frame_and_interval,reconstruct_video,encode_video
from moviepy import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# 加载环境变量
load_dotenv()

class Qwen25_Omni_Sound:
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
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise IOError("Could not read first frame of video.")
        
        nframes, interval = get_max_frame_and_interval(
            self.max_tokens, cap, frame, self.model_h, self.model_w, self.model_max_tokens, self.model_min_tokens
        )

        if original_fps <= 0:
            original_fps = 30.0  # 默认帧率

        time_between_frames_sec = interval / original_fps
        target_fps = 1.0 / time_between_frames_sec if time_between_frames_sec > 0 else 1.0

        cap.release()

        base64_frames = extract_frames_base64(video_path, nframes=nframes, interval=interval)

        clip = VideoFileClip(video_path)
        audio = clip.audio  # 获取音频对象
        audio_path = None
        if audio is not None:
            audio_path = "extracted_audio.wav"
            audio.write_audiofile(audio_path)

        reconstruct_video(base64_frames, audio_path, target_fps, "reconstruct_video.mp4")

        base64_video = encode_video("reconstruct_video.mp4")

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
                            {"type": "video_url", 
                             "video_url":{
                                 "url": f"data:;base64,{base64_video}"}},
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
