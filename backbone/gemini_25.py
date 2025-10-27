import os
import sys
import cv2
import base64
from openai import OpenAI
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# 加载环境变量
load_dotenv()


class Gemini_25:
    def __init__(self, model_name_or_path="gemini-2.5-flash", max_tokens=30000):
        # 确保使用 DashScope 的 API 密钥和基础 URL
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name_or_path
        self.max_tokens = max_tokens

    def extract_frames_base64(self, video_path):
        """
        从视频中以每秒1帧的固定频率抽取帧，并将其编码为Base64字符串列表。

        Args:
            video_path (str): 视频文件的路径。

        Returns:
            list: 包含每个抽取帧的Base64编码字符串的列表。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        # 获取视频的原始帧率
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # 计算每秒1帧所需的帧间隔。例如，如果原始视频是 24 FPS，则间隔为 24 帧。
        frame_interval = int(round(original_fps))

        base64_frames = []
        count = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # 仅在达到抽帧间隔时保存一帧
            if count % frame_interval == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                base64_str = base64.b64encode(buffer).decode("utf-8")
                base64_frames.append(base64_str)
            
            count += 1
        
        cap.release()

        # --- 确保至少有一帧的关键逻辑 ---
        if not base64_frames:
            # 如果因为视频过短而没有抽取到帧，则回退到抽取第一帧
            cap_fallback = cv2.VideoCapture(video_path)
            ret, first_frame = cap_fallback.read()
            cap_fallback.release()
            
            if ret:
                _, buffer = cv2.imencode(".jpg", first_frame)
                base64_str = base64.b64encode(buffer).decode("utf-8")
                base64_frames.append(base64_str)
            else:
                # 如果连第一帧都无法读取，则抛出错误
                raise IOError("Failed to extract even a single frame from the video.")

        return base64_frames
    

    def get_completion(self, system_prompt, user_prompt, video_path):

        # 直接调用 extract_frames_base64，它现在独立处理 1 FPS 逻辑
        base64_frames = self.extract_frames_base64(video_path)
        
        if not base64_frames:
            print("警告: 未从视频中抽取到任何帧。")
            return "Error: No frames extracted from video."
        
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
