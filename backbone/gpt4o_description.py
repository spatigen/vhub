import os
import sys
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from backbone_utils import extract_frames_base64, get_max_frame_and_interval

# 加载环境变量
load_dotenv()

class GPT4o_Description:
    def __init__(self, model_name_or_path="gpt-4o", max_tokens=120000):
        # 从环境变量读取 OpenAI API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model_name_or_path
        self.max_tokens = max_tokens

    def get_completion(self, system_prompt, user_prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                ],
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"

