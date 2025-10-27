import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# 加载环境变量
load_dotenv()

class Qwen25_7:
    def __init__(self, model_name_or_path="qwen2.5-7b-instruct", max_tokens=128000):
        # 确保使用 DashScope 的 API 密钥和基础 URL
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name_or_path
        self.max_tokens = max_tokens


    def get_completion(self, system_prompt, user_prompt):

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
                        "content": user_prompt
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
