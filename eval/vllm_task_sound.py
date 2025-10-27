"""
Using video-llm to do our tasks
"""
import argparse
import os
import sys
import torch
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from humor_benchmark.bert_QA import score_QA
from humor_benchmark.bertsocre import score_explanation
from humor_benchmark.accuracy import score_caption
from humor_benchmark.generate_caption import generate_caption_videos
from humor_benchmark.generate_explanation import generate_explanation_videos
from humor_benchmark.generate_QA import generate_QA_videos
from humor_benchmark.matching_question import generate_matchingQ_with_correct_choice
from humor_benchmark.open_ended_QA import process_qa_videos


# 基础模型类（保持不变）
class VLLMTaskSoundModel(torch.nn.Module):
    """
    Task API. Send in a Video LLM model and the device to do the task.
    """
    def __init__(self, model, device):
        super().__init__()
        self.device = device
        self.model = model

    def forward(self, instruction, videos):
        return self.generate(instruction, videos)

    def generate(self, instruction, video_path):
        """
        Generate according to tasks

        Args:
            instruction (dict): A dictionary containing task instructions.
            videos (str): Path to the video file.
        """
        task_type = instruction.get("task")
        
        # build prompt
        system_prompt, user_prompt = self.get_prompt(task_type, video_path, instruction=instruction)

        # --- 调用 API 获取响应 ---
        raw_response = self.model.get_completion(system_prompt, user_prompt, video_path)

        # Post-process the answer to clean up the task-specific prefixes
        post_processed_response = self.post_process_response(task_type, raw_response)
        
        return post_processed_response

    def get_prompt(self, task_type="matching", video_path=None, instruction=None):
        """
        Returns a prompt template based on the task type.

        Args:
            task_type (str): The type of task. Options are 'matching', 'explanation', 'QA'.
            video_path (str): The path or URL to the video.
            instruction (dict): Additional instructions for the task.

        Returns:
            system_prompt (str)
            user_prompt (str)
        """
        if task_type == "QA":
            question = instruction.get("question")
            system_prompt = "You are a helpful AI assistant. You can analyze videos and answer questions about their content. Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
            user_prompt = (
                f"Here's a humorous video. "
                f"Based on the its visual and audio information, answer the following question: {question}\n\n"
                "Output format:\n"
                "Answer: <answer>\n\n"
            )
        
        elif task_type == "explanation":
            system_prompt = "You are a helpful AI assistant specialized in video understanding and humor analysis. You can explain jokes clearly and naturally based on video content and video description. Please respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
            user_prompt = (
                f"Here's a humorous video. "
                "Your job is to explain why the video is humorous in 2-3 sentences as if you were explaning to a friend who doesn't get the joke yet. "
                "Respond with a 2-3 sentence explanation of the joke and how it relates to the video.\n\n"
                "Output format:\n"
                "Explanation: <answer>\n\n"
            )
        elif task_type == "matching":
            question = instruction.get("question")
            system_prompt = "You are a helpful AI assistant. You can analyze videos and answer questions about their content. Please only output in the specified format. No extra text."
            user_prompt =  (f"Along with visual and audio information in the video. And {question}\n"
                "Please respond with response with the option letter only.\n\n"
                "Output format:\n"
                "Answer: <answer>\n\n"
                )

        else:
            raise ValueError("Invalid task type. Choose from 'matching', 'explanation', 'QA'.")

        return system_prompt, user_prompt

    def post_process_response(self, task_type, raw_response):
        """
        Post-processes the model response based on the task type.

        Args:
            task_type (str): The type of task. Options are 'matching', 'explanation', 'QA'.
            response (str): The raw response from the model.

        Returns:
            processed_response (str)
        """
        if task_type == "QA":
            return raw_response.replace("Answer: ", "").strip()
        elif task_type == "explanation":
            return raw_response.replace("Explanation: ", "").strip()
        elif task_type == "matching":
            return raw_response.replace("Answer: ", "").strip()
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a model on the humor benchmark.")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to the directory containing video files.")
    parser.add_argument("--model_name", type=str, default="Qwen_Omni", help="Name of the model to use.")
    parser.add_argument("--questions_csv", type=str, required=True, help="Path to the CSV file with questions.")
    parser.add_argument("--cand_file", type=str, required=True, help="Path to the candidate answers file.")
    parser.add_argument("--ref_file", type=str, required=True, help="Path to the reference answers file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--task", type=str, required=True, choices=["explanation", "matching", "QA","Process_matching","Process_QA"],
                        help="The evaluation task to perform (explanation, matching, or QA).")
    
    args = parser.parse_args()

    if args.model_name == "Qwen2.5-Omni":
        from backbone import Qwen25_Omni
        model = Qwen25_Omni()
    else:
        raise ValueError(f"Model {args.model_name} not supported.")

    model = VLLMTaskModel(model, device="cuda")
    metrics = {}
    if args.task == "explanation":
        print("Running explanation task...")
        # 生成解释
        generate_explanation_videos(args.video_dir, args.questions_csv, model, args.cand_file)
        
        # 评估解释
        bert_p_mean, bert_r_mean, bert_f1_mean, meteor_mean = score_explanation(args.cand_file, args.ref_file)
        metrics = {
            "Metric": ["BERT_Precision", "BERT_Recall", "BERT_F1", "METEOR"],
            "Value": [bert_p_mean, bert_r_mean, bert_f1_mean, meteor_mean]
        }
    
    elif args.task == "matching":
        print("Running matching task...")
        # 生成匹配问题
        generate_caption_videos(args.video_dir, args.questions_csv, model, args.cand_file)
        
        caption_accuracy = score_caption(args.cand_file, args.ref_file)
        metrics = {
            "Metric": ["Matching_Accuracy"],
            "Value": [caption_accuracy]
        }

    elif args.task == "QA":
        print("Running QA task...")
        # 生成QA答案
        generate_QA_videos(args.video_dir, args.questions_csv, model, args.cand_file)
        
        # 评估QA答案
        bert_p_mean, bert_r_mean, bert_f1_mean, meteor_mean = score_QA(args.cand_file, args.ref_file)
        metrics = {
            "Metric": ["BERT_Precision", "BERT_Recall", "BERT_F1", "METEOR"],
            "Value": [bert_p_mean, bert_r_mean, bert_f1_mean, meteor_mean]
        }
    elif args.task == "Process_QA":
        print("Process QA videos...")

        process_qa_videos(args.video_dir, args.questions_csv, model)
    elif args.task == "Process_matching":
        print("Process matchingQ_with_correct_choice...")

        generate_matchingQ_with_correct_choice(args.video_dir, args.questions_csv, model)

    # 写入 CSV 文件
    if metrics:
        print(f"Saving evaluation metrics to {args.output_csv}...")
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for i in range(len(metrics["Metric"])):
                writer.writerow([metrics["Metric"][i], metrics["Value"][i]])
        print(f"Evaluation metrics saved to {args.output_csv}")
    
    print("Evaluation completed.")
