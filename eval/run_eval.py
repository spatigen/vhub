import argparse
import os
import csv
import sys
from vllm_task import VLLMTaskModel

# 将项目根目录添加到 Python 搜索路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
backbone_dir = os.path.join(project_root, 'backbone')
sys.path.append(backbone_dir)
from humor_benchmark.autodq import evaluate_from_csv
from humor_benchmark.bert_QA import score_QA
from humor_benchmark.bertsocre import score_explanation
from humor_benchmark.accuracy import score_caption
from humor_benchmark.generate_caption import generate_caption_videos
from humor_benchmark.generate_explanation import generate_explanation_videos
from humor_benchmark.generate_QA import generate_QA_videos
from humor_benchmark.matching_question import generate_matchingQ_with_correct_choice
from humor_benchmark.open_ended_QA import process_qa_videos
from backbone import Qwen25_Omni
from backbone import Gemini_20
from backbone import Qwen25_VL
from backbone import Gemini_25
from backbone import GPT4o
from backbone import VideoLlama3
from backbone import VideoSALMONN2
from backbone import MINICPM26_o
from backbone import InternVL35


sys.path.append(os.getcwd())

CKPT_DIR = 'checkpoints'

def load_backbone(TESTING_MODEL):
    """
    Loads a specified model based on the TESTING_MODEL string.
    
    Args:
        TESTING_MODEL (str): The name of the model to load.
        
    Returns:
        The loaded model object.
    """
    if TESTING_MODEL == "Qwen2.5-Omni":
        model = Qwen25_Omni()
    elif TESTING_MODEL == "Qwen2.5-VL":
        model = Qwen25_VL()
    elif TESTING_MODEL == "Gemini2.0-flash":
        model = Gemini_20()
    elif TESTING_MODEL == "Gemini2.5-flash":
        model = Gemini_25()
    elif TESTING_MODEL == "GPT-4o":
        model = GPT4o()
    elif TESTING_MODEL == "Video-llama3":
        model = VideoLlama3()
    elif TESTING_MODEL == "InterVL 3.5":
        model = InternVL35()
    elif TESTING_MODEL == "Minicpm 2.6-o":
        model = MINICPM26_o()
    elif TESTING_MODEL == "video SALMONN 2":
        model = VideoSALMONN2()
    else:
        raise ValueError(f"Model {TESTING_MODEL} not recognized.")
        
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the humor benchmark.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to the directory containing video files.")
    parser.add_argument("--questions_csv", type=str, required=True, help="Path to the CSV file with questions.")
    parser.add_argument("--cand_file", type=str, required=False, help="Path to the candidate answers file.") # 模型输出的结果
    parser.add_argument("--ref_file", type=str, required=False, help="Path to the reference answers file.") # 标准对比的数据集
    parser.add_argument("--output_csv", type=str, required=False, help="Path to save the output CSV file.") # 存放指标结果
    parser.add_argument("--task", type=str, required=True, choices=["explanation", "matching", "QA","Process_QA", "Process_matching"],
                        help="The evaluation task to perform (explanation, matching, or QA).")
    
    args = parser.parse_args()

    # 实例化模型
    backbone= load_backbone(args.model_name)
    model = VLLMTaskModel(backbone, device="cuda")

    # 检查文件和目录是否存在
    if not os.path.exists(args.video_dir):
        print(f"Video directory not found: {args.video_dir}")
        return
    if not os.path.exists(args.questions_csv):
        print(f"CSV file not found: {args.questions_csv}")
        return
    
    if args.cand_file is not None:
        # 检查生成和评估所需的文件是否存在
        if not os.path.exists(args.cand_file):
            print(f"Candidate file not found: {args.cand_file}")
            # 创建父目录（如果不存在）
            os.makedirs(os.path.dirname(args.cand_file), exist_ok=True)
            # 创建空文件
            with open(args.cand_file, 'w', encoding='utf-8') as f:
                pass

    if args.ref_file is not None:
        if not os.path.exists(args.ref_file):
            print(f"Reference file not found: {args.ref_file}")
            os.makedirs(os.path.dirname(args.ref_file), exist_ok=True)
            # 创建空文件
            with open(args.ref_file, 'w', encoding='utf-8') as f:
                pass
    if args.output_csv is not None:
        if not os.path.exists(args.output_csv):
            print(f"Reference file not found: {args.output_csv}")
            os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
            # 创建空文件
            with open(args.output_csv, 'w', encoding='utf-8') as f:
                pass

    metrics = {}
    
    # 根据任务类型执行不同的逻辑
    if args.task == "explanation":
        print("Running explanation task...")
        # 生成解释
        generate_explanation_videos(args.video_dir, args.questions_csv, model, args.cand_file)
        
        # 评估解释
        evaluate_from_csv(
        ref_csv_path=args.ref_file,
        cand_csv_path=args.cand_file,
        ref_column="humor_explanation",
        cand_column="explanation",
        dataset_name="humor_ex",
        verbose=False,
        save_dir="./results/autodq"
        )
        
        bert_p_mean, bert_r_mean, bert_f1_mean, meteor_mean, sentbert_mean = score_explanation(args.cand_file, args.ref_file)
        metrics = {
            "Metric": ["BERT_Precision", "BERT_Recall", "BERT_F1", "METEOR", "SentBert"],
            "Value": [bert_p_mean, bert_r_mean, bert_f1_mean, meteor_mean, sentbert_mean]
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
        bert_p_mean, bert_r_mean, bert_f1_mean, meteor_mean, sentbert_mean = score_QA(args.cand_file, args.ref_file)
        metrics = {
            "Metric": ["BERT_Precision", "BERT_Recall", "BERT_F1", "METEOR", "SentBert"],
            "Value": [bert_p_mean, bert_r_mean, bert_f1_mean, meteor_mean, sentbert_mean]
        }
    elif args.task == "Process_QA":
        print("Process QA videos...")
        process_qa_videos(args.video_dir, args.questions_csv, model)
    elif args.task == "Process_matching":
        print("Process matchingQ_with_correct_choice...")
        generate_matchingQ_with_correct_choice(args.video_dir, args.questions_csv, model)
    elif args.task == "Process_QA":
        print("Process QA videos...")

        process_qa_videos(args.video_dir, args.questions_csv, model)
    elif args.task == "Process_matching":
        print("Process matchingQ_with_correct_choice...")

        generate_matchingQ_with_correct_choice(args.video_dir, args.questions_csv, model)
    else:
        print(f"Unknown task: {args.task}")
        return
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

if __name__ == "__main__":
    main()
