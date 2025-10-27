#!/bin/bash

# 获取脚本的父目录（即项目根目录）并切换到该目录
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# Get the script's parent directory (the project root) and switch to it.
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# Define the log file path, which is relative to the project root.
LOG_FILE="./evaluation_$(date +%Y-%m-%d_%H-%M-%S).log"

# 定义参数，路径都相对于项目根目录
MODEL_NAME="Minicpm 2.6-o"
VIDEO_DIR="./all_data" # 假设视频在这个目录
QUESTIONS_CSV="./all_data.csv"
CAND_FILE="./results_video_sound/explanation/minicpm26.csv" # 模型输出的结果
REF_FILE="./all_data.csv" # 假设参考文件在这个目录
OUTPUT_CSV="./metric_video_sound/explanation/minicpm26.csv" # 模型和标准答案之间的评估结果
TASK="explanation" # 任务

# Run the Python script.
# The `tee` command redirects the standard output to both the console and the log file.
# The `-a` flag appends to the log file instead of overwriting it.
echo "Starting evaluation with log output to $LOG_FILE" | tee -a "$LOG_FILE"
echo "---------------------------------------------------" | tee -a "$LOG_FILE"

# 运行 Python 脚本
# 注意：这里需要指定 Python 脚本的相对路径
python ./eval/run_eval_sound.py \
    --model_name "$MODEL_NAME" \
    --video_dir "$VIDEO_DIR" \
    --questions_csv "$QUESTIONS_CSV" \
    --cand_file "$CAND_FILE" \
    --ref_file "$REF_FILE" \
    --output_csv "$OUTPUT_CSV" \
    --task "$TASK" 2>&1 | tee -a "$LOG_FILE"

echo "---------------------------------------------------" | tee -a "$LOG_FILE"
echo "Evaluation completed. Log file saved at $LOG_FILE" | tee -a "$LOG_FILE"
