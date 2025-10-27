# 完成explanation任务，生成explanation

import os
import csv
import time

def generate_explanation_videos(video_dir, questions_csv, model, output_file):
    # 初始化：首次运行时创建文件并写入表头
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file_name', 'explanation']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        print(f"Initialized output file: {output_file}")
    except Exception as e:
        print(f"Error initializing output file: {e}")
        return
        
    try:
        with open(questions_csv, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for i, row in enumerate(reader):
                filename = row['file_name']
                video_description = row['video_description']

                # 移除可能的文件扩展名
                base_filename = filename.replace('.mp4', '')
                video_path = os.path.join(video_dir, base_filename + ".mp4")

                if os.path.exists(video_path):
                    print(f"Processing {i+1}: {base_filename}.mp4")

                    # 封装 instruction 字典
                    instruction_dict = {
                        "task": "explanation",
                        "video_description": video_description,
                        "background_knowledge": row['background_knowledge']
                        # 可以添加其他需要的字段，但在这个任务中 video_description 就够了
                    }

                    # 调用模型的 generate 方法
                    explanation = model.generate(instruction=instruction_dict, video_path=video_path)

                    result_row = {
                        'file_name': base_filename, 
                        'explanation': explanation, 
                    }
                    # 立即写入CSV（追加模式）
                    try:
                        with open(output_file, 'a', newline='', encoding='utf-8') as out_csv:
                            writer = csv.DictWriter(out_csv, fieldnames=['file_name', 'explanation'])
                            writer.writerow(result_row)
                        print(f"Saved result for {base_filename}")
                    except Exception as write_e:
                        print(f"Error writing row to CSV: {write_e}")

                    print(f"Response: {explanation[:100]}...")  # 打印部分响应用于调试
                        
                    # 添加延迟以避免API速率限制
                    time.sleep(2)
                else:
                    print(f"Video file not found: {video_path}")
                    result_row = {
                        'file_name': base_filename, 
                        'explanation': "Video file not found", 
                    }

                    try:
                        with open(output_file, 'a', newline='', encoding='utf-8') as out_csv:
                            writer = csv.DictWriter(out_csv, fieldnames=['file_name', 'explanation'])
                            writer.writerow(result_row)
                        print(f"Saved 'not found' result for {base_filename}")
                    except Exception as write_e:
                        print(f"Error writing 'not found' row to CSV: {write_e}")
        
    except Exception as e:
        print(f"Error processing videos: {e}")
