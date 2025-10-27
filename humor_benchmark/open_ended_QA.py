# 生成Open-ended QA，包括QA_Q, QA_A, QA_Type

import os
import csv
import time
import time
import pandas as pd


def create_qa_prompt(video_url,video_description, humor_explanation):
    prompt = (
        f"These are frames from a video."
        "And you'll be given a description of a video and an explanation of why it's humorous to watch. "
        "Based on given information, generate a Video Reasoning QA pair, try to make answer only as phrases. And avoid using unpronouncable punctuation or emojis. Let’s think step by step.\n"
        "Additionally, classify this question into one of the following categories using the concise definitions provided:\n"
        "Descriptive question: Involves factual details such as location or count\n"
        "Temporal question: Involves time-related aspects (e.g., previous, after)\n"
        "Causal question: Involves reasons or explanations (e.g., why, how)\n\n"
        "Example 1:\n"
        "Description:\n"
        "Two hands are stretched out, one hand holding KFC chicken nuggets and the other hand holding seeds. In the distance, a chicken runs over, but the chicken prefers to eat the KFC chicken.\n"
        "Explanation: The chicken surprisingly likes to eat KFC chicken, which is unexpected and a bit funny. The man realizes something is wrong and tries to push the chicken pieces away with his hand, which adds to the humor with a sense of panic.\n\n"
        "Question: What does the man holding in his hand?\n"
        "Answer: KFC chicken nuggets and seeds.\n"
        "Type: Descriptive\n\n"
        "Example 2:\n"
        "Description: A man poured red liquid into the water, and a group of fish came to snatch the food. Another man poured beer into the water, and a group of men came to snatch the food like fish.\n"
        "Explanation: The portrait of people snatching food like fish humorously reflects the attraction of beer to men, and the connection between them is very funny.\n\n"
        "Question: After the man poured beer into the water, what happened?\n"
        "Answer: A group of men came.\n"
        "Type: Temporal\n\n"
        "Example 3:\n"
        "Description: A woman was lying on the handrail of an escalator while moving down. A man saw her, and lying on the handrail on the other side, and as a result, there was no barrier on that side, and he fell directly down the escalator.\n"
        "Explanation: The man tried to show off by imitating others, but ended up falling hard, which made people find it funny.\n\n"
        "Question: Why does the man fall off on the other side of the handrail?\n"
        "Answer: There was no barrier.\n"
        "Type: Causal\n\n"
        "Output format:\n"
        "Question: <question>\n"
        "Answer: <answer>\n"
        "Type: <type>\n\n"
        f"Video Description: {video_description}\n"
        f"Humor Explanation: {humor_explanation}\n"
    )
    return prompt


def process_qa_videos(video_dir, questions_csv, model):
    
    results = []
    
    df_original = pd.read_csv(questions_csv, encoding='utf-8-sig')

    try:
        with open(questions_csv, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for i, row in enumerate(reader):
                filename = row['file_name']
                video_description = row['video_description']
                humor_explanation = row['humor_explanation']
                    
                # 移除可能的文件扩展名
                base_filename = filename.replace('.mp4', '')
                video_path = os.path.join(video_dir, base_filename + ".mp4")

                if os.path.exists(video_path):
                    video_url = video_path
                    
                    question = create_qa_prompt(video_url,video_description, humor_explanation)
                    print(f"Processing {i+1}: {base_filename}.mp4")
                    instruction_dict = {
                        "task": "Process_QA",
                        "user_prompt": question
                    }
                    answer = model.generate(instruction=instruction_dict, video_path=video_path)
                    # 调用Qwen API分析视频
                    index1 = answer.find('Answer')
                    index2 = answer.find('Type')
                    answer_VQ_Q = answer[10:index1-1].strip('"\'')
                    answer_VQ_A = answer[index1+8:index2-1].strip('"\'')
                    answer_VQ_Type = answer[index2+6:].strip('"\'')

                    results.append({
                        'file_name': base_filename,
                        'VQ_Q': answer_VQ_Q, 
                        'VQ_A': answer_VQ_A,
                        'VQ_Type': answer_VQ_Type, 
                    })
                        
                    print(f"Response: {answer[:100]}...")  # 打印部分响应用于调试
                        
                    # 添加延迟以避免API速率限制
                    time.sleep(2)
                else:
                    print(f"Video file not found: {video_path}")
                    results.append({
                        'file_name': base_filename, 
                        'VQ_Q': "Video file not found", 
                        'VQ_A': "Video file not found",
                        'VQ_Type': "Video file not found"
                    })

    except Exception as e:
        print(f"Error processing videos: {e}")
        
    # Step 2: 将结果转为 DataFrame
    df_results = pd.DataFrame(results)

    # Step 3: 以 'file_name' 为键，合并原始数据和结果数据
    # 使用 left join 保证原始行顺序和完整性
    df_merged = df_original.merge(df_results, on='file_name', how='left')

    # Step 4: 保存回原文件（建议先备份）
    backup_file = questions_csv.replace('.csv', '_backup1.csv')
    df_original.to_csv(backup_file, index=False, encoding='utf-8-sig')
    print(f"Original file backed up to: {backup_file}")

    df_merged.to_csv(questions_csv, index=False, encoding='utf-8-sig')
    print(f"Results merged and saved back to: {questions_csv}")

