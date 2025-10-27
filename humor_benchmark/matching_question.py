# 生成Caption Matching，包括CaptionQ和correct_choice

import pandas as pd
import random
import os

def create_matching_prompt(video_description, descriptive_captions):
    prompt = (
        f"These are frames from a video. And I will provide a description of the video and a list of descriptive captions that break down what happens in it. "
        "Your task is to write a caption in one sentences from the video creator's perspective — something you would write to attract viwers.\n\n"
        
        "Requirements:\n"
        "- Please ensure it is related to the video content.\n"
        "- Write as if you're sharing it with an audience (e.g., use 'this' or 'me' naturally).\n"
        "- Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis.\n"

        "Output format:\n"
        "Caption: <caption>\n\n"
        
        f"Video description: {video_description}\n"
        f"Descriptive captions: {descriptive_captions}"
    )
    return prompt



def generate_matchingQ_with_correct_choice(video_dir, input_csv, model, seed=42):
    """
    根据指定的CSV：
    1. 收集每行(视频)的所有 caption，构建全局 caption 池。
    2. 如果本行有至少一个幽默caption，则从其中随机选择一个作为正确选项，
       再从全局池中抽取4个干扰项(排除同视频caption)，打乱后放在5个选项中。
    3. 新建列 'CaptionQ'，存放问题 + 5个选项；
       新建列 'correct_choice'，存放正确选项对应的A/B/C/D/E。
    4. 输出新的CSV文件。
    """
    random.seed(seed)
    
    # 1. 读取CSV
    df = pd.read_csv(input_csv)
    
    # 用于存储每个file_name对应的所有非空caption
    file_captions_dict = {}
    # 用于存储全局的所有caption，后面做干扰项池
    all_captions = []
    
    # 准备存放新生成的列数据
    new_column_captionQ = []
    new_column_correct_choice = []
    all_captions = []
    file_correct_caption_dict = {}

    # 3. 生成correct_caption
    for idx, row in df.iterrows():
        # 找到本行幽默caption
        hum_caps = []
        ent_caps = []
        captions_for_this_file = []
        for col in ["humorous_caption_1", "humorous_caption_2", "humorous_caption_3"]:
            c = str(row[col]).strip()
            if c and c.lower() != 'nan':
                captions_for_this_file.append(c)
                hum_caps.append(c)
                all_captions.append(c)
        
        for col in ["entertaining_caption_1", "entertaining_caption_2", "entertaining_caption_3"]:
            c = str(row[col]).strip()
            if c and c.lower() != 'nan':
                ent_caps.append(c)        

        filename = str(row['file_name'])
        video_url = os.path.join(video_dir, filename + ".mp4")

        # 如果没有幽默caption，则对应列留空
        if not hum_caps:
            caps_text = "\n".join(ent_caps)
            video_description = row['video_description']
            question = create_matching_prompt(video_description, caps_text)
            instruction_dict = {
                "task": "Process_matching",
                "user_prompt": question
            }
            correct_caption = model.generate(instruction=instruction_dict, video_path=video_url)
            # correct_caption = summarize_caption(video_url, question, model)[9:].strip('"\'')
            file_correct_caption_dict[filename] = correct_caption
            captions_for_this_file.append(correct_caption)
            all_captions.append(correct_caption)
        
        file_captions_dict[filename] = captions_for_this_file

    
    # 4. 对每行进行处理
    for idx, row in df.iterrows():
        # 找到本行幽默caption
        hum_caps = []
        captions_for_this_file = []
        for col in ["humorous_caption_1", "humorous_caption_2", "humorous_caption_3"]:
            c = str(row[col]).strip()
            if c and c.lower() != 'nan':
                captions_for_this_file.append(c)
                hum_caps.append(c)
        
        filename = str(row['file_name'])

        if not hum_caps:
                correct_caption = file_correct_caption_dict[filename]
        else:
            # 从本行的幽默caption中随机选择一个作为正确选项
            correct_caption = random.choice(hum_caps)

        # 复制全局caption池
        global_captions_pool = list(all_captions)
        
        # 从干扰池中排除本行的所有caption，避免干扰选项与本行视频重复
        this_file_captions = file_captions_dict[filename]
        for cap in this_file_captions:
            while cap in global_captions_pool:
                global_captions_pool.remove(cap)
        
        # 随机抽取4个干扰项
        if len(global_captions_pool) < 4:
            # 池子不足4个的情况可以特殊处理，这里简单示例补空串
            distractors = random.sample(global_captions_pool, k=len(global_captions_pool))
            while len(distractors) < 4:
                distractors.append("（无可用干扰项）")
        else:
            distractors = random.sample(global_captions_pool, 4)
        
        # 组成5个选项
        five_options = [correct_caption] + distractors
        
        # 打乱顺序
        random.shuffle(five_options)
        
        # 生成标签A~E
        labels = ["A", "B", "C", "D", "E"]
        
        # 找到正确选项在five_options中的位置
        correct_index = five_options.index(correct_caption)
        correct_label = labels[correct_index]  # A/B/C/D/E
        
        option_str_list = []
        for label, opt in zip(labels, five_options):
            option_str_list.append(f"{label}. {opt}")
        
        options_text = "\n".join(option_str_list)
        question_text = (
            "You will see five captions — only one of which was written about the video. "
            "Pick which of the five choices truly corresponds to the humor in the video.\n"
            f"{options_text}"
        )
        
        # 将生成的内容添加到列表中
        new_column_captionQ.append(question_text)
        new_column_correct_choice.append(correct_label)
    
    # 4. 将新列加入 DataFrame
    backup_file = input_csv.replace('.csv', '_backup2.csv')
    df.to_csv(backup_file, index=False, encoding='utf-8-sig')
    
    df["CaptionQ"] = new_column_captionQ
    df["correct_choice"] = new_column_correct_choice
    
    # 5. 输出到新的 CSV
    df.to_csv(input_csv, index=False)
    print(f"处理完成，结果已保存到 {input_csv}")

