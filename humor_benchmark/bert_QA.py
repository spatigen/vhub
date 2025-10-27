# 计算bertscore

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score
from bert_score import score
import numpy as np
from sentence_transformers import SentenceTransformer, util
# nltk.download('punkt_tab', quiet=True)  # quiet=True 可减少输出

def score_QA(cand_file, ref_file):
    cand_df = pd.read_csv(cand_file)
    ref_df = pd.read_csv(ref_file, encoding='utf-8-sig')

    # 2. 将 gemini_test_result.csv 中的 file_name -> explanation 建立字典，方便索引
    cand_dict = dict(zip(cand_df['file_name'], cand_df['VQ_A']))

    # 3. 待比较的文本对
    ref_str = [] 
    cand_str = []  

    for _, row in ref_df.iterrows():
        file_name = row['file_name']
        correct_answer = row['VQ_A'] if pd.notnull(row['VQ_A']) else ""
        model_answer = cand_dict.get(file_name, "")

        model_answer = str(model_answer) if pd.notna(model_answer) else ''

        ref_str.append(correct_answer)
        cand_str.append(model_answer)
        

    # 计算分数
    P, R, F1 = score(cand_str, ref_str, model_type="bert-base-uncased", lang="en", num_layers=12)

    # 计算 METEOR 指标
    meteor = []
    for ref, cand in zip(ref_str, cand_str):
        candidates = [w.lower() for w in word_tokenize(cand)]
        references = [[w.lower() for w in word_tokenize(ref)]]
        score1 = meteor_score.meteor_score(references, candidates)
        meteor.append(score1)
    
    # 计算 SentBert指标
    model = SentenceTransformer('all-MiniLM-L6-v2')
    SentBert = []
    for ref, cand in zip(ref_str, cand_str):
        embedding1 = model.encode(ref, convert_to_tensor=True)
        embedding2 = model.encode(cand, convert_to_tensor=True)
        # 计算余弦相似度
        cosine_sim = util.cos_sim(embedding1, embedding2)
        SentBert.append(cosine_sim)


    bert_p_mean = P.mean().item()
    bert_r_mean = R.mean().item()
    bert_f1_mean = F1.mean().item()
    meteor_mean = np.mean(meteor).item()
    scores = [t.item() for t in SentBert]
    sentbert_mean = sum(scores) / len(scores)


    return bert_p_mean, bert_r_mean, bert_f1_mean, meteor_mean, sentbert_mean
