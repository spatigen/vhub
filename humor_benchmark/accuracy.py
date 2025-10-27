import pandas as pd
def score_caption(cand_file, ref_file):

    cand_df = pd.read_csv(cand_file)
    ref_df = pd.read_csv(ref_file)

    cand_choice = cand_df['correct_choice']
    ref_choice = ref_df['correct_choice']

    if len(cand_choice) != len(ref_choice):
        raise ValueError("预测和标签的样本数量必须一致")

    correct = sum(1 for pred, label in zip(cand_choice, ref_choice) if pred == label)
    accuracy = correct / len(ref_choice)
    return accuracy
