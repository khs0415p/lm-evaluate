import os
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
from config import Config
from evaluate import Evaluator

from transformers import AutoTokenizer, AutoModelForCausalLM


def main(config):
    sub_categories = config.sub_categories
    categories = config.categories
    # load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("================= Start evaluation =================")
    total_text, total_label = [], []
    sub_group = defaultdict(list)
    print(f"{'Subject':<30}{'':<15}{'Acc':<5}")
    print("--------------------------------------------------")
    for subject in sub_categories.keys():
        test_df = pd.read_csv(os.path.join(config.test_dir, subject + '.csv'))
        # load evaluator
        evaluator = Evaluator(config, subject, test_df, model, tokenizer, device)
        text_acc, _, label_acc = evaluator.eval()

        total_text.append(text_acc)
        total_label.append(label_acc)

        sub_group[sub_categories[subject][0]].append(label_acc)
    print("==================================================")
    print(f"{'Method':<40}{'Acc':<5}")
    print("--------------------------------------------------")

    total_text_acc = np.mean(total_text)
    total_label_acc = np.mean(total_label)

    print(f"{'Text match':<40}{total_text_acc:<5.3f}")
    print(f"{'Label match':<40}{total_label_acc:<5.3f}")
    print("==================================================")

    print(f"{'Sub-group':<40}{'Acc':<5}")
    print("--------------------------------------------------")
    for k, v in sub_group.items():
        sub_acc = np.mean(v)
        print(f"{k:<40}{sub_acc:<5.3f}")
    print("==================================================")

    print(f"{'Group':<40}{'Acc':<5}")
    print("--------------------------------------------------")
    for k, v in categories.items():
        group_lst = []
        for sub in v:
            group_lst.extend(sub_group[sub])
        group_acc= np.mean(group_lst)
        print(f"{k:<40}{group_acc:<5.3f}")
    print("==================================================")
    print("--------------------------------------------------")
    ppl_score = evaluator.get_ppl_score()
    print(f"{'Perplexity':<40}{'Score':<5}")
    print(f"{ppl_score:>45.3f}")
    print("==================================================")


if __name__ == "__main__":
    config = Config('./config.json')
    main(config)
