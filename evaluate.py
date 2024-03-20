import os
import ast
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from tqdm import tqdm
from datasets import load_dataset



class Evaluator:
    def __init__(self, config, subject, df, model, tokenizer, device) -> None:
        self.config = config
        self.subject = subject
        self.df = df
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.label_to_text = {0:'A', 1:'B', 2:'C', 3:'D'}
        self.label_indices = torch.tensor(
            [
                self.tokenizer('A').input_ids[-1],                 
                self.tokenizer('B').input_ids[-1],
                self.tokenizer('C').input_ids[-1],
                self.tokenizer('D').input_ids[-1]
            ]
        )
        self.prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(self.subject.replace("_", " "))
        self.load_data(self.config.few_shot)

        if self.config.ppl:
            self.ppl_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            self.ppl_tensor = self.tokenizer("\n\n".join(self.ppl_raw['text']), return_tensors='pt')


    def load_data(self, few_shot=True):
        self.dev_df = pd.read_csv(os.path.join(self.config.dev_dir, self.subject + '.csv'))
        if few_shot:
            for _, row in self.dev_df.iterrows():
                self.prompt += self.make_example(row)


    def make_example(self, row, end=False):
        example_format = "{}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer: {}\n\n"
        if end:
            example_format = "{}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer:"
        lst = ast.literal_eval(row.choices)

        return example_format.format(
            row.question,
            lst[0],
            lst[1],
            lst[2],
            lst[3],
            self.label_to_text[row.answer]
        )


    @torch.no_grad()
    def get_ppl_score(self):
        self.model.eval()
        max_length = self.model.config.max_position_embeddings
        stride = self.config.stride
        seq_len = self.ppl_tensor.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating the perplexity..."):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = self.ppl_tensor.input_ids[:, begin_loc: end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, -trg_len] = -100

            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl


    @torch.no_grad()
    def eval(self):
        self.model.eval()
        text_match, prob_match, label_prob_match  = [], [], []
        for i in range(self.df.shape[0]):
            row = self.df.iloc[i]
            label = row['answer']

            prompt_end = self.make_example(row, end=True)
            prompt = self.prompt + prompt_end

            inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
            output = self.model(**inputs)
            last_token_logit = output.logits[:, -1, :].flatten()

            label_prob = nn.functional.softmax(last_token_logit[self.label_indices]).detach().cpu().numpy()
            label_prob_pred = np.argmax(label_prob)
            label_prob_match.append(label_prob_pred == label)

            all_prob = nn.functional.softmax(last_token_logit).detach().cpu().numpy()
            all_prob_pred = np.argmax(all_prob)
            prob_match.append(all_prob_pred == self.label_indices[label])

            gen_token = self.tokenizer.batch_decode(
                self.model.generate(
                    **inputs,
                    temperature=0.0,
                    max_new_tokens=1
                )
            )[0][-1]
            text_match.append(self.label_to_text[label] == gen_token)

        text_acc = np.mean(text_match)
        prob_acc = np.mean(prob_match)
        label_prob_acc = np.mean(label_prob_match)
        print(f"{self.subject:<30}{'Text match':<15}{text_acc:<5.3f}")
        print(f"{'':<30}{'Label match':<15}{label_prob_acc:<5.3f}")
        print(f"{'':<30}{'Last logits':<15}{prob_acc:<5.3f}\n")

        return text_acc, prob_acc, label_prob_acc