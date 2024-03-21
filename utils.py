import os
import pandas as pd
from datasets import load_dataset


def group_data(data):
    groups = {col : []for col in set(data['subject'])}
    for row in data:
        groups[row['subject']].append(
            [
                row['question'],
                row['subject'],
                row['choices'],
                row['answer']
            ]
        )

    return groups


def save_data(data, mode='test'):
    for k, v in data.items():
        pd.DataFrame(v, columns = ['question', 'subject', 'choices', 'answer']).to_csv(f'data/{mode}/{k}.csv', index=False)


def make_data(name='cais/mmlu'):
    test, dev = load_dataset(name, 'all', split=['test', 'dev'])
    grouped_test = group_data(test)
    grouped_dev = group_data(dev)

    save_data(grouped_test)
    save_data(grouped_dev, 'dev')

if __name__ == "__main__":
    make_data()