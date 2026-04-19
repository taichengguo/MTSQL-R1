import os, sys
import json
import pdb
import sqlite3
import argparse
from tqdm import tqdm
import pandas as pd
import re

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n")
        return extract_again(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def compute_score(solution_str, ground_truth, **kwargs) -> float:
    """
    solution_str: answer, may contains some illustrations
    """
    reward = 0.0
    try:
        answer = extract_answer(solution_str)
        if answer == ground_truth:
            reward = 1.0
    except Exception as e:
        print(e)

    return reward

if __name__ == '__main__':
    # test score: execution match
    # df = pd.read_parquet("~/Desktop/verl/data/text2sql/train.parquet")
    # gt = df['reward_model'].tolist()
    # gt = [i['ground_truth'] for i in gt]
    # for i in gt:
    #     pred_sql = i.split("\t")[0]
    #     print(compute_score(pred_sql, i))

    # test score: exact match
    raw_text = "SELECT COUNT(*) FROM Employee WHERE salary > (SELECT AVG(salary) FROM Employee);"






