import os, sys
import json
import pdb
import sqlite3
import argparse
from tqdm import tqdm
from verl.utils.reward_score.eval.exec_eval import eval_exec_match
from verl.utils.reward_score.eval.evaluation import eval_exact_match_score
from verl.utils.reward_score.eval.process_sql import get_schema, Schema, get_sql
import pandas as pd
import re

def extract_code(text):
    # Using regex to find content between <code> and </code>
    pattern = r'<answer_sql>(.*?)</answer_sql>'
    try:
        matches = re.findall(pattern, text, re.DOTALL)
        sql =  matches[-1]
        sql = sql.replace("\n", " ")
        sql = sql.strip()
        return sql
    except Exception as e:
        print(f"extracting error: {e}, results is: {text}")
        try:
            pattern = r'<think>(.*?)</think>'
            matches = re.findall(pattern, text, re.DOTALL)
            sql =  matches[-1]
            sql = sql.replace("\n", " ")
            sql = sql.strip()
            return sql
        except Exception as e:
            err = text.replace("\n", " ")
            print(f"extracting still error: {e}, results is: {err}")
            return text

# eval F1 score of the parsed SQL overlap
def eval_f1_score_of_sql_overlap(pred_sql, gt_sql, db):
    # parse each sql with the database
    # TODO: set to your local path to the CoSQL/SParC database directory
    db_dir = f"PATH/TO/database/cosql/database/{db}"
    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if '.sqlite' in basename]
    db_path = db_paths[0]
    schema = get_schema(db_path)
    schema = Schema(schema)
    p_sql = get_sql(schema, pred_sql)
    g_sql = get_sql(schema, gt_sql)
    p_terms = set()
    g_terms = set()
    if p_sql is None or g_sql is None:
        return 0.0
    for key in p_sql:
        p_terms.add(str(p_sql[key]))
    for key in g_sql:
        g_terms.add(str(g_sql[key]))
    common_terms = p_terms.intersection(g_terms)
    if len(common_terms) == 0:
        return 0.0
    precision = len(common_terms) / len(p_terms)
    recall = len(common_terms) / len(g_terms)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# eval the execution column name
# def eval_execution_column_name(pred_sql, gt_sql, db):
#     # eval the overlap of the column names in the select clause


# Verification reward
# please write python code to extract <> previous <tool call> SQL
# pass SQL reward
# elif not pass SQL
def get_process_reward_for_verify_action(solution_str, ground_truth, **kwargs) -> float:
    """
    markers:
    exec
    1. <tool> SQL cannot be executed, but pass / not pass => 0 / 1
    2. <tool> SQL results are null; and not same to gt SQL, but pass / not pass => 0, 0.5
    3. other cases can be exected, pass = 1, not pass = 0

    coherence:
    1.   F1 score
    2.   1-F1 score
    """


    return 0.0



# Gen SQL reward
# Propose: <tool call> SQL
# Refinement: <not pass> <tool call>/<end SQL>
# Steps
def get_process_reward_for_generate_action(solution_str, ground_truth, **kwargs) -> float:
    # markers: <tool call>
    # 1. For each tool call, check the f1 score between the proposed SQL and the ground truth SQL
    # if SQL is the same, only calculate once
    # reward = sum of f1 score for each tool call
    # Avg. f1 score for each tool call
    # 2. Improvement Tool gain: f1(new) - f1(old)

    return 0.0


def compute_score(solution_str, ground_truth, **kwargs) -> float:
    """
    solution_str: answer, may contains some illustrations
    """
    gt_array = ground_truth.split("\t")
    g_str, db = gt_array
    extracted_sql = extract_code(solution_str)

    # print("="*10)
    # print(f"Predicted: {extracted_sql}")
    # print(f"Ground truth: {g_str}")
    # print(f"DB: {db}")
    # print("="*10)

    exec_score = 0
    exact_score = 0

    try:
        # exec_score = eval_exec_match(db, extracted_sql, g_str, False, False, False)
        exact_score = eval_exact_match_score(db, extracted_sql, g_str, dataset="cosql")
    except Exception as e:
        print(f"Reward Score Error: {e}; prediction SQL is: {extracted_sql}")
    return exec_score + exact_score

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
    print(extract_code(raw_text))






