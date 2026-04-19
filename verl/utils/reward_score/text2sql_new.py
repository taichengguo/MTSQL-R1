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
import re
import json
from typing import List, Tuple

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
    db_dir = f"./database/cosql/database/{db}"
    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if '.sqlite' in basename]
    db_path = db_paths[0]
    schema = get_schema(db_path)
    schema = Schema(schema)
    p_sql = get_sql(schema, pred_sql)
    g_sql = get_sql(schema, gt_sql)
    if p_sql is None or g_sql is None:
        return 0.0

    f1_list = []
    all_keys = set(p_sql.keys()) | set(g_sql.keys())  # union of keys

    for key in all_keys:
        g_terms = set([str(g_sql[key])]) if key in g_sql and g_sql[key] else set()
        p_terms = set([str(p_sql[key])]) if key in p_sql and p_sql[key] else set()

        # skip if both are empty (no signal for this key)
        if not g_terms and not p_terms:
            continue

        common_terms = p_terms.intersection(g_terms)
        precision = len(common_terms) / len(p_terms) if p_terms else 0.0
        recall = len(common_terms) / len(g_terms) if g_terms else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_list.append(f1)

    return sum(f1_list) / len(f1_list) if f1_list else 0.0


# eval the execution column name
# def eval_execution_column_name(pred_sql, gt_sql, db):
#     # eval the overlap of the column names in the select clause
import re
from typing import Optional

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", flags=re.DOTALL | re.IGNORECASE)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", flags=re.DOTALL | re.IGNORECASE)
_EXEC_VERIFY_RE = re.compile(r"<exec_verify>\s*(pass|no_pass)\s*</exec_verify>", flags=re.IGNORECASE)

def _strip_tool_responses(text: str) -> str:
    return _TOOL_RESPONSE_RE.sub("", text)

def _first_exec_verify_outside_responses(text_after_call: str) -> Optional[str]:
    cleaned = _strip_tool_responses(text_after_call)
    m = _EXEC_VERIFY_RE.search(cleaned)
    return m.group(1).lower() if m else None

def _nearest_following_tool_response(solution_str: str, start_idx: int) -> Optional[str]:
    for m in _TOOL_RESPONSE_RE.finditer(solution_str):
        if m.start() >= start_idx:
            return m.group(1)
    return None

# 简单的错误兜底（你之前版本）
def _classify_exec_status_from_response(resp_text: Optional[str]) -> str:
    if not resp_text:
        return "ok"
    t = resp_text.strip().lower()

    # 轻量强化：即便没有 "error" 字样，也捕捉常见的 sqlite 报错短语
    if ("error" in t
        or "no such table" in t
        or "no such column" in t
        or "ambiguous column name" in t
        or "syntax error" in t):
        return "error"

    # 更稳妥的“空结果”识别
    if _looks_empty_sqlite(t):
        return "null"

    return "ok"

_EMPTY_LIST_RE   = re.compile(r"\[\s*\]")  # 匹配 [] 或 [   ]
_EMPTY_PHRASES_RE = re.compile(
    r"(?:\bno\s+rows?\b|"
    r"\b0\s+rows?\b|"
    r"\breturned\s+0\s+rows?\b|"
    r"\bempty\s+(?:result|set|list)\b|"
    r"\bresult\s+set\s+is\s+empty\b|"
    r"\bno\s+data\b)",
    re.IGNORECASE
)
_EMPTY_NUM_RE = re.compile(
    r"\b(?:rows?|rowcount|count)\s*:\s*0\b", re.IGNORECASE
)

def _looks_empty_sqlite(text_lower: str) -> bool:
    """
    轻量空结果识别：
      1) []（包含空白）
      2) 常见空集合短语
      3) 结构化计数：rows: 0 / rowcount: 0 / count: 0
    """
    return (
        bool(_EMPTY_LIST_RE.search(text_lower)) or
        bool(_EMPTY_PHRASES_RE.search(text_lower)) or
        bool(_EMPTY_NUM_RE.search(text_lower))
    )

def _safe_get_sql_from_tool_json(call_obj) -> Optional[str]:
    try:
        return call_obj.get("arguments", {}).get("code")
    except Exception:
        return None


# Verification reward
# please write python code to extract <> previous <tool call> SQL
# pass SQL reward
# elif not pass SQL
def get_process_reward_for_verify_action(solution_str, ground_truth, db, normalize=True):
    """
    Iteratate the solution_str from the beginning to the end
    If meet tool_call, extract the SQL and the tool name, and calculate reward
    if name is exec_sql,  add it to exec reward
    If name is memory_retrieve, add it to coherence reward
    Fianlly return the sum of exec reward and coherence reward

    Concretely:
    SQL = Extract the SQL between <tool_call> ... </tool_call>, the name of the dict between <tool_call> ... </tool_call> is exec_sql
    Pass_or_not = first extract value after this tool call and  exclude <tool_response> ... </tool_response>, then extract first values between  <exec_verify>pass</exec_verify> from other strings, values are either pass or no_pass
    1. <tool> SQL cannot be executed, but pass = 0 / not pass 1
    2. <tool> SQL can be executed but results are null and not obtain the same results as ground-truth, but pass, reward=0 / not pass => 0.5
    3. <tool> SQL can be executed, pass = 1, not pass = 0

    Coherence:
    SQL = Extract the SQL between <tool_call> ... </tool_call>, the name of the dict between <tool_call> ... </tool_call> is memory_retrieve
    Pass_or_not = first extract value after the <tool_call> memory_retrieve </tool_call> and   exclude <tool_response> ... </tool_response>, then extract the first value between  <exec_verify>pass</exec_verify> from other strings, values are either pass or no_pass
    1.   If pass, reward = F1 score between SQL and ground truth
    2.   If not pass, reward = 1 - F1 score between SQL and ground truth
    """
    exec_reward = 0.0
    coherence_reward = 0.0
    exec_count = 0
    coherence_count = 0

    for m in _TOOL_CALL_RE.finditer(solution_str):
        call_json_raw = m.group(1).strip()
        call_end_idx = m.end()
        try:
            call_obj = json.loads(call_json_raw)
        except Exception:
            continue

        name = (call_obj.get("name") or "").lower()
        sql_in_call = _safe_get_sql_from_tool_json(call_obj)

        after_call_text = solution_str[call_end_idx:]
        token = _first_exec_verify_outside_responses(after_call_text)
        pass_bool = (token == "pass")

        if name == "exec_sql":
            exec_count += 1
            resp_text = _nearest_following_tool_response(solution_str, call_end_idx)
            status = _classify_exec_status_from_response(resp_text)
            if status == "error":
                exec_reward += 0.0 if pass_bool else 1.0
            elif status == "null":
                exec_reward += 0.0 if pass_bool else 0.5
            else:  # ok
                exec_reward += 1.0 if pass_bool else 0.0

        elif name == "memory_retrieve":
            coherence_count += 1
            try:
                f1 = float(eval_f1_score_of_sql_overlap(sql_in_call, ground_truth, db)) if sql_in_call else 0.0
            except Exception:
                f1 = 0.0
            coherence_reward += f1 if pass_bool else (1.0 - f1)

    if normalize:
        if exec_count > 0:
            exec_reward /= exec_count
        if coherence_count > 0:
            coherence_reward /= coherence_count
    return float(exec_reward), float(coherence_reward)

def _extract_all_sql_codes(solution_str: str) -> List[str]:
    """
    提取所有 <tool_call> ... </tool_call> 中的 arguments.code 字段，不筛选工具名。
    """
    sqls = []
    # 找到所有 <tool_call> 块
    for block in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", solution_str, flags=re.DOTALL):
        try:
            obj = json.loads(block)
        except Exception:
            try:
                cleaned = re.sub(r",\s*([}\]])", r"\1", block)
                obj = json.loads(cleaned)
            except Exception:
                continue
        # 不管 name 是什么，只要 arguments 里有 code 就提取
        args = obj.get("arguments", {})
        if isinstance(args, dict) and "code" in args and isinstance(args["code"], str):
            sqls.append(args["code"])
    return sqls

# Gen SQL reward
# Propose: <tool call> SQL
# Refinement: <not pass> <tool call>/<end SQL>
# Steps
def get_process_reward_for_generate_action(solution_str: str,
                                           ground_truth: str, db) -> Tuple[float, float]:

    # markers: <tool call>
    # 1. For each tool call, check the f1 score between the proposed SQL and the ground truth SQL
    # if SQL is the same, only calculate once
    # reward = sum of f1 score for each tool call
    # 2. Improvement Tool gain: f1(new) - f1(old)

    """
    计算过程奖励:
    1. avg_f1: 所有工具调用的 F1 平均值
    2. improvement_gain: 反映随着工具调用次数增加是否逐渐逼近 ground_truth 的趋势
    """
    improvement_mode = "cumpos_norm"

    sql_candidates = _extract_all_sql_codes(solution_str)
    if not sql_candidates:
        return 0.0, 0.0

    # 调用 schema-aware F1
    f1_list = [eval_f1_score_of_sql_overlap(sql, ground_truth, db) for sql in sql_candidates]
    avg_f1 = sum(f1_list) / len(f1_list)

    if len(f1_list) > 1:
        deltas = [f1_list[i] - f1_list[i-1] for i in range(1, len(f1_list))]
    else:
        deltas = []

    if not deltas:
        improvement_gain = 0.0
    else:
        if improvement_mode == "net":
            improvement_gain = f1_list[-1] - f1_list[0]
        elif improvement_mode == "cumpos":
            improvement_gain = sum(max(0.0, d) for d in deltas)
        elif improvement_mode == "cumpos_norm":
            improvement_gain = sum(max(0.0, d) for d in deltas) / len(deltas)

    return avg_f1, improvement_gain


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
        exec_score = eval_exec_match(db, extracted_sql, g_str, False, False, False)
        exact_score = eval_exact_match_score(db, extracted_sql, g_str, dataset="cosql")
    except Exception as e:
        print(f"Reward Score Error: {e}; prediction SQL is: {extracted_sql}")

    # finally return the weighted sum of different scores
    # weight is from kwargs which is an array
    weights = kwargs.get("weights", [0.5, 0.5])


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
    # raw_text = "SELECT COUNT(*) FROM Employee WHERE salary > (SELECT AVG(salary) FROM Employee);"
    # print(extract_code(raw_text))

    # test eval_f1_score_of_sql_overlap function
    pred_sql = "SELECT dept_name, AVG(salary) FROM instructor GROUP BY dept_name"
    gt_sql = "SELECT dept_name, AVG(salary) FROM instructor GROUP BY dept_name"
    db = "college_2"
    f1 = eval_f1_score_of_sql_overlap(pred_sql, gt_sql, db)
    print(f"F1 score of SQL overlap: {f1}")







