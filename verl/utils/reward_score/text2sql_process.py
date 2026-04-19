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
from text2sql import extract_code

# eval F1 score of the parsed SQL overlap
def eval_f1_score_of_sql_overlap(pred_sql, gt_sql, db):
    # parse each sql with the database
    try:
        # TODO: set to your local path to the CoSQL/SParC database directory
        db_dir = f"PATH/TO/database/cosql/database/{db}"
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

    except Exception as e:
        print(f"Error in eval_f1_score_of_sql_overlap: {e}")
        return 0.0

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
    return m.group(1).lower() if m else "pass"

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
    if ("no such table" in t
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
                exec_reward += 0.0 if pass_bool else 0.1
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
                                           ground_truth: str, db) -> float:

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
        return 0.0

    # 调用 schema-aware F1
    f1_list = [eval_f1_score_of_sql_overlap(sql, ground_truth, db) for sql in sql_candidates]
    if len(f1_list) == 0:
        return 0.0

    avg_f1 = sum(f1_list) / len(f1_list)

    return avg_f1


def compute_score(solution_str, ground_truth, **kwargs):
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

    exec_score = 0.0
    exact_score = 0.0

    try:
        exec_score = eval_exec_match(db, extracted_sql, g_str, False, False, False)
        exact_score = eval_exact_match_score(db, extracted_sql, g_str, dataset="cosql")
    except Exception as e:
        print(f"Reward Score Error: {e}; prediction SQL is: {extracted_sql}")

    # add get_process_reward_for_verify_action
    try:
        exec_verify_reward, coherence_reward  = get_process_reward_for_verify_action(solution_str, g_str, db)
        exec_verify_reward = 0.1 * exec_verify_reward
        coherence_reward = 0.1 * coherence_reward
    except Exception as e:
        print(f"Reward Score Error in get_process_reward_for_verify_action: {e}")
        exec_verify_reward = 0.0
        coherence_reward = 0.0

    # add get_process_reward_for_generate_action
    process_reward_for_generate = get_process_reward_for_generate_action(solution_str, g_str, db)
    process_reward_for_generate = 0.8 * process_reward_for_generate

    # weights = kwargs.get("weights", [0.5, 0.5])

    # return a score dict
    return_dict = {
        "score": exec_score + exact_score + exec_verify_reward + coherence_reward + process_reward_for_generate,
        "outcome_reward": exec_score + exact_score,
        "exec_verify_reward": exec_verify_reward,
        "coherence_reward": coherence_reward,
        "process_reward_for_generate": process_reward_for_generate
    }
    # convert all values to float
    for k in return_dict:
        return_dict[k] = float(return_dict[k])
    return return_dict

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
    example_str = """
<think>
Okay, let's tackle this question: "Which models are the most popular?" The user wants the SQL query that would answer this, not the actual answer. Let me think about how to approach this.

First, I need to understand what "most popular" means here. Typically, that would refer to the models with the highest number of cars produced or the most sales. Looking at the database schema, there's a table called cars_data which has entries for each car, and it references car_names via MakeId. The model_list table links Maker (which is a car_makers.Id) to Model. 

So, to find the most popular models, I probably need to count how many times each model appears in the cars_data table. Because each entry in cars_data represents a car, the count of entries per model would indicate popularity.

The steps would be:

1. Join the cars_data table with car_names to get the model names. The cars_data.Id is a foreign key to car_names.MakeId.
2. Then, group by the model and count the number of entries per model.
3. Order the results in descending order of the count to get the most popular first.
4. Maybe limit the results to the top ones, but the question doesn't specify a number, so perhaps just order and let the user decide.

Wait, the question says "which models are the most popular," so maybe just ordering by count descending is sufficient. However, the exact SQL might need to select the model name and the count.

Looking at the tables again: car_names has Model and Make columns. The cars_data table has Id (which is MakeId in car_names) and other fields. So joining cars_data with car_names on cars_data.Id = car_names.MakeId would give the model name.

So the SQL would be:

SELECT car_names.Model, COUNT(*) as count
FROM cars_data
JOIN car_names ON cars_data.Id = car_names.MakeId
GROUP BY car_names.Model
ORDER BY count DESC;

But wait, the user might want to see the model name and the count. Also, maybe the model is in the model_list table. Wait, let me check the schema again.

Looking at the model_list table: it has ModelId, Maker (which is a car_makers.Id), and Model. The car_names table has MakeId, Model, and Make. The cars_data table's Id is a foreign key to car_names.MakeId. So the Model in car_names is the actual model name, like "chevrolet chevelle malibu".

So joining cars_data with car_names on Id = MakeId would get the model name. Then grouping by that model and counting the number of cars in cars_data for each model would give the popularity.

Alternatively, maybe the model_list table is needed. But the cars_data table's Id is a foreign key to car_names.MakeId, which is a MakeId. So the model name is in car_names.Model. Therefore, the join between cars_data and car_names is correct.

So the SQL would be as above. But the user might want to use table aliases for clarity. Also, the question says to translate the question to SQL, not execute it, but the user requires using the exec_sql tool at least once. So I need to generate the SQL, then call exec_sql to verify.

Wait, but the user's note says to call exec_sql at least once. So I need to generate the SQL, then execute it via the tool, check if it's correct, and adjust if needed.

But since I don't have actual data, maybe the exec_sql tool will return some results. However, the previous example had a correct SQL for counting car models per maker. Let me check if my SQL is correct.

Another thing: the previous question was about counting car models per maker, which used model_list and car_makers. But this question is about models' popularity, which is based on the number of cars in cars_data. So the approach is different.

So the SQL I wrote earlier should be correct. Let me write that and call exec_sql.
</think>

<tool_call>
{"name": "exec_sql", "arguments": {"code": "SELECT car_names.Model, COUNT(*) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId GROUP BY car_names.Model ORDER BY COUNT(*) DESC;"}}
</tool_call>
user
<tool_response>

Recap:  
- Current question: Question: Which models are the most popular?  
- Generated SQL: SELECT car_names.Model, COUNT(*) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId GROUP BY car_names.Model ORDER BY COUNT(*) DESC;  
- SQL execution results (truncated to 200 characters): The sql results example is: [('ford', 53), ('chevrolet', 48), ('plymouth', 32), ('amc', 29), ('dodge', 28), ('toyota', 26), ('datsun', 23), ('volkswagen', 22), ('buick', 17), ('pontiac', 16), ('honda'  

Now please:  
1. Verify whether the SQL execution results are valid:  
   - Check if the SQL runs without errors.  
   - Check if the returned columns exist in the schema and are relevant to the question.  
   - Check if the results contain unexpected NULL values, empty sets, or error messages.  

2. After verifying, output:  
   - <exec_verify>pass</exec_verify> if the results are valid and consistent with the schema.  
   - <exec_verify>no_pass</exec_verify> if the results show errors, irrelevant columns, or invalid values.  

3. If <exec_verify>no_pass</exec_verify>, think step by step, refine the SQL and provide a corrected SQL and then execute it via re-calling ``exec_sql`` tool again via <tool_call>. Repeat until you get valid results.
4. If <exec_verify>pass</exec_verify>,  You have to call `memory_retrieve` tool via <tool_call>  at least once to ensure the current generated SQL is coherent with the historical memory. 
</tool_response>
assistant
<think>
Okay, let's check the SQL execution results. The generated SQL is:

SELECT car_names.Model, COUNT(*) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId GROUP BY car_names.Model ORDER BY COUNT(*) DESC;

The results show models like 'ford', 'chevrolet', etc., with counts. This seems valid. The columns 'Model' and the count are relevant to the question about popular models. No errors mentioned, and the results make sense. So, the execution is valid.

Now, I need to call memory_retrieve to ensure coherence with historical data. The previous correct SQL was about counting models per maker using model_list and car_makers. This query is different because it's about popularity based on cars_data, which is a separate approach. Since the user's question is about popularity, this SQL is appropriate. So, memory_retrieval should confirm it's coherent.
</think>

<tool_call>
{"name": "memory_retrieve", "arguments": {"code": "SELECT car_names.Model, COUNT(*) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId GROUP BY car_names.Model ORDER BY COUNT(*) DESC;"}}
</tool_call>
user
<tool_response>

You are a coherence verifier for Multi-turn Text2SQL.

Current Question: Question: Which models are the most popular?  
Proposed SQL: SELECT car_names.Model, COUNT(*) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId GROUP BY car_names.Model ORDER BY COUNT(*) DESC;  
The execution results of the proposed SQL: The sql results example is: [('ford', 53), ('chevrolet', 48), ('plymouth', 32), ('amc', 29), ('dodge', 28), ('toyota', 26), ('datsun', 23), ('volkswagen', 22), ('buick', 17), ('pontiac', 16), ('honda'

Memory (historical information in order):  

== Turn 0 ==
Question: Question: How many car models are produced by each maker? List the count and the maker full name.
Ground-Truth SQL: SELECT Count ( * )  ,  T2.FullName  FROM MODEL_LIST AS T1 JOIN CAR_MAKERS AS T2 ON T1.Maker   =   T2.Id GROUP BY T2.id
Parsed Elements for each term: {'from': {'table_units': [('table_unit', '__model_list__'), ('table_unit', '__car_makers__')], 'conds': [(False, 2, (0, (0, '__model_list.maker__', False), None), (0, '__car_makers.id__', False), None)]}, 'select': ['__all__', '__car_makers.fullname__'], 'where': [], 'groupBy': [(0, '__car_makers.id__', False)], 'having': [], 'orderBy': [], 'limit': None, 'intersect': None, 'union': None, 'except': None}
SQL Results (truncated to 50 characters): [(1, 'American Motor Company'), (2, 'Volkswagen'), (1, 'BMW'), (5, 'General Motors'), (3, 'Ford Motor Company'), (4, 'Chrysler'), (1, 'Citroen'), (2, 'Nissan Motors'), (1, 'Fiat'), (1, 'Honda'), (1, 'Mazda'), (2, 'Daimler Benz'), (1, 'Opel'), (1, 'Peugeaut'), (1, 'Renault'), (1, 'Saab'), (1, 'Subaru'), (2, 'Toyota'), (1, 'Triumph'), (1, 'Volvo'), (1, 'Kia Motors'), (1, 'Hyundai')]
== Turn 0 ==
      

Your tasks:  
1. Verify whether the Proposed SQL is coherent with the Current Question and the Memory, based on the relation between the Current Question and Historical Questions.  
   - If the Current Question introduces changes (new columns, conditions, ordering, etc.), SQL should update accordingly.  
   - If not, SQL must remain consistent with the Historical Questions.  

Step-by-step reasoning checklist:  
   1. First parse the Proposed SQL into its components (SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, JOINs).
   2. Check tables are consistent with context.  
   3. Check selected columns match current and historical intent.  
   4. Check conditions (WHERE/GROUP/HAVING) reflect the relation between current and past questions.  
   5. Check ordering (ORDER BY) is preserved unless explicitly changed.  
   6. Verify that joins and table relationships follow the established context.
   7. Make sure the SQL and the execution results of the proposed SQL answer the current question while remaining logically coherent with the conversation history and execution results.

2. After verifying, output one of the following:  
   - `<memory_verify>pass</memory_verify>` if coherent.  
   - `<memory_verify>no_pass</memory_verify>` if not coherent.  

3. If `no_pass`: explain issues, think step by step to refine SQL, and then please call `exec_sql` tool again via <tool_call> to check the corrected SQL and get the execution results. Repeat until you get `pass`. 
4. If `pass`: return the final SQL inside `<answer_sql>...</answer_sql>`.  

Note finally you should return the final SQL inside `<answer_sql>...</answer_sql>

</tool_response>
assistant
<think>
Okay, let's go through the verification step by step.

1. **Parse the Proposed SQL**: The SQL selects `car_names.Model` and counts the number of entries from `cars_data` joined with `car_names` on `Id = MakeId`. It groups by the model and orders by the count in descending order. This should give the most popular models based on the number of cars.

2. **Check Tables**: The tables involved are `cars_data` and `car_names`, which are part of the schema. The join condition is correct as `cars_data.Id` references `car_names.MakeId`.

3. **Selected Columns**: The `Model` column from `car_names` is selected, which is the correct column for the model name. The count is aggregated, which makes sense for determining popularity.

4. **Conditions and Grouping**: The grouping by `car_names.Model` is appropriate to count each model's occurrences. The ordering by count descending will list the most popular first.

5. **Ordering**: The `ORDER BY COUNT(*) DESC` ensures the results are sorted from most to least popular, which aligns with the question's request for "most popular" models.

6. **Joins and Relationships**: The join between `cars_data` and `car_names` is correct based on the schema. The previous question involved joining `model_list` and `car_makers`, but this question is about popularity based on `cars_data`, so the tables and joins are appropriate here.

7. **Execution Results**: The results show models with their counts, which matches the expected output. There's no mention of errors, so the SQL runs correctly.

Comparing with the historical question, which counted models per maker using `model_list` and `car_makers`, this query is different but appropriate for the new question about popularity. The current SQL correctly uses `cars_data` to count occurrences of each model.

The SQL is coherent with the current question and the schema. The execution results are valid and relevant. No issues found.
</think>

<answer_sql>SELECT car_names.Model, COUNT(*) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId GROUP BY car_names.Model ORDER BY COUNT(*) DESC;</answer_sql>
"""


    # pred_sql = "SELECT dept_name, AVG(salary) FROM instructor GROUP BY dept_name"
    # gt_sql = "SELECT dept_name, AVG(salary) FROM instructor GROUP BY dept_name"
    db = "car_1"
    # f1 = eval_f1_score_of_sql_overlap(pred_sql, gt_sql, db)
    # print(f"F1 score of SQL overlap: {f1}")

    gt_sql = "SELECT car_names.Model FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId GROUP BY car_names.Model ORDER BY COUNT(*) DESC;"
    # print(get_process_reward_for_generate_action(example_str, gt_sql, db))
    print(get_process_reward_for_verify_action(example_str, gt_sql, db))






