import os
import re
import asyncio
import sqlite3
import threading
from typing import Tuple, Any, List, Set
from itertools import product
from collections import defaultdict
import tqdm
import random
import time
import pickle as pkl
import subprocess
from itertools import chain
import re
import sqlparse
from typing import List, Tuple, Set, Iterator, Dict, Any, Union
from sqlparse.sql import Comparison, Identifier
from sqlparse.tokens import Whitespace
import itertools
from collections import namedtuple
import signal
import time

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Query execution timed out")

Token = namedtuple('Token', ['ttype', 'value'])
VALUE_NUM_SYMBOL = 'VALUERARE'
QUOTE_CHARS = {'`', '\'', '"'}


def tokenize(query: str) -> List[Token]:
    tokens = list([Token(t.ttype, t.value) for t in sqlparse.parse(query)[0].flatten()])
    return tokens


def join_tokens(tokens: List[Token]) -> str:
    return ''.join([x.value for x in tokens]).strip().replace('  ', ' ')


def round_trip_test(query: str) -> None:
    tokens = tokenize(query)
    reconstructed = ''.join([token.value for token in tokens])
    assert query == reconstructed, "Round trip test fails for string %s" % query


def postprocess(query: str) -> str:
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    return query


# strip_query, reformat_query and replace values
# were implemented by Yu Tao for processing CoSQL
def strip_query(query: str) -> Tuple[List[str], List[str]]:
    query_keywords, all_values = [], []

    # then replace all stuff enclosed by "" with a numerical value to get it marked as {VALUE}

    # Tao's implementation is commented out here.
    """
    str_1 = re.findall("\"[^\"]*\"", query)
    str_2 = re.findall("\'[^\']*\'", query)
    values = str_1 + str_2
        """

    toks = sqlparse.parse(query)[0].flatten()
    values = [t.value for t in toks if t.ttype == sqlparse.tokens.Literal.String.Single or t.ttype == sqlparse.tokens.Literal.String.Symbol]


    for val in values:
        all_values.append(val)
        query = query.replace(val.strip(), VALUE_NUM_SYMBOL)

    query_tokenized = query.split()
    float_nums = re.findall("[-+]?\d*\.\d+", query)
    all_values += [qt for qt in query_tokenized if qt in float_nums]
    query_tokenized = [VALUE_NUM_SYMBOL if qt in float_nums else qt for qt in query_tokenized]

    query = " ".join(query_tokenized)
    int_nums = [i.strip() for i in re.findall("[^tT]\d+", query)]

    all_values += [qt for qt in query_tokenized if qt in int_nums]
    query_tokenized = [VALUE_NUM_SYMBOL if qt in int_nums else qt for qt in query_tokenized]
    # print int_nums, query, query_tokenized

    for tok in query_tokenized:
        if "." in tok:
            table = re.findall("[Tt]\d+\.", tok)
            if len(table) > 0:
                to = tok.replace(".", " . ").split()
                to = [t.lower() for t in to if len(t) > 0]
                query_keywords.extend(to)
            else:
                query_keywords.append(tok.lower())

        elif len(tok) > 0:
            query_keywords.append(tok.lower())
    return query_keywords, all_values


def reformat_query(query: str) -> str:
    query = query.strip().replace(";", "").replace("\t", "")
    query = ' '.join([t.value for t in tokenize(query) if t.ttype != sqlparse.tokens.Whitespace])
    t_stars = ["t1.*", "t2.*", "t3.*", "T1.*", "T2.*", "T3.*"]
    for ts in t_stars:
        query = query.replace(ts, "*")
    return query


def replace_values(sql: str) -> Tuple[List[str], Set[str]]:
    sql = sqlparse.format(sql, reindent=False, keyword_case='upper')
    # sql = re.sub(r"(<=|>=|!=|=|<|>|,)", r" \1 ", sql)
    sql = re.sub(r"(T\d+\.)\s", r"\1", sql)
    query_toks_no_value, values = strip_query(sql)
    return query_toks_no_value, set(values)


# extract the non-value tokens and the set of values
# from a sql query
def extract_query_values(sql: str) -> Tuple[List[str], Set[str]]:
    reformated = reformat_query(query=sql)
    query_value_replaced, values = replace_values(reformated)
    return query_value_replaced, values


# plug in the values into query with value slots
def plugin(query_value_replaced: List[str], values_in_order: List[str]) -> str:
    q_length = len(query_value_replaced)
    query_w_values = query_value_replaced[:]
    value_idx = [idx for idx in range(q_length) if query_value_replaced[idx] == VALUE_NUM_SYMBOL.lower()]
    assert len(value_idx) == len(values_in_order)

    for idx, value in zip(value_idx, values_in_order):
        query_w_values[idx] = value
    return ' '.join(query_w_values)


# a generator generating all possible ways of
# filling values into predicted query
def plugin_all_permutations(query_value_replaced: List[str], values: Set[str]) -> Iterator[str]:
    num_slots = len([v for v in query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
    for values in itertools.product(*[list(values) for _ in range(num_slots)]):
        yield plugin(query_value_replaced, list(values))


# given the gold query and the model prediction
# extract values from the gold, extract predicted sql with value slots
# return 1) number of possible ways to plug in gold values and 2) an iterator of predictions with value plugged in
def get_all_preds_for_execution(gold: str, pred: str) -> Tuple[int, Iterator[str]]:
    _, gold_values = extract_query_values(gold)
    pred_query_value_replaced, _ = extract_query_values(pred)
    num_slots = len([v for v in pred_query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
    num_alternatives = len(gold_values) ** num_slots
    return num_alternatives, plugin_all_permutations(pred_query_value_replaced, gold_values)


def remove_distinct(s):
    toks = [t.value for t in list(sqlparse.parse(s)[0].flatten())]
    return ''.join([t for t in toks if t.lower() != 'distinct'])


def extract_all_comparison_from_node(node: Token) -> List[Comparison]:
    comparison_list = []
    if hasattr(node, 'tokens'):
        for t in node.tokens:
            comparison_list.extend(extract_all_comparison_from_node(t))
    if type(node) == Comparison:
        comparison_list.append(node)
    return comparison_list


def extract_all_comparison(query: str) -> List[Comparison]:
    tree = sqlparse.parse(query)[0]
    comparison_list = extract_all_comparison_from_node(tree)
    return comparison_list


def extract_toks_from_comparison(comparison_node: Comparison) -> List[Token]:
    tokens = [t for t in comparison_node.tokens if t.ttype != Whitespace]
    return tokens


def extract_info_from_comparison(comparison_node: Comparison) -> Dict[str, Any]:
    tokens = extract_toks_from_comparison(comparison_node)
    left, op, right = tokens

    returned_dict = {
        'left': left,
        'op': op.value,
        'right': right
    }

    if type(left) != Identifier:
        return returned_dict

    table = None
    if len(left.tokens) == 3 and re.match('^[tT][0-9]$', left.tokens[0].value) is None:
        table = left.tokens[0].value.lower()
    col = left.tokens[-1].value

    if type(right) == Identifier:
        if len(right.tokens) == 1 and type(right.tokens[0]) == sqlparse.sql.Token:
            right_val = right.tokens[0].value
        else:
            return returned_dict
    elif type(right) == sqlparse.sql.Token:
        right_val = right.value
    else:
        return returned_dict

    returned_dict['table_col'], returned_dict['val'] = (table, col.upper()), process_str_value(right_val)

    return returned_dict


def extract_all_comparison_from_query(query: str) -> List[Dict[str, Any]]:
    comparison_list = extract_all_comparison(query)
    return [extract_info_from_comparison(c) for c in comparison_list]


def extract_typed_value_in_comparison_from_query(query: str) -> List[Tuple[Tuple[Union[str, None], str], str]]:
    cmps = extract_all_comparison_from_query(query)
    typed_values = [(cmp['table_col'], cmp['val']) for cmp in cmps if 'table_col' in cmp]
    for table, col, val1, val2 in re.findall('(?:([^\.\s]*)\.)?([^\.\s]+) between ([^\s;]+) and ([^\s;]+)', query, re.IGNORECASE):
        if table == '':
            table = None
        else:
            table = table.lower()
        col = col.upper()
        for v in [val1, val2]:
            typed_values.append(((table, col), v))
    return typed_values


def process_str_value(v: str) -> str:
    if len(v) > 0 and v[0] in QUOTE_CHARS:
        v = v[1:]
    if len(v) > 0 and v[-1] in QUOTE_CHARS:
        v = v[:-1]
    for c in QUOTE_CHARS:
        v = v.replace(c + c, c)
    return v




threadLock = threading.Lock()
TIMEOUT = 60
EXEC_TMP_DIR = 'tmp/'

def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


# unorder each row in the table
# [result_1 and result_2 has the same bag of unordered row]
# is a necessary condition of
# [result_1 and result_2 are equivalent in denotation]
def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


# return whether two bag of relations are equivalent
def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


# check whether two denotations are correct
def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    if len(result1) == 0 and len(result2) == 0:
        return True

    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        return False

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        return False

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


def replace_cur_year(query: str) -> str:
    return re.sub(
        "YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE
    )


# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path: str):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


def exec_on_db_(sqlite_path: str, query: str) -> Tuple[str, Any]:
    query = replace_cur_year(query)
    cursor = get_cursor_from_path(sqlite_path)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        cursor.connection.close()
        return "result", result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        return "exception", e

def exec_on_db(
    sqlite_path: str, query: str, process_id: str = "", timeout: int = TIMEOUT
) -> Tuple[str, Any]:
    # signal.signal(signal.SIGALRM, timeout_handler)
    # signal.alarm(timeout)  # Set the timeout

    try:
        result = exec_on_db_(sqlite_path, query)
        # signal.alarm(0)  # Disable the alarm
        return result
    except TimeoutError:
        return ('exception', f"Query execution timed out after {timeout} seconds")
    except Exception as e:
        return ("exception", e)


# postprocess the model predictions to avoid execution errors
# e.g. removing spaces between ">" and "="
def postprocess(query: str) -> str:
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    return query


def exec_sql(db: str, p_str: str):
    try:
        p_str = postprocess(p_str)
        # if not False:
        #     p_str = remove_distinct(p_str)
    except Exception as e:
        p_flag = "SQLexception"
        return_mesg = f"SQL parse error: {e}"
        return p_flag, return_mesg

    # execute sql and obtain results
    # TODO: set to your local path to the CoSQL/SParC database directory
    db_dir = f"PATH/TO/database/cosql/database/{db}"

    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if '.sqlite' in basename]
    ranger = db_paths
    for db_path in ranger:
        p_flag, p_denotation = exec_on_db(db_path, p_str)

    if p_flag == "exception":
        return_mesg = f"Predicted sql has error on database. Error is: {p_denotation}"
        # print(f"{db} - {p_str} has error: {p_denotation}??")
    else:
        return_mesg = f"The sql results example is: {p_denotation}"

    return p_flag, return_mesg


# approximate whether p_str and g_str are semantically equivalent
# db is the database path
# we are going to evaluate whether they are equivalent in all the databases
# that are in the same directory as db
# 0 if denotationally equivalent
# 1 otherwise
# the meaning of each auxillary argument can be seen in the parser definition in evaluation.py
def eval_exec_match(db: str, p_str: str, g_str: str, plug_value: bool, keep_distinct: bool, progress_bar_for_each_datapoint: bool) -> int:
    # post-process the prediction.
    # e.g. removing spaces between ">" and "="
    p_str, g_str = postprocess(p_str), postprocess(g_str)
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)

    # we decide whether two denotations are equivalent based on "bag semantics"
    # https://courses.cs.washington.edu/courses/cse444/10sp/lectures/lecture16.pdf
    # if there is order by in query, then we assume order of the rows matter
    # order by might also be used to find the max/min instead of sorting,
    # but in that case the result mostly only contains one row and hence order_matters does not make a difference
    order_matters = 'order by' in g_str.lower()

    # find all databases in the same directory
    # TODO what I changed
    db_dir = os.path.dirname(db)
    db_dir = f"database/cosql/database/{db}"
    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if '.sqlite' in basename]

    preds = [p_str]
    # if plug in value (i.e. we do not consider value prediction correctness)
    # enumerate all ways to plug in values in the gold query to the model predictions
    # otherwise, we only evaluate the predicted query with its own value prediction
    if plug_value:
        _, preds = get_all_preds_for_execution(g_str, p_str)
        # we did not add this line in our EMNLP work
        # this reduces "false negatives" when value is substituted
        preds = chain([p_str], preds)

    for pred in preds:

        pred_passes = 1
        # compare the gold and predicted denotations on each database in the directory
        # wrap with progress bar if required
        if progress_bar_for_each_datapoint:
            ranger = tqdm.tqdm(db_paths)
        else:
            ranger = db_paths

        for db_path in ranger:
            g_flag, g_denotation = asyncio.run(exec_on_db(db_path, g_str))
            p_flag, p_denotation = asyncio.run(exec_on_db(db_path, pred))

            # we should expect the gold to be succesfully executed on the database
            # assert g_flag != 'exception', 'gold query %s has error on database file %s' % (g_str, db_path)
            if g_flag == 'exception': return 0
            # wrong if execution fails
            if p_flag == 'exception':
                pred_passes = 0

            # if denotations are not equivalent, the prediction must be wrong
            elif not result_eq(g_denotation, p_denotation, order_matters=order_matters):
                pred_passes = 0
            if pred_passes == 0:
                break

        # the model prediction has the same denotation as the gold for all databases
        if pred_passes == 1:
            return 1

    # none of the predictions passed
    return 0
