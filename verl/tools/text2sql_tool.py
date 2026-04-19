
import logging
import os
import pdb
from typing import Any, Optional, Tuple
from uuid import uuid4

from verl.utils.reward_score import gsm8k

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.eval.exec_eval import exec_sql
import ast

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Text2sqlTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
            "db_id": kwargs.get("create_kwargs", {}).get("db_id", None),
            "current_q": kwargs.get("create_kwargs", {}).get("current_q", None),
            "memory": kwargs.get("create_kwargs", {}).get("memory", None),
            "history_len": kwargs.get("create_kwargs", {}).get("history_len", 0),
        }
        self._db_id = kwargs.get("create_kwargs", {}).get("db_id", None)
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        print(f"kwargs is: {kwargs}")
        db = self._instance_dict[instance_id]["db_id"]
        current_q = self._instance_dict[instance_id]["current_q"]
        memory = self._instance_dict[instance_id]["memory"]
        history_len =  self._instance_dict[instance_id]["history_len"]

        if not isinstance(code, str):
            code = str(code)
        # flag = "results"
        # return_msg = "hahha"
        # print(code + "|" + db)
        flag,  return_msg = exec_sql(db, code)
        return_msg = return_msg[:200]

        memory = ast.literal_eval(memory)
        memory_str = ""
        # expand the memory array to a string
        memory_info = memory[1]
        for idx, atuple in enumerate(memory_info):
            cleaned_parsed = {k: v for k, v in atuple[2].items() if k != 'sql_results'}
            try:
                memory_str += f"""
        == Turn {idx} ==
        Question: {atuple[0]}
        Ground-Truth SQL: {atuple[1]}
        Parsed Elements for each term: {cleaned_parsed}
        == Turn {idx} ==
            """
            except:
                memory_str += f"""
        == Turn {idx} ==
        Question: {atuple[0]}
        Ground-Truth SQL: {atuple[1]}
        Parsed Elements for each term: {cleaned_parsed}
        == Turn {idx} ==
                        """

        if flag == "SQLexception":
            print(f"SQL parse error: {db} | {code}")

        tool_reward = 0.0

        # split by history_len
        if history_len == 0:
            add_str = " please note return the final SQL inside `<answer_sql>...</answer_sql>`. "
        else:
            add_str = f""" please do the following thing:
You are a coherence verifier for Multi-turn Text2SQL.
Given the Memory (historical information in order):  
{memory_str}  

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

3. If `<memory_verify>no_pass</memory_verify>`: explain issues, think step by step refine SQL, and then please do functional calling and re-call `exec_sql` tool again via <tool_call>  to execute the corrected SQL and get the execution results. Repeat until you get `pass`. 
4. If `<memory_verify>pass</memory_verify>`: return the final SQL inside `<answer_sql>...</answer_sql>`.  

Note finally you should return the final SQL inside `<answer_sql>...</answer_sql>
"""
            add_str = "You have to call `memory_retrieve` tool via <tool_call>  at least once to ensure the current generated SQL is coherent with the historical memory. "


        msg_and_the_following_instructions = f"""
Recap:  
- Current question: {current_q}  
- Generated SQL: {code}  
- SQL execution results (truncated to 200 characters): {return_msg}  

Now please:  
1. Verify whether the SQL execution results are valid:  
   - Check if the SQL runs without errors.  
   - Check if the returned columns exist in the schema and are relevant to the question.  
   - Check if the results contain unexpected NULL values, empty sets, or error messages.  

2. After verifying, output:  
   - <exec_verify>pass</exec_verify> if the results are valid and consistent with the schema.  
   - <exec_verify>no_pass</exec_verify> if the results show errors, irrelevant columns, or invalid values.  

3. If <exec_verify>no_pass</exec_verify>, think step by step, refine the SQL and provide a corrected SQL and then execute it via re-calling ``exec_sql`` tool again via <tool_call>. Repeat until you get valid results.
4. If <exec_verify>pass</exec_verify>,  """ + add_str

        # return  return_msg + "return the final SQL inside `<answer_sql>...</answer_sql>", tool_reward, {"query_flag": flag}

        return ToolResponse(text=msg_and_the_following_instructions), tool_reward, {"query_flag": flag}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Multi-turn reward
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
