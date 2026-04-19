
import logging
import os
import pdb
from typing import Any, Optional, Tuple
from uuid import uuid4

from verl.utils.reward_score import gsm8k

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from verl.tools.eval.exec_eval import exec_sql

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
import ast


class MemoryRetriever(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        code = parameters.get("code", "")
        memory = kwargs['memory']
        db = kwargs['db_id']

        if not isinstance(code, str):
            code = str(code)

        flag,  return_msg = exec_sql(db, code)
        execution_results = return_msg[:200]

        memory = ast.literal_eval(memory)

        current_q = memory[0]

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
SQL Results (truncated to 50 characters): {atuple[2]["sql_results"][:50]}
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

        # db = parameters.get("db", "")
        if not isinstance(code, str):
            code = str(code)
        return_msg = f"""
You are a coherence verifier for Multi-turn Text2SQL.

Current Question: {current_q}  
Proposed SQL: {code}  
The execution results of the proposed SQL: {execution_results}

Memory (historical information in order):  
{memory_str}  
Each memory entry includes:  
- Historical question  
- SQL  
- Parsed elements (columns, conditions, etc.)  
- SQL results (truncated to 50 characters)  

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

3. If `no_pass`: explain issues, think step by step refine SQL, and then please call `text2sql` tool to check the corrected SQL and get the execution results. Repeat until you get `pass`. 
4. If `pass`: return the final SQL inside `<answer_sql>...</answer_sql>`.  

Note finally you should return the final SQL inside `<answer_sql>...</answer_sql>
"""
        tool_reward = 0.0
        # return_msg = f"return the final SQL inside `<answer_sql>...</answer_sql>"
        return  return_msg, tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Multi-turn reward
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
