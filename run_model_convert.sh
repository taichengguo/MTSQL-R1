set -x

# f1-score two fields
#model_path=checkpoints/verl_grpo_proj1/qwen3_safety_f1_score/global_step_75/actor

# f1-score all fields
#model_path=checkpoints/verl_grpo_proj1/qwen3_safety_f1_score_all_fields/global_step_220/actor
#model_path=checkpoints/verl_grpo_proj1/qwen3_safety_f1_score_all_fields/global_step_320/actor  # best


# f1-score joint
#model_path=checkpoints/verl_grpo_proj1/qwen3_safety_f1_score_joint/global_step_45/actor

#model_path=checkpoints/verl_grpo_proj1/qwen3_safety_f1_score_joint/global_step_130/actor


################################################################ v2

# three fields
#model_path=checkpoints/verl_grpo_proj1/qwen3_safety_f1_score_three_fields_v2/global_step_80/actor
#model_path=checkpoints/verl_grpo_proj1/qwen3_safety_f1_score_three_fields_v2/global_step_120/actor


# All fields - Default
#model_path=checkpoints/verl_grpo_proj1/qwen3_safety_f1_score_v2/global_step_380/actor

# All fields - Encourage Exploration
#model_path=checkpoints/verl_grpo_proj1/qwen3_safety_f1_score_v2_pt/global_step_280/actor

#TARGET_FOLDER=safety_ckpt

################################################################ Text2SQL


#model_path=checkpoints/verl_grpo_text2sql/qwen3_1.7b_text2sql_07192/global_step_60/actor
TARGET_FOLDER=text2sql_ckpt
#model_path=checkpoints/verl_grpo_text2sql/cosql_17B/global_step_130/actor
#model_path=checkpoints/verl_grpo_text2sql_sparc/sparc_4B/global_step_150/actor
#model_path=checkpoints/verl_grpo_text2sql_sparc/sparc_4B/global_step_180/actor
#model_path=checkpoints/verl_grpo_text2sql/cosql_4B/global_step_190/actor
#model_path=checkpoints/verl_grpo_text2sql_sparc/sparc_17B/global_step_40/actor
#model_path=checkpoints/verl_grpo_text2sql_sparc/sparc_17B/global_step_20/actor
model_path=checkpoints/verl_grpo_text2sql/cosql_17B/global_step_120/actor



python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir $model_path \
    --target_dir $TARGET_FOLDER
