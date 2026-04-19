set -x

ulimit -n 65535
#rollout_mode="async"
#if [ "$rollout_mode" = "async" ]; then
#    export VLLM_USE_V1=1
#    return_raw_chat="True"
#fi

DATA_FOLDER="data/cosql"
#DATA_FOLDER="data/sparc"
#DATA_FOLDER="data/gsm8k_our_test/"

# Dataset: Except Simple
# Except correct
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

#MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
# SET VLLM to v1 for async


# Important: One model

#MODEL_PATH="Qwen/Qwen3-0.6B"
#MODEL_PATH="Qwen/Qwen3-1.7B"
#MODEL_PATH="Qwen/Qwen3-4B"
#MODEL_PATH="Qwen/Qwen3-8B"
#MODEL_PATH="Qwen/Qwen3-14B"
#MODEL_PATH="Qwen/Qwen3-32B"

#MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"


#MODEL_PATH="./text2sql_ckpt"


# Multi-turns multi-tools call
# 14B -> data;   1.7B RL
# Multiple times: Duplicate the training dataset
# 50 times
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Model list
model_list=(
#  "../llamafactory/saves/sql/cosql_round1/checkpoint-1784"
#  "../llamafactory/saves/sql/cosql_round2/checkpoint-2000"
  "../llamafactory/saves/sql/cosql_round3_4B/checkpoint-2054"
##  "../llamafactory/saves/sql/cosql_round1_17B/checkpoint-1782"
#  "../llamafactory/saves/sql/cosql_round2_17B/checkpoint-2000"
#
#  "../llamafactory/saves/sql/sparc_round1_4B/checkpoint-2798"
#
#  "../llamafactory/saves/sql_8ktoken/cosql_round3_17B/checkpoint-4108"
#    "Qwen/Qwen3-4B"

#    "Qwen/Qwen3-14B"
)


# Reward function list
reward_function_list=(
  "verl/utils/reward_score/text2sql_exec.py"
#  "verl/utils/reward_score/text2sql_exact.py"
#  "verl/utils/reward_score/text2sql.py"
  # Add more reward functions here as needed
  # "/path/to/another/reward_function.py"
  # "/path/to/third/reward_function.py"
)

for DATA_FOLDER in  "data/cosql"
do
  for MODEL_PATH in "${model_list[@]}"
  do
    for REWARD_FUNCTION in "${reward_function_list[@]}"
    do
      echo "Evaluating MODEL_PATH: $MODEL_PATH on DATA_FOLDER: $DATA_FOLDER with REWARD_FUNCTION: $REWARD_FUNCTION"
      CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
      --config-path="$CONFIG_PATH" \
      --config-name='text2sql_multiturn_grpo' \
      custom_reward_function.path=$REWARD_FUNCTION  \
      algorithm.adv_estimator=grpo \
      data.train_files=$DATA_FOLDER/test.parquet \
      data.val_files=$DATA_FOLDER/test.parquet  \
      data.train_batch_size=64 \
      data.max_prompt_length=4000 \
      data.max_response_length=8000 \
      actor_rollout_ref.rollout.max_num_batched_tokens=13000 \
      data.filter_overlong_prompts=True \
      data.truncation='error' \
      data.return_raw_chat=True \
      actor_rollout_ref.rollout.multi_turn.enable=True \
      actor_rollout_ref.model.path=$MODEL_PATH \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=64 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=0.001 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=sglang \
      actor_rollout_ref.rollout.mode=async \
      actor_rollout_ref.rollout.multi_turn.enable=True \
      actor_rollout_ref.rollout.multi_turn.format=hermes \
      actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
      actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
      actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8000 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
      actor_rollout_ref.rollout.n=1 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      algorithm.use_kl_in_reward=False \
      trainer.critic_warmup=0 \
      trainer.logger=['console','wandb'] \
      trainer.project_name='verl_grpo_text2sql' \
      trainer.experiment_name="qwen3_eval_text2sqlf" \
      trainer.val_before_train=True \
      trainer.n_gpus_per_node=4 \
      trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=-1 \
      actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/text2sql_tool_config.yaml" \
      trainer.total_epochs=1 \
      data.shuffle=False \
      trainer.validation_data_dir="./base_traj/"
    done
  done
done

#      actor_rollout_ref.rollout.mode=async \
#      actor_rollout_ref.rollout.mode=async \
#      actor_rollout_ref.rollout.mode=async \