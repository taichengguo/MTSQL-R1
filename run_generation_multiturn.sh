set -x

ulimit -n 65535
data="cosql"
#data="cosql"
#data="sparc"
c_round=2

DATA_FOLDER="data/$data"
#DATA_FOLDER="data/gsm8k_our_test/"

# Dataset: Except Simple
# Except correct
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

#MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"

#MODEL_PATH="Qwen/Qwen3-0.6B"
#MODEL_PATH="Qwen/Qwen3-1.7B"
#MODEL_PATH="Qwen/Qwen3-4B"
#MODEL_PATH="Qwen/Qwen3-8B"
#MODEL_PATH="Qwen/Qwen3-14B"


# 1st round model for cosql
if [ "$data" = "cosql" ]; then
  MODEL_PATH="../llamafactory/saves/sql/cosql_round1_4B/checkpoint-1756"
#  MODEL_PATH="../llamafactory/saves/sql/cosql_round2/checkpoint-2000"

  # 4B-final
#  MODEL_PATH="../llamafactory/saves/sql/${data}_round3_4B/checkpoint-2054"

  # 1.7B-final
#  MODEL_PATH="../llamafactory/saves/sql/${data}_round3_17B/checkpoint-4108"

elif [ "$data" = "sparc" ]; then
  MODEL_PATH="../llamafactory/saves/sql/sparc_round1_4B/checkpoint-2798"
else
  echo "Unknown dataset: $data"
  exit 1
fi


# if cround >= 2, then parquet_data = 1
parquet_data=$DATA_FOLDER/train_sft.parquet
if [ "$c_round" -ge 2 ]; then
  # assign DATA_INDEX to c_round-1
  DATA_INDEX=$((c_round - 1))
  parquet_data="../llamafactory/data/${data}still_error_add${DATA_INDEX}.parquet"
  echo "Using parquet_data: $parquet_data"
fi

resp_length=4000

# if c_round == 4, still use $DATA_FOLDER/train_sft.parquet
if [ "$c_round" -eq 4 ]; then
  parquet_data="$DATA_FOLDER/train_filtered_sft.parquet"
  echo "Using parquet_data: $parquet_data"
  # change c_round to 4_4B
  c_round="4_4B_filter"
fi

#MODEL_PATH="../llamafact/saves/sql/tool_longcot_score2_round3_hybrid/checkpoint-1713"

# Multi-turns multi-tools call
# 14B -> data;   1.7B RL
# Multiple times: Duplicate the training dataset
# 50 times
# set tempature to high
#for i in {1..100}
# TODO: 1. add two rewards sum  2. add temperature 3. train_sft data
#for i in {20..1..-1}
#for i in 8 9
for i in {1..20}
do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name='text2sql_multiturn_grpo' \
  custom_reward_function.path=verl/utils/reward_score/text2sql.py  \
  algorithm.adv_estimator=grpo \
  data.train_files=${parquet_data} \
  data.val_files=${parquet_data}  \
  data.train_batch_size=64 \
  data.max_prompt_length=4000 \
  data.max_response_length=${resp_length} \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  actor_rollout_ref.rollout.mode=async \
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
  trainer.total_epochs=0 \
  data.shuffle=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  trainer.validation_data_dir="./${data}_warmstart/round${c_round}/$i/"
  sleep 1m
done