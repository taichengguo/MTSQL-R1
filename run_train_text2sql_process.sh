set -x

ulimit -n 65535
data="cosql"
#data="sparc"
DATA_FOLDER="data/$data"
#DATA_FOLDER="PATH/TO/data/gsm8k/"

# Dataset: Except Simple
# Except correct
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

#MODEL_PATH="Qwen/Qwen3-0.6B"
#MODEL_PATH="Qwen/Qwen3-1.7B"
MODEL_PATH="../llamafactory/saves/sql/cosql_round3_llama3B/checkpoint-6162"


#clip_ratio_low=0.2
#clip_ratio_high=0.28

# Prev - best model
#tag="4B"
#tag="17B"
tag="3B"
#MODEL_PATH="../llamafactory/saves/sql/${data}_round3_${tag}/checkpoint-2054"  # 4B
#MODEL_PATH="../llamafactory/saves/sql/${data}_round3_${tag}/checkpoint-4108"  #17B

# From 0 to 2
#DATA_LABEL=2
DATA_LABEL=0


#ray job submit --address="http://127.0.0.1:8265" \
#    --runtime-env=verl/trainer/runtime_env.yaml \
#    --no-wait \
#    -- \

# TODO NOTE we remove KL loss for 17B
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='text2sql_multiturn_grpo' \
    custom_reward_function.path=verl/utils/reward_score/text2sql_process.py  \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_FOLDER/train_rl${DATA_LABEL}.parquet \
    data.val_files=$DATA_FOLDER/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=4000 \
    data.max_response_length=8000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=34000 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true  \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=34000 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_text2sql' \
    trainer.experiment_name="${data}_${tag}" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.validation_data_dir="./${data}_${tag}_rollouts_sql_train/" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/text2sql_tool_config.yaml" \
    trainer.total_epochs=60  > train_sql.log 2>&1  $@


#> train_sql.log 2>&1
# For process reward, record the reward by adding the rollout dir
#trainer.rollout_data_dir="${data}_${DATA_LABEL}_${tag}_rollouts_realtrain/"