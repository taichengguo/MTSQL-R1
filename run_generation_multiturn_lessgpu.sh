set -x

ulimit -n 65535
DATA_FOLDER="data/text2sql"
#DATA_FOLDER="PATH/TO/data/gsm8k/"

# Dataset: Except Simple
# Except correct
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"


#MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"

MODEL_PATH="Qwen/Qwen3-0.6B"
#MODEL_PATH="Qwen/Qwen3-1.7B"
#MODEL_PATH="Qwen/Qwen3-14B"



CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='text2sql_multiturn_grpo' \
    custom_reward_function.path=verl/utils/reward_score/text2sql.py  \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_FOLDER/train.parquet \
    data.val_files=$DATA_FOLDER/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=50 \
    data.max_prompt_length=2500 \
    data.max_response_length=3096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_text2sql' \
    trainer.experiment_name='qwen3_generation3' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/text2sql_tool_config.yaml" \
    trainer.total_epochs=0 \
    trainer.validation_data_dir="./rollouts/"


#> train_sql.log 2>&1
# Message: 'Prompt {} has length {} greater than max_prompt_len {}'
# Increase max_prompt_length
