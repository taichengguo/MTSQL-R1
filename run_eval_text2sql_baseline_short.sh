set -x

ulimit -n 65535
DATA_FOLDER="data/text2sql"
#DATA_FOLDER="data/gsm8k_our_test/"

# Dataset: Except Simple
# Except correct
PROJECT_DIR="$(pwd)"
#MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"

#MODEL_PATH="Qwen/Qwen3-0.6B"
#MODEL_PATH="Qwen/Qwen3-1.7B"
MODEL_PATH="Qwen/Qwen3-4B"
#MODEL_PATH="Qwen/Qwen3-8B"
#MODEL_PATH="Qwen/Qwen3-14B"
#MODEL_PATH="Qwen/Qwen3-32B"



#MODEL_PATH="./text2sql_ckpt"
#"../llamafactory/saves/baseline_cosql/checkpoint-3480"
#"../llamafactory/saves/baseline_cosql/checkpoint-3480"

#"../llamafactory/saves/baseline_sparc/checkpoint-6032"

#"data/text2sql"  "data/sparc"

# For model_path in 14b and 1.7b

for MODEL_PATH in   "Qwen/Qwen3-8B"
do
  for DATA_FOLDER in  "data/cosql"  "data/sparc"
  do
    echo "Evaluating MODEL_PATH: $MODEL_PATH on DATA_FOLDER: $DATA_FOLDER"
    # Multi-turns multi-tools call
    # 14B -> data;   1.7B RL
    # Multiple times: Duplicate the training dataset
    # 50 times
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=verl/utils/reward_score/text2sql.py  \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_FOLDER/enforce_sql_test.parquet \
    data.val_files=$DATA_FOLDER/enforce_sql_test.parquet  \
    data.train_batch_size=64 \
    data.max_prompt_length=4000 \
    data.max_response_length=1000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
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
    trainer.experiment_name="qwen3_eval_text2sqlf_short" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=0 \
    data.shuffle=False \
    trainer.validation_data_dir="./baseline_rollout/"
    #> train_sql.log 2>&1
    #data.val_batch_size=10 \
    #    data.val_batch_size=50 \
  done
done