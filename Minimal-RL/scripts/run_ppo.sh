set -x
export VLLM_ATTENTION_BACKEND=XFORMERS

data=numina_math
project_name=em-raft
algorithm=ppo
model=Qwen2.5-Math-1.5B
model_name_or_path=Qwen/$model
policy_loss=plusplus
n=1
experiment_name=${model}-${algorithm}-${policy_loss}-${data}-n${n}
GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}
train_path=./data/$data/train.parquet
test_path=./data/math500/test.parquet

train_files="['$train_path']"
test_files="['$test_path']"

mkdir -p logs/${project_name}

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_name_or_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.policy_loss=$policy_loss \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$model_name_or_path \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$my_world_size \
    trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
    trainer.val_before_train=True \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 2>&1 | tee -a logs/${project_name}/${experiment_name}.log
