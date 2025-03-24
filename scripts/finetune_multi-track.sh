lr=1e-4
exp=exp
batch_size=4
e=15
n_prompts=10
task_type=multi-track
model_type=$1
exp_dir=music_jam_${model_type}_${n_prompts}_${task_type}

python -m shoelace.slakh2100_training.train_single_gpu \
      --learning_rate=$lr \
      --experiment_folder=$exp \
      --batch_size=$batch_size \
      --epoch=$e \
      --exp_name=$exp_dir \
      --duration=10.24 \
      --task_type=${task_type} \
      --n_prompts=${n_prompts} \
      --model_type=${model_type}