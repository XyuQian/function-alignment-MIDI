lr=1e-4
exp=exp
batch_size=4
e=15
mask_type=bi_mask
exp_dir=bi_direct_medium_${mask_type}_5_tasks

CUDA_VISIBLE_DEVICES=0 python -m shoelace.actual_shoelace.bi_direct_5_tasks.train_single_gpu \
      --learning_rate=$lr \
      --experiment_folder=$exp \
      --batch_size=$batch_size \
      --epoch=$e \
      --exp_name=$exp_dir \
      --duration=15.36 \
      --mask_type=${mask_type}
