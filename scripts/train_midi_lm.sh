lr=8e-5
exp=exp
batch_size=12
e=2
w=4
exp_dir=midi_lm_continue

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m shoelace.midi_lm.train \
      --learning_rate=$lr \
      --experiment_folder=$exp \
      --batch_size=$batch_size \
      --epoch=$e \
      --world_size=$w \
      --exp_name=$exp_dir
