lr=1e-4
exp=exp
batch_size=4
e=30
w=1
exp_dir=midi_2_audio

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m shoelace.shoelace.midi_2_audio.finetune \
      --learning_rate=$lr \
      --experiment_folder=$exp \
      --batch_size=$batch_size \
      --epoch=$e \
      --world_size=$w \
      --exp_name=$exp_dir
