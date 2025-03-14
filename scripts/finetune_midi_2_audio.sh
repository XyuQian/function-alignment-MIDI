lr=1e-4
exp=exp
batch_size=16
e=50
w=1
exp_dir=midi_2_audio_small

CUDA_VISIBLE_DEVICES=0 python -m shoelace.actual_shoelace.midi_2_audio.finetune_single_gpu \
      --learning_rate=$lr \
      --experiment_folder=$exp \
      --batch_size=$batch_size \
      --epoch=$e \
      --exp_name=$exp_dir
