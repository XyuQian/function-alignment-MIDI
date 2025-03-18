lr=1e-4
exp=exp
batch_size=4
e=20
mask_type=random
exp_dir=midi_2_audio_medium_${mask_type}

CUDA_VISIBLE_DEVICES=0 python -m shoelace.actual_shoelace.midi_2_audio.train_single_gpu \
      --learning_rate=$lr \
      --experiment_folder=$exp \
      --batch_size=$batch_size \
      --epoch=$e \
      --exp_name=$exp_dir \
      --duration=15.36 \
      --mask_type=${mask_type}
