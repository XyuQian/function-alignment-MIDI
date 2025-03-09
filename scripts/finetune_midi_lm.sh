lr=1e-4
exp=exp
batch_size=8
e=30
exp_dir=midi_lm_piano_cover

python -m shoelace.midi_lm.finetune.finetune_single_gpu \
      --learning_rate=$lr \
      --experiment_folder=$exp \
      --batch_size=$batch_size \
      --epoch=$e \
      --exp_name=$exp_dir
