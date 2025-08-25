CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch \
--nproc_per_node=6 \
--master_port 12324 \
downstream_phase/run_phase_training.py \
--suffix "baseline" \
--batch_size 8 \
--epochs 50 \
--save_ckpt_freq 10 \
--model surgformer_HTA \
--pretrained_path pretrain_params/timesformer_base_patch16_224_K400.pyth \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path "/nfs/scratch/xjiangbh/video_tokenpruning/dataset/AutoLaparo_Task1/" \
--eval_data_path "/nfs/scratch/xjiangbh/video_tokenpruning/dataset/AutoLaparo_Task1/" \
--nb_classes 7 \
--data_strategy online \
--output_mode key_frame \
--num_frames 16 \
--sampling_rate 4 \
--data_set AutoLaparo \
--data_fps 1fps \
--output_dir "results/AutoLaparo/" \
--log_dir "results/AutoLaparo/" \
--num_workers 10 \
--dist_eval \
--no_auto_resume \
# --enable_deepspeed \

