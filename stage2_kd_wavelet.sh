python stage2_kd_wavelet.py \
    --batch_size 8 \
    --num_steps 187480 \
    --lr 0.00005 \
    --train_datasets HRWSI BlendedMVS tartan_air \
    --save_dir checkpoint_kd_naive\
    --ckpt /home/dgist/project/AsymKD_large_proj/best_checkpoint_depth_latent1_avg/depth_latent1_avg_best_checkpoint.pth