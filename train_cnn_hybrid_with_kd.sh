python train_cnn_hybrid_with_kd.py \
    --batch_size 16 \
    --num_steps 187480 \
    --lr 0.00005 \
    --train_datasets HRWSI BlendedMVS tartan_air \
    --ckpt /home/wodon326/project/AsymKD_large_proj/best_checkpoint_depth_latent1_avg/depth_latent1_avg_best_checkpoint.pth \
    --save_dir depth_latent1_cnn_hybrid_with_kd