python train_naive_kd_more_data.py \
    --batch_size 8 \
    --num_steps 1000000 \
    --lr 0.00005 \
    --train_datasets HRWSI BlendedMVS tartan_air ImageNet_1k \
    --ckpt /home/wodon326/project/AsymKD_large_proj/best_checkpoint_depth_latent1_avg/depth_latent1_avg_best_checkpoint.pth \
    --save_dir depth_latent1_2way_cbam_trans_adapter_kd_naive_residual \
    --train_style trans