python train.py \
    --batch_size 16 \
    --num_steps 187480 \
    --lr 0.00005 \
    --train_datasets HRWSI BlendedMVS tartan_air \
    --save_dir depth_latent1_cls_token_ver2 \
    --restore_ckpt /home/wodon326/project/AsymKD_large_proj_cls_token/best_checkpoint_depth_latent1_avg/depth_latent1_avg_best_checkpoint.pth