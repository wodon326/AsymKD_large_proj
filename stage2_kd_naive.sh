CUDA_VISIBLE_DEVICES=0,1,2,3 python stage2_kd_naive.py \
    --batch_size 16 \
    --num_steps 187480 \
    --lr 0.00005 \
    --train_datasets HRWSI BlendedMVS tartan_air \
    --save_dir checkpoint_kd_naive\
    --ckpt /home/wodon326/project/AsymKD_large_proj/best_checkpoint_depth_latent1_avg/depth_latent1_avg_best_checkpoint.pth \
    --restore_ckpt /home/wodon326/project/AsymKD_large_proj/checkpoint_kd_naive/8250_AsymKD_new_loss.pth