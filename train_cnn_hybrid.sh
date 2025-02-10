CUDA_VISIBLE_DEVICES=4,5,6,7 python train_cnn_hybrid.py \
    --batch_size 16 \
    --num_steps 187480 \
    --lr 0.00005 \
    --train_datasets HRWSI BlendedMVS tartan_air \
    --save_dir depth_latent1_cnn_hybrid_only_gt