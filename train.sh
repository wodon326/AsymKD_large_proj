CUDA_VISIBLE_DEVICES=3,4 python train.py \
    --batch_size 16 \
    --num_steps 187480 \
    --lr 0.00005 \
    --train_datasets HRWSI BlendedMVS tartan_air