CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
    --batch_size 8 \
    --num_steps 187480 \
    --lr 0.00005 \
    --train_datasets HRWSI BlendedMVS tartan_air