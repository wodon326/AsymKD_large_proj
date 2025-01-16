set -e
set -x

CUDA_VISIBLE_DEVICES=6 python AsymKD_evaluate_affine_inv_gpu.py \
    --model metric3d \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_scannet_val.yaml \
    --alignment least_square \
    --output_dir output/scannet
