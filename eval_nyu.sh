set -e
set -x

CUDA_VISIBLE_DEVICES=7 python AsymKD_evaluate_affine_inv_gpu.py \
    --model metric3d \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_nyu_test.yaml \
    --alignment least_square \
    --output_dir output/nyu_test
