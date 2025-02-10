set -e
set -x

CUDA_VISIBLE_DEVICES=0 python AsymKD_evaluate_affine_inv_gpu.py \
    --model depth_anything_large_v2 \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_kitti_eigen_test.yaml \
    --alignment least_square_disparity \
    --output_dir output/kitti_eigen_test
