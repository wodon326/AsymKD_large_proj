set -e
set -x

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model kd_latent1_avg \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_kitti_eigen_test.yaml \
    --alignment least_square_disparity \
    --output_dir output/kitti_eigen_test \
    --checkpoint_dir /home/dgist/project/AsymKD_large_proj/checkpoint_depth_latent1_avg_ver
