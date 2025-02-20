set -e
set -x

python AsymKD_evaluate_ddp_cache_ver.py \
    --model depth_latent1_avg_cls_token \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_kitti_eigen_test.yaml \
    --alignment least_square_disparity \
    --output_dir output/kitti_eigen_test \
    --checkpoint_dir /home/dgist/project/AsymKD_large_proj/checkpoint_depth_latent1_cls_token