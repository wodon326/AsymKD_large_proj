set -e
set -x

# python AsymKD_evaluate_affine_inv_gpu_ddp.py \
#     --model depth_latent1_avg \
#     --base_data_dir ~/data/AsymKD \
#     --dataset_config config/data_kitti_eigen_test.yaml \
#     --alignment least_square_disparity \
#     --output_dir output/kitti_eigen_test \
#     --checkpoint_dir /home/wodon326/project/AsymKD_large_proj/checkpoint_depth_latent1_avg_ver

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model depth_latent1_avg \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_nyu_test.yaml \
    --alignment least_square_disparity \
    --output_dir output/nyu_test \
    --checkpoint_dir /home/wodon326/project/AsymKD_large_proj/checkpoint_depth_latent1_avg_ver

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model depth_latent1_avg \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_eth3d.yaml \
    --alignment least_square_disparity \
    --output_dir output/eth3d \
    --alignment_max_res 1024 \
    --checkpoint_dir /home/wodon326/project/AsymKD_large_proj/checkpoint_depth_latent1_avg_ver

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model depth_latent1_avg \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_diode_all.yaml \
    --alignment least_square_disparity \
    --output_dir output/diode \
    --checkpoint_dir /home/wodon326/project/AsymKD_large_proj/checkpoint_depth_latent1_avg_ver

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model depth_latent1_avg \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_scannet_val.yaml \
    --alignment least_square_disparity \
    --output_dir output/scannet \
    --checkpoint_dir /home/wodon326/project/AsymKD_large_proj/checkpoint_depth_latent1_avg_ver