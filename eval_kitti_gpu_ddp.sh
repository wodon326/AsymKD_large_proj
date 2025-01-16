set -e
set -x

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model bfm_compress \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_kitti_eigen_test.yaml \
    --alignment least_square_disparity \
    --output_dir output/kitti_eigen_test \
    --checkpoint_dir /home/dgist/project/AsymKD_VIT_Adapter_large_proj/checkpoints_new_loss_001_smooth
