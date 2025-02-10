import cv2
import torch
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr

def resize_to_nearest_divisible(img, divisor=14):
    # Get the original size of the image
    original_width, original_height = img.size
    
    # Calculate the new size that is divisible by the divisor
    new_width = (original_width // divisor) * divisor
    new_height = (original_height // divisor) * divisor
    
    # Apply the resize transform
    resize_transform = transforms.Resize((new_height, new_width))
    resized_img = resize_transform(img)
    
    return resized_img

def preprocess_image(image_path, divisor=14):
    """
    Load and preprocess an image from the given path.
    """
    img = Image.open(image_path).convert("RGB")
    img = resize_to_nearest_divisible(img, divisor)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor.to(device)

def process_images_in_folder(folder_path, model, save_dir):
    """
    Process all images in the given folder, passing them through the model and visualizing features.
    """
    image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(('png', 'jpg', 'jpeg'))]
    os.makedirs(save_dir, exist_ok=True)
    
    for image_path in image_paths:
        img_tensor = preprocess_image(image_path)
        with torch.no_grad():
            student_intermediate_feature, compress_feat = model.feature_visualize(img_tensor, reshape_to_image = False)
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        os.makedirs(os.path.join(save_dir,base_filename), exist_ok=True)

        compress_feat = compress_feat.squeeze(0).cpu().detach().numpy()
        student_intermediate_feature = student_intermediate_feature.squeeze(0).cpu().detach().numpy()

        num_channels = student_intermediate_feature.shape[0]
        # 1. 유사성 분석
        cosine_similarities = [1 - cosine(student_intermediate_feature[i], compress_feat[i]) for i in range(num_channels)]

        plt.figure(figsize=(6, 5))
        plt.plot(cosine_similarities, label='Channel wise cosine_similarities', alpha=0.7)
        plt.legend()
        plt.title("Channel-wise cosine_similarities")
        plt.xlabel("Channel Index")
        plt.ylabel("Mean Value")
        plt.savefig(os.path.join(save_dir,base_filename,"Channel_wise_cosine_similarities.png"))


        # 7. 채널별 MSE 시각화
        mse_per_channel = np.mean((student_intermediate_feature - compress_feat) ** 2, axis=0)
        euclidean_distances = mse_per_channel
        plt.figure(figsize=(12, 5))
        plt.plot(mse_per_channel, label='Channel-wise MSE', alpha=0.7)
        plt.legend()
        plt.title("Channel-wise Mean Squared Error (MSE)")
        plt.xlabel("Channel Index")
        plt.ylabel("MSE Value")
        plt.savefig(os.path.join(save_dir,base_filename,"Channel_wise_mse.png"))

        # 2. 통계 분석
        mean_student, std_student = np.mean(student_intermediate_feature), np.std(student_intermediate_feature)
        mean_compress, std_compress = np.mean(compress_feat), np.std(compress_feat)


        # 6. 채널별 평균 및 분산 시각화
        channel_means_student = np.mean(student_intermediate_feature, axis=0)
        channel_stds_student = np.std(student_intermediate_feature, axis=0)
        channel_means_compress = np.mean(compress_feat, axis=0)
        channel_stds_compress = np.std(compress_feat, axis=0)

        plt.figure(figsize=(12, 5))
        plt.plot(channel_means_student, label='student_feature Mean', alpha=0.7)
        plt.plot(channel_means_compress, label='compress_feat Mean', alpha=0.7)
        plt.legend()
        plt.title("Channel-wise Mean")
        plt.xlabel("Channel Index")
        plt.ylabel("Mean Value")
        plt.savefig(os.path.join(save_dir,base_filename,"Channel_wise_Mean.png"))

        plt.figure(figsize=(12, 5))
        plt.plot(channel_stds_student, label='student_feature Std Dev', alpha=0.7)
        plt.plot(channel_stds_compress, label='compress_feat Std Dev', alpha=0.7)
        plt.legend()
        plt.title("Channel-wise Standard Deviation")
        plt.xlabel("Channel Index")
        plt.ylabel("Standard Deviation")
        plt.savefig(os.path.join(save_dir,base_filename,"Channel_wise_Std.png"))

        # 3. 상관관계 분석
        pearson_corr, _ = pearsonr(student_intermediate_feature.flatten(), compress_feat.flatten())
        spearman_corr, _ = spearmanr(student_intermediate_feature.flatten(), compress_feat.flatten())

        # 4. PCA 차원 축소 후 시각화
        pca = PCA(n_components=2)
        pca_features_student = pca.fit_transform(student_intermediate_feature)
        pca_features_compress = pca.transform(compress_feat)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_features_student[:, 0], pca_features_student[:, 1], label='student_feature', alpha=0.6)
        plt.scatter(pca_features_compress[:, 0], pca_features_compress[:, 1], label='compress_feat', alpha=0.6)
        plt.legend()
        plt.title("PCA Visualization of Features")
        plt.savefig(os.path.join(save_dir,base_filename,"PCA.png"))

        # 5. t-SNE 시각화
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_features = tsne.fit_transform(np.vstack((student_intermediate_feature, compress_feat)))

        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_features[:num_channels, 0], tsne_features[:num_channels, 1], label='student_feature', alpha=0.6)
        plt.scatter(tsne_features[num_channels:, 0], tsne_features[num_channels:, 1], label='compress_feat', alpha=0.6)
        plt.legend()
        plt.title("t-SNE Visualization of Features")
        plt.savefig(os.path.join(save_dir,base_filename,"t-SNE.png"))

        # 6. 분석 결과 출력
        
        print_str = ''
        print_str += f"Cosine Similarity (mean): {np.mean(cosine_similarities):.4f}\n"
        print_str += f"Euclidean Distance (mean): {np.mean(euclidean_distances):.4f}\n"
        print_str += f"student_intermediate_feature - Mean: {mean_student:.4f}, Std: {std_student:.4f}\n"
        print_str += f"compress_feat - Mean: {mean_compress:.4f}, Std: {std_compress:.4f}\n"
        print_str += f"Pearson Correlation: {pearson_corr:.4f}\n"
        print_str += f"Spearman Correlation: {spearman_corr:.4f}"
        metrics_filename = f"{base_filename}.txt"

        _save_to = os.path.join(save_dir,base_filename, metrics_filename)
        with open(_save_to, "a") as f:
            f.write(f'{print_str}\n')



####model load#####
import sys
import os

# 현재 스크립트의 위치에서 상위 디렉토리 추가
CODE_SPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_SPACE)
os.chdir(CODE_SPACE)
from AsymKD.dpt_latent1_avg_ver import AsymKD_compress_latent1_avg_ver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AsymKD_compress_latent1_avg_ver().to(device)
restore_ckpt = '/home/wodon326/project/AsymKD_large_proj/best_checkpoint_depth_latent1_avg/depth_latent1_avg_best_checkpoint.pth'
if restore_ckpt is not None:
    checkpoint = torch.load(restore_ckpt, map_location=device)
    model_state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key = k.replace('module.', '')
        if new_key in model_state_dict:
            new_state_dict[new_key] = v

    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

# Example Usage
image_folder = "./feature_visualize/image_folder"
save_dir = "./feature_visualize/feature_analysis_output"
process_images_in_folder(image_folder, model, save_dir)