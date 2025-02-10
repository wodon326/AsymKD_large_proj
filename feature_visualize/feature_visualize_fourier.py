import cv2
import torch
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

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

def visualize_feature_map(feature_map, save_dir, base_filename, title="Feature Map"):
    """
    Visualize and save each channel of the feature map separately.
    feature_map shape: (B, C, H, W)
    """
    # (B, C, H, W) -> (C, H, W) & 채널 평균
    avg_feature_map = feature_map.squeeze(0).mean(dim=0).cpu().detach().numpy()
    feature_map_np = feature_map.squeeze(0).cpu().detach().numpy()  # (C, H, W)
    num_channels = feature_map_np.shape[0]
    
    channel_save_dir = os.path.join(save_dir, title, base_filename)
    os.makedirs(channel_save_dir, exist_ok=True)
    
    # 채널 평균 시각화
    plt.figure()
    plt.imshow(avg_feature_map, cmap='viridis')
    plt.axis("off")
    plt.title(f"{title} - Channel avg")
    plt.savefig(os.path.join(channel_save_dir, f"{base_filename}_avg.png"))
    plt.close()

    # 개별 채널 시각화
    for i in range(num_channels):
        plt.figure()
        plt.imshow(feature_map_np[i], cmap='viridis')
        plt.axis("off")
        plt.title(f"{title} - Channel {i}")
        plt.savefig(os.path.join(channel_save_dir, f"{base_filename}_channel_{i}.png"))
        plt.close()

def visualize_feature_map_fourier_sub_ver(student_feature_map,compress_feature_map, save_dir, base_filename, title="Feature Map Fourier Spectrum"):
    """
    주어진 feature map에 대해 2D Fourier Transform(푸리에 스펙트럼)을 
    채널 평균과 개별 채널 각각 시각화하여 저장합니다.
    feature_map shape: (B, C, H, W)
    """
    # (B, C, H, W) -> (C, H, W)
    feature_map = compress_feature_map - student_feature_map
    feature_map = feature_map.squeeze(0)  # (C, H, W)
    channel_save_dir = os.path.join(save_dir, title, base_filename)
    os.makedirs(channel_save_dir, exist_ok=True)
    
    # 1) 채널 평균에 대한 Fourier
    avg_feature_map = feature_map.mean(dim=0)  # (H, W)
    fft_result = torch.fft.fftn(avg_feature_map, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft_result)
    magnitude_spectrum = torch.abs(fft_shifted)
    log_magnitude_spectrum = torch.log(magnitude_spectrum + 1e-8)  # log scale

    plt.figure()
    plt.imshow(log_magnitude_spectrum.cpu().detach().numpy(), cmap='viridis')
    plt.axis("off")
    plt.title(f"{title} - Channel sub_ver avg")
    plt.savefig(os.path.join(channel_save_dir, f"{base_filename}_fft_avg_sub_ver.png"))
    plt.close()

def visualize_feature_map_fourier(feature_map, save_dir, base_filename, title="Feature Map Fourier Spectrum"):
    """
    주어진 feature map에 대해 2D Fourier Transform(푸리에 스펙트럼)을 
    채널 평균과 개별 채널 각각 시각화하여 저장합니다.
    feature_map shape: (B, C, H, W)
    """
    # (B, C, H, W) -> (C, H, W)
    feature_map = feature_map.squeeze(0)  # (C, H, W)
    channel_save_dir = os.path.join(save_dir, title, base_filename)
    os.makedirs(channel_save_dir, exist_ok=True)
    
    # 1) 채널 평균에 대한 Fourier
    avg_feature_map = feature_map.mean(dim=0)  # (H, W)
    fft_result = torch.fft.fftn(avg_feature_map, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft_result)
    magnitude_spectrum = torch.abs(fft_shifted)
    log_magnitude_spectrum = torch.log(magnitude_spectrum + 1e-8)  # log scale

    plt.figure()
    plt.imshow(log_magnitude_spectrum.cpu().detach().numpy(), cmap='viridis')
    plt.axis("off")
    plt.title(f"{title} - Channel avg")
    plt.savefig(os.path.join(channel_save_dir, f"{base_filename}_fft_avg.png"))
    plt.close()

    # 2) 개별 채널에 대한 Fourier
    feature_map_np = feature_map.cpu().detach().numpy()  # (C, H, W)
    num_channels = feature_map_np.shape[0]
    for i in range(num_channels):
        channel_data = torch.from_numpy(feature_map_np[i])
        fft_result = torch.fft.fftn(channel_data, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft_result)
        magnitude_spectrum = torch.abs(fft_shifted)
        log_magnitude_spectrum = torch.log(magnitude_spectrum + 1e-8)

        plt.figure()
        plt.imshow(log_magnitude_spectrum.cpu().detach().numpy(), cmap='viridis')
        plt.axis("off")
        plt.title(f"{title} - Channel {i}")
        plt.savefig(os.path.join(channel_save_dir, f"{base_filename}_fft_channel_{i}.png"))
        plt.close()

def process_images_in_folder(folder_path, model, save_dir):
    """
    Process all images in the given folder, passing them through the model and visualizing features.
    """
    image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(('png', 'jpg', 'jpeg'))]
    os.makedirs(save_dir, exist_ok=True)
    
    for image_path in image_paths:
        img_tensor = preprocess_image(image_path)
        with torch.no_grad():
            # model.feature_visualize -> (student_intermediate_feature, compress_feat) 반환
            student_intermediate_feature, compress_feat = model.feature_visualize(img_tensor)
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # 1) 원본 Feature Map 시각화
        # visualize_feature_map(student_intermediate_feature, save_dir, base_filename, "Student Feature Map")
        # visualize_feature_map(compress_feat, save_dir, base_filename, "Compressed Feature Map")
        
        # 2) Fourier Spectrum 시각화
        visualize_feature_map_fourier_sub_ver(student_intermediate_feature, compress_feat, save_dir, base_filename, "Sub Feature Map Fourier")
        # visualize_feature_map_fourier(student_intermediate_feature, save_dir, base_filename, "Student Feature Map Fourier")
        # visualize_feature_map_fourier(compress_feat, save_dir, base_filename, "Compressed Feature Map Fourier")

#### model load ####
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
image_folder = "/home/wodon326/project/AsymKD_large_proj/feature_visualize/image_folder/temp"
save_dir = "./feature_visualize/feature_visualize_Fourier_output"
process_images_in_folder(image_folder, model, save_dir)
