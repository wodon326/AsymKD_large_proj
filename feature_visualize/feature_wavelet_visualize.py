import cv2
import torch
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

from pytorch_wavelets import DWTForward, DWTInverse

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

def wavelet_transform(feature_map):
        
        xfm = DWTForward(J=3, mode='zero',wave='haar').to(device)
        # self.ixfm = DWTInverse(mode='zero',wave='haar')
        # Split
        cA, (cH, cV, cD) = xfm(feature_map)
        # print(f'cA.shape : {cA.shape}')
        # print(f'cH.shape : {cH.shape}')
        # print(f'cV.shape : {cV.shape}')
        # print(f'cD.shape : {cD.shape}')
        # # quit()
        # predict the masks
        slice_cH = cH.sum(2) # [N, C, H, W]
        slice_cV = cV.sum(2) # [N, C, H, W]
        slice_cD = cD.sum(2) # [N, C, H, W]
        return [slice_cH, slice_cV, slice_cD], ['slice_cH', 'slice_cV', 'slice_cD']



def visualize_wavelet_feature_map(feature_map, save_dir, base_filename, title="Wavelet_Feature Map"):
    """
    Visualize and save each channel of the feature map separately.
    """
    wavelet_feature_map_arr, name_arr = wavelet_transform(feature_map)

    for wavelet_feature_map, name in zip(wavelet_feature_map_arr,name_arr):
        avg_wavelet_feature_map = wavelet_feature_map.squeeze(0).mean(dim=0).cpu().detach().numpy()
        wavelet_feature_map = wavelet_feature_map.squeeze(0).cpu().detach().numpy()  # Remove batch dim
        num_channels = wavelet_feature_map.shape[0]
        
        channel_save_dir = os.path.join(save_dir,title, base_filename)
        os.makedirs(channel_save_dir, exist_ok=True)
        
        plt.figure()
        plt.imshow(avg_wavelet_feature_map, cmap='viridis')
        plt.axis("off")
        plt.title(f"{name+title} - Channel avg")
        plt.savefig(os.path.join(channel_save_dir, f"{name+base_filename}_avg.png"))
        plt.close()

        for i in range(num_channels):
            plt.figure()
            plt.imshow(wavelet_feature_map[i], cmap='viridis')
            plt.axis("off")
            plt.title(f"{title} - Channel {i}")
            plt.savefig(os.path.join(channel_save_dir, f"{name+base_filename}_channel_{i}.png"))
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
            print(f'img_tensor {img_tensor.shape}')
            student_intermediate_feature, compress_feat = model.feature_visualize(img_tensor)
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        visualize_wavelet_feature_map(student_intermediate_feature, save_dir, base_filename, "Student Feature Map")
        visualize_wavelet_feature_map(compress_feat, save_dir, base_filename, "Compressed Feature Map")

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
save_dir = "./feature_visualize/wavelet_feature_visualize_output"
process_images_in_folder(image_folder, model, save_dir)