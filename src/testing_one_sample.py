# testing_one_sample.py
import torch
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import json
import sys
import os

# Import các module từ code gốc
from models import *
from utils.helper_funcs import (
    load_config,
    save_sampling_results_as_imgs,
    get_model_path,
    get_conf_name,
    print_config,
    draw_boundary,
    mean_of_list_of_tensors,
)
from forward.forward_schedules import ForwardSchedule
from reverse.reverse_process import sample
from modules.transforms import DiffusionTransform
from common.logging import get_logger
from argument import get_argparser, sync_config

# Tắt cảnh báo
import warnings
warnings.filterwarnings("ignore")

def preprocess_image(image_path, input_size, device):
    """Tiền xử lý ảnh giống hệt pipeline training"""
    # Load ảnh
    img = Image.open(image_path).convert('RGB')
    
    # Áp dụng transform giống dataset
    DT = DiffusionTransform((input_size, input_size))
    transform = DT.get_forward_transform_img()
    
    # Thêm batch dimension và chuyển sang device
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Chuẩn hóa
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    
    return img_tensor

def main(args):
    # Load config
    config = load_config(args.config_file)
    config = sync_config(config, args)
    
    # Khởi tạo logger
    logger = get_logger(
        filename=f"{config['model']['name']}_single_test", 
        dir=f"logs/{config['dataset']['name']}"
    )
    print_config(config, logger)
    
    # Thiết lập device
    device = torch.device(config["run"]["device"])
    logger.info(f"Device is <{device}>")
    
    # Khởi tạo diffusion schedule
    forward_schedule = ForwardSchedule(**config["diffusion"]["schedule"])
    
    # Khởi tạo model
    Net = globals()[config["model"]["class"]]
    model = Net(**config["model"]["params"]).to(device)
    
    # Load weights
    if config["testing"]["model_weigths"]["overload"]: 
        best_model_path = config["testing"]["model_weigths"]["file_path"]
    else:
        best_model_path = get_model_path(name=get_conf_name(config), dir=config["model"]["save_dir"])
    
    logger.info(f"Loading model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location="cpu")
    
    # Xử lý EMA
    if config["training"]["ema"]["use"]:
        from ema_pytorch import EMA
        ema = EMA(model=model, **config["training"]["ema"]["params"])
        ema.load_state_dict(checkpoint["ema"])
        model = ema.ema_model
        logger.info("Loaded EMA model")
    else:
        model.load_state_dict(checkpoint["model"])
        logger.info("Loaded vanilla model")
    
    model.eval()
    
    # Tiền xử lý ảnh đầu vào
    input_tensor = preprocess_image(
        image_path=args.image_path,
        input_size=config["dataset"]["input_size"],
        device=device
    )
    
    # Inference
    ensemble = config["testing"]["ensemble"]
    timesteps = config["diffusion"]["schedule"]["timesteps"]
    
    samples_list, mid_samples_list = [], []
    all_samples_list = []
    
    with torch.no_grad():
        for en in range(ensemble):
            samples = sample(
                forward_schedule,
                model,
                images=input_tensor,
                out_channels=1,
                desc=f"Ensemble {en+1}/{ensemble}",
            )
            samples_list.append(samples[-1][:, :1, :, :])
            mid_samples_list.append(samples[-int(0.1 * timesteps)][:, :1, :, :])
            all_samples_list.append([s[:, :1, :, :] for s in samples])
    
    # Tổng hợp kết quả
    preds = mean_of_list_of_tensors(samples_list)
    mid_preds = mean_of_list_of_tensors(mid_samples_list)
    
    # Lưu kết quả
    if args.save_dir:
        save_sampling_results_as_imgs(
            input_tensor,
            [Path(args.image_path).stem],  # ID
            preds,
            all_samples_list,
            middle_steps_of_sampling=8,
            save_dir=args.save_dir,
            dataset_name=config["dataset"]["name"].upper(),
            result_id="single_infer",
            img_ext="png",
            save_mat=True,
        )
        logger.info(f"Results saved to {args.save_dir}")
    
    # Hiển thị kết quả
    if args.show:
        plt.figure(figsize=(12, 4))
        
        # Ảnh gốc
        plt.subplot(131)
        plt.imshow(input_tensor[0].cpu().permute(1,2,0))
        plt.title("Input Image")
        plt.axis('off')
        
        # Prediction
        plt.subplot(132)
        plt.imshow(preds[0,0].cpu(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
        
        # Overlay
        plt.subplot(133)
        plt.imshow(input_tensor[0].cpu().permute(1,2,0))
        plt.imshow(preds[0,0].cpu(), alpha=0.5, cmap='jet')
        plt.title("Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, required=True, 
                       help="Path to config file")
    parser.add_argument("-i", "--image_path", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("-s", "--save_dir", type=str, default=None,
                       help="Directory to save results")
    parser.add_argument("--show", action="store_true",
                       help="Show results with matplotlib")
    
    args = parser.parse_args()
    
    main(args)