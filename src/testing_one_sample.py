import sys
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import shutil
import warnings
from torch.utils.tensorboard import SummaryWriter
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
from torchvision import transforms
from modules.transforms import DiffusionTransform
from common.logging import get_logger
from argument import get_argparser, sync_config
from ema_pytorch import EMA
from tqdm import tqdm

warnings.filterwarnings("ignore")

from torch.utils.data import Dataset


# ------------------- Args & Config --------------------
argparser = get_argparser()
args = argparser.parse_args(sys.argv[1:])

config = load_config(args.config_file)
config = sync_config(config, args)

# Logger & Writer
logger = get_logger(
    filename=f"{config['model']['name']}_test_single", 
    dir=f"logs/{config['dataset']['name']}"
)
print_config(config, logger)

writer = SummaryWriter(f"{config['run']['writer_dir']}/{config['model']['name']}")
jet = plt.get_cmap("jet")

device = torch.device(config["run"]["device"])
logger.info(f"Device: {device}")

# ------------------ Output Dirs -----------------------
Path(config["model"]["save_dir"]).mkdir(exist_ok=True)
ID = get_conf_name(config)

if config["testing"]["result_imgs"]["save"]:
    save_dir = Path(config["testing"]["result_imgs"]["dir"]) / ID
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

# ------------------ Image Preprocessing ----------------
def preprocess_image(img_path, image_size=128):
    img_pil = Image.open(img_path).convert('RGB')
    original_size = img_pil.size
    # Bước 1: Đọc ảnh và chuyển thành numpy array, sau đó chuyển thành tensor
    img_array = np.asarray(img_pil)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()  # Chuyển sang CxHxW
    
    # Bước 2: Áp dụng các phép biến đổi (resize)
    img_transform = transforms.Compose([
        transforms.Resize(
            size=[image_size, image_size],
            interpolation=transforms.functional.InterpolationMode.BILINEAR,
        ),
    ])
    
    # Áp dụng transform cho img_tensor
    img_tensor = img_transform(img_tensor)
    
    # Bước 3: Normalize ảnh
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    
    # Bước 4: Chuyển đổi về numpy
    img_numpy = img_tensor.numpy()
    
    # Bước 5: Chuyển lại thành tensor
    img_tensor = torch.tensor(img_numpy)
    
    # Bước 6: Sử dụng DiffusionTransform
    DT = DiffusionTransform((image_size, image_size))
    img_tensor = DT.get_forward_transform_img()(img_tensor)  # Áp dụng transform từ DiffusionTransform
    
    # Bước 7: Sử dụng nan_to_num để thay thế NaN bằng giá trị 0.5
    img_tensor = torch.nan_to_num(img_tensor, nan=0.5)
    
    # Bước 8: Trả về kết quả
    return img_tensor, original_size


# ------------------ Load Model -------------------------
Net = globals()[config["model"]["class"]]
model = Net(**config["model"]["params"])
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

ema = None
if config["training"].get("ema", {}).get("use", False):
    logger.info("Using EMA")
    ema = EMA(model=model, **config["training"]["ema"]["params"])
    ema.to(device)

# Load weights
# if config["testing"]["model_weigths"]["overload"]:
#     best_model_path = config["testing"]["model_weigths"]["file_path"]
# else:
#     best_model_path = get_model_path(name=ID, dir=config["model"]["save_dir"])
best_model_path = args.best_model_path if args.best_model_path else get_model_path(name=ID, dir=config["model"]["save_dir"])

if not os.path.isfile(best_model_path):
    logger.exception(f"Model file not found: {best_model_path}")

checkpoint = torch.load(best_model_path, map_location="cpu")
if ema:
    ema.load_state_dict(checkpoint["ema"])
    model = ema.ema_model
else:
    model.load_state_dict(checkpoint["model"])
model.eval()

# ----------------- Inference ---------------------------
timesteps = config["diffusion"]["schedule"]["timesteps"]
ensemble = args.ensemble_number
forward_schedule = ForwardSchedule(**config["diffusion"]["schedule"])

image_path = args.image_path
if not os.path.isfile(image_path):
    logger.error(f"Image not found: {image_path}")
    sys.exit(1)
base_name = os.path.splitext(os.path.basename(image_path))[0]

input_tensor, original_size = preprocess_image(image_path, config["dataset"]["input_size"])
input_tensor = input_tensor.to(device).unsqueeze(0)  # Shape: [1, C, H, W]

samples_list, mid_samples_list, all_samples_list = [], [], []
for en in range(ensemble):
    samples = sample(
        forward_schedule,
        model,
        images=input_tensor,
        out_channels=1,
        desc=f"ensemble {en+1}/{ensemble}",
    )
    samples_list.append(samples[-1][:, :1, :, :].to(device))
    mid_samples_list.append(samples[-int(0.1 * timesteps)][:, :1, :, :].to(device))
    all_samples_list.append([s[:, :1, :, :] for s in samples])

preds = mean_of_list_of_tensors(samples_list)
mid_preds = mean_of_list_of_tensors(mid_samples_list)

# ----------------- Visualization -----------------------
def write_imgs(imgs, prds, mid_prds, id, dataset):
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    img_grid = torchvision.utils.make_grid(imgs)
    prd_grid = torchvision.utils.make_grid(prds)
    
    mid_prds_jet = torch.zeros_like(imgs)
    for i, mid_prd in enumerate(mid_prds.detach().cpu().numpy()):
        t = jet(mid_prd[0]).transpose(2, 0, 1)[:-1, :, :]
        t = np.log(t + 0.1)
        t = (t - t.min()) / (t.max() - t.min())
        mid_prds_jet[i, :, :, :] = torch.tensor(t)
    
    mid_prd_grid = torchvision.utils.make_grid(mid_prds_jet)
    res_grid = draw_boundary(torch.where(prd_grid > 0, 1, 0), img_grid, (0, 0, 255))
    
    img_msk_prd_grid = torch.concat(
        [
            img_grid,
            mid_prd_grid,
            torch.tensor(res_grid).to(device),
        ],
        dim=1,
    )
    
    writer.add_image(f"{dataset}/Test:{id}", img_msk_prd_grid, 0)

write_imgs(
    input_tensor,
    preds,
    mid_preds,
    id=f"{ID}_E{ensemble}_{base_name}",
    dataset=config["dataset"]["name"].upper()
)

# Save results if configured
if config["testing"]["result_imgs"]["save"]:
    orig_size = original_size
    if isinstance(orig_size, torch.Tensor):
        orig_size = tuple(orig_size.tolist())
    orig_size = orig_size[::-1]
    save_sampling_results_as_imgs(
        input_tensor,
        [base_name],
        preds,
        [s for s in all_samples_list],
        middle_steps_of_sampling=8,
        save_dir=config["testing"]["result_imgs"]["dir"],
        dataset_name=config["dataset"]["name"].upper(),
        result_id=f"{ID}_E{ensemble}",
        img_ext="png",
        save_mat=True,
        original_size=orig_size,
        base_name=base_name,
    )

logger.info(f"✅ Done processing image: {args.image_path}")
