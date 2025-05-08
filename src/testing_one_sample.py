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
from modules.transforms import DiffusionTransform
from common.logging import get_logger
from argument import get_argparser, sync_config
from ema_pytorch import EMA

warnings.filterwarnings("ignore")

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
def preprocess_image(image_path, input_size, device):
    img = Image.open(image_path).convert('RGB')
    transform = DiffusionTransform((input_size, input_size)).get_forward_transform_img()
    img_tensor = transform(img).unsqueeze(0).to(device)
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    return img_tensor

input_tensor = preprocess_image(
    image_path=args.image_path,
    input_size=config["dataset"]["input_size"],
    device=device
)

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
if config["testing"]["model_weigths"]["overload"]:
    best_model_path = config["testing"]["model_weigths"]["file_path"]
else:
    best_model_path = get_model_path(name=ID, dir=config["model"]["save_dir"])

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
ensemble = config["testing"]["ensemble"]
forward_schedule = ForwardSchedule(**config["diffusion"]["schedule"])

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
def write_imgs(imgs, prds, mid_prds, step, id, dataset):
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

    writer.add_image(f"{dataset}/Test:{id}", img_msk_prd_grid, step)

write_imgs(
    input_tensor,
    preds,
    mid_preds,
    step=0,
    id=f"{ID}_E{ensemble}",
    dataset=config["dataset"]["name"].upper()
)

if config["testing"]["result_imgs"]["save"]:
    save_sampling_results_as_imgs(
        input_tensor,
        ["single_input"],
        preds,
        all_samples_list,
        middle_steps_of_sampling=8,
        save_dir=config["testing"]["result_imgs"]["dir"],
        dataset_name=config["dataset"]["name"].upper(),
        result_id=f"{ID}_E{ensemble}",
        img_ext="png",
        save_mat=True,
    )

logger.info("✅ Done testing single image.")
