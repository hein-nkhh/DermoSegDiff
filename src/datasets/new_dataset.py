import os
import glob
import numpy as np
import torch
import tifffile
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from utils.helper_funcs import (
    calc_edge,
    calc_distance_map,
    normalize
)


np_normalize = lambda x: (x-x.min())/(x.max()-x.min()) if (x.max()-x.min()) != 0 else x


class DermaVQADataset(Dataset):
    def __init__(self,
                 mode,
                 data_dir=None,
                 one_hot=True,
                 image_size=224,
                 aug=None,
                 aug_empty=None,
                 transform=None,
                 img_transform=None,
                 msk_transform=None,
                 add_boundary_mask=False,
                 add_boundary_dist=False,
                 logger=None,
                 **kwargs):
        self.print = logger.info if logger else print
        
        # pre-set variables
        self.data_dir = data_dir if data_dir else "/kaggle/input/imageclefmed-mediqa-magic-2025"
        self.images_dir = os.path.join(self.data_dir, "images_final/images_final")
        self.masks_dir = os.path.join(self.data_dir, "dermavqa-seg-trainvalid (2)/dermavqa-seg-trainvalid/dermavqa-segmentations")

        # input parameters
        self.one_hot = one_hot
        self.image_size = image_size
        self.aug = aug
        self.aug_empty = aug_empty
        self.transform = transform
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.mode = mode

        self.add_boundary_mask = add_boundary_mask
        self.add_boundary_dist = add_boundary_dist

        data_preparer = PrepareDermaVQA(
            data_dir=self.data_dir, 
            mode=self.mode,
            image_size=self.image_size, 
            logger=logger
        )
        data = data_preparer.get_data()
        
        self.imgs = torch.tensor(data["x"])
        self.msks = torch.tensor(data["y"]) if data["y"] is not None else None  # Handle case where no masks are available

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_id = idx
        img = self.imgs[idx]
    
        # Mask for non-test modes
        msk = self.msks[idx] if self.msks is not None else None  # Check if masks are available
    
        if self.one_hot and msk is not None:
            msk = (msk - msk.min()) / (msk.max() - msk.min() + 1e-8)
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)
    
        if self.aug:
            if self.mode == "train":
                img_ = np.uint8(torch.moveaxis(img*255, 0, -1).detach().numpy())
                msk_ = np.uint8(torch.moveaxis(msk*255, 0, -1).detach().numpy()) if msk is not None else None
                augmented = self.aug(image=img_, mask=msk_)
                img = torch.moveaxis(torch.tensor(augmented['image'], dtype=torch.float32), -1, 0)
                msk = torch.moveaxis(torch.tensor(augmented['mask'], dtype=torch.float32), -1, 0) if msk is not None else None
            elif self.aug_empty:
                img_ = np.uint8(torch.moveaxis(img*255, 0, -1).detach().numpy())
                msk_ = np.uint8(torch.moveaxis(msk*255, 0, -1).detach().numpy()) if msk is not None else None
                augmented = self.aug_empty(image=img_, mask=msk_)
                img = torch.moveaxis(torch.tensor(augmented['image'], dtype=torch.float32), -1, 0)
                msk = torch.moveaxis(torch.tensor(augmented['mask'], dtype=torch.float32), -1, 0) if msk is not None else None
            img = img.nan_to_num(127)
            img = normalize(img)
            msk = msk.nan_to_num(0) if msk is not None else None
            msk = normalize(msk) if msk is not None else None
    
        if self.add_boundary_mask or self.add_boundary_dist:
            msk_ = np.uint8(torch.moveaxis(msk*255, 0, -1).detach().numpy()) if msk is not None else None
                
        if self.add_boundary_mask and msk is not None:
            boundary_mask = calc_edge(msk_, mode='canny')
            msk = torch.concatenate([msk, torch.tensor(boundary_mask).unsqueeze(0)], dim=0)
    
        if self.add_boundary_dist and msk is not None:
            boundary_mask = boundary_mask if self.add_boundary_mask else calc_edge(msk_, mode='canny')
            distance_map = calc_distance_map(boundary_mask, mode='l2')
            distance_map = distance_map/(self.image_size*1.4142)
            distance_map = np.clip(distance_map, a_min=0, a_max=0.2)
            distance_map = (1-np_normalize(distance_map))*255 
            msk = torch.concatenate([msk, torch.tensor(distance_map).unsqueeze(0)], dim=0)
    
        if self.img_transform:
            img = self.img_transform(img)
        if self.msk_transform and msk is not None:
            msk = self.msk_transform(msk)
    
        img = img.nan_to_num(0.5)
        msk = msk.nan_to_num(-1) if msk is not None else None
        
        sample = {"image": img, "mask": msk, "id": data_id} if msk is not None else {"image": img, "id": data_id}
        return sample


class PrepareDermaVQA:
    def __init__(self, data_dir, mode, image_size, logger=None):
        self.print = logger.info if logger else print
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.mode = mode
        
        # Update paths to match your dataset structure
        self.images_dir = os.path.join(self.data_dir, "images_final/images_final/images_" + self._get_mode_folder())
        self.masks_dir = os.path.join(self.data_dir, "dermavqa-seg-trainvalid (2)/dermavqa-seg-trainvalid/dermavqa-segmentations", self._get_mode_folder())
        
        self.saved_data_dir = "/kaggle/input/dermo-degsiff-training-part1-128x128-16-batch/dermo-degsiff/imageclefmed-mediqa-magic-2025/np"
        

        # Đường dẫn mới để lưu nếu chưa có dữ liệu trước đó
        self.npy_dir = "/kaggle/working/imageclefmed-mediqa-magic-2025/np"
        os.makedirs(self.npy_dir, exist_ok=True)
        
    def _get_mode_folder(self):
        """Convert mode to folder name"""
        mode_map = {
            "train": "train",
            "tr": "train",
            "val": "valid",
            "vl": "valid",
            "test": "test",  # For test mode
            "te": "test"
        }
        return mode_map.get(self.mode, "train")

    def __get_data_path(self):
        mode_str = self._get_mode_folder()
        x_path = f"{self.npy_dir}/X_{mode_str}_{self.image_size}x{self.image_size}.npy"
        y_path = f"{self.npy_dir}/Y_{mode_str}_{self.image_size}x{self.image_size}.npy"
        return {"x": x_path, "y": y_path}

    def __get_image_mask_pairs(self):
        """Get matched image and mask pairs"""
        if self.mode == 'test':
            # For the test mode, we only need images, no masks
            image_files = glob.glob(os.path.join(self.images_dir, "*.jpg"))
            return [(img_path, None) for img_path in image_files]  # No masks for test
        else:
            # For train and val, we need both images and masks
            mask_files = glob.glob(os.path.join(self.masks_dir, "*.tiff"))
            pairs = []
            
            for mask_path in mask_files:
                mask_filename = os.path.basename(mask_path)
                # Extract image base name from mask file
                image_base = "_".join(mask_filename.split("_")[:3])
                image_path = os.path.join(self.images_dir, f"{image_base}.jpg")
                
                if os.path.exists(image_path):
                    pairs.append((image_path, mask_path))
            
            return pairs

    def __get_transforms(self):
        # Transform for image
        img_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=[self.image_size, self.image_size],
                    interpolation=transforms.functional.InterpolationMode.BILINEAR,
                ),
            ]
        )
        # Transform for mask
        msk_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=[self.image_size, self.image_size],
                    interpolation=transforms.functional.InterpolationMode.NEAREST,
                ),
            ]
        )
        return {"img": img_transform, "msk": msk_transform}

    def is_data_existed(self):
        for k, v in self.__get_data_path().items():
            if not os.path.isfile(v):
                return False
        return True

    def prepare_data(self):
        data_path = self.__get_data_path()
        self.transforms = self.__get_transforms()

        # Get all image-mask pairs
        pairs = self.__get_image_mask_pairs()
        self.print(f"Found {len(pairs)} image-mask pairs for {self.mode} set")

        # Gathering images and masks (or no masks for test)
        imgs = []
        msks = []
        
        for img_path, mask_path in tqdm(pairs):
            try:
                # Read image with numpy
                img_array = np.asarray(Image.open(img_path).convert('RGB'))
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()  # Convert to CxHxW
                
                if self.mode != 'test' and mask_path:  # Only process masks for non-test sets
                    # Read mask with tifffile
                    mask_array = tifffile.imread(mask_path)
                    if len(mask_array.shape) > 2:  # If mask has multiple channels
                        mask_array = mask_array[:, :, 0]  # Take first channel
                    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()  # Add channel dimension
                    
                    # Apply transforms to mask
                    mask_tensor = self.transforms["msk"](mask_tensor)
                    
                    # Normalize mask
                    mask_tensor = (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min() + 1e-8)
                    
                    msks.append(mask_tensor.numpy())
                
                # Apply image transforms
                img_tensor = self.transforms["img"](img_tensor)
                
                # Normalize image
                img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
                
                imgs.append(img_tensor.numpy())
            except Exception as e:
                self.print(f"Error processing {img_path} and {mask_path if mask_path else ''}: {e}")
                continue

        X = np.array(imgs)
        Y = np.array(msks) if msks else None

        # Check dir
        # Path(self.npy_dir).mkdir(exist_ok=True)
        
        # Update file paths
        save_path_x = os.path.join(self.npy_dir, os.path.basename(data_path["x"]))
        save_path_y = os.path.join(self.npy_dir, os.path.basename(data_path["y"]))
        
        # Ensure there's valid data
        if X.size == 0:
            raise ValueError("Error: No valid images found. Check dataset paths!")
        
        # For test data, skip saving masks
        if self.mode != 'test' and msks:
            np.save(save_path_y, Y)  # Save masks if available
        
        self.print(f"Saving data... (X shape: {X.shape}, Y shape: {Y.shape if Y is not None else 'No Y file'})")
        
        # Save the data
        np.save(save_path_x, X)  # Save images
        
        self.print(f"Saved at:\n  X: {save_path_x}\n  Y: {save_path_y if msks else ''}")

    def get_data(self):
        # Chỉ load train và valid từ đường dẫn đã lưu trước
        if self.mode in ["train", "val"]:
            # self.npy_dir = self.saved_data_dir // Sử dụng đường dẫn đã lưu trước
            self.npy_dir = "/kaggle/working/imageclefmed-mediqa-magic-2025/np" # Đường dẫn mới để lưu nếu chưa có dữ liệu trước đó
        elif self.mode == "test":
            self.npy_dir = "/kaggle/working/imageclefmed-mediqa-magic-2025/np"
        
        # Tạo thư mục nếu chưa có
        os.makedirs(self.npy_dir, exist_ok=True)
        
        data_path = self.__get_data_path()

        self.print("Checking for pre-saved files...")
        if not self.is_data_existed():
            self.print(f"There are no pre-saved files for {self.mode}.")
            self.print(f"Preparing {self.mode} data...")
            self.prepare_data()
        else:
            self.print(f"Found pre-saved files at {self.npy_dir}")

        self.print("Loading...")
        X = np.load(data_path["x"])
        Y = np.load(data_path["y"]) if os.path.isfile(data_path["y"]) else None
        self.print(f"Loaded X ({X.shape}) and Y ({'No Y file' if Y is None else Y.shape}) npy format")

        return {"x": X, "y": Y}
