from datasets.new_dataset import DermaVQADataset
from torch.utils.data import DataLoader
from modules.transforms import DiffusionTransform, DataAugmentationTransform
import albumentations as A
import glob


def get_dermavqa(config, logger=None, verbose=False):
    if logger: 
        print = logger.info
    else:
        print = print

    INPUT_SIZE = config["dataset"]["input_size"]
    DT = DiffusionTransform((INPUT_SIZE, INPUT_SIZE))
    AUGT = DataAugmentationTransform((INPUT_SIZE, INPUT_SIZE))
    
    # Paths for DermaVQA dataset
    img_dir = "images_final/images_final/images_final/images_train"
    msk_dir = "dermavqa-seg-trainvalid (2)/dermavqa-seg-trainvalid/dermavqa-segmentations/train"
    img_path_list = glob.glob(f"{config['dataset']['data_dir']}/{img_dir}/IMG_*.jpg")
    
    pixel_level_transform = AUGT.get_pixel_level_transform(config["augmentation"], img_path_list=img_path_list)
    spacial_level_transform = AUGT.get_spacial_level_transform(config["augmentation"])
    tr_aug_transform = A.Compose([
        A.Compose(pixel_level_transform, p=config["augmentation"]["levels"]["pixel"]["p"]), 
        A.Compose(spacial_level_transform, p=config["augmentation"]["levels"]["spacial"]["p"])
    ], p=config["augmentation"]["p"])

    # ----------------- dataset --------------------
    if config["dataset"]["class_name"] == "DermaVQADataset":
        # preparing training dataset
        tr_dataset = DermaVQADataset(
            mode="train",  # Use "train" instead of "tr"
            data_dir=config["dataset"]["data_dir"],
            one_hot=False,
            image_size=config["dataset"]["input_size"],
            aug=tr_aug_transform,
            img_transform=DT.get_forward_transform_img(),
            msk_transform=DT.get_forward_transform_msk(),
            add_boundary_mask=config["dataset"].get("add_boundary_mask", False),
            add_boundary_dist=config["dataset"].get("add_boundary_dist", False),
            logger=logger,
            data_scale=config["dataset"].get("data_scale", "full")
        )
        
        vl_dataset = DermaVQADataset(
            mode="val",  # Use "val" instead of "vl"
            data_dir=config["dataset"]["data_dir"],
            one_hot=False,
            image_size=config["dataset"]["input_size"],
            img_transform=DT.get_forward_transform_img(),
            msk_transform=DT.get_forward_transform_msk(),
            add_boundary_mask=config["dataset"].get("add_boundary_mask", False),
            add_boundary_dist=config["dataset"].get("add_boundary_dist", False),
            logger=logger,
            data_scale=config["dataset"].get("data_scale", "full")
        )
        
        te_dataset = DermaVQADataset(
            mode="test",  # Use "test" instead of "te"
            data_dir=config["dataset"]["data_dir"],
            one_hot=False,
            image_size=config["dataset"]["input_size"],
            img_transform=DT.get_forward_transform_img(),
            msk_transform=DT.get_forward_transform_msk(),
            add_boundary_mask=config["dataset"].get("add_boundary_mask", False),
            add_boundary_dist=config["dataset"].get("add_boundary_dist", False),
            logger=logger,
            data_scale=config["dataset"].get("data_scale", "full")
        )
    else:
        message = "In the config file, `dataset>class_name` should be 'DermaVQADataset'"
        if logger: 
            logger.exception(message)
        else:
            raise ValueError(message)

    if verbose:
        print("DermaVQA Dataset:")
        print(f"├──> Length of training_dataset:    {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset:  {len(vl_dataset)}")
        print(f"└──> Length of test_dataset:        {len(te_dataset)}")

    # prepare train dataloader
    tr_dataloader = DataLoader(tr_dataset, **config["data_loader"]["train"])

    # prepare validation dataloader
    vl_dataloader = DataLoader(vl_dataset, **config["data_loader"]["validation"])

    # prepare test dataloader
    te_dataloader = DataLoader(te_dataset, **config["data_loader"]["test"])

    return {
        "tr": {"dataset": tr_dataset, "loader": tr_dataloader},
        "vl": {"dataset": vl_dataset, "loader": vl_dataloader},
        "te": {"dataset": te_dataset, "loader": te_dataloader},
    }


# Commented test code for visualization
"""
# test and visualize the input data
from utils.helper_funcs import show_sbs
for sample in tr_dataloader:
    img = sample['image']
    msk = sample['mask']
    print("Training")
    print(img.shape, msk.shape)
    show_sbs(img[0], msk[0])
    break

for sample in vl_dataloader:
    img = sample['image']
    msk = sample['mask']
    print("Validation")
    show_sbs(img[0], msk[0])
    break

for sample in te_dataloader:
    img = sample['image']
    msk = sample['mask']
    print("Test")
    show_sbs(img[0], msk[0])
    break
"""