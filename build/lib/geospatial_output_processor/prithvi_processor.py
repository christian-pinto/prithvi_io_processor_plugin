import datetime
import re
from einops import rearrange
import numpy as np
import rasterio
import torch
from typing import Any, List, Optional, Union
from vllm.inputs.data import MultiModalPromptType, PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.multimodal_data_processors.interface import MultimodalDataProcessor
from terratorch.datamodules import Sen1Floods11NonGeoDataModule
from vllm.config import VllmConfig
from vllm.outputs import MultiModalRequestOutput, ImageRequestOutput
import albumentations

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
OFFSET = 0
PERCENTILE = 99

DEFAULT_INPUT_INDICES = [1, 2, 3, 8, 11, 12]

datamodule_config = {
    "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    "batch_size": 16,
    "constant_scale": 0.0001,
    "data_root": "/dccstor/geofm-finetuning/datasets/sen1floods11",
    "drop_last": True,
    "no_data_replace": 0.0,
    "no_label_replace": -1,
    "num_workers": 8,
    "test_transform": [
        albumentations.Resize(
            always_apply=False, height=448, interpolation=1, p=1, width=448
        ),
        albumentations.pytorch.ToTensorV2(
            transpose_mask=False, always_apply=True, p=1.0
        ),
    ],
}

def save_geotiff(image, output_path: str, meta: dict):
    """Save multi-band image in Geotiff file.

    Args:
        image: np.ndarray with shape (bands, height, width)
        output_path: path where to save the image
        meta: dict with meta info.
    """

    with rasterio.open(output_path, "w", **meta) as dest:
        for i in range(image.shape[0]):
            dest.write(image[i, :, :], i + 1)

    return

def _convert_np_uint8(float_image: torch.Tensor):
    image = float_image.numpy() * 255.0
    image = image.astype(dtype=np.uint8)

    return image

def read_geotiff(file_path: str):
    """Read all bands from *file_path* and return image + meta info.

    Args:
        file_path: path to image file.

    Returns:
        np.ndarray with shape (bands, height, width)
        meta info dict
    """

    with rasterio.open(file_path) as src:
        img = src.read()
        meta = src.meta
        try:
            coords = src.lnglat()
        except Exception:
            # Cannot read coords
            coords = None

    return img, meta, coords

def load_image(
    file_paths: list[str],
    mean: list[float] = None,
    std: list[float] = None,
    indices: Union[list[int], None] = None,
):
    """Build an input example by loading images in *file_paths*.

    Args:
        file_paths: list of file paths .
        mean: list containing mean values for each band in the
              images in *file_paths*.
        std: list containing std values for each band in the
             images in *file_paths*.

    Returns:
        np.array containing created example
        list of meta info for each image in *file_paths*
    """

    imgs = []
    metas = []
    temporal_coords = []
    location_coords = []

    for file in file_paths:
        img, meta, coords = read_geotiff(file)

        # Rescaling (don't normalize on nodata)
        img = np.moveaxis(img, 0, -1)  # channels last for rescaling
        if indices is not None:
            img = img[..., indices]
        if mean is not None and std is not None:
            img = np.where(img == NO_DATA, NO_DATA_FLOAT, (img - mean) / std)

        imgs.append(img)
        metas.append(meta)
        if coords is not None:
            location_coords.append(coords)

        try:
            match = re.search(r"(\d{7,8}T\d{6})", file)
            if match:
                year = int(match.group(1)[:4])
                julian_day = match.group(1).split("T")[0][4:]
                if len(julian_day) == 3:
                    julian_day = int(julian_day)
                else:
                    julian_day = (
                        datetime.datetime.strptime(julian_day, "%m%d")
                        .timetuple()
                        .tm_yday
                    )
                temporal_coords.append([year, julian_day])
        except Exception as e:
            print(f"Could not extract timestamp for {file} ({e})")

    imgs = np.stack(imgs, axis=0)  # num_frames, H, W, C
    imgs = np.moveaxis(imgs, -1, 0).astype("float32")  # C, num_frames, H, W
    imgs = np.expand_dims(imgs, axis=0)  # add batch di

    return imgs, temporal_coords, location_coords, metas

class PrithviOutputProcessor(MultimodalDataProcessor):
            
    def __init__(self, vllm_config: VllmConfig):
        
        super().__init__(vllm_config)

        self.datamodule = Sen1Floods11NonGeoDataModule(
            data_root=datamodule_config["data_root"],
            batch_size=datamodule_config["batch_size"],
            num_workers=datamodule_config["num_workers"],
            bands=datamodule_config["bands"],
            drop_last=datamodule_config["drop_last"],
            test_transform=datamodule_config["test_transform"],
        )
        self.img_size = 512
        self.h1 = 1
        self.w1 = 1
        self.original_h = 512
        self.original_w = 512
        self.batch_size = 1
        self.metadata = None
        self.channels = None

    def pre_process(self, prompt: MultiModalPromptType,
                    request_id: Optional[str] = None,) -> List[PromptType]:
        

        input_data, temporal_coords, location_coords, meta_data = load_image(
            file_paths=[prompt["data"]],
            indices=DEFAULT_INPUT_INDICES,
        )

        self.meta_data = meta_data[0]

        if input_data.mean() > 1:
            input_data = input_data / 10000  # Convert to range 0-1

        self.channels = [
            datamodule_config["bands"].index(b) for b in ["RED", "GREEN", "BLUE"]
        ]  # BGR -> RGB

        self.original_h, self.original_w = input_data.shape[-2:]
        pad_h = (self.img_size - (self.original_h % self.img_size)) % self.img_size
        pad_w = (self.img_size - (self.original_w % self.img_size)) % self.img_size
        input_data = np.pad(
            input_data, ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
        )

    
        batch = torch.tensor(input_data)
        windows = batch.unfold(3, self.img_size, self.img_size).unfold(4, self.img_size, self.img_size)
        self.h1, self.w1 = windows.shape[3:5]
        windows = rearrange(
            windows, "b c t h1 w1 h w -> (b h1 w1) c t h w", h=self.img_size, w=self.img_size
        )

        # Split into batches if number of windows > batch_size
        num_batches = windows.shape[0] // self.batch_size if windows.shape[0] > self.batch_size else 1
        windows = torch.tensor_split(windows, num_batches, dim=0)

        if temporal_coords:
            temporal_coords = torch.tensor(temporal_coords).unsqueeze(0)
        else:
            temporal_coords = None
        if location_coords:
            location_coords = torch.tensor(location_coords[0]).unsqueeze(0)
        else:
            location_coords = None

        prompts = []
        for window in windows:
        # Apply standardization
            window = self.datamodule.test_transform(image=window.squeeze().numpy().transpose(1, 2, 0))
            window = self.datamodule.aug(window)["image"]
            prompts.append(
                {
                    "prompt_token_ids": [1],
                    "multi_modal_data": {
                        "pixel_values": window.to(torch.float16)[0],
                        "location_coords": location_coords
                    }
                }
            )

        return prompts
    
    def post_process(self,
                    model_out: list[Optional[PoolingRequestOutput]],
                    request_id: Optional[str] = None, ) -> MultiModalRequestOutput:
        print("This is the Geospatial plugin hidden states processor:")            

        y_hat = model_out[0].outputs.data.argmax(dim=1)
        pred = torch.nn.functional.interpolate(y_hat.unsqueeze(1).float(),
                                               size=self.img_size,
                                               mode="nearest",)

        
        # Build images from patches
        pred_imgs = rearrange(
            pred,
            "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
            h=self.img_size,
            w=self.img_size,
            b=1,
            c=1,
            h1=self.h1,
            w1=self.w1,
        )


        # Cut padded area back to original size
        pred_imgs = pred_imgs[..., :self.original_h, :self.original_w]

        # Squeeze (batch size 1)
        pred_imgs = pred_imgs[0]

        #create temp file
        file_path = "/workspace/vllm_max/file_test_output_processor.tiff"
        meta_data = {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': 0,
                     'width': 512, 'height': 512, 'count': 1,
                     'compress': 'lzw',
                     }
        save_geotiff(_convert_np_uint8(pred_imgs), file_path, meta_data)

        return ImageRequestOutput(type="path", format="tiff", data=file_path)