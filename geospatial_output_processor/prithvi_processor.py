from vllm.v1.engine.hidden_states_processor import HiddenStatesProcessor
from typing import Any
from einops import rearrange
import torch
import rasterio
import numpy as np

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

class PrithviOutputProcessor(HiddenStatesProcessor):
    img_size = 512
    h1 = 1
    w1 = 1
    original_h = 512
    original_w = 512

    def apply(self, data: torch.Tensor) -> Any:
        print("This is the Geospatial plugin hidden states processor:")            

        y_hat = data.argmax(dim=1)
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
        file_path = "/workspace/file_test_output_processor.tiff"
        meta_data = {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': 0,
                     'width': 512, 'height': 512, 'count': 1,
                     'compress': 'lzw',
                     }
        save_geotiff(_convert_np_uint8(pred_imgs), file_path, meta_data)
    