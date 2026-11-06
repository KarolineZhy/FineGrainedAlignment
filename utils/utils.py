from dataclasses import dataclass, field
from typing import Optional 
import torch
from contextlib import suppress
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np

@dataclass
class DataArguments:
    modality: str = "us"
    data_path: str = field(default=None,
                        metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    max_seq_length: int = 77*4-60
    base_seq_length: int = 77
    base_image_size: int = 224
    add_box_loss: bool = field(default=False)
    use_hard_neg: bool = field(default=False)
    spatial: bool = field(default=False)

def parse_config_to_data_args(cfg) -> DataArguments:
    def clean(val):
        if isinstance(val, str):
            if val.lower() == "none":
                return None
            if val.startswith("'") and val.endswith("'"):
                return val[1:-1]  # strip quotes
        return val

    args_dict = {k: clean(v) for k, v in cfg.items()}
    return DataArguments(**args_dict)

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype



def get_autocast(precision):
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress



def normalize_and_tensorize_boxes(bbox, image_width, image_height, feature_size=14):
    x, y, w, h = bbox
    
    # 正则化边界框坐标到0-1之间
    x1 = (x / image_width)*feature_size
    y1 = (y / image_height)*feature_size
    x2 = ((x + w) / image_width)*feature_size
    y2 = ((y + h) / image_height)*feature_size

    newbox = [[0, x1, y1, x2, y2]]
    
    # 转换为张量
    boxes_tensor = torch.tensor(newbox, dtype=torch.float32)

    return boxes_tensor


def plot_heatmap(array, title=""):
    fig, ax = plt.subplots()
    cax = ax.imshow(array, cmap='viridis')
    ax.set_title(title)
    fig.colorbar(cax)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    image = Image.open(buf)
    return torch.tensor(np.array(image)).permute(2, 0, 1) / 255.0  # shape: (3, H, W)
