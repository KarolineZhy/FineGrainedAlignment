import json
import os
import random
import torch
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional
from torch.utils.data import Dataset, DataLoader
import glob
import argparse
import random
import pickle
from transformers import AutoTokenizer, AutoImageProcessor
local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)
from dataset.transform import image_transform

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    base_image_size: int = field(default=512, metadata={"help": "Base image size."})
    


class MRIFinegrainedDataset(Dataset):
    """Dataset for Fine Grained Pretraining of MRI Images"""

    def __init__(self, data_path: str, data_args: DataArguments, img_preprocess_vlm=None, img_preprocess_vfm=None,tokenizer=None):
        super(MRIFinegrainedDataset, self).__init__()
        self.data_path = data_path
        self.text_paths = sorted([path for path in glob.glob(os.path.join(self.data_path, "*", "*.json")) if not path.endswith('_llm.json')])
        # self.text_paths = sorted(glob.glob(os.path.join(self.data_path, "*", "*_spatial_llm.json")))
        
        #### filter text path if only {'bbox_info': []} 
        self.text_paths = [path for path in self.text_paths if len(json.load(open(path, "r"))['bbox_info']) > 0]
        # self.text_paths = [path for path in self.text_paths if len(json.load(open(path, "r"))['box_info']) > 0]
        
        # self.text_paths = self.text_paths[:3000]
        
        self.data_args = data_args
        self.preprocess_vlm = img_preprocess_vlm
        self.preprocess_vfm = img_preprocess_vfm
        
        self.tokenizer = tokenizer
        
        self.base_image_size = data_args.base_image_size
        self.max_length = 77
        self.max_anns = 5
        ## mri
        # self.text_paths = self.text_paths[:10000]
        # self.class_name_map = {
        #     'Liver': 1,
        #     'Spleen': 2,
        #     'Kidney': 3,
        #     'Kidney': 4,
        # }

        ### ct
        self.class_name_map  = {
            'Liver': 1,
            'Spleen': 2,
            'Kidney': 3,
            'Kidney': 4,
            'Stomach': 5,
            'Gallbladder': 6,
            'Pancreas': 8,
        }


        
    def __len__(self):
        return len(self.text_paths)
        
    def denorm(self, box, image_size):
        return [box[0] * image_size[0], box[1] * image_size[1], box[2] * image_size[0], box[3] * image_size[1]]
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data = {}
        
        #----------------- load relevant files -----------------#
        text_path = self.text_paths[i]
        image_path = text_path.replace(".json", ".png")
        # image_path = text_path.replace("_spatial_llm.json", ".png")
        text = json.load(open(text_path, "r"))
    
        
        #----------------- process image -----------------#
        image = Image.open(image_path)
        image_tensor = self.preprocess_vlm(image)
        image_tensor_vfm = self.preprocess_vfm(image)
        
        #----------------- process general description -----------------#
        gen_desc = text["general_description"]
        gen_desc_embedding =  torch.tensor(self.tokenizer([gen_desc], max_length=self.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)
        
        #----------------- process box -----------------#
        anatomy_captions = []
        anatomy_labels = []
        crop_images = []
        bbox_info = text['bbox_info']
        # bbox_info = text['box_info']
        total_num = self.max_anns
        valid_num = min(len(bbox_info), self.max_anns)
        boxes_template = torch.zeros((total_num, 4), device=image_tensor.device)
        
        for i in range(total_num):
            if i < valid_num:
                bbox_data = bbox_info[i]
                box = bbox_data['box']
                anatomy_caption = bbox_data['anatomy']
                anatomy_label = torch.tensor([self.class_name_map[anatomy_caption]], device=image_tensor.device)
                box_denorm = self.denorm(box, image.size)
                crop_image = image.crop(box_denorm)
                crop_image_tensor = self.preprocess_vlm(crop_image)
                
            else:
                box = [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.000000000]
                anatomy_caption = ""
                anatomy_label = torch.tensor([-1], device=image_tensor.device)
                crop_image_tensor = torch.zeros((3, self.base_image_size, self.base_image_size), device=image_tensor.device)
                
            box_tensor = torch.tensor(box[:4])
            boxes_template[i] = box_tensor
            
            if box[0] > box[2] or box[1] > box[3]:
                raise ValueError("Box coordinates are invalid.")

            
            anatomy_caption = torch.tensor(self.tokenizer([anatomy_caption], max_length=self.max_length, padding="max_length", truncation=True).
                                        input_ids, dtype=torch.long, device=image_tensor.device)   
            
            anatomy_captions.append(anatomy_caption)
            anatomy_labels.append(anatomy_label)
            crop_images.append(crop_image_tensor)
        
        anatomy_captions = torch.cat(anatomy_captions,dim=0)
        anatomy_labels = torch.cat(anatomy_labels,dim=0)
        crop_images = torch.stack(crop_images,dim=0)
        bbox_num = torch.tensor([valid_num], device=image_tensor.device)
        
        # ----------------- assign all the processed data -----------------#
        data["image"] = image_tensor
        data["image_vfm"] = image_tensor_vfm
        data["general_desc"] = gen_desc_embedding

        data['anatomy_captions'] = anatomy_captions
        data['anatomy_labels'] = anatomy_labels
        data['box_infos'] = boxes_template
        data['box_nums'] = bbox_num
        
        data['captions'] = gen_desc
        data['box_crop_images'] = crop_images
        return data
    
        
@dataclass
class DataCollatorMRIFinegrainedDataset(object):
    """Collate examples for fine-grained fine-tuning."""
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        batch = {}
        
        # process image
        images = [instance['image'] for instance in instances]
        batch['image'] = torch.stack(images)
        images_vfm = [instance['image_vfm'] for instance in instances]
        batch['image_vfm'] = torch.stack(images_vfm)
        
        # process general description
        texts = [instance['general_desc'] for instance in instances]
        batch['general_desc'] = torch.cat(texts,dim=0)
        
        # process bbox info + box texts
        box_anatomy_captions = [instance['anatomy_captions'] for instance in instances]
        batch['box_anatomy_captions'] = torch.cat(box_anatomy_captions,dim=0)
        box_anatomy_labels = [instance['anatomy_labels'] for instance in instances]
        batch['box_anatomy_labels'] = torch.cat(box_anatomy_labels,dim=0)
        box_infos = [instance['box_infos'] for instance in instances]
        batch['box_infos'] = torch.cat(box_infos,dim=0)
        box_nums = [instance['box_nums'] for instance in instances]
        batch['box_nums'] = torch.cat(box_nums, dim=0)
        box_crop_images = [instance['box_crop_images'] for instance in instances]
        batch['box_crop_images'] = torch.cat(box_crop_images,dim=0)
        

        # process captions
        captions = [instance['captions'] for instance in instances]
        # Concatenate all sentences in each instance to form a single tensor
        batch['captions'] = captions
        
        
        return batch
