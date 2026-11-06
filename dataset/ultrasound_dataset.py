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
    


class UltrasoundFinegrainedDataset(Dataset):
    """Dataset for Fine Grained Pretraining of Ultrasound Images"""

    def __init__(self, data_path: str, data_args: DataArguments, img_preprocess_vlm=None, img_preprocess_vfm=None,tokenizer=None):
        super(UltrasoundFinegrainedDataset, self).__init__()
        self.data_path = data_path
        if not data_args.spatial:
            self.text_paths = sorted([
                    path for path in glob.glob(os.path.join(self.data_path, "*", "*.json"))
                    if not path.endswith("_spatial.json")
                ])
        else:
            self.text_paths = sorted([
                    path for path in glob.glob(os.path.join(self.data_path, "*", "*_spatial.json"))
                ])
            # self.text_paths = sorted([
            #         path for path in glob.glob(os.path.join(self.data_path, "*", "*_direct.json"))
            #     ])
            
        with open('dataset/selected_list.txt', "r", encoding="utf-8") as f:
            filenames = [line.strip() for line in f if line.strip()]
        
        
        self.text_paths = [
            path for path in self.text_paths
            if os.path.splitext(os.path.basename(path))[0].split('_')[0] in filenames
        ]
        print("======Now Using Json File: ", self.text_paths[0], "========")
        
        self.data_args = data_args
        self.preprocess_vlm = img_preprocess_vlm
        self.preprocess_vfm = img_preprocess_vfm
        
        self.tokenizer = tokenizer
        
        self.base_image_size = data_args.base_image_size
        self.max_length = 77
        self.max_anns = 5
        self.class_name_map = {
            'carotid artery': 0,
            'jugular vein': 1,
            'trachea': 2,
            'thyroid gland': 3,
            'vertebral body': 4,
        }

        
    def __len__(self):
        return len(self.text_paths)
        
    def denorm(self, box, image_size):
        return [box[0] * image_size[0], box[1] * image_size[1], box[2] * image_size[0], box[3] * image_size[1]]
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data = {}
        
        #----------------- load relevant files -----------------#
        text_path = self.text_paths[i]
        if text_path.endswith("_direct.json"):
            image_path = text_path.replace("_direct.json", ".png")
            text = json.load(open(text_path, "r"))
        elif not self.data_args.spatial:
            image_path = text_path.replace(".json", ".png")
            bbox_path = text_path.replace(".json", ".pkl")
            bbox = pickle.load(open(bbox_path, "rb"))
            text = json.load(open(text_path, "r"))
        else:
            image_path = text_path.replace("_spatial.json", ".png")
            bbox_path = text_path.replace("_spatial.json", ".pkl")
            bbox = pickle.load(open(bbox_path, "rb"))
            text = json.load(open(text_path, "r"))
        
        #----------------- process image -----------------#
        image = Image.open(image_path)
        # image = image.resize((self.base_image_size, self.base_image_size))
        # image_tensor = self.preprocess.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = self.preprocess_vlm(image)
        image_tensor_vfm = self.preprocess_vfm(image)
        
        #----------------- process general description -----------------#
        gen_desc = text["general_description"]
        gen_desc_embedding =  torch.tensor(self.tokenizer([gen_desc], max_length=self.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)
        gen_desc_spatial = text["general_description_spatial"]
        gen_desc_spatial_embedding =  torch.tensor(self.tokenizer([gen_desc_spatial], max_length=self.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)
        
        
        #----------------- process box -----------------#
        orig_captions = []
        crop_captions = []
        anatomy_captions = []
        anatomy_labels = []
        crop_images = []
        bbox_info = text['box_info']
        total_num = self.max_anns
        valid_num = min(len(bbox_info), self.max_anns)
        boxes_template = torch.zeros((total_num, 4), device=image_tensor.device)
        
        for i in range(total_num):
            if i < valid_num:
                bbox_data = bbox_info[i]
                box = bbox_data['box']
                orig_caption = bbox_data['orig_description']
                crop_caption = bbox_data['crop_description']
                anatomy_caption = bbox_data['anatomy']
                anatomy_label = torch.tensor([self.class_name_map[anatomy_caption]], device=image_tensor.device)

                box_denorm = self.denorm(box, image.size)
                crop_image = image.crop(box_denorm)
                crop_image_tensor = self.preprocess_vlm(crop_image)
                
            
            else:
                box = [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.000000000]
                orig_caption = ""
                crop_caption = ""
                anatomy_caption = ""
                anatomy_label = torch.tensor([-1], device=image_tensor.device)
                crop_image_tensor = torch.zeros((3, self.base_image_size, self.base_image_size), device=image_tensor.device)
                
            box_tensor = torch.tensor(box[:4])
            boxes_template[i] = box_tensor
            

            if box[0] > box[2] or box[1] > box[3]:
                raise ValueError("Box coordinates are invalid.")

            
            orig_caption = torch.tensor(self.tokenizer([orig_caption], max_length=self.max_length, padding="max_length", truncation=True).
                                        input_ids, dtype=torch.long, device=image_tensor.device)   
            crop_caption = torch.tensor(self.tokenizer([crop_caption], max_length=self.max_length, padding="max_length", truncation=True).
                                        input_ids, dtype=torch.long, device=image_tensor.device)   
            anatomy_caption = torch.tensor(self.tokenizer([anatomy_caption], max_length=self.max_length, padding="max_length", truncation=True).
                                        input_ids, dtype=torch.long, device=image_tensor.device)   
            
            
            orig_captions.append(orig_caption)
            crop_captions.append(crop_caption)
            anatomy_captions.append(anatomy_caption)
            anatomy_labels.append(anatomy_label)
            crop_images.append(crop_image_tensor)
        
        orig_captions = torch.cat(orig_captions,dim=0)
        crop_captions = torch.cat(crop_captions,dim=0)
        anatomy_captions = torch.cat(anatomy_captions,dim=0)
        anatomy_labels = torch.cat(anatomy_labels,dim=0)
        crop_images = torch.stack(crop_images,dim=0)
        bbox_num = torch.tensor([valid_num], device=image_tensor.device)
        
        #----------------- process neg prompts -----------------#
        anatomy_neg_prompts = text['negation']['anatomy']
        organ_neg_prompts = text['negation']['organ']
        possible_neg_prompts = []
        for neg_prompt in anatomy_neg_prompts:
            possible_neg_prompts.append(neg_prompt)
        for neg_prompt in organ_neg_prompts:
            possible_neg_prompts.append(neg_prompt)
        
        neg_prompt_count = min(self.max_anns, len(possible_neg_prompts))
        selected_neg_prompts = random.sample(possible_neg_prompts, neg_prompt_count)
        neg_text = [gen_desc, *selected_neg_prompts]
        neg_prompts_tensor = torch.tensor(
            self.tokenizer(neg_text, max_length=self.max_length, padding="max_length", truncation=True).input_ids,
            dtype=torch.long,
            device=image_tensor.device
        )
        # ----------------- assign all the processed data -----------------#
        data["image"] = image_tensor
        data["image_vfm"] = image_tensor_vfm
        data["general_desc"] = gen_desc_embedding
        data["general_desc_spatial"] = gen_desc_spatial_embedding
        data['box_orig_captions'] = orig_captions
        data['box_crop_captions'] = crop_captions
        data['anatomy_captions'] = anatomy_captions
        data['anatomy_labels'] = anatomy_labels
        data['box_infos'] = boxes_template
        data['box_nums'] = bbox_num
        data['neg_texts'] = neg_prompts_tensor
        data['neg_nums'] = torch.tensor([1])
        data['captions'] = gen_desc
        data['captions_spatial'] = gen_desc_spatial
        data['box_crop_images'] = crop_images
        return data
    
        
@dataclass
class DataCollatorUltrasoundFinegrainedDataset(object):
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
        texts_spatial = [instance['general_desc_spatial'] for instance in instances]
        batch['general_desc_spatial'] = torch.cat(texts_spatial,dim=0)
        
        # process bbox info + box texts
        box_orig_captions = [instance['box_orig_captions'] for instance in instances]
        batch['box_orig_captions'] = torch.cat(box_orig_captions,dim=0)
        box_crop_captions = [instance['box_crop_captions'] for instance in instances]
        batch['box_crop_captions'] = torch.cat(box_crop_captions,dim=0)
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
        
        # process neg texts
        neg_texts = [instance['neg_texts'] for instance in instances]
        batch['neg_texts'] = torch.cat(neg_texts, dim=0)
        neg_nums = [instance['neg_nums'] for instance in instances]
        batch['neg_nums'] = torch.cat(neg_nums, dim=0) 
        # To handle captions that are individual sentences, we need to process them similarly to other text data.
        # Assuming each instance has a 'captions' key containing a list of sentences, we will concatenate them.
        
        # process captions
        captions = [instance['captions'] for instance in instances]
        captions_spatial = [instance['captions_spatial'] for instance in instances]
        # Concatenate all sentences in each instance to form a single tensor
        batch['captions'] = captions
        batch['captions_spatial'] = captions_spatial
        
        
        return batch
        

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        batch = {}
        images = [instance['image'] for instance in instances]
        batch['image'] = torch.stack(images)
        texts = [instance['text'] for instance in instances]
        batch['text_long'] = torch.cat(texts,dim=0)
        short_texts = [instance['short_text'] for instance in instances]
        batch['text_short'] = torch.cat(short_texts,dim=0)
        
        batch["add_box_loss"] = instances[0]["add_box_loss"]
        batch["use_hard_neg"] = instances[0]["use_hard_neg"]
        
        if batch["add_box_loss"]:
            box_texts = [instance['box_texts'] for instance in instances]
            batch['box_texts'] = torch.cat(box_texts,dim=0)
            box_infos = [instance['box_infos'] for instance in instances]
            batch['box_infos'] = torch.cat(box_infos,dim=0)
            box_nums = [instance['box_nums'] for instance in instances]
            batch['box_nums'] = torch.cat(box_nums, dim=0)
        if batch["use_hard_neg"] :
            hard_texts = []
            for instance in instances:
                if instance['hard_texts'] != None:
                    hard_texts.append(instance['hard_texts'])

            batch['hard_texts'] = torch.cat(hard_texts,dim=0)
            hard_infos = [instance['hard_infos'] for instance in instances]
            batch['hard_infos'] = torch.cat(hard_infos,dim=0)
            hard_nums = [instance['hard_nums'] for instance in instances]
            batch['hard_nums'] = torch.cat(hard_nums, dim=0)                

        return batch
