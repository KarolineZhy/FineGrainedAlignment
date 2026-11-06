import torch
import torch.nn as nn
import math

from transformers import CLIPConfig,AutoConfig
from typing import Any, Optional, Tuple, Union
import torch.distributed.nn as nn_dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union
import matplotlib.pyplot as plt
import json
import pickle
import torch.distributed as dist
from torch.nn import AvgPool2d
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from tqdm import tqdm
from model.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from torch import nn
from sentence_transformers import SentenceTransformer as SBERT
import lightning as L
import math
from torchvision.ops import roi_align
from loss.ultraclip_loss import clip_loss, pairwise_contrastive_loss, hard_contrastive_loss, ClipInModalityLoss
from model.ultraclip import UltraCLIP
from loss.ultraclip_loss import ClipInModalityLoss
from skimage.transform import resize
import wandb
from scipy.ndimage import gaussian_filter
import glob
import os
from PIL import Image, ImageDraw

from model.ultraclip import PureAttentionPoolingBlock, GlobalAwareFusion


class UltraCLIPLightningModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.model = UltraCLIP.from_pretrained(config.model.model_name_or_path)
        self.model.train()
    
        # load config
        self.config = config
        self.add_box_loss = config.model.add_box_loss
        self.use_hard_neg = config.model.use_hard_neg
        self.attention_mask = None
        self.position_ids = None
        self.output_attentions = None
        self.output_hidden_states = None
        self.return_dict = None
        self.nl_semantic_supervision = config.model.nl_semantic_supervision
    
        self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(self.device)
        self.vfm.eval()
        # freeze the vfm
        for param in self.vfm.parameters():
            param.requires_grad = False
        
        if self.nl_semantic_supervision:
            self.clip_in_modality_loss = ClipInModalityLoss(
                local_loss=config.model.local_loss,
                gather_with_grad=config.model.gather_with_grad,
                cache_labels=True,
                rank=config.model.rank,
                world_size=config.model.world_size,
                use_horovod=config.model.horovod,
                alpha=config.model.alpha,
                beta=config.model.beta,
                nl_semantic_supervision=config.model.nl_semantic_supervision,
                separate_text=config.model.separate_text,
                separate_image=config.model.separate_image
            )
            self.sbert = SBERT('all-mpnet-base-v2')
            self.sbert.eval()
            # freeze the sbert
            for param in self.sbert.parameters():
                param.requires_grad = False
        else:
            self.sbert = None
            self.clip_in_modality_loss = None
            
        # logging config
        self.loss_history = []
        
    def training_step(self, batch, batch_idx):
        
        # for mri dataset
        image = batch['image']
        batch_size = image.shape[0]
        image_vfm = batch['image_vfm']
        
        general_desc = batch['general_desc']
        box_anatomy_captions = batch['box_anatomy_captions']
        box_anatomy_labels = batch['box_anatomy_labels']
        box_infos = batch['box_infos']
        box_nums = batch['box_nums']
        # neg_texts = batch['neg_texts']
        # neg_nums = batch['neg_nums']
        captions = batch['captions']
        general_desc_spatial = general_desc.clone()
        box_orig_captions = box_anatomy_captions.clone()
        box_crop_captions = box_anatomy_captions.clone()
        neg_texts = box_anatomy_captions.clone()
        neg_nums = torch.tensor([1])
        captions_spatial = captions

        # image = batch['image']
        # image_vfm = batch['image_vfm']
        # batch_size = image.shape[0]
        # general_desc = batch['general_desc']
        # general_desc_spatial = batch['general_desc_spatial']
        # box_orig_captions = batch['box_orig_captions']
        # box_crop_captions = batch['box_crop_captions']
        # box_anatomy_captions = batch['box_anatomy_captions']
        # box_anatomy_labels = batch['box_anatomy_labels']
        # box_infos = batch['box_infos']
        # box_nums = batch['box_nums']
        # neg_texts = batch['neg_texts']
        # neg_nums = batch['neg_nums']
        # captions = batch['captions']
        # captions_spatial = batch['captions_spatial']
        
        # model forward
        loss, vis = self.model(
            # general description
            general_desc = general_desc,
            general_desc_spatial = general_desc_spatial,
            image = image,
            image_vfm = image_vfm,
            
            # box input
            box_infos = box_infos,
            box_orig_captions = box_orig_captions,
            box_crop_captions = box_crop_captions,
            box_anatomy_captions = box_anatomy_captions,
            box_anatomy_labels = box_anatomy_labels,
            box_nums = box_nums,
            
            # hard neg input
            neg_texts = neg_texts,
            neg_nums = neg_nums,
            
            # semantic loss input
            captions = captions,
            captions_spatial = captions_spatial,
            sbert = self.sbert,
            cal_semantic_loss = self.clip_in_modality_loss,
            
            # vfm input
            vfm = self.vfm,
            
            # other input
            attention_mask = self.attention_mask,
            position_ids = self.position_ids,
            output_attentions = self.output_attentions,
            output_hidden_states = self.output_hidden_states,
            return_dict = self.return_dict
        )
            
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 50:
            self.loss_history.pop(0)
        avg_loss = sum(self.loss_history) / len(self.loss_history)
        self.log("loss_avg50", avg_loss, on_step=True, logger=True, batch_size=batch_size, sync_dist=True)
        
        self.logger.experiment.log({
            "text_level_similarity": wandb.Image(vis["text_level_similarity"])
        })
        
        self.logger.experiment.log({
            "image_level_similarity": wandb.Image(vis["image_level_similarity"])
        })
        self.logger.experiment.log({
            "semantic_features_similarity": wandb.Image(vis["semantic_features_similarity"])
        })
        self.logger.experiment.log({
            "bbox_similarity": wandb.Image(vis["bbox_similarity"])
        })
        self.logger.experiment.log({
            "image_vis": wandb.Image(vis["vis_sim"])
        })

        return loss
    

        
    def validation_step(self, batch, batch_idx):
        pass
    
    def on_test_start(self):
        # test config
        self.test_mode = self.config.test.test_mode
        pass
    
    def setup(self, stage: Optional[str] = None):
        pass   

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        pass
    
    def test_one_image(self, image, text_tokens, text):
        
        image_features = self.model.vision_model(
                pixel_values=image,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                interpolate_pos_encoding=True,
        )
        image_features = image_features[1]
        image_features = self.model.visual_projection(image_features)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        text_features = self.model.text_model(
                input_ids=text_tokens,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                walk_short_pos=True,
            )
        text_features = text_features[1]
        text_features = self.model.text_projection(text_features)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        probs = (image_features @ text_features.T).sigmoid()
        print(probs)
        print(text[torch.argmax(probs)])
        
    @torch.no_grad()
    def test_image_retrieval_given_image(self, retrival_image, image_list, img_preprocess, save_path):
        retrival_image = Image.open(retrival_image)
        retrival_image.save(os.path.join(save_path, 'retrival_image.png'))
        retrival_image_tensor = img_preprocess(retrival_image).unsqueeze(0).cuda()
        retrival_image_features = self.model.vision_model(
                pixel_values=retrival_image_tensor,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                interpolate_pos_encoding=True,
        )
        image_features_retrival = retrival_image_features[1]
        image_features_retrival = self.model.visual_projection(image_features_retrival)
        image_features_retrival = image_features_retrival / image_features_retrival.norm(p=2, dim=-1, keepdim=True)
        similarity_list = []
        image_list = [img for img in glob.glob(os.path.join(image_list, '*.png')) if not img.endswith('_pred.png')]
        for image_path in image_list:
            image = Image.open(image_path)
            image_tensor = img_preprocess(image).unsqueeze(0).cuda()
            image_features = self.model.vision_model(
                pixel_values=image_tensor,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                interpolate_pos_encoding=True,
            )
            image_features = image_features[1]
            image_features = self.model.visual_projection(image_features)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)    
            sim = (image_features_retrival @ image_features.T).sigmoid()
            similarity_list.append(sim)
        
        similarity_list = torch.cat(similarity_list, dim=0)
        print(similarity_list)
        print(image_list[torch.argmax(similarity_list)])
        retrieved_image = image_list[torch.argmax(similarity_list)]
        save_path = os.path.join(save_path, os.path.basename(retrieved_image))
        image = Image.open(retrieved_image)
        image.save(save_path)
        # return retrieved_image
    
    # encode text
    @torch.no_grad()
    def encode_text(self, text_tokens):
        text_features = self.model.text_model(
            input_ids=text_tokens,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            return_dict=self.return_dict,
            walk_short_pos=True,
        )
        text_features = text_features[1]
        text_features = self.model.text_projection(text_features)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    # encode image
    @torch.no_grad()
    def encode_image(self, image_tensor):
        image_features = self.model.vision_model(
            pixel_values=image_tensor,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            return_dict=self.return_dict,
            interpolate_pos_encoding=True,
        )
        image_features = image_features[1]
        image_features = self.model.visual_projection(image_features)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    
            
    @torch.no_grad()
    def test_text_retrieval_given_images(self, images, image_list, text_tokens, text):
        
        image_features = self.model.vision_model(
                pixel_values=images,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                interpolate_pos_encoding=True,
        )
        image_features = image_features[1]
        image_features = self.model.visual_projection(image_features)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        
        text_features = self.model.text_model(
                input_ids=text_tokens,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                walk_short_pos=True,
            )
        text_features = text_features[1]
        text_features = self.model.text_projection(text_features)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        probs = ( text_features @ image_features.T).sigmoid()
        print(probs)
        print(image_list[torch.argmax(probs)])
    
    def get_image_text_similarity(self, image_features, text_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (text_features @ image_features.T).sigmoid()
    
    @torch.no_grad()
    def test_image_retrieval_given_text(self, folder_path, img_preprocess, text_tokens, save_path):
        similarity_list = []
        sim_max = 0
        image_list = [img for img in glob.glob(os.path.join(folder_path, '*.png')) if not img.endswith('_pred.png')]
        text_features = text_features = self.model.text_model(
                input_ids=text_tokens,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                walk_short_pos=True,
            )
        text_features = text_features[1]
        text_features = self.model.text_projection(text_features)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        for image_path in tqdm(image_list):
            image = Image.open(image_path)
            image_tensor = img_preprocess(image).unsqueeze(0).cuda()
            image_features = self.model.vision_model(
                pixel_values=image_tensor,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                interpolate_pos_encoding=True,
            )
            image_features = image_features[1]
            image_features = self.model.visual_projection(image_features)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            similarity = self.get_image_text_similarity(image_features, text_features)
            
            if similarity > sim_max:
                sim_max = similarity
                image.save(os.path.join(save_path, os.path.basename(image_path)))
                
    @torch.no_grad()
    def test_text_retrival_given_image(self, folder_path, img_preprocess, text_tokens, text, gt, save_path, save_vis=True, data_root=""):
        # image_list = [img for img in glob.glob(os.path.join(folder_path, '*.png')) if not img.endswith('_pred.png')]
        if isinstance(folder_path, list):
            image_list = folder_path
            root_path = os.path.join(data_root, '')
            image_list = [os.path.join(root_path, image_path) for image_path in image_list]
        else:
            image_list = [img for img in glob.glob(os.path.join(folder_path, '*.png')) if not img.endswith('_pred.png')]
        text_features = self.model.text_model(
                input_ids=text_tokens,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                walk_short_pos=True,
            )
        text_features = text_features[1]
        text_features = self.model.text_projection(text_features)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        total = 0
        correct = 0
        retrieval_results = {}
        for image_path in tqdm(image_list):
            total += 1
            image = Image.open(image_path)
            image_tensor = img_preprocess(image).unsqueeze(0).cuda()
            image_features = self.model.vision_model(
                pixel_values=image_tensor,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                interpolate_pos_encoding=True,
            )
            image_features = image_features[1]
            image_features = self.model.visual_projection(image_features)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            similarity = self.get_image_text_similarity(image_features, text_features)

            retrieved_text = text[torch.argmax(similarity)]
            # For list input (MRI/CT), strip root_path to get the relative key matching gt dict
            if isinstance(folder_path, list):
                img_key = image_path.replace(root_path, '', 1)
            else:
                img_key = os.path.splitext(os.path.basename(image_path))[0]
            if retrieved_text in gt[img_key]:
                correct += 1
            # 记录 gt 和 pred
            retrieval_results[img_key] = {
                "gt": gt[img_key],
                "pred": retrieved_text,
                "pred_rank": similarity.squeeze().tolist(),
                "all_texts": text,
            }
            # save the image with the retrieved_text written on the image
            if save_vis and save_path:
                image = image.convert("RGB")
                draw = ImageDraw.Draw(image)
                draw.text((10, 10), retrieved_text, fill=(255, 255, 255))
                image.save(os.path.join(save_path, os.path.basename(image_path)))
        return correct / total, retrieval_results
        
    def test_dense_features(self, image_pil, image_input, caption_input, save_path, eval_grid_size=32, save_vis=True):
        image = image_pil.convert('RGB')
        image = image.resize((image_input.shape[2], image_input.shape[3]))

        with torch.no_grad():
            dense_image_feature = self.model.get_image_dense_features(image_input)
            text_feature = self.model.get_text_features(caption_input, walk_short_pos=True)

            text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
            dense_image_feature = dense_image_feature / dense_image_feature.norm(p=2, dim=-1, keepdim=True)

        similarity = dense_image_feature.squeeze() @ text_feature.squeeze().T
        similarity = similarity.cpu().numpy()
        
        patch_size = int(math.sqrt(similarity.shape[0]))
        similarity = similarity.reshape((patch_size, patch_size))
        
        if patch_size != eval_grid_size:
            similarity = resize(similarity, (eval_grid_size, eval_grid_size), order=1, anti_aliasing=True)
        
        similarity = resize(similarity, (image_input.shape[2], image_input.shape[3]), order=1, anti_aliasing=True)
        similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
        
        # Apply Gaussian smoothing
        smoothed_similarity = gaussian_filter(similarity, sigma=2)
        # smoothed_similarity = similarity
        # similarity = similarity ** 4
        # print('postprocessing setting: order=1, sigma=2, similarity order=0')
        # Apply thresholding to create a segmentation mask
        threshold = 0.8  # You can adjust this threshold value as needed
        seg_map = smoothed_similarity > threshold
    
        if save_vis and save_path:
            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(image)
            plt.imshow(similarity, cmap='jet', alpha=0.5)
            plt.title("Similarity Map")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(image)
            plt.imshow(seg_map, cmap='gray', alpha=0.5)
            plt.title("Segmentation Mask")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        return similarity
    
    # def test_dense_features(self, image_pil, image_input, caption_input, save_path):
    #     # print(save_path)
    #     image = image_pil.convert('RGB')
    #     image = image.resize((image_input.shape[2], image_input.shape[3]))

        
    #     with torch.no_grad():
    #         dense_image_feature = self.model.get_image_dense_features(image_input)
    #         text_feature = self.model.get_text_features(caption_input, walk_short_pos=True)
            
    #         text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
    #         dense_image_feature = dense_image_feature / dense_image_feature.norm(p=2, dim=-1, keepdim=True)

    #     similarity = dense_image_feature.squeeze() @ text_feature.squeeze().T
        
    #     similarity = similarity.cpu().numpy()
        
    #     patch_size = int(math.sqrt(similarity.shape[0]))
    #     original_shape = (patch_size, patch_size)
    #     similarity = similarity.reshape(original_shape) 
        
        
    #     similarity = resize(similarity,(image_input.shape[2], image_input.shape[3]), order=1, anti_aliasing=True)
    #     similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
        
    #     # Apply Gaussian smoothing
    #     smoothed_similarity = gaussian_filter(similarity, sigma=2)
    #     # similarity = similarity ** 4
    #     # Apply thresholding to create a segmentation mask
    #     threshold = 0.8  # You can adjust this threshold value as needed
    #     seg_map = smoothed_similarity > threshold
    
    #     plt.figure(figsize=(18, 6))

    #     plt.subplot(1, 3, 1)
    #     plt.imshow(image)
    #     plt.title("Original Image")
    #     plt.axis('off')

    #     plt.subplot(1, 3, 2)
    #     plt.imshow(image)
    #     plt.imshow(similarity, cmap='jet', alpha=0.5)
    #     plt.title("Similarity Map")
    #     plt.axis('off')

    #     plt.subplot(1, 3, 3)
    #     plt.imshow(image)
    #     plt.imshow(seg_map, cmap='gray', alpha=0.5)
    #     plt.title("Segmentation Mask")
    #     plt.axis('off')

    #     plt.tight_layout()
    #     plt.show()
    #     # plt.savefig("attention_map_with_segmentation.png")
    #     plt.savefig(save_path)
    #     plt.close()
    #     return similarity
    
    @torch.no_grad()
    def test_text_image_retrieval(self, folder_path, img_preprocess, tokenizer, gt):
        correct = 0
        retrieval_results = {}
        ##### Processing text features #####
        text_tokens = []
        retrival_text = gt.keys()
        
        for text, gt_img in gt.items():
            # print(text)
            text_token = torch.tensor(tokenizer([text], max_length=77, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=self.device)
            text_features = self.model.text_model(
                input_ids=text_token,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                walk_short_pos=True,
            )
            text_features = text_features[1]
            text_features = self.model.text_projection(text_features)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            ##### Processing image features #####
            image_features_list = []
            save_dict = {}
            image_list = [img for img in glob.glob(os.path.join(folder_path, '*.png'))]
            for image_path in tqdm(image_list):
                image = Image.open(image_path)
                image_tensor = img_preprocess(image).unsqueeze(0).cuda()
                image_features = self.model.vision_model(
                    pixel_values=image_tensor,
                    output_attentions=self.output_attentions,
                    output_hidden_states=self.output_hidden_states,
                    return_dict=self.return_dict,
                    interpolate_pos_encoding=True,
                )
                image_features = image_features[1]
                image_features = self.model.visual_projection(image_features)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                image_features_list.append(image_features)
                save_dict[os.path.basename(image_path)] = image_features
            image_features_list = torch.cat(image_features_list, dim=0)
            # save a dict file with json
            os.makedirs("results", exist_ok=True)
            with open("results/image_feature.pkl", "wb") as f:
                pickle.dump(save_dict, f)
                print('image feature saved')
        
            ##### Computing text-image similarity #####
            text_image_similarity = text_features @ image_features_list.T
            ## calculate the top 1 text image similarity
            retrieved_image = text_image_similarity.topk(1, dim=1)
            basename = os.path.basename(image_list[retrieved_image.indices]).replace('.png', '')
            print(basename)
            print(gt_img)
            if basename in gt_img:
                correct += 1
            # 记录 gt 和 pred
            # top-k retrieved images
            topk = text_image_similarity.topk(min(10, len(image_list)), dim=1)
            pred_images = [os.path.basename(image_list[idx]).replace('.png', '') for idx in topk.indices[0]]
            retrieval_results[text] = {
                "gt": gt_img,
                "pred_top1": basename,
                "pred_topk": pred_images,
                "scores": topk.values[0].cpu().tolist(),
            }
        print(correct / len(retrival_text))
        return correct / len(retrival_text), retrieval_results

    @torch.no_grad()
    def test_text_image_retrieval_subset(self, folder_path, img_preprocess, tokenizer, subset_data):
        """T2I retrieval with pre-defined candidate subsets."""
        total_correct = 0
        total_tasks = 0
        retrieval_results = {}

        # Step 1: Pre-encode all images in folder (cache to avoid redundant computation)
        print("Pre-encoding all candidate images...")
        image_feature_cache = {}
        # Search both flat and nested directory structures
        image_list = sorted(
            glob.glob(os.path.join(folder_path, '*.png')) +
            glob.glob(os.path.join(folder_path, '**', '*.png'), recursive=True)
        )
        # Deduplicate (flat results may overlap with recursive results)
        image_list = sorted(set(image_list))
        for image_path in tqdm(image_list):
            image = Image.open(image_path)
            image_tensor = img_preprocess(image).unsqueeze(0).cuda()
            image_features = self.model.vision_model(
                pixel_values=image_tensor,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                interpolate_pos_encoding=True,
            )
            image_features = image_features[1]
            image_features = self.model.visual_projection(image_features)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            feat = image_features.squeeze(0)
            # Cache with multiple key formats to match different subset json styles:
            # - relative path: "subdir/image_slice_148.png" (CT/MRI style)
            # - basename without ext: "1747817603.772205" (US style)
            rel_path = os.path.relpath(image_path, folder_path)
            basename_no_ext = os.path.basename(image_path).replace('.png', '')
            image_feature_cache[rel_path] = feat
            if basename_no_ext != rel_path:
                image_feature_cache[basename_no_ext] = feat
        print(f"Cached features for {len(image_list)} images.")

        # Step 2: Evaluate each text query and its subsets
        for text, task_info in subset_data["tasks"].items():
            print(f"\nText: {text[:80]}...")
            text_token = torch.tensor(
                tokenizer([text], max_length=77, padding="max_length", truncation=True).input_ids,
                dtype=torch.long, device=self.device)
            text_features = self.model.text_model(
                input_ids=text_token,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                return_dict=self.return_dict,
                walk_short_pos=True,
            )
            text_features = text_features[1]
            text_features = self.model.text_projection(text_features)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            text_correct = 0
            subset_results = []
            for subset in task_info["subsets"]:
                candidates = subset["candidates"]
                gt_ids = set(subset["gt"])

                # Gather candidate image features
                candidate_features = []
                valid_candidates = []
                for cid in candidates:
                    if cid in image_feature_cache:
                        candidate_features.append(image_feature_cache[cid])
                        valid_candidates.append(cid)
                    else:
                        print(f"  Warning: candidate {cid} not found in cache, skipping.")

                if len(candidate_features) == 0:
                    continue

                candidate_features = torch.stack(candidate_features, dim=0)  # [N, D]
                similarity = text_features @ candidate_features.T  # [1, N]

                top1_idx = similarity.argmax(dim=1).item()
                top1_id = valid_candidates[top1_idx]
                is_correct = top1_id in gt_ids

                if is_correct:
                    text_correct += 1
                    total_correct += 1
                total_tasks += 1

                subset_results.append({
                    "subset_id": subset["subset_id"],
                    "gt": subset["gt"],
                    "pred_top1": top1_id,
                    "correct": is_correct,
                    "top1_score": similarity[0, top1_idx].item(),
                })

            retrieval_results[text] = {
                "num_subsets": len(task_info["subsets"]),
                "correct": text_correct,
                "subsets": subset_results,
            }
            print(f"  Correct: {text_correct}/{len(task_info['subsets'])}")

        accuracy = total_correct / total_tasks if total_tasks > 0 else 0
        print(f"\nTotal T2I Subset: {total_correct}/{total_tasks} = {accuracy:.4f}")
        return accuracy, retrieval_results

    def test_dense_features_quantitative(self, image_pil, image_input, caption_input, savename):
        # print(save_path)
        image = image_pil.convert('RGB')
        shape = image_input.shape
        image = image.resize((shape[2], shape[3]))

        
        with torch.no_grad():
            dense_image_feature = self.model.get_image_dense_features(image_input)
            text_feature = self.model.get_text_features(caption_input, walk_short_pos=True)
            
            text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
            dense_image_feature = dense_image_feature / dense_image_feature.norm(p=2, dim=-1, keepdim=True)

        similarity = dense_image_feature.squeeze() @ text_feature.squeeze().T
        
        similarity = similarity.cpu().numpy()
        
        patch_size = int(math.sqrt(similarity.shape[0]))
        original_shape = (patch_size, patch_size)
        similarity = similarity.reshape(original_shape) 
        
        similarity = resize(similarity,(image_input.shape[2], image_input.shape[3]), order=1, anti_aliasing=True)
        similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
        
        # Apply Gaussian smoothing
        smoothed_similarity = gaussian_filter(similarity, sigma=2)
        similarity = smoothed_similarity ** 2
        save_overlay(np.array(image), similarity, f"{savename}.png")

        return similarity
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=0.001)
        return optimizer

import cv2
import numpy as np

def save_overlay(image, overlay, save_path, alpha=0.5, cmap='jet'):
    # image: HxWx3 RGB
    # overlay: HxW normalized [0,1] or raw float

    # Convert overlay to heatmap
    overlay_norm = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
    overlay_uint8 = np.uint8(overlay_norm * 255)
    heatmap = cv2.applyColorMap(overlay_uint8, cv2.COLORMAP_JET)

    # convert to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # blend
    blended = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    # save
    cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


    
def recall_at_k(sim, k):
    pred_true = 0
    N = sim.size(0)
    for i in range(N):
        pred = sim[i]
        topk = pred.topk(k).indices
        # 对应的5条ground-truth文本的index范围是 5*i ... 5*i+4
        true_indices = torch.arange(5*i, 5*i+5, device=sim.device)
        if (topk.unsqueeze(1) == true_indices).any():
            pred_true += 1
    return pred_true / N
    
def recall_at_k_t2i(sim, k):
    pred_true = 0
    N5 = sim.size(0)
    for i in range(N5):
        pred = sim[i]
        topk = pred.topk(k).indices
        true_img_idx = i // 5
        if true_img_idx in topk:
            pred_true += 1
    return pred_true / N5


@torch.no_grad
def zeroshot_classifier(model, classnames, templates, tokenizer, device, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if isinstance(classname, list):
                clsname = classname[0]
            else:
                clsname = classname
            texts = [template.format(clsname) for template in templates]  # format with class

            caption_input = torch.tensor(tokenizer(texts, max_length=args.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)
            caption_input = caption_input.to(model.device)
            walk_short_pos = True
            if args.max_length>100:
                walk_short_pos = False

            class_embeddings = model.get_text_features(caption_input,walk_short_pos=walk_short_pos)

            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            del class_embeddings
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device) # [512, 80]
    return zeroshot_weights

