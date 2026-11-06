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
from model.modeling_clip import CLIPModel, CLIPTextTransformer, CLIPVisionTransformer, CLIPOutput, CLIPAttention, CLIPMLP

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
from typing import Callable

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from torch import nn

import seaborn as sns
import matplotlib.pyplot as plt
import math
from torchvision.ops import roi_align
from utils.utils import plot_heatmap

from loss.ultraclip_loss import ClipInModalityLoss


class GlobalAwareFusion(nn.Module):
    def __init__(self, dim=512, num_heads=8, num_global_tokens=10):
        super().__init__()
        self.num_global_tokens = num_global_tokens
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, patch_tokens, global_tokens):
        """
        patch_tokens: [B, 196, 512]
        global_tokens: [10, 512] (learnable, shared across batch)
        """
        B = patch_tokens.size(0)

        # Expand global tokens to batch
        global_tokens_expanded = global_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 10, 512]

        # Apply cross-attention: Q=patch, K/V=global
        out, _ = self.attn(query=patch_tokens, key=global_tokens_expanded, value=global_tokens_expanded)
        # Output: [B, 196, 512], same as input patch_tokens

        return out



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)




class PureAttentionPoolingBlock(nn.Module):
    """
    Just a pure attn_pooling implementation, without ln_post, without projection, no mormalized_final
    """

    def __init__(
            self,
            context_dim: int,
            n_head: int = 8,
            norm_layer: Callable = LayerNorm,
            need_weights: bool = False
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True, add_zero_attn=True)
        #self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(context_dim)
        self.ln_k = norm_layer(context_dim)
        self.ln_v = norm_layer(context_dim)
        self.need_weights=need_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, output_attn_weights=False, average_attn_weights=True):
        batch_size, seg_length, embed_dim = k.size()
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        if self.need_weights or output_attn_weights:
            out, attn_weights = self.attn(q, k, v, need_weights=True, average_attn_weights=average_attn_weights)
            return out, attn_weights
        else:
            out = self.attn(q, k, v, need_weights=False)[0]
        # we can directly normalize the output, without setting a flag
        #return F.normalize(out, dim=-1)
            return out



class UltraCLIP(CLIPModel):
    config_class = CLIPConfig
    main_input_name = "text_long"

    def __init__(self, config):
        super(CLIPModel, self).__init__(config)
        

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        text_config.eos_token_id = 49407
        text_config.pad_token_id = 49407
        text_config.bos_token_id = 49406

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        
        self.text_model = CLIPTextTransformer(text_config)
        ## frozen text model
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)

        
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.text_filip_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_finegraind = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_hardneg = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        

        self.embed_dim = text_config.hidden_size
        self.world_size = 0

        # Initialize weights and apply final processing
        self.post_init()
        

    def copy_weight(self,):
        with torch.no_grad():
            self.text_filip_projection.weight.data.copy_(self.text_projection.weight.data)  

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=True
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features
    
    def get_image_box_roi_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        box_info=None,
    ) -> torch.FloatTensor:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            interpolate_pos_encoding=True
        )

        bs = pixel_values.shape[0]
        length = vision_outputs[0].shape[1]-1
        h = int(math.sqrt(length))
        w = h

        feature_map = vision_outputs.hidden_states[-2]#[:, 1:, :]
        feature_map = self.forward_without_attn(feature_map)[:, 1:]

        feature_map = self.vision_model.post_layernorm(feature_map)
        feature_map = self.visual_projection(feature_map)
        
        feature_map = feature_map.view(bs, h, w, -1).permute(0, 3, 1, 2)
        x_rois = roi_align(feature_map.type(torch.float32),box_info, (1, 1), 1.0, -1, True)[..., 0, 0]

        return x_rois

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        walk_short_pos: Optional[bool] = True,
        use_bbox: Optional[bool] = False
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pos_flag = walk_short_pos or use_bbox

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            walk_short_pos=pos_flag,
        )
        pooled_output = text_outputs[1]

        if walk_short_pos:
            text_features = self.text_projection(pooled_output)
        else:
            text_features = self.text_filip_projection(pooled_output)           

        return text_features

    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        # print("normed_boxes, ", normed_boxes.shape)
        for boxes in normed_boxes:
            # print("boxes, ", boxes)
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes.type(torch.float32))
        return denormed_boxes

    def forward_without_attn(self, x):
        # get last layer 
        residual = x
        x = self.vision_model.encoder.layers[-1].layer_norm1(x)

        x = F.linear(input=x, weight=self.vision_model.encoder.layers[-1].self_attn.v_proj.weight, bias=self.vision_model.encoder.layers[-1].self_attn.v_proj.bias)
        x = self.vision_model.encoder.layers[-1].self_attn.out_proj(x)
        x = residual+x

        residual = x
        x = self.vision_model.encoder.layers[-1].layer_norm2(x)
        x = self.vision_model.encoder.layers[-1].mlp(x)
        x = residual + x

        return x

    def get_image_dense_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        box_info=None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            interpolate_pos_encoding=True
        )

        bs = pixel_values.shape[0]
        length = vision_outputs[0].shape[1]-1
        h = int(math.sqrt(length))
        w = h

        feature_map = vision_outputs.hidden_states[-2]#[:, 1:, :]
        feature_map = self.forward_without_attn(feature_map)[:, 1:]

        feature_map = self.vision_model.post_layernorm(feature_map)
        feature_map = self.visual_projection(feature_map)

        return feature_map
        


    def clip_loss(self, image_features_long, text_features_long, rank, image):

        image_feat_all_long = torch.cat(nn_dist.all_gather(image_features_long), dim=0) # gather with grad

        if text_features_long is not None:
            text_feat_all_long = torch.cat(nn_dist.all_gather(text_features_long), dim=0)

            sim_i2tl = torch.matmul(image_features_long, text_feat_all_long.T)
            sim_tl2i = torch.matmul(image_feat_all_long, text_features_long.T)
            sim_tl2i = sim_tl2i.T

            sim_i2tl = self.logit_scale.exp() * sim_i2tl
            sim_tl2i = self.logit_scale.exp() * sim_tl2i

        bs = image_features_long.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long).to(image.device)

        loss_itcl = None
        if text_features_long is not None:
            loss_itcl = (
                F.cross_entropy(sim_i2tl, targets, label_smoothing=0.0)
                + F.cross_entropy(sim_tl2i, targets, label_smoothing=0.0)
            ) / 2

        return loss_itcl
    
        
    

    def pairwise_contrastive_loss(self, image_features_long, text_features_long, device, logit_scale=1.0):
        batch_size, c = image_features_long.shape
        labels = torch.eye(batch_size, device=device, dtype=torch.float)#.repeat(batch_size, 1)
        logits_per_image = logit_scale.exp() * image_features_long @ text_features_long.T
        logits_per_text = logit_scale.exp() * text_features_long @ image_features_long.T
        temp1 = F.cross_entropy(logits_per_text, labels)
        temp2 = F.cross_entropy(logits_per_image, labels)

        loss = (temp1+temp2)/2
        return loss
    
    def hard_contrastive_loss(self, image_feat, neg_text_feat, max_neg, device, logit_scale=1.0):
        batch_size, c = image_feat.shape
        neg_text_feat = neg_text_feat.reshape(batch_size, max_neg, -1)
        labels = torch.zeros(batch_size, device=device, dtype=torch.long)#.repeat(batch_size, 1)
        predict = logit_scale.exp() * torch.einsum('bp,bdp->bd', image_feat, neg_text_feat)
        loss = F.cross_entropy(predict, labels)
        return loss
    
    def get_text_embeds(self, box_crop_captions, attention_mask, position_ids, output_attentions, output_hidden_states, return_dict):
        bbox_text_outputs = self.text_model(
                input_ids=box_crop_captions,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                walk_short_pos=True,
        )
        bbox_text_embeds = bbox_text_outputs[1]
        bbox_text_embeds = self.text_projection(bbox_text_embeds)
        bbox_text_embeds = bbox_text_embeds / bbox_text_embeds.norm(p=2, dim=-1, keepdim=True)
        return bbox_text_embeds
    
    def get_better_dense_features(self, image, vision_outputs):
        feature_map = vision_outputs.hidden_states[-2]
        feature_map = self.forward_without_attn(feature_map)[:, 1:]
        feature_map = self.vision_model.post_layernorm(feature_map)
        feature_map = self.visual_projection(feature_map)
        
        bs = image.shape[0]
        length = vision_outputs[0].shape[1]-1
        h = int(math.sqrt(length))
        w = h
        feature_map = feature_map.view(bs, h, w, -1).permute(0, 3, 1, 2)
        
        return feature_map
    

    def bbox_iou(self, box1, box2):
        """
        box1: [..., 4]  (x1,y1,x2,y2)
        box2: [..., 4]
        return: IoU [..., ...]
        """
        # Expand dims to broadcast
        box1 = box1.unsqueeze(-2)  # [..., 1, 4]
        box2 = box2.unsqueeze(-3)  # [..., 1, 4] -> [..., N, 1, 4] broadcast

        # Intersection
        inter_x1 = torch.max(box1[..., 0], box2[..., 0])
        inter_y1 = torch.max(box1[..., 1], box2[..., 1])
        inter_x2 = torch.min(box1[..., 2], box2[..., 2])
        inter_y2 = torch.min(box1[..., 3], box2[..., 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union = area1 + area2 - inter_area

        return inter_area / union.clamp(min=1e-6)  # IoU

    def batch_bbox_similarity(self,batch_box, batch_label):
        """
        batch_box: [B, N, 4]
        batch_label: [B, N, 1]
        return: [B, B] similarity matrix
        """
        B, N, _ = batch_box.shape
        sim_matrix = torch.zeros(B, B)
        
        for i in range(B):
            for j in range(B):
                # 标签匹配 mask
                label_i = batch_label[i]  # [N, 1]
                label_j = batch_label[j]  # [N, 1]
                
                mask = (label_i == label_j.T).float()  # [N, N]
                
                # IoU
                ious = self.bbox_iou(batch_box[i], batch_box[j])  # [N, N]
                
                # 只保留同类的 IoU
                ious = ious * mask
                
                # 最大匹配作为相似度
                sim = ious.max(dim=1)[0].mean()  # 可以用平均最大 IoU
                if i == j:
                    sim = 1.0
                sim_matrix[i, j] = sim
                
        return sim_matrix

    
    def xyxy_to_xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        (x1, y1, x2, y2) --> (xc, yc, w, h)

        Args:
            boxes: shape [N, 4], [x1, y1, x2, y2]
        Returns:
            shape [N, 4], [xc, yc, w, h]
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([xc, yc, w, h], dim=1)
    
    def coord_to_token_index(self, row, col, H=32, W=32):
        """
        know row、col --> calculate token_index after flatten
        Args:
            row: int, 
            col: int, 
            H, W: int, int
        Returns:
            token_index: int
        """
        return row * W + col
    
    def mse_loss(self, text_image_similarity_map, bbox_vfm_similarity_map):
        loss = F.mse_loss(text_image_similarity_map, bbox_vfm_similarity_map)
        return loss
    
    def remap_to_image(self, token_index, image_size, feature_map_size):
        downsample_factor = image_size // feature_map_size
        row = token_index // feature_map_size
        col = token_index % feature_map_size
        original_row = row * downsample_factor
        original_col = col * downsample_factor
        return original_row, original_col
    
    def mse_kldiv_loss(self, text_image_similarity_map, bbox_vfm_similarity_map):
        loss = F.mse_loss(text_image_similarity_map, bbox_vfm_similarity_map) 
        # + 0.1 * F.kl_div(
        #     F.log_softmax(bbox_vfm_similarity_map, dim=-1),
        #         F.softmax(text_image_similarity_map, dim=-1),
        #     reduction='batchmean'
        # )
        return loss
    
    def plt_vis_vfm_vlm(self, image, center_token_idx, text_image_similarity_map, bbox_vfm_similarity_map):
        
        img_resolution = image.shape[2]
        patch_size = 16
        H, W = img_resolution // patch_size, img_resolution // patch_size
        # img_resolution = 512
        rand_idx = torch.randint(0, len(center_token_idx), (1,))
        center_token_idx_vis = center_token_idx[rand_idx].detach().cpu().numpy()
        text_image_similarity_map_vis = text_image_similarity_map[rand_idx].view(H,W).detach().cpu().numpy()
        bbox_vfm_similarity_map_vis = bbox_vfm_similarity_map[rand_idx].view(H,W).detach().cpu().numpy()
        
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        
        image_vis = image[rand_idx].squeeze(0).detach().cpu().numpy()
        
        image_vis = np.transpose(image_vis, (1, 2, 0))
        image_vis = image_vis * std + mean   # 反标准化到 [0,1]
        image_vis = (image_vis * 255).clip(0,255).astype(np.uint8)
        original_row, original_col = self.remap_to_image(center_token_idx_vis, img_resolution, H)

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))  
        ax[0].imshow(image_vis)
        ax[0].axis('off')
        ax[0].set_title('Raw Image')
        
        ax[0].scatter(original_col, original_row, color='red', s=100, edgecolor='yellow', marker='o')  # 标记位置

        im1 = ax[1].imshow(bbox_vfm_similarity_map_vis, cmap='jet', interpolation='nearest')
        ax[1].axis('off')
        ax[1].set_title('vfm similarities')
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(text_image_similarity_map_vis, cmap='jet', interpolation='nearest')
        ax[2].axis('off')
        ax[2].set_title('vlm similarities')
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        return fig
    
    
    def forward(
        self,
        
        # general input
        general_desc: Optional[torch.LongTensor] = None,
        general_desc_spatial: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        image_vfm: Optional[torch.FloatTensor] = None,
        
        # box input
        box_infos: Optional[torch.FloatTensor] = None,
        box_orig_captions: Optional[torch.LongTensor] = None,
        box_crop_captions: Optional[torch.LongTensor] = None,
        box_anatomy_captions: Optional[torch.LongTensor] = None,
        box_anatomy_labels: Optional[torch.LongTensor] = None,
        box_nums: Optional[torch.LongTensor] = None,
        box_crop_images: Optional[torch.FloatTensor] = None,
        
        global_aware: Optional[bool] = False,
        global_local_constrain: Optional[bool] = False,
        attention_pooling: Optional[bool] = False,
        attention_model: Optional[nn.Module] = None,
        learnable_global_tokens: Optional[torch.Tensor] = None,
        
        # hard neg input
        hard_infos: Optional[torch.FloatTensor] = None,
        neg_texts: Optional[torch.LongTensor] = None,
        neg_nums: Optional[torch.LongTensor] = None,
        
        # intra-modal semantic loss input
        captions: Optional[torch.LongTensor] = None,
        captions_spatial: Optional[torch.LongTensor] = None,
        sbert=None,
        cal_semantic_loss=None,
        
        # vfm input
        use_vfm: Optional[bool] = False,
        vfm: Optional[torch.FloatTensor] = None,
        
        
        # other input
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        
    ) -> Union[Tuple, CLIPOutput]:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0 
        
        # print(image.shape)
        vision_outputs = self.vision_model(
                pixel_values=image,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                interpolate_pos_encoding=True
        )
        
        
        # get image embeds
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # general text embeds
        general_text_embeds = self.get_text_embeds(general_desc, attention_mask, position_ids, output_attentions, output_hidden_states, return_dict)
        general_spatial_text_embeds = self.get_text_embeds(general_desc_spatial, attention_mask, position_ids, output_attentions, output_hidden_states, return_dict)
        
        # contrastive loss
        loss_itcl_spatial = self.clip_loss(image_embeds, general_spatial_text_embeds, rank, image)
        loss_itcl = self.clip_loss(image_embeds, general_text_embeds, rank, image)
        loss = loss_itcl + loss_itcl_spatial

        # get better low-level features
        feature_map = self.get_better_dense_features(image, vision_outputs) 
        
        # box feature extraction  
        bs = image.shape[0]
        box_size = box_infos.shape[-1]
        box_infos = box_infos.reshape(bs, -1, box_size)
        original_bboxes = self._denormalize_boxes(box_infos, feature_map)
        x_rois = roi_align(feature_map.type(torch.float32), original_bboxes, (1, 1), 1.0, -1, True)[..., 0, 0]
        
        bbox_vision_outputs = x_rois.type(torch.bfloat16)
        bbox_image_embeds = bbox_vision_outputs
        bbox_image_embeds = bbox_image_embeds / bbox_image_embeds.norm(p=2, dim=-1, keepdim=True)   
        
        
        # box text embeds
        bbox_text_embeds_anatomy = self.get_text_embeds(box_anatomy_captions, attention_mask, position_ids, output_attentions, output_hidden_states, return_dict)
        bbox_text_embeds_crop = self.get_text_embeds(box_crop_captions, attention_mask, position_ids, output_attentions, output_hidden_states, return_dict)
        bbox_text_embeds_orig = self.get_text_embeds(box_orig_captions, attention_mask, position_ids, output_attentions, output_hidden_states, return_dict)
        
        # use anatomy name to describe
        bbox_text_embeds = bbox_text_embeds_anatomy

        # box related loss
        box_loss_weight = 0.5
        bs = box_nums.shape[0]
        resolution = image.shape[2]
        bbox_size = int(bbox_text_embeds.shape[0]/bs)
        box_weight = torch.zeros([bs, bbox_size], device=bbox_text_embeds.device)
        
        per_img_box_num = int(bbox_image_embeds.shape[0] / image_embeds.shape[0])
        image_expand = image.unsqueeze(1).repeat(1, per_img_box_num, 1, 1, 1)
        image_expand = image_expand.reshape(bs*per_img_box_num, 3, resolution, resolution)
        for i in range(bs):
            valid_count = int(box_nums[i])
            box_weight[i][:valid_count] = 1
        
        box_weight = box_weight.reshape(1, bbox_text_embeds.shape[0]).squeeze()
        select_index = box_weight.nonzero()
        bbox_text_embeds_selected = bbox_text_embeds[select_index,:].squeeze()
        bbox_image_embeds_selected = bbox_image_embeds[select_index,:].squeeze()
        bbox_text_embeds_selected_crop = bbox_text_embeds_crop[select_index,:].squeeze()
        bbox_text_embeds_selected_orig = bbox_text_embeds_orig[select_index,:].squeeze()
        
        image_expand_selected = image_expand[select_index,:].squeeze()
        
        
        loss_bbox_itcl = self.pairwise_contrastive_loss(bbox_image_embeds_selected, bbox_text_embeds_selected, bbox_image_embeds.device, self.logit_scale_finegraind)
        loss_bbox_itcl_crop = self.pairwise_contrastive_loss(bbox_image_embeds_selected, bbox_text_embeds_selected_crop, bbox_image_embeds.device, self.logit_scale_finegraind)
        loss_bbox_itcl_orig = self.pairwise_contrastive_loss(bbox_image_embeds_selected, bbox_text_embeds_selected_orig, bbox_image_embeds.device, self.logit_scale_finegraind)
        # loss_bbox_itcl = loss_bbox_itcl + 0.1*loss_bbox_itcl_crop + 0.1*loss_bbox_itcl_orig
        # loss_bbox_itcl = loss_bbox_itcl + loss_bbox_itcl_crop + loss_bbox_itcl_orig
        loss_bbox_itcl = loss_bbox_itcl + loss_bbox_itcl_orig 
        
        loss = loss + box_loss_weight*loss_bbox_itcl 
        
        
        # patch token similarity constraint between vlm and vfm
        H, W = feature_map.shape[2], feature_map.shape[3]
        bbox_text_embeds = bbox_text_embeds.reshape(bs, bbox_text_embeds.shape[0]//bs, -1)
        original_bboxes_reshape = torch.stack(original_bboxes, dim=0)
        vfm_feature_map = vfm.get_intermediate_layers(image_vfm, reshape=True)[0].flatten(2, 3)
        normed_vfm_feats= F.normalize(vfm_feature_map, dim=1)
        vfm_similarity = torch.einsum("b c m, b c n -> b m n", normed_vfm_feats, normed_vfm_feats)
        vfm_similarity = vfm_similarity.unsqueeze(1).repeat(1, per_img_box_num, 1, 1)
        vfm_similarity = vfm_similarity.reshape(vfm_similarity.shape[0]*vfm_similarity.shape[1], vfm_similarity.shape[2], vfm_similarity.shape[3])
        
        # get bbox center
        x1, y1, x2, y2 = original_bboxes_reshape[...,0], original_bboxes_reshape[...,1], original_bboxes_reshape[...,2], original_bboxes_reshape[...,3]
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        row = yc.long().clamp(0, H-1)
        col = xc.long().clamp(0, W-1)
        
        # get vfm feature using the center
        center_token_idx = self.coord_to_token_index(row, col, H=H, W=W)
        center_token_idx = center_token_idx.reshape(center_token_idx.shape[0]*center_token_idx.shape[1])
        bbox_vfm_similarity_map = vfm_similarity[torch.arange(bs*per_img_box_num), center_token_idx, :] 
        
        # Calculate text-image similarity map for each bbox
        vlm_feature_map = feature_map.flatten(2, 3)
        vlm_feature_map = F.normalize(vlm_feature_map, dim=1)
        bbox_text_embeds = F.normalize(bbox_text_embeds, dim=1)
        
        # Compute similarity map
        text_image_similarity_map = torch.einsum("b c n, b t c -> b t n", vlm_feature_map, bbox_text_embeds).reshape(bs*per_img_box_num, -1)
        
        # perform selection
        text_image_similarity_map = text_image_similarity_map[select_index,:].squeeze()
        bbox_vfm_similarity_map = bbox_vfm_similarity_map[select_index,:].squeeze()
        center_token_idx = center_token_idx[select_index].squeeze()
        
        # adding pow
        power = 4.0
        bbox_vfm_similarity_map = torch.pow(bbox_vfm_similarity_map, power)
        
        loss_vfm_vlm = self.mse_kldiv_loss(text_image_similarity_map, bbox_vfm_similarity_map)
        
        loss = loss + loss_vfm_vlm
        
        # visualize bbox center feature similarity
        text_image_similarity_map_vis = torch.pow(text_image_similarity_map, power)
        fig = self.plt_vis_vfm_vlm(image_expand_selected, center_token_idx, text_image_similarity_map_vis, bbox_vfm_similarity_map)
        
        # semantic loss
        
        # get bbox similarity
        box_anatomy_labels_reshape = box_anatomy_labels.reshape(bs, 5, -1).squeeze().to(device=self.device)
        
        bbox_similarity = self.batch_bbox_similarity(original_bboxes_reshape, box_anatomy_labels_reshape).to(device=self.device)
        
        if sbert is not None:
            # in_modality_loss_weight = 0.5
            in_modality_loss_weight = 1
            
            semantic_features = sbert.encode(sentences=captions_spatial, show_progress_bar=False)
            semantic_features = torch.from_numpy(semantic_features)
            semantic_features = semantic_features.to(device=self.device, non_blocking=True)
            semantic_loss = cal_semantic_loss(image_embeds, general_spatial_text_embeds, self.logit_scale, output_dict=True, semantic_features=semantic_features, bbox_similarity=bbox_similarity)
            # semantic_loss = cal_semantic_loss(image_embeds, general_spatial_text_embeds, self.logit_scale, output_dict=True, semantic_features=semantic_features, bbox_similarity=None)
            #  semantic_loss = cal_semantic_loss(image_embeds, general_spatial_text_embeds, self.logit_scale, output_dict=True, semantic_features=semantic_features)
            
            in_modality_loss = semantic_loss['inModality_loss']
            loss = loss + in_modality_loss_weight*in_modality_loss
            # print('in_modality_loss', in_modality_loss_weight*in_modality_loss)
        
        text_level_similarity = general_spatial_text_embeds @ general_spatial_text_embeds.T
        text_level_similarity = text_level_similarity.detach().cpu().numpy()
        heatmap_text = plot_heatmap(text_level_similarity, title="Text Level Similarity")
        
        image_level_similarity = image_embeds @ image_embeds.T
        image_level_similarity = image_level_similarity.detach().cpu().numpy()
        heatmap_img = plot_heatmap(image_level_similarity, title="Image Level Similarity")
        
        semantic_features_similarity = semantic_features @ semantic_features.T
        semantic_features_similarity = semantic_features_similarity.detach().cpu().numpy()
        heatmap_semantic = plot_heatmap(semantic_features_similarity, title="Semantic Features Similarity")
        
        heatmap_bbox = plot_heatmap(bbox_similarity.detach().cpu().numpy(), title="Bbox Similarity")
        
        vis = {
            "text_level_similarity": heatmap_text,
            "image_level_similarity": heatmap_img,
            "semantic_features_similarity": heatmap_semantic,
            "bbox_similarity": heatmap_bbox,
            "vis_sim": fig
        }
        return loss, vis


class FeatureFusionAttention(nn.Module):
    def __init__(self, d_model=512, num_layers=2, nhead=8, dim_feedforward=512):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # Important for (N, Seq_Len, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, img_feats, bbox_feats):
        """
        img_feats:    [N, D]
        bbox_feats:   [N, Box_N, D]
        returns:
        global_aware_bbox_feats: [N, Box_N, D]
        """
        N, Box_N, D = bbox_feats.shape
        

        # Expand img_feats to [N, 1, D] so it can be prepended
        cls_tokens = img_feats.unsqueeze(1)  # [N, 1, D]

        # Concatenate along sequence dimension
        tokens = torch.cat([cls_tokens, bbox_feats], dim=1)  # [N, 1 + Box_N, D]

        # Pass through transformer
        tokens_out = self.encoder(tokens)  # [N, 1 + Box_N, D]

        # Extract only bbox features (skip the cls token at index 0)
        global_aware_feats = tokens_out[:, 1:, :]  # [N, Box_N, D]
        
        return global_aware_feats