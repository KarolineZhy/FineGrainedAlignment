import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


import torch
import torch.nn.functional as F
import torch.distributed.nn as nn_dist

# def clip_loss(image_features_long, text_features_long, text_features_short, rank, image, logit_scale):


#     image_feat_all_long = torch.cat(nn_dist.all_gather(image_features_long), dim=0)#gather with grad

#     if text_features_long is not None:
#         text_feat_all_long = torch.cat(nn_dist.all_gather(text_features_long), dim=0)

#     text_feat_all_short = torch.cat(nn_dist.all_gather(text_features_short), dim=0)
    
#     if text_features_long is not None:
#         sim_i2tl = torch.matmul(image_features_long, text_feat_all_long.T)
#         sim_tl2i = torch.matmul(image_feat_all_long, text_features_long.T)
#         sim_tl2i = sim_tl2i.T

#     sim_i2ts = torch.matmul(image_features_long, text_feat_all_short.T)
#     sim_ts2i = torch.matmul(image_feat_all_long, text_features_short.T)
#     sim_ts2i = sim_ts2i.T

    
#     if text_features_long is not None:
#         sim_i2tl = logit_scale.exp() * sim_i2tl
#         sim_tl2i = logit_scale.exp() * sim_tl2i


#     sim_i2ts = logit_scale.exp() * sim_i2ts
#     sim_ts2i = logit_scale.exp() * sim_ts2i

    
#     bs = image_features_long.size(0)
#     targets = torch.linspace(rank * bs,rank * bs + bs - 1, bs, dtype=torch.long).to(image.device)

#     loss_itcl = None
#     if text_features_long is not None:
#         loss_itcl = (
#                 F.cross_entropy(sim_i2tl, targets, label_smoothing=0.0)
#                 + F.cross_entropy(sim_tl2i, targets, label_smoothing=0.0)
#             ) / 2
    
#     loss_itcs = (
#             F.cross_entropy(sim_i2ts, targets, label_smoothing=0.0)
#             + F.cross_entropy(sim_ts2i, targets, label_smoothing=0.0)
#         ) / 2

#     return loss_itcl, loss_itcs

def clip_loss(self, image_features_long, text_features_long, rank, image):
    
    image_feat_all_long = self.all_gather(image_features_long)

    if text_features_long is not None:
        text_feat_all_long = self.all_gather(text_features_long)

        sim_i2tl = torch.matmul(image_features_long, text_feat_all_long.transpose(0, 1))
        sim_tl2i = torch.matmul(image_feat_all_long, text_features_long.transpose(0, 1))
        sim_tl2i = sim_tl2i.transpose(0, 1)

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


def pairwise_contrastive_loss(image_features_long, text_features_long, device, logit_scale=1.0):
    batch_size, c = image_features_long.shape
    labels = torch.eye(batch_size, device=device, dtype=torch.float)#.repeat(batch_size, 1)
    logits_per_image = logit_scale.exp() * image_features_long @ text_features_long.T
    logits_per_text = logit_scale.exp() * text_features_long @ image_features_long.T
    temp1 = F.cross_entropy(logits_per_text, labels)
    temp2 = F.cross_entropy(logits_per_image, labels)

    loss = (temp1+temp2)/2
    return loss

def hard_contrastive_loss(image_features_long, text_features_long, device, logit_scale=1.0):
    batch_size, c = image_features_long.shape
    text_features_long = text_features_long.reshape(batch_size, 11, -1)
    labels = torch.zeros(batch_size, device=device, dtype=torch.long)#.repeat(batch_size, 1)
    predict = logit_scale.exp() * torch.einsum('bp,bdp->bd', image_features_long, text_features_long)
    loss = F.cross_entropy(predict, labels)
    return loss

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features



class ClipInModalityLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            alpha=1.0,
            beta=0.5,
            n_epoch=30,
            nl_semantic_supervision=False,
            separate_text=True,
            separate_image=False
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        
        self.alpha = alpha
        self.beta = beta
        
        self.nl_semantic_supervision = nl_semantic_supervision
        self.separate_text = separate_text
        self.separate_image = separate_image

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = image_features @ all_image_features.T
                logits_per_text = text_features @ all_text_features.T

                logits_image_text = image_features @ text_features.T
                size = logits_per_image.shape[0]

                logscale_logits_image_text = logit_scale * image_features @ all_text_features.T
                logscale_logits_text_image = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = all_image_features @ all_image_features.T
                logits_per_text = all_text_features @ all_text_features.T

                logits_image_text = all_image_features @ all_text_features.T
                size = logits_per_image.shape[0]
                
                logscale_logits_image_text = logit_scale * all_image_features @ all_text_features.T
                logscale_logits_text_image = logscale_logits_image_text.T 
        else:
            logits_per_image = image_features @ image_features.T
            logits_per_text = text_features @ text_features.T
            
            logits_image_text = image_features @ text_features.T
            size = logits_per_image.shape[0]
            
            logscale_logits_image_text = logit_scale * image_features @ text_features.T
            logscale_logits_text_image = logit_scale * text_features @ image_features.T
        
        
        return logits_per_image, logits_per_text, logits_image_text, logscale_logits_image_text, logscale_logits_text_image

    def forward(self, image_features, text_features, logit_scale, output_dict=False, semantic_features=None, bbox_similarity=None):
        device = image_features.device
        logits_per_image, logits_per_text, logits_image_text, logscale_logits_image_text, logscale_logits_text_image = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        if self.nl_semantic_supervision:

            semantic_features = semantic_features / semantic_features.norm(dim=-1, keepdim=True)
            semantic_sim = semantic_features @ semantic_features.T
            if bbox_similarity is not None:
                
                # Original
                semantic_sim = (semantic_sim + bbox_similarity) / 2
            
                # Ablation A
                # 1.5 * semantic_sim + 0.5 * bbox_similarity more semantic
                # semantic_sim = 2 * semantic_sim + 0.5 * bbox_similarity
                # semantic_sim = semantic_sim / 2
                
                # Ablation B
                # 0.5 * semantic_sim + 1.5 * bbox_similarity more bbox
                # semantic_sim = 0.5 * semantic_sim + 2 * bbox_similarity
                # semantic_sim = semantic_sim / 2
                # semantic_sim = bbox_similarity + bbox_similarity
                # semantic_sim = semantic_sim / 2
                
                
                
            # interagted_similarity = (semantic_sim + bbox_similarity) / 2
            semantic_sim = 1 - semantic_sim

            device = logits_per_image.get_device()
            size = logits_per_image.shape[0]

            logits_paired_text_image = torch.mul(logits_image_text, torch.eye(size).to(device))

            if self.separate_text:
                logits_per_text = torch.mul(logits_per_text, semantic_sim)
                logits_per_text = logits_per_text + logits_paired_text_image
                logscale_logits_per_text = logit_scale * logits_per_text

            if self.separate_image:
                logits_per_image = torch.mul(logits_per_image, semantic_sim)
                logits_per_image = logits_per_image + logits_paired_text_image
                logscale_logits_per_image = logit_scale * logits_per_image

            if self.separate_text and self.separate_image:
                inModality_loss = self.beta*(F.cross_entropy(logscale_logits_per_text, labels) +
                        F.cross_entropy(logscale_logits_per_image, labels))
            elif self.separate_text:
                inModality_loss = self.beta*(F.cross_entropy(logscale_logits_per_text, labels))
            elif self.separate_image:
                inModality_loss = self.beta*(F.cross_entropy(logscale_logits_per_image, labels))
            
        else:
            logscale_logits_image = logit_scale * logits_per_image
            logscale_logits_text = logit_scale * logits_per_text

            inModality_loss = self.beta*(
                F.cross_entropy(logscale_logits_image, labels) +
                F.cross_entropy(logscale_logits_text, labels)
                )
        
        clip_loss = self.alpha*((
            F.cross_entropy(logscale_logits_image_text, labels) +
            F.cross_entropy(logscale_logits_text_image, labels)))


        total_loss = inModality_loss + clip_loss
        return {"total_loss": total_loss, "clip_loss": clip_loss, "inModality_loss": inModality_loss} if output_dict else total_loss