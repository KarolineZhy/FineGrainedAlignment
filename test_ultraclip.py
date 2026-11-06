#### EVALUATE UltraCLIP - Unified test script for US/MRI/CT
#### Supports batch evaluation across checkpoints with metrics-only mode
import argparse
from argparse import ArgumentParser
from typing import Any
from collections import defaultdict
import re

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, CLIPImageProcessor
from pyhocon import ConfigFactory
import lightning as L
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.ultraclip_lighting import UltraCLIPLightningModule
import os
from torchvision.transforms import transforms as T
from dataset.transform import image_transform
import glob
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from skimage.transform import resize
import json


# ========================= Modality-specific configurations =========================

MODALITY_CONFIG = {
    "us": {
        "single_test_path": "{data_root}/ultrasound_data/demo9/1747325223.072807.png",
        "single_test_captions": [
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery.",
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery, jugular vein.",
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery, jugular vein, trachea.",
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery, jugular vein, thyroid gland.",
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery, jugular vein, thyroid gland, trachea.",
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery, thyroid gland.",
        ],
        "referring_captions": ["carotid artery", "jugular vein", "thyroid gland", "trachea"],
        "i2t_captions": [
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery.",
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery, jugular vein.",
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery, thyroid gland, trachea.",
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery, jugular vein, thyroid gland.",
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery, jugular vein, thyroid gland, trachea.",
            "The thyroid ultrasound image contains the following anatomy structures: carotid artery, thyroid gland.",
        ],
        "retrieval_gt_path": "{data_root}/ultrasound_data/rebuttal/new_retrieval_test/gt_i2t.json",
        "retrieval_folder": "{data_root}/ultrasound_data/rebuttal/new_retrieval_test",
        "retrieval_data_root": "{data_root}",
        "retrieval_t2i_gt_path": "{data_root}/ultrasound_data/rebuttal/new_retrieval_test/gt_t2i.json",
        "retrieval_t2i_folder": "{data_root}/ultrasound_data/rebuttal/new_retrieval_test",
        "image_retrieval_text": "The thyroid gland is left and inferior to the carotid artery and not contiguous with the jugular vein. The carotid artery is right and superior to the thyroid gland, and contiguous to the jugular vein. The jugular vein is superior and right to the thyroid gland, contiguous but superior to the carotid artery.",
        "image_retrieval_folder": "{data_root}/ultrasound_data/retrieval",
        "image_retrieval_image": "{data_root}/ultrasound_data/retrieval/1747994803.944078.png",
        "image_retrieval_source_folder": "{data_root}/ultrasound_data/demo8",
        "subset_json_path": "{data_root}/ultrasound_data/rebuttal/new_retrieval_test/retrieval_t2i_subset.json",
        "subset_folder": "{data_root}/ultrasound_data/rebuttal/new_retrieval_test",
        "seg_test_cases": [
            {"caption": "carotid artery", "seg_image_folder": "{data_root}/ultrasound_data/segmentation/carotid-artery/jpg", "seg_gt_folder": "{data_root}/ultrasound_data/segmentation/carotid-artery/seg"},
            {"caption": "thyroid gland", "seg_image_folder": "{data_root}/ultrasound_data/segmentation/thyroid-gland/jpg", "seg_gt_folder": "{data_root}/ultrasound_data/segmentation/thyroid-gland/seg"},
            {"caption": "carotid artery", "seg_image_folder": "{data_root}/ultrasound_data/CCA/US images", "seg_gt_folder": "{data_root}/ultrasound_data/CCA/Expert mask images"},
        ],
        "seg_gt_assertion": "mask",
        "seg_resize_target": "similarity",
        "seg_skip_single_value": False,
    },
    "mri": {
        "single_test_path": None,
        "single_test_captions": [],
        "referring_captions": [],
        "retrieval_gt_path": "{data_root}/mri_data_retrieval_test/result_json.json",
        "retrieval_folder": None,
        "retrieval_data_root": "{data_root}",
        "retrieval_t2i_gt_path": "{data_root}/mri_data_retrieval_test/result_json.json",
        "retrieval_t2i_folder": None,
        "image_retrieval_text": None,
        "image_retrieval_folder": None,
        "image_retrieval_image": None,
        "image_retrieval_source_folder": None,
        "subset_json_path": "{data_root}/mri_data_retrieval_test/retrieval_subset.json",
        "subset_folder": "{data_root}/mri_data_retrieval_test",
        "seg_test_cases": [
            {"caption": "Liver", "seg_image_folder": "{data_root}/mri_seg_data_test/Liver/img", "seg_gt_folder": "{data_root}/mri_seg_data_test/Liver/seg"},
            {"caption": "Spleen", "seg_image_folder": "{data_root}/mri_seg_data_test/Spleen/img", "seg_gt_folder": "{data_root}/mri_seg_data_test/Spleen/seg"},
            {"caption": "Kidney", "seg_image_folder": "{data_root}/mri_seg_data_test/Kidney/img", "seg_gt_folder": "{data_root}/mri_seg_data_test/Kidney/seg"},
        ],
        "seg_gt_assertion": "label",
        "seg_resize_target": "gt",
        "seg_skip_single_value": False,
    },
    "ct": {
        "single_test_path": None,
        "single_test_captions": [
            "The abdominal CT image contains the following anatomy structures: liver.",
            "The abdominal CT image contains the following anatomy structures: liver, spleen.",
            "The abdominal CT image contains the following anatomy structures: liver, spleen, stomach.",
            "The abdominal CT image contains the following anatomy structures: liver, spleen, kidney.",
            "The abdominal CT image contains the following anatomy structures: liver, spleen, kidney, stomach.",
            "The abdominal CT image contains the following anatomy structures: liver, spleen, kidney, stomach, gallbladder.",
            "The abdominal CT image contains the following anatomy structures: liver, spleen, kidney, stomach, gallbladder, pancreas.",
            "The abdominal CT image contains the following anatomy structures: liver, spleen, kidney, stomach, pancreas.",
            "The abdominal CT image contains the following anatomy structures: liver, spleen, stomach, pancreas.",
        ],
        "referring_captions": ["liver", "spleen", "kidney", "stomach", "gallbladder", "pancreas"],
        "retrieval_gt_path": "{data_root}/ct_data_retrieval_test/result_json.json",
        "retrieval_folder": None,
        "retrieval_data_root": "{data_root}",
        "retrieval_t2i_gt_path": "{data_root}/ct_data_retrieval_test/result_json.json",
        "retrieval_t2i_folder": None,
        "image_retrieval_text": None,
        "image_retrieval_folder": None,
        "image_retrieval_image": None,
        "image_retrieval_source_folder": None,
        "subset_json_path": "{data_root}/ct_data_retrieval_test/retrieval_subset.json",
        "subset_folder": "{data_root}/ct_data_retrieval_test",
        "seg_test_cases": [
            {"caption": "Liver", "seg_image_folder": "{data_root}/ct_seg_data_sel/Liver/img", "seg_gt_folder": "{data_root}/ct_seg_data_sel/Liver/seg"},
            {"caption": "Spleen", "seg_image_folder": "{data_root}/ct_seg_data_sel/Spleen/img", "seg_gt_folder": "{data_root}/ct_seg_data_sel/Spleen/seg"},
            {"caption": "Kidney", "seg_image_folder": "{data_root}/ct_seg_data_sel/Kidney/img", "seg_gt_folder": "{data_root}/ct_seg_data_sel/Kidney/seg"},
            {"caption": "Stomach", "seg_image_folder": "{data_root}/ct_seg_data_sel/Stomach/img", "seg_gt_folder": "{data_root}/ct_seg_data_sel/Stomach/seg"},
            {"caption": "Gallbladder", "seg_image_folder": "{data_root}/ct_seg_data_sel/Gallbladder/img", "seg_gt_folder": "{data_root}/ct_seg_data_sel/Gallbladder/seg"},
            {"caption": "Pancreas", "seg_image_folder": "{data_root}/ct_seg_data_sel/Pancreas/img", "seg_gt_folder": "{data_root}/ct_seg_data_sel/Pancreas/seg"},
        ],
        "seg_gt_assertion": "label",
        "seg_resize_target": "gt",
        "seg_skip_single_value": True,
    },
}


# ========================= Retrieval metrics =========================

def compute_retrieval_metrics(image_to_text_dict, text_to_image_dict, text_embeds, image_embeds, ks=[1, 5, 10]):
    texts = list(text_to_image_dict.keys())
    images = list(image_to_text_dict.keys())
    sim = cosine_similarity(text_embeds, image_embeds)

    results = {f"R@{k}": [] for k in ks}
    ranks = []

    for i, txt in enumerate(texts):
        true_imgs = text_to_image_dict[txt]
        true_indices = [images.index(img) for img in true_imgs if img in images]
        sorted_indices = np.argsort(-sim[i])
        rank = min([np.where(sorted_indices == ti)[0][0] for ti in true_indices])
        ranks.append(rank + 1)
        for k in ks:
            results[f"R@{k}"].append(1 if rank < k else 0)

    metrics = {k: np.mean(v) * 100 for k, v in results.items()}
    metrics["MeanRank"] = float(np.mean(ranks))
    metrics["MedianRank"] = float(np.median(ranks))
    return metrics


# ========================= GT assertion helpers =========================

def check_seg_gt_match(seg_image_name, seg_gt_name, assertion_type):
    if assertion_type == "mask":
        assert (seg_image_name == seg_gt_name.replace(".png", "").replace("mask_", "")
                or seg_image_name == seg_gt_name), \
            f"Mismatch: {seg_image_name} vs {seg_gt_name}"
    elif assertion_type == "label":
        assert seg_image_name == seg_gt_name.replace("label", "image"), \
            f"Mismatch: {seg_image_name} vs {seg_gt_name}"
    else:
        assert seg_image_name == seg_gt_name, \
            f"Mismatch: {seg_image_name} vs {seg_gt_name}"


# ========================= Checkpoint utilities =========================

def parse_epoch_from_path(ckpt_path):
    """Extract epoch number from checkpoint filename like 'epoch=7-step=6904.ckpt'."""
    m = re.search(r'epoch=(\d+)', os.path.basename(ckpt_path))
    return int(m.group(1)) if m else -1


def get_checkpoint_list(conf, checkpoint_dir=None):
    """Return list of (epoch, ckpt_path) sorted by epoch."""
    if checkpoint_dir:
        ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, '*.ckpt')))
        return sorted([(parse_epoch_from_path(c), c) for c in ckpts], key=lambda x: x[0])
    else:
        path = conf.test.model_path
        return [(parse_epoch_from_path(path), path)]


def resolve_data_root(value, data_root):
    """Replace the {data_root} placeholder in every path of a modality config."""
    if isinstance(value, dict):
        return {k: resolve_data_root(v, data_root) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_data_root(v, data_root) for v in value]
    if isinstance(value, str):
        return value.replace("{data_root}", data_root)
    return value


# ========================= Single checkpoint evaluation =========================

def evaluate_checkpoint(model_path, conf, mcfg, tokenizer, img_preprocess, save_vis=False,
                        save_dir=None, multi_ckpt=False):
    """Evaluate a single checkpoint. Returns a dict of all metrics.

    Per-sample retrieval results are written to ``save_dir`` (flat layout for
    a single checkpoint, ``save_dir/epoch{N}/`` when multiple checkpoints are
    being evaluated to avoid overwrites). Files written:
      - i2t_retrieval_results.json
      - t2i_retrieval_results.json
      - t2i_subset_retrieval_results.json
    """
    model = UltraCLIPLightningModule.load_from_checkpoint(model_path, config=conf).cuda()
    model.eval()

    epoch = parse_epoch_from_path(model_path)
    metrics = {"model_path": model_path, "epoch": epoch}

    if save_dir is not None:
        if multi_ckpt:
            ckpt_save_dir = os.path.join(save_dir, f"epoch{epoch}")
        else:
            ckpt_save_dir = save_dir
        os.makedirs(ckpt_save_dir, exist_ok=True)
    else:
        ckpt_save_dir = None

    def _dump(filename, payload):
        if ckpt_save_dir is None:
            return
        out = os.path.join(ckpt_save_dir, filename)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved: {out}")

    # ---- I2T Retrieval ----
    if hasattr(conf.test, 'text_retrieval_given_image') and conf.test.text_retrieval_given_image:
        gt_path = mcfg["retrieval_gt_path"]
        with open(gt_path, 'r') as f:
            image_to_text_dict = json.load(f)

        # Detect format: US uses {image_id: [captions]} with a fixed candidate list;
        # MRI/CT uses {image_path: caption} where we derive unique captions from values.
        first_val = next(iter(image_to_text_dict.values()))
        is_us_style = isinstance(first_val, list)

        if is_us_style:
            unique_captions = mcfg.get("i2t_captions", mcfg["single_test_captions"])
            folder_arg = mcfg["retrieval_folder"]  # pass folder, not list of paths
        else:
            unique_captions = sorted(set(image_to_text_dict.values()))
            folder_arg = list(image_to_text_dict.keys())

        text_tokens_retrieval = []
        for caption in unique_captions:
            text_token = torch.tensor(
                tokenizer([caption], max_length=77, padding="max_length", truncation=True).input_ids,
                dtype=torch.long).cuda()
            text_tokens_retrieval.append(text_token)
        text_tokens_retrieval = torch.cat(text_tokens_retrieval, dim=0)

        save_folder = None
        result = model.test_text_retrival_given_image(
            folder_arg, img_preprocess, text_tokens_retrieval, unique_captions,
            image_to_text_dict, save_folder, save_vis=False, data_root=mcfg["retrieval_data_root"])
        if isinstance(result, tuple):
            i2t_correct, i2t_results = result
        else:
            i2t_correct, i2t_results = result, None
        metrics["i2t_accuracy"] = i2t_correct
        print(f"  I2T Accuracy: {i2t_correct:.4f}")
        if i2t_results is not None:
            _dump("i2t_retrieval_results.json", i2t_results)

    # ---- T2I Retrieval ----
    if hasattr(conf.test, 'image_retrieval_given_text') and conf.test.image_retrieval_given_text:
        gt_path = mcfg["retrieval_t2i_gt_path"]
        data_root = mcfg["retrieval_data_root"]
        with open(gt_path, 'r') as f:
            gt_raw = json.load(f)

        # Detect format: US gt is {text: [image_ids]}; MRI/CT gt is {image_path: caption}.
        first_val = next(iter(gt_raw.values()))
        is_us_style = isinstance(first_val, list)

        if is_us_style:
            result = model.test_text_image_retrieval(
                mcfg["retrieval_t2i_folder"], img_preprocess, tokenizer, gt_raw)
            if isinstance(result, tuple):
                t2i_correct, t2i_results = result
            else:
                t2i_correct, t2i_results = result, None
            metrics["t2i_accuracy"] = t2i_correct
            print(f"  T2I Accuracy: {t2i_correct:.4f}")
            if t2i_results is not None:
                _dump("t2i_retrieval_results.json", t2i_results)
        else:
            image_to_text_dict = gt_raw
            text_to_image_dict = defaultdict(list)
            for img, txt in image_to_text_dict.items():
                text_to_image_dict[txt].append(img)

            unique_captions = sorted(text_to_image_dict.keys())
            text_tokens_t2i = []
            for caption in unique_captions:
                text_token = torch.tensor(
                    tokenizer([caption], max_length=77, padding="max_length", truncation=True).input_ids,
                    dtype=torch.long).cuda()
                text_tokens_t2i.append(text_token)
            text_tokens_t2i = torch.cat(text_tokens_t2i, dim=0).cuda()
            text_features = model.encode_text(text_tokens_t2i)

            image_list = list(image_to_text_dict.keys())
            image_tensors = []
            batch_size = 32
            for image_path in image_list:
                img = Image.open(os.path.join(data_root, image_path)).convert("RGB")
                image_tensors.append(img_preprocess(img).unsqueeze(0).cuda())
            image_tensors = torch.cat(image_tensors, dim=0).cuda()

            image_features = []
            for i in range(0, image_tensors.size(0), batch_size):
                batch_tensor = image_tensors[i:i + batch_size].cuda()
                batch_features = model.encode_image(batch_tensor)
                image_features.append(batch_features.cpu())
            image_features = torch.cat(image_features, dim=0)

            text_features_np = text_features.detach().cpu().numpy()
            image_features_np = image_features.detach().cpu().numpy()

            t2i_metrics = compute_retrieval_metrics(image_to_text_dict, text_to_image_dict, text_features_np, image_features_np)
            for k, v in t2i_metrics.items():
                metrics[f"t2i_{k}"] = v
            print(f"  T2I: {t2i_metrics}")

            # Build per-caption results in the same shape the US path emits, so
            # downstream significance tests can read either uniformly.
            sim_mat = cosine_similarity(text_features_np, image_features_np)
            top_k_save = min(10, sim_mat.shape[1])
            t2i_results = {}
            for i, caption in enumerate(unique_captions):
                order = np.argsort(-sim_mat[i])
                topk_idx = order[:top_k_save]
                pred_topk = [image_list[j] for j in topk_idx]
                t2i_results[caption] = {
                    "gt": text_to_image_dict[caption],
                    "pred_top1": image_list[int(order[0])],
                    "pred_topk": pred_topk,
                    "scores": [float(sim_mat[i, j]) for j in topk_idx],
                }
            _dump("t2i_retrieval_results.json", t2i_results)

    # ---- T2I Subset Retrieval ----
    if hasattr(conf.test, 'image_retrieval_given_text_subset') and conf.test.image_retrieval_given_text_subset and mcfg["subset_json_path"]:
        subset_json_path = mcfg["subset_json_path"]
        subset_folder = mcfg["subset_folder"]
        if os.path.exists(subset_json_path):
            with open(subset_json_path, 'r') as f:
                subset_data = json.load(f)
            correct_t2i_subset, t2i_subset_results = model.test_text_image_retrieval_subset(
                subset_folder, img_preprocess, tokenizer, subset_data)
            metrics["t2i_subset_accuracy"] = correct_t2i_subset
            print(f"  T2I Subset Accuracy: {correct_t2i_subset:.4f}")
            if t2i_subset_results is not None:
                _dump("t2i_subset_retrieval_results.json", t2i_subset_results)

    # ---- Segmentation AUROC ----
    if hasattr(conf.test, 'test_segmentation_auroc') and conf.test.test_segmentation_auroc:
        test_cases = mcfg["seg_test_cases"]
        seg_results = {}
        seg_per_sample = {}   # per-image AUROC/AUPR for paired significance testing

        for test_case in test_cases:
            caption = test_case["caption"]
            seg_image_folder = test_case["seg_image_folder"]
            seg_gt_folder = test_case["seg_gt_folder"]

            seg_images = sorted(
                glob.glob(os.path.join(seg_image_folder, '*.jpg')) +
                glob.glob(os.path.join(seg_image_folder, '*.png'))
            )
            seg_gts = sorted(glob.glob(os.path.join(seg_gt_folder, '*.png')))

            if len(seg_images) != len(seg_gts):
                print(f"  WARNING: Image/GT count mismatch for {caption}: {len(seg_images)} vs {len(seg_gts)}")
                continue

            gt_list = []
            seg_list = []
            ids_list = []
            count = 0

            caption_input = torch.tensor(
                tokenizer([caption], max_length=77, padding="max_length", truncation=True).input_ids,
                dtype=torch.long, device='cuda')

            for seg_image, seg_gt in tqdm(zip(seg_images, seg_gts), total=len(seg_images), desc=caption):
                check_seg_gt_match(
                    os.path.basename(seg_image), os.path.basename(seg_gt),
                    mcfg["seg_gt_assertion"])

                img = Image.open(seg_image).convert("RGB")
                img_tensor = img_preprocess(img).unsqueeze(0).cuda()

                gt = Image.open(seg_gt).convert("L")
                gt = np.array(gt)

                if mcfg["seg_skip_single_value"] and np.unique(gt).shape[0] == 1:
                    continue

                count += 1
                similarity = model.test_dense_features(img, img_tensor, caption_input, None, save_vis=False)

                if mcfg["seg_resize_target"] == "similarity":
                    similarity = resize(similarity, gt.shape, order=0, preserve_range=True, anti_aliasing=False)
                else:
                    gt = resize(gt, similarity.shape, order=0, preserve_range=True, anti_aliasing=False)

                gt_list.append(np.array(gt))
                seg_list.append(similarity)
                ids_list.append(os.path.basename(seg_image))

            if len(seg_list) == 0:
                continue

            dataset_key = seg_gt_folder.split("/")[-2]
            group = f"{caption}_{dataset_key}"

            # per-image AUROC/AUPR (paired unit = "<caption>_<dataset>/<image>") for significance.
            # AUROC is undefined when an image's gt has a single class -> stored as None.
            for sid, g, s in zip(ids_list, gt_list, seg_list):
                y_true = (np.array(g) > 0).astype(np.uint8).flatten()
                y_score = np.array(s).flatten()
                if y_true.min() == y_true.max():
                    img_auroc, img_aupr = None, None
                else:
                    img_auroc = float(roc_auc_score(y_true, y_score))
                    img_aupr = float(average_precision_score(y_true, y_score))
                seg_per_sample[f"{group}/{sid}"] = {"auroc": img_auroc, "aupr": img_aupr}

            seg_list = np.array(seg_list)
            gt_list = np.array(gt_list)
            auroc = roc_auc_score(gt_list.flatten(), seg_list.flatten())
            gt_list_aupr = (gt_list > 0).astype(np.uint8)
            aupr = average_precision_score(gt_list_aupr.flatten(), seg_list.flatten())

            seg_results[group] = {"auroc": float(auroc), "aupr": float(aupr), "count": count}
            print(f"  {caption} ({dataset_key}): AUROC={auroc:.4f}, AUPR={aupr:.4f}, n={count}")

        metrics["segmentation"] = seg_results
        if seg_per_sample:
            _dump("seg_per_sample_results.json", seg_per_sample)
        # Compute average AUROC/AUPR across all test cases
        if seg_results:
            avg_auroc = np.mean([v["auroc"] for v in seg_results.values()])
            avg_aupr = np.mean([v["aupr"] for v in seg_results.values()])
            metrics["seg_avg_auroc"] = float(avg_auroc)
            metrics["seg_avg_aupr"] = float(avg_aupr)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return metrics


# ========================= Summary table =========================

def print_summary_table(all_metrics):
    """Print a comparison table across epochs and highlight the best."""
    if not all_metrics:
        return

    print("\n" + "=" * 100)
    print("SUMMARY - All Checkpoints")
    print("=" * 100)

    # Collect all metric keys (flat)
    header_keys = []
    if "i2t_accuracy" in all_metrics[0]:
        header_keys.append("i2t_accuracy")
    if "t2i_R@1" in all_metrics[0]:
        header_keys.extend(["t2i_R@1", "t2i_R@5", "t2i_R@10"])
    if "t2i_subset_accuracy" in all_metrics[0]:
        header_keys.append("t2i_subset_accuracy")
    if "seg_avg_auroc" in all_metrics[0]:
        header_keys.extend(["seg_avg_auroc", "seg_avg_aupr"])

    # Print header
    header = f"{'Epoch':>6}"
    for k in header_keys:
        header += f"  {k:>18}"
    print(header)
    print("-" * len(header))

    # Print rows
    best = {k: (-1, -float('inf')) for k in header_keys}
    for m in all_metrics:
        row = f"{m['epoch']:>6}"
        for k in header_keys:
            val = m.get(k, 0)
            row += f"  {val:>18.4f}"
            if val > best[k][1]:
                best[k] = (m['epoch'], val)
        print(row)

    # Print best
    print("-" * len(header))
    print("BEST:")
    for k in header_keys:
        epoch, val = best[k]
        print(f"  {k}: epoch={epoch}, value={val:.4f}")

    # Overall best (average rank across metrics)
    epoch_scores = defaultdict(list)
    for k in header_keys:
        sorted_epochs = sorted(all_metrics, key=lambda m: m.get(k, 0), reverse=True)
        for rank, m in enumerate(sorted_epochs):
            epoch_scores[m['epoch']].append(rank)
    best_overall = min(epoch_scores.items(), key=lambda x: np.mean(x[1]))
    print(f"\n  >>> BEST OVERALL: epoch={best_overall[0]} (avg rank={np.mean(best_overall[1]):.1f})")
    print("=" * 100)


# ========================= Main =========================

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conf", type=str, default="configs/test_ultraclip.conf")
    parser.add_argument("--modality", type=str, required=True, choices=["us", "mri", "ct"])
    parser.add_argument("--data_root", type=str, default="data", help="Root folder of the test data")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing multiple checkpoints. If set, evaluates all ckpts and compares.")
    parser.add_argument("--save_vis", action="store_true", default=False,
                        help="Save visualization images (disabled by default)")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        conf_text = f.read()
    conf = ConfigFactory.parse_string(conf_text)

    modality = args.modality
    mcfg = resolve_data_root(MODALITY_CONFIG[modality], args.data_root)

    tokenizer = AutoTokenizer.from_pretrained(conf.model.base_model)
    img_preprocess = image_transform(
        image_size=conf.dataset.base_image_size,
        is_train=False,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        fill_color=0)

    # Get checkpoint list
    checkpoint_list = get_checkpoint_list(conf, args.checkpoint_dir)
    print(f"Evaluating {len(checkpoint_list)} checkpoint(s) for modality={modality}")
    for epoch, path in checkpoint_list:
        print(f"  epoch={epoch}: {path}")
    print()

    # Compute save_dir up-front so per-sample retrieval JSONs land in the
    # same directory as the eval_results summary.
    if args.checkpoint_dir:
        model_name = os.path.basename(os.path.normpath(args.checkpoint_dir))
    else:
        model_name = os.path.basename(os.path.dirname(conf.test.model_path))
    save_dir = f"results/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    multi_ckpt = len(checkpoint_list) > 1

    # Evaluate each checkpoint
    all_metrics = []
    for epoch, ckpt_path in checkpoint_list:
        print(f"\n{'='*80}")
        print(f"Evaluating epoch={epoch}: {os.path.basename(ckpt_path)}")
        print(f"{'='*80}")

        metrics = evaluate_checkpoint(
            ckpt_path, conf, mcfg, tokenizer, img_preprocess,
            save_vis=args.save_vis, save_dir=save_dir, multi_ckpt=multi_ckpt)
        all_metrics.append(metrics)

    results_path = os.path.join(save_dir, f"eval_results_{modality}.json")
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {results_path}")

    # Print summary
    print_summary_table(all_metrics)
