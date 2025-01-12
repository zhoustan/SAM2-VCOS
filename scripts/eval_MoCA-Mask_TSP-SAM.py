import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from PIL import Image

import random
import re
from new_eval import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sam2.build_sam import build_sam2_video_predictor
import json
import pandas as pd


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def eval_cls_mask_refine_vcos(cls):
    """
    Refine VCOS models by using the prediction as the mask prompt.

    Args:
        cls: class name (video name) to evaluate.

    Returns:
        metrics: calculated metrics.
    """
    video_dir = test_data_folder + cls + '/Imgs/'
    gt_dir = test_data_folder + cls + '/GT/'
    VCOS_dir = VCOS_result_folder + cls + '/'

    # process gt
    gt_files = [
        p for p in os.listdir(gt_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    gt_files_sorted = sorted(gt_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    gt_to_eval = []
    for i in gt_files_sorted:
        img_np = np.array(Image.open(gt_dir + i))
        gt_to_eval.append(img_np)
    gt_to_eval = np.array(gt_to_eval)

    VCOS_files = [
        p for p in os.listdir(VCOS_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    VCOS_files_sorted = sorted(VCOS_files, key=lambda x: int(re.findall(r'\d+', x)[0]))


    real_ann_idx = len(gt_to_eval)-1
        
    mask = np.array(Image.open(VCOS_dir + VCOS_files_sorted[-1]))
        
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    ann_frame_idx = real_ann_idx  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state, 
        frame_idx=ann_frame_idx, 
        obj_id=ann_obj_id, 
        mask=mask
    )
    
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    # since sam2 skip reverse tracking if starting from frame 0
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    prediction_to_eval = []
    for out_frame_idx in range(0, len(frame_names)):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            prediction_to_eval.append(out_mask[0])
    prediction_to_eval = np.array(prediction_to_eval)

    metrics = calculate_metrics(prediction_to_eval, gt_to_eval)
    print(cls, np.nanmean(metrics['IoU Results']['iou']['curve']))
    return metrics


# path of the test data
test_data_folder = "/path/of/MoCA-Mask/"
VCOS_result_folder = "/path/of/VCOS_prediction/"
# get all classes
classes = os.listdir(test_data_folder)

prompt_type = 'mask'

model_cfg = "sam2_hiera_b+.yaml"
model_type = "hiera_base_plus"
sam2_checkpoint = "../checkpoints/sam2_%s.pt"%(model_type)
sam2 = torch.load(sam2_checkpoint)
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
file = '../eval_result/MoCA_Mask_%s_refine_vcos.json'%(model_type)
print(file)
with open(file, 'w') as f:
    results = {}
    for cls in classes:
        metric = eval_cls_mask_refine_vcos(cls)
        metric = convert_ndarray_to_list(metric)
        results[cls] = metric
    json.dump(results, f, indent=4)
  
results_list = []  
results = parse_result_json(file)
results['model_type'] = model_type
results['prompt_type'] = 'auto'
results['frame'] = 0
results_list.append(results)

df_results = pd.DataFrame(results_list)
columns_order = ['model_type', 'prompt_type', 'frame'] + [col for col in df_results.columns if col not in ['model_type', 'prompt_type', 'frame']]
df_results = df_results[columns_order]
df_results = df_results.drop(columns=['BIoU', 'TIoU', 'Boundary Accuracy'])
df_results = df_results.round(3)
pd.options.display.max_columns = None
pd.options.display.max_rows = None
print(df_results)
print("="*50)
