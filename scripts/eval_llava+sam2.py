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

import os
import sys
import logging
import time
import argparse
import tempfile
from pathlib import Path
from typing import List, Any, Union

import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

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

def eval_cls_bbox_frame_0(cls):
    """
    Eval with box prompt generated by LLaVA on frame 0.

    Args:
        cls: class name (video name) to evaluate.

    Returns:
        metrics: calculated metrics.
    """
    video_dir = test_data_folder + cls + '/Imgs/'
    gt_dir = test_data_folder + cls + '/GT/'

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

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    real_ann_idx = 0
    
    inputs = processor(images=Image.open(video_dir + frame_names[0]), text=prompt, return_tensors='pt').to(0, torch.float16)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    output = processor.decode(output[0], skip_special_tokens=True)
    box = eval(output.split('ASSISTANT: ')[-1])
    print(box)
    Y, X = gt_to_eval[real_ann_idx].shape
    # Return the bounding box in XYXY format
    bbox = np.array([box[0]*X, box[1]*Y, box[2]*X, box[3]*Y])
    
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    ann_frame_idx = real_ann_idx  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=bbox
    )


    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
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


model_path = '/path/of/models--llava-hf--llava-1.5-7b-hf/'

model = LlavaForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    # load_in_8bit=True,
)
model = model.to(device)
processor = AutoProcessor.from_pretrained(model_path)
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "Please provide the coordinates of the bounding box where the animal is camouflaged in the picture."},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# path of the test data
test_data_folder = "/path/of/MoCA-Mask/"
# get all classes
classes = os.listdir(test_data_folder)

prompt_type = 'box'

model_cfg = "sam2_hiera_b+.yaml"
model_type = "hiera_base_plus"
sam2_checkpoint = "../checkpoints/sam2_%s.pt"%(model_type)
sam2 = torch.load(sam2_checkpoint)
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    

file = '../eval_result/MoCA_Mask_llava_%s_%s_frame_0.json'%(model_type, prompt_type)
print(file)
with open(file, 'w') as f:
    results = {}
    for cls in classes:
        metric = eval_cls_bbox_frame_0(cls)
        metric = convert_ndarray_to_list(metric)
        results[cls] = metric
    json.dump(results, f, indent=4)
    

results_list = []
file = '../eval_result/MoCA_Mask_llava_%s_%s_frame_0.json'%(model_type, prompt_type)
results = parse_result_json(file)
results['model_type'] = model_type
results['prompt_type'] = prompt_type
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
