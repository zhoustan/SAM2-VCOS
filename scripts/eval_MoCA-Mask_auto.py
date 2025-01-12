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

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
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

def get_bbox_from_mask(mask):
    # Find indices where mask is True
    indices = np.where(mask)
    
    if len(indices[0]) == 0 or len(indices[1]) == 0:
        # No mask found, return None
        return None
    
    # Get the min and max for both rows and columns
    ymin, xmin = np.min(indices[0]), np.min(indices[1])
    ymax, xmax = np.max(indices[0]), np.max(indices[1])
    
    return [xmin, ymin, xmax, ymax]
    
def iou(bbox1, bbox2):
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    boxBArea = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def find_max_iou(bbox_from_mask, pred_boxes):
    max_iou = 0
    max_iou_bbox = None
    max_idx = 0
    idx = 0
    for bbox in pred_boxes:
        current_iou = iou(bbox_from_mask, bbox.tolist())
        if current_iou > max_iou:
            max_iou = current_iou
            max_iou_bbox = bbox.tolist()
            max_idx = idx
        idx += 1
    return max_iou, max_iou_bbox, max_idx



def eval_cls_auto(cls):
    """
    Eval with auto-mode.

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
    img = np.array(Image.open(video_dir + frame_names[real_ann_idx]).convert("RGB"))
    masks = mask_generator.generate(img)

    while len(masks) == 0:
        real_ann_idx += 1
        img = np.array(Image.open(video_dir + frame_names[real_ann_idx]).convert("RGB"))
        masks = mask_generator.generate(img)

    outputs = {"pred_logits": [], "pred_boxes": [], "mask": [], "scores": []}
    for mask_dict in masks:
        mask = mask_dict["segmentation"]
        score = mask_dict["predicted_iou"]
        x, y, w, h = bbox = mask_dict["bbox"]
        outputs["pred_boxes"].append([x, y, x + w, y + h])
        outputs["pred_logits"].append(0)
        outputs["mask"].append(mask)
        outputs["scores"].append(score)
    
    outputs["pred_boxes"] = torch.tensor(outputs["pred_boxes"], dtype=torch.float32).view(1, -1, 4)
    outputs["pred_logits"] = torch.tensor(outputs["pred_logits"], dtype=torch.float32).view(1, -1, 1)
    
    bbox_from_mask = get_bbox_from_mask(gt_to_eval[real_ann_idx])
    
    if bbox_from_mask:
        print(bbox_from_mask)
        max_iou_value, max_iou_bbox, max_idx = find_max_iou(bbox_from_mask, outputs['pred_boxes'][0])
    
    matched_mask = outputs['mask'][max_idx]

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    ann_frame_idx = real_ann_idx
    ann_obj_id = 1
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state, 
        frame_idx=ann_frame_idx, 
        obj_id=ann_obj_id, 
        mask=matched_mask
    )


    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx,reverse=True):
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
# get all classes
classes = os.listdir(test_data_folder)

model_cfg = "sam2_hiera_b+.yaml"
model_type = "hiera_base_plus"
sam2_checkpoint = "../checkpoints/sam2_%s.pt"%(model_type)
sam2 = torch.load(sam2_checkpoint)
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)



file = '../eval_result/MoCA_Mask_%s_auto.json'%(model_type)
print(file)
with open(file, 'w') as f:
    results = {}
    for cls in classes:
        metric = eval_cls_auto(cls)
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
