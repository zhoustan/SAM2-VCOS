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

import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
from mmengine import Config
import transformers
from transformers import BitsAndBytesConfig

from mllm.dataset.process_function import PlainBoxFormatter
from mllm.dataset.builder import prepare_interactive
from mllm.utils import draw_bounding_boxes
from mllm.models.builder.build_shikra import load_pretrained_shikra

log_level = logging.DEBUG
transformers.logging.set_verbosity(log_level)
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

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

model_path = '/path/of/shikra-7b'
model_args = Config(dict(
    type='shikra',
    version='v1',
    device_map='auto',
    # checkpoint config
    cache_dir=None,
    model_name_or_path=model_path,
    vision_tower=r'openai/clip-vit-large-patch14',
    pretrain_mm_mlp_adapter=None,

    # model config
    mm_vision_select_layer=-2,
    model_max_length=2048,

    # finetune config
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,
    freeze_mm_mlp_adapter=False,

    # data process config
    is_multimodal=True,
    sep_image_conv_front=False,
    image_token_len=256,
    mm_use_im_start_end=True,

    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    process_func_args=dict(
        conv=dict(type='ShikraConvProcess'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='ShikraTextProcess'),
        image=dict(type='ShikraImageProcessor'),
    ),

    conv_args=dict(
        conv_template='vicuna_v1.1',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=None),
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
))

training_args = Config(dict(
    bf16=False,
    fp16=True,
    device='cuda',
    fsdp=None,
))
load_in_8bit = False

if load_in_8bit:
    quantization_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
        )
    )
else:
    quantization_kwargs = dict()

model, preprocessor = load_pretrained_shikra(model_args, training_args, **quantization_kwargs,device_map='auto')
if not getattr(model, 'is_quantized', False):
    model.to(dtype=torch.float16, device=torch.device('cuda'))
if not getattr(model.model.vision_tower[0], 'is_quantized', False):
    model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device('cuda'))
print(
    f"LLM device: {model.device}, is_quantized: {getattr(model, 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
print(
    f"vision device: {model.model.vision_tower[0].device}, is_quantized: {getattr(model.model.vision_tower[0], 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")

def call_shikra(image_path, user_input):
    preprocessor['target'] = {'boxes': PlainBoxFormatter()}
    tokenizer = preprocessor['text']
    pil_image = Image.open(image_path).convert("RGB")
    ds = prepare_interactive(model_args, preprocessor)
    ds.set_image(pil_image)
    ds.append_message(role=ds.roles[0], message=user_input)
    model_inputs = ds.to_model_input()
    model_inputs['images'] = model_inputs['images'].to("cuda")
    model_inputs['input_ids'] = model_inputs['input_ids'].to("cuda")
    gen_kwargs = dict(
                    use_cache=True,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=200,
                    top_p=1.0,
                    temperature=1.0,
                )
    input_ids = model_inputs['input_ids']
    st_time = time.time()
    with torch.inference_mode():
        with torch.autocast(dtype=torch.float32, device_type='cuda'):
            output_ids = model.generate(**model_inputs, **gen_kwargs)
    # print(f"done generated in {time.time() - st_time} seconds")
    input_token_len = input_ids.shape[-1]
    response = tokenizer.batch_decode(output_ids[:, input_token_len:])[0]
    print(f"response: {response}")

    match = re.search(r"\[([0-9.,]+)\]", response)

    if match:
        extracted_list_str = match.group(0)  # Includes the brackets
        extracted_list = match.group(1).split(',')  # Extract and split numbers
        extracted_list = [float(x) for x in extracted_list]
        return extracted_list
    else:
        return [0,0,1,1]




def eval_cls_bbox_frame_0(cls):
    """
    Eval with box prompt generated by Shikra on frame 0.

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
    
    image_path = video_dir + frame_names[0]
    box = call_shikra(image_path, user_input)
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
    # since sam2 skip reverse tracking if starting from frame 0
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


user_input = """Please provide the coordinates of the bounding box where the animal is camouflaged in the picture."""

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
    

file = '../eval_result/MoCA_Mask_shikra_%s_%s_frame_0.json'%(model_type, prompt_type)
print(file)
with open(file, 'w') as f:
    results = {}
    for cls in classes:
        metric = eval_cls_bbox_frame_0(cls)
        metric = convert_ndarray_to_list(metric)
        results[cls] = metric
    json.dump(results, f, indent=4)
    

results_list = []
file = '../eval_result/MoCA_Mask_shikra_%s_%s_frame_0.json'%(model_type, prompt_type)
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
