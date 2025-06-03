from torchvision import transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
import torch
from utils.videoReader import VideoLoader
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from data.video_dataset import VideoDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, get_rank
import torch.nn as nn
import json

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def split_batch(clip_indexes, batch_size):
    # n_batch=len(clip_indexes)//batch_size
    indices=np.arange(batch_size,len(clip_indexes),step=batch_size)
    batch_indexes=np.split(clip_indexes,indices,axis=0)
    # batch_indexes.append(clip_indexes[-1:])
    return batch_indexes

def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, distributed = False):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    with torch.no_grad():
        outputs = model(image, captions=[caption]*len(image))
    logits = outputs["pred_logits"].sigmoid()  # (nq, 256)
    boxes = outputs["pred_boxes"]  # (nq, 4)

    # filter output
    logits_filt_batch = logits.cpu().clone()
    boxes_filt_batch = boxes.cpu().clone()
    filt_mask_batch = logits_filt_batch.max(dim=2)[0] > box_threshold
    pred_phrase_batch = []
    box_batch = []
    for logits_filt, boxes_filt, filt_mask in zip(logits_filt_batch, boxes_filt_batch, filt_mask_batch):
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        box_batch.append(boxes_filt)
        # get phrase
        if distributed:
            tokenlizer = model.module.tokenizer
        else:
            tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
        pred_phrase_batch.append(pred_phrases)
    return box_batch, pred_phrase_batch

def get_class_labels(class_files, exclude_cls):
    exclude_list = []
    with open(exclude_cls) as f:
        for line in f:
            exclude_list.append(line.strip())
    if isinstance(class_files, list):
        class_labels = []
        for class_file in class_files:
            with open(class_file, "r") as f:
                for l in f:
                    c = l.strip().split(',')[0]
                    if c not in class_labels and c not in exclude_list:
                        class_labels.append(c)
    else:
        with open(class_file, "r") as f:
            for l in f:
                c = l.strip().split(',')[0]
                if c not in class_labels and c not in exclude_list:
                    class_labels.append(c)
    prompt = ''
    for c in class_labels:
        prompt = prompt + c + '.' 
    return prompt


def get_grounding_output_for_video(device,model, video_path, class_files, 
                                  output_dir, box_threshold,
                                  text_threshold=None, batch_size=16):
    transform = T.Compose(
        [
            # T.ToTensor(),
            T.Resize(800, max_size=1333),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    caption = get_class_labels(class_files)
    vid = os.path.basename(video_path).split('.')[0]
    try:
        vr = VideoLoader(video_path,2,transforms=transform)
        clip_indexes=vr.get_clip_indexes(2,2)
        batch_indexes = split_batch(clip_indexes,batch_size)
        # vid_dir = os.path.join(output_dir, vid)
        # os.makedirs(vid_dir, exist_ok=True)
        result_dict = {}
        for clip_index in batch_indexes:
            clip_index = np.concatenate(clip_index, axis=0)
            clip, tensor = vr.read_clip(clip_index)
            tensor=tensor.to(device)
            boxes, pred_phrases = get_grounding_output(model, tensor, 
                                                    caption, box_threshold, 
                                                    text_threshold, distributed=True)
            for i,(image_pil, box, pred_phrase) in enumerate(zip(clip, boxes, pred_phrases)):
                image_pil = Image.fromarray(image_pil)
                size = image_pil.size
                pred_dict = {
                    "boxes": box,
                    "size": [size[1], size[0]],  # H,W
                    "labels": pred_phrase,
                }
                box_dict = {
                    'boxes': box.tolist(),
                    'labels': pred_phrase
                }
                result_dict[str(clip_index[i])] = box_dict
        
        with open(os.path.join(output_dir, f"{vid}.json"), "w") as f:
            json.dump(result_dict, f)
            f.close()
        return boxes, pred_phrases
    
    except Exception as e:
        print(e)
        print(f'{video_path} cannot be read by decord')
        return None, None