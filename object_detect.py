import torch
import argparse
import os
from data.video_dataset import VideoDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, get_rank
import torch.nn as nn
from model.grouding_dino import *
init_process_group('nccl')

def main(args):
    
    model = load_model(args.config_file, args.checkpoint_path)
    local_rank =get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    model = model.to(device)
    if os.path.isdir(args.video_path):
        video_dataset = VideoDataset(args.video_path)
        sampler = DistributedSampler(video_dataset,shuffle=False)
        dataloader = DataLoader(video_dataset,batch_size=1,shuffle= False,sampler=sampler)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank)
        for video_path in dataloader:
            boxes, pred_phrases = get_grounding_output_for_video(device, model,
                                                                 video_path[0], args.cls_file,
                                                                 args.output_dir,
                                                                 args.box_threshold,
                                                                 args.text_threshold, args.bs)
            
    else:
        boxes, pred_phrases = get_grounding_output_for_video(device, model,
                                                            args.video_path, args.cls_file,
                                                            args.output_dir,
                                                            args.box_threshold,
                                                            args.text_threshold,
                                                            args.bs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Detect Anything in videos", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--video_path", "-v", type=str, required=True, help="path to image file")
    parser.add_argument("--cls_file", "-l", nargs='+', type=str, required=True, help="path to class label files, split by spaces")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument(
        "--bs", type=int, default=64, help="clips one batch"
    )
    parser.add_argument('--local_rank', type=int, default=None)
    args = parser.parse_args()
    main(args)
