import os
import logging

logger = logging.getLogger(__name__)

root_dir = '/chenyaofo/project/videocap/result/msrvtt'
dir_list = os.listdir(root_dir)

clip_cap_list = [ l for l in dir_list if 'clip_cap' in l and 'uni_sample' not in l]
video_cap_list = [ l for l in dir_list if 'video_cap' in l]

print(f'{len(clip_cap_list)*len(video_cap_list)} tasks to run')

for clip_cap in clip_cap_list:
    clip_cap_dir = os.path.join(root_dir, clip_cap)
    for video_cap in video_cap_list:
        video_cap_dir = os.path.join(root_dir, video_cap)
        cmd= f'python match/match_clip_story.py \
                  --dataset msrvtt \
                  --video_cap {video_cap_dir} \
                  --clip_cap {clip_cap_dir} \
                  --bs 256'
        print(cmd)
        os.system(cmd)