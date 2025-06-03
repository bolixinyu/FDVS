import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.distributed import init_process_group, get_rank
# from data.AnetCap import AnetCaption
# from data.MSRVTT import MSRVTT
# from data.YouCook import YouCook
from prompt import Prompter  
from utils.videoReader import VideoLoader
import yaml
import pdb
import json
from tqdm import tqdm
from data import *
from utils.register import _dict as DataCls

task_func = {}
def registe_func(name):
    def decorator(func):
        task_func[name]=func
        return func
    return decorator

@registe_func('clip_caption_videochat')
def clipcaption_w_chat(video_path, prompter, config, result_dir):
    try: 
        vr = VideoLoader(video_path, config['Data']['fps'])
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    base_name = os.path.basename(video_path)
    dataset_name = config['Data']['set_name']
    if dataset_name=='anet':
        vid = base_name[2:13]
    else:
        vid = base_name.split('.')[0]
        
    clip_indexes = vr.get_clip_indexes_by_kframe(config['Data']['clip_length'],config['Data']['clip_stride'])
    
    dict_list = []
    for c in clip_indexes:
        try:
            clip_dict,_ = prompter.clip_caption(vr, c)
            dict_list.append(clip_dict)
        except Exception as e:
            print(e)
            print(f'-----------{vid}------------')
            return
    with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        json.dump(dict_list, f)

@registe_func('object_det')
def atomic_object(video_path, prompter, config, result_dir):
    dict_list = {}
    try:
        vr = VideoLoader(video_path, config['Data']['fps'])
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    base_name = os.path.basename(video_path)
    if config['Data']['set_name']=='anet':
        vid = base_name[2:13]
    else:
        vid = base_name.split('.')[0]
    clip_indexes = vr.get_clip_indexes_by_kframe(config['Data']['clip_length'],config['Data']['clip_stride'])
    # import pdb
    # pdb.set_trace()

    # task_name = config['Task']['name']
    # result_dir = config['Task']['result_dir'][task_name]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for c in clip_indexes:
        try:
            _,object_dict = prompter.clip_object_prompt(vr, c)
            dict_list.update(object_dict)
        except Exception as e:
            print(e)
            print(vid)
    with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        json.dump(dict_list, f)

@registe_func('action_rec')
def atomic_act_rec(video_path, prompter, config, result_dir):
    try:
        vr = VideoLoader(video_path, config['Data']['fps'])
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    base_name = os.path.basename(video_path)
    if config['Data']['set_name']=='anet':
        vid = base_name[2:13]
    else:
        vid = base_name.split('.')[0]
    clip_indexes = vr.get_clip_indexes_by_kframe(config['Data']['clip_length'],config['Data']['clip_stride'])

    # task_name = config['Task']['name']
    # result_dir = config['Task']['result_dir'][task_name]
    action_list=[]
    for c in clip_indexes:
        try:
            _,action_cls = prompter.clip_action_prompt(vr, c)
            action_dict = {'s':c[0].tolist(),'e':c[-1].tolist(),'cls':action_cls}
            action_list.append(action_dict)
        except Exception as e:
            print(e)
            print(vid)
    with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        json.dump(action_list, f)

@registe_func('image_cap')
def atomic_img_cap(video_path, prompter, config, result_dir):
    try:
        vr = VideoLoader(video_path, config['Data']['fps'])
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    base_name = os.path.basename(video_path)
    if config['Data']['set_name']=='anet':
        vid = base_name[2:13]
    else:
        vid = base_name.split('.')[0]
    clip_indexes = vr.get_clip_indexes_by_kframe(config['Data']['clip_length'],config['Data']['clip_stride'])
    
    # if not vid == "11699242@N07_8289244544_f659afaea0":
    #     return
    # import pdb
    # pdb.set_trace()
    # task_name = config['Task']['name']
    # result_dir = config['Task']['result_dir'][task_name]
    caption_list={}
    for c in clip_indexes:
        try:
            img_cap = prompter.clip_img_cap(vr, c)
            caption_list.update(img_cap)
        except Exception as e:
            print(e)
            print(vid)
    with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        json.dump(caption_list, f)

@registe_func('clip_cap_lavila')
def clip_cap_lavila(video_path, prompter, config, result_dir):
    try:
        vr = VideoLoader(video_path, config['Data']['fps'])
    except Exception as e:
        print(f'-----------{video_path}---------------')
    base_name = os.path.basename(video_path)
    if config['Data']['set_name']=='anet':
        vid = base_name[2:13]
    else:
        vid = base_name.split('.')[0]
    clip_indexes = vr.get_clip_indexes_by_kframe(4,config['Data']['clip_stride'])
    try:
        prompts = prompter.clip_caption_lavila(vr, clip_indexes)
        # -------
        # total_num, selected_num = prompter.vicuna_caption_clip(action_dict, object_dict, caption_dict,clip_indexes,vr)
        # with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        #     json.dump({'total_num': total_num, 'selected_num': selected_num},f)
        # return
        # -------
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        json.dump(prompts, f)

@registe_func('videochat')
def videochat_cap_video(video_path, prompter, config, result_dir):
    try:
        vr = VideoLoader(video_path, config['Data']['fps'])
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    base_name = os.path.basename(video_path)
    if config['Data']['set_name']=='anet':
        vid = base_name[2:13]
    else:
        vid = base_name.split('.')[0]
    clip_indexes = vr.uniformly_sample_video(config['Data']['clip_length'])
    # task_name = config['Task']['name']
    # result_dir = config['Task']['result_dir'][task_name]
    if get_rank()==0:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    caption = prompter.videochat_cap_video(vr, clip_indexes)
    with open(os.path.join(result_dir,vid+'.txt'), 'w+',encoding='utf-8') as f:
        f.write(caption)
        f.close()

@registe_func('atomic_clipcap_videochat')
def atomic_clipcap_videochat(video_path, prompter, config, result_dir):
    base_name = os.path.basename(video_path)
    dataset_name = config['Data']['set_name']
    if dataset_name=='anet':
        vid = base_name[2:13]
    else:
        vid = base_name.split('.')[0]
    
    try:
        vr = VideoLoader(video_path, config['Data']['fps'])
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    
    action_dir = os.path.join(config['Task']['result_dir'],f'{dataset_name}/action_rec')
    object_dir = os.path.join(config['Task']['result_dir'],f'{dataset_name}/object_det')
    caption_dir = os.path.join(config['Task']['result_dir'],f'{dataset_name}/image_cap')
    action_file = os.path.join(action_dir,vid+'.json')
    object_file = os.path.join(object_dir, vid+'.json')
    caption_file = os.path.join(caption_dir, vid+'.json')

    with open(action_file,'r') as f:
        action_dict = json.load(f)
        f.close()
    with open(object_file,'r') as f:
        object_dict = json.load(f)
        f.close()
    with open(caption_file,'r') as f:
        caption_dict = json.load(f)
        f.close()
    try:
        prompts = prompter.videochat_atomic_clipcap(vr, action_dict, object_dict, caption_dict)
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    # prompts = prompter.vicuna_caption_clip(action_dict, object_dict, caption_dict)
    with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        json.dump(prompts, f)
        
@registe_func('clip_cap')
def vicuna_caption_clip(video_path, prompter, config, result_dir):
    try:
        vr = VideoLoader(video_path, config['Data']['fps'])
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    clip_indexes = vr.get_clip_indexes_by_kframe(config['Data']['clip_length'],config['Data']['clip_stride'])
    base_name = os.path.basename(video_path)
    dataset_name = config['Data']['set_name']
    if dataset_name=='anet':
        vid = base_name[2:13]
    else:
        vid = base_name.split('.')[0]
    action_dir = os.path.join(config['Task']['result_dir'], f'{dataset_name}/action_rec')
    object_dir = os.path.join(config['Task']['result_dir'],'rebuttal_exp',f'{dataset_name}/object_det_frame32')
    caption_dir = os.path.join(config['Task']['result_dir'],'rebuttal_exp',f'{dataset_name}/image_cap_frame32')
    action_file = os.path.join(action_dir,vid+'.json')
    object_file = os.path.join(object_dir, vid+'.json')
    caption_file = os.path.join(caption_dir, vid+'.json')

    with open(action_file,'r') as f:
        action_dict = json.load(f)
        f.close()
    with open(object_file,'r') as f:
        object_dict = json.load(f)
        f.close()
    with open(caption_file,'r') as f:
        caption_dict = json.load(f)
        f.close()
    # prompts = prompter.vicuna_caption_clip(action_dict, object_dict, caption_dict,clip_indexes,vr)
    try:
        prompts = prompter.vicuna_caption_clip(action_dict, object_dict, caption_dict,clip_indexes,vr)
        # -------
        # total_num, selected_num = prompter.vicuna_caption_clip(action_dict, object_dict, caption_dict,clip_indexes,vr)
        # with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        #     json.dump({'total_num': total_num, 'selected_num': selected_num},f)
        # return
        # -------
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        json.dump(prompts, f)
        
@registe_func('clip_cap_llama2')
def llama2_caption_clip(video_path, prompter, config, result_dir):
    try:
        vr = VideoLoader(video_path, config['Data']['fps'])
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    clip_indexes = vr.get_clip_indexes_by_kframe(config['Data']['clip_length'],config['Data']['clip_stride'])
    base_name = os.path.basename(video_path)
    dataset_name = config['Data']['set_name']
    if dataset_name=='anet':
        vid = base_name[2:13]
    else:
        vid = base_name.split('.')[0]
    action_dir = os.path.join(config['Task']['result_dir'],f'{dataset_name}/action_rec')
    object_dir = os.path.join(config['Task']['result_dir'],f'{dataset_name}/object_det')
    caption_dir = os.path.join(config['Task']['result_dir'],f'{dataset_name}/image_cap')
    action_file = os.path.join(action_dir,vid+'.json')
    object_file = os.path.join(object_dir, vid+'.json')
    caption_file = os.path.join(caption_dir, vid+'.json')

    with open(action_file,'r') as f:
        action_dict = json.load(f)
        f.close()
    with open(object_file,'r') as f:
        object_dict = json.load(f)
        f.close()
    with open(caption_file,'r') as f:
        caption_dict = json.load(f)
        f.close()
    # prompts = prompter.llama_caption_clip(action_dict, object_dict, caption_dict,clip_indexes,vr)
    try:
        prompts = prompter.llama_caption_clip(action_dict, object_dict, caption_dict,clip_indexes,vr)
        # -------
        # total_num, selected_num = prompter.vicuna_caption_clip(action_dict, object_dict, caption_dict,clip_indexes,vr)
        # with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        #     json.dump({'total_num': total_num, 'selected_num': selected_num},f)
        # return
        # -------
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        json.dump(prompts, f)
        
@registe_func('video_cap')
def vicuna_caption_video(video_path, prompter, config, result_dir):
    try:
        vr = VideoLoader(video_path,30)
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    base_name = os.path.basename(video_path)
    dataset_name = config['Data']['set_name']
    if dataset_name=='anet':
        vid = base_name[2:13]
    else:
        vid = base_name.split('.')[0]
        
    clip_cap_dir = os.path.join(config['Task']['result_dir'],'rebuttal_exp',f'{dataset_name}','clip_cap_frame32')
    if vid+'.json' in os.listdir(clip_cap_dir):
        clip_cap_file = os.path.join(clip_cap_dir, vid+'.json')
    else:
        return
    with open(clip_cap_file,'r') as f:
        clip_cap_dict = json.load(f)
        f.close()
    if len(clip_cap_dict)==0:
        print(video_path)
    prompts, captions = prompter.vicuna_caption_video(clip_cap_dict, vr, config['Task']['semantic_reduce'])
    try:
        # #------------------------for hard sample------------------
        # with open('./tmp/hard_samples.txt','r') as f:
        #     hard_list = [l.strip() for l in f]
        #     f.close()
        # if vid in hard_list:
        #     prompts, captions = prompter.vicuna_caption_video(clip_cap_dict, vr.vr.get_avg_fps(), config['Task']['semantic_reduce'])
        #     with open(os.path.join('./tmp/hard_clip_caps',vid+'.txt'),'w') as f:
        #         f.write(prompts)
        # return
        # #---------------------------------------------------------

        prompts, captions = prompter.vicuna_caption_video(clip_cap_dict, vr, config['Task']['semantic_reduce'])
        # # selecte_res = prompter.vicuna_caption_video(clip_cap_dict, vr, config['Task']['semantic_reduce'])
        # with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        #     json.dump(selecte_res, f)
        # return
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    # prompts = prompter.vicuna_caption_clip(action_dict, object_dict, caption_dict)
    with open(os.path.join(result_dir,vid+'.txt'), 'w+', encoding='utf-8') as f:
        f.write(captions)

@registe_func('video_cap_llama')
def llama_caption_video(video_path, prompter, config, result_dir):
    try:
        vr = VideoLoader(video_path,30)
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    base_name = os.path.basename(video_path)
    dataset_name = config['Data']['set_name']
    if dataset_name=='anet':
        vid = base_name[2:13]
    else:
        vid = base_name.split('.')[0]
        
    clip_cap_dir = os.path.join(config['Task']['result_dir'],f'{dataset_name}','clip_cap_interpretable_all_visual_reduce')
    if vid+'.json' in os.listdir(clip_cap_dir):
        clip_cap_file = os.path.join(clip_cap_dir, vid+'.json')
    else:
        return
    with open(clip_cap_file,'r') as f:
        clip_cap_dict = json.load(f)
        f.close()
    if len(clip_cap_dict)==0:
        print(video_path)
    try:
        # #------------------------for hard sample------------------
        # with open('./tmp/hard_samples.txt','r') as f:
        #     hard_list = [l.strip() for l in f]
        #     f.close()
        # if vid in hard_list:
        #     prompts, captions = prompter.vicuna_caption_video(clip_cap_dict, vr.vr.get_avg_fps(), config['Task']['semantic_reduce'])
        #     with open(os.path.join('./tmp/hard_clip_caps',vid+'.txt'),'w') as f:
        #         f.write(prompts)
        # return
        # #---------------------------------------------------------

        prompts, captions = prompter.llama_caption_video(clip_cap_dict, vr, config['Task']['semantic_reduce'])
        # # selecte_res = prompter.vicuna_caption_video(clip_cap_dict, vr, config['Task']['semantic_reduce'])
        # with open(os.path.join(result_dir,vid+'.json'), 'w+') as f:
        #     json.dump(selecte_res, f)
        # return
    except Exception as e:
        print(f'-----------{video_path}---------------')
        print(e)
        return
    # prompts = prompter.vicuna_caption_clip(action_dict, object_dict, caption_dict)
    with open(os.path.join(result_dir,vid+'.txt'), 'w+', encoding='utf-8') as f:
        f.write(captions)
    
    

def main(config):
    rank_id = get_rank() if config['Infer']['distributed'] else 0
    device = torch.device('cuda', rank_id)
    torch.cuda.set_device(rank_id)
    task_name = config['Task']['name']
    dataset_name = config['Data']['set_name']
    tag = config['Task']['tag']
    result_dir = os.path.join(config['Task']['result_dir'],'rebuttal_exp/', f'{dataset_name}/{task_name+tag}')#'./result/youcook2/videochat_cap'
    if rank_id==0:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    dataset = DataCls[dataset_name](config['Data']['data_paths'][dataset_name]['file'],
                    config['Data']['data_paths'][dataset_name]['video_dir'],
                    result_dir)
    sampler = DistributedSampler(dataset, shuffle=False) if config['Infer']['distributed'] else None
    data_loader = DataLoader(dataset, 1, shuffle=False, sampler=sampler)
    prompter = Prompter(config)
    # if config['Infer']['verbose'] is not None:
    #     verbose_dir = config['Infer']['verbose']
    #     if not os.path.exists(verbose_dir):
    #         os.makedirs(verbose_dir)
    
    # # ------
    # total_num = 0
    # selected_num = 0
    # ------
    for video_path in tqdm(data_loader, desc=f'{dataset_name},{task_name+tag}'):
        task_func[task_name](video_path[0],prompter, config,result_dir)
        
        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config_smil.yaml')
    parser.add_argument('--local_rank', type=int, default=None)
    args = parser.parse_args()
    
    config = yaml.load(open(args.config,'r'), yaml.FullLoader)
    if config['Infer']['distributed']:
        init_process_group('nccl')
    main(config)