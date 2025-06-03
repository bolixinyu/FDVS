from model.grouding_dino import *
from action_recognize import action_recognize
from model.load_internvideo import load_intern_action, kinetics_classnames, transform_action
from utils.videoReader import VideoLoader, select_frames_byCos
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms._transforms_video as transforms_video
import os
import yaml
import math
from collections import OrderedDict
from utils.config import Config
from utils.easydict import EasyDict
from model.videochat import VideoChat
from model.captioner import Captioner
from torch.distributed import get_rank
from torch.nn.parallel.distributed import DistributedDataParallel
# import pdb
# from model.blip2model.model.blip2_opt import Blip2OPT
# from model.blip2model.processor.blip_processors import BlipImageEvalProcessor
from omegaconf import OmegaConf
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPImageProcessor, CLIPModel, CLIPTokenizer
from model.vicuna.model.model_adapter import load_vicuna
import numpy as np
from text2vec import SentenceModel
import json
from queue import Queue
from match.angle_emb import AnglE, Prompts
from transformers import LlamaTokenizer, LlamaForCausalLM
from model.lavila.data.video_transforms import Permute
from model.lavila.data.datasets import get_frame_ids, video_loader_by_frames
from model.lavila.models.models import VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL
from model.lavila.models.tokenizer import MyGPT2Tokenizer

B_TOK, E_TOK = '<s>', '</s>'
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
prompt_template = f'{B_TOK}{B_INST} {B_SYS}You are a helpful expert in video analysis based on given video information. When User provides information about the content in a video. You should summarize the content of the video based on the information provided by the user in a 100 word summary.{E_SYS} {E_INST}'\
    f' Sure, I can help you with that! {E_TOK} {B_TOK} {B_INST}<video info>{E_INST}Here is a summary of the video in 100 words: this video'
#Describe the content of the video based on the following information: 
class Prompter:
    def __init__(self,config):
        config['Infer']['distributed'] = False
        self.config = config
        # pdb.set_trace()
        self.disctributed = self.config['Infer']['distributed']
        if self.disctributed:
            self.rank_id = get_rank()
        self.agents = self.load_models()
        self.video_transform = transform_action()
        self.toPIL = T.ToPILImage()

        self.img_transform = T.Compose(
            [
                # T.ToTensor(),
                T.Resize(800, max_size=1333),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self.prompt_template = self.config['Template']['prompt']
        self.object_template = self.config['Template']['object']
        self.action_template = self.config['Template']['action']
        self.caption_template = self.config['Template']['caption']


    def load_models(self):
        if self.disctributed:
            self.device = torch.device('cuda', self.rank_id)
        else:
            self.device = torch.device('cuda') if self.config['Device'] == 'cuda' else torch.device('cpu')
        agents = {}
        
        if 'text_encoder' in self.config['Model']['AtomicAgents']:
            text_encoder = SentenceModel(self.config['Model']['SentenceEncoder']['model_path'], device=self.device)
            agents['text_encoder']=text_encoder
        
        if 'clip' in self.config['Model']['AtomicAgents']:
            clip_model = CLIPModel.from_pretrained(self.config['Model']['CLIP']['model_path'])
            agents['clip'] = clip_model.to(self.device)
            self.clip_processor={}
            self.clip_processor['img_processor'] = CLIPImageProcessor.from_pretrained(self.config['Model']['CLIP']['model_path'])
            self.clip_processor['tokenizer'] = CLIPTokenizer.from_pretrained(self.config['Model']['CLIP']['model_path'])
        
        if 'action_rec' in self.config['Model']['AtomicAgents']:
            agents['action_rec'] = load_intern_action(self.device, self.config['Model']['ActionRecognizer']['model_path'])
        
        if 'object_det' in self.config['Model']['AtomicAgents']:
            detector = load_model(self.config['Model']['ObjectDetector']['config'],self.config['Model']['ObjectDetector']['checkpoint_path'])
            agents['object_det'] = detector.to(self.device).eval()
            self.object_cls = get_class_labels(self.config['Model']['ObjectDetector']['cls_file'], self.config['Model']['ObjectDetector']['exclude_cls'])
        if 'videochat' in self.config['Model']['AtomicAgents']:
            cfg = Config.from_file(self.config['Model']['VideoChat']['config'])
            video_chat = VideoChat(cfg.model)
            video_chat = video_chat.to(self.device)
            agents['videochat'] = video_chat.eval()
            agents['videochat'] = Captioner(agents['videochat'], self.config['Model']['VideoChat']['generation'], self.device)
        
        if 'vicuna' in self.config['Model']['AtomicAgents']:
            model_path = self.config['Model']['Vicuna']['model_path']
            vicuna, tokenizer = load_vicuna(model_path)
            vicuna = vicuna.to(self.device)
            agents['vicuna']={'model':vicuna.eval(),'tokenizer':tokenizer}
        
        if 'angle' in self.config['Model']['AtomicAgents']:
            pretrain_weights = self.config['Model']['Angle']['pretrianed_weights']
            model_path = self.config['Model']['Angle']['model_path']
            embeder = AnglE.from_pretrained(pretrain_weights, pretrained_lora_path = model_path)
            embeder.set_prompt(prompt=Prompts.A)
            embeder.backbone.to(self.device)
            embeder.backbone.eval()
            agents['angle'] = embeder
        
        if 'llama' in self.config['Model']['AtomicAgents']:
            llama = LlamaForCausalLM.from_pretrained(self.config['Model']['LlaMa2']['model_path'],torch_dtype=torch.float16)
            # llama.eval()
            tokenizer = LlamaTokenizer.from_pretrained(self.config['Model']['LlaMa2']['model_path'])
            tokenizer.add_special_tokens(
                {
                    "pad_token": "<PAD>",
                }
                )
            llama = llama.to(self.device)
            agents['llama']={'model':llama.eval(),'tokenizer':tokenizer}
        if 'lavila' in self.config['Model']['AtomicAgents']:
            model_path = self.config['Model']['Lavila']['model_path']
            clip_path = self.config['Model']['Lavila']['clip_path']
            gpt2_path = self.config['Model']['Lavila']['gpt2_xl_path']
            ckpt = torch.load(model_path, map_location='cpu')
            state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                state_dict[k.replace('module.', '')] = v
            
            model = VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
                text_use_cls_token=False,
                project_embed_dim=256,
                gated_xattn=True,
                timesformer_gated_xattn=False,
                freeze_lm_vclm=False,      # we use model.eval() anyway
                freeze_visual_vclm=False,  # we use model.eval() anyway
                num_frames=4,
                drop_path_rate=0.,
                clip_path = clip_path,
                gpt2_path = gpt2_path,
            )
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self.device)
            model.eval()
            tokenizer = MyGPT2Tokenizer(gpt2_path,add_bos=True)
            processor = T.Compose([
                # T.ToTensor(),
                Permute([3, 0, 1, 2]),
                T.Resize(336),
                T.CenterCrop(336),
                transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
            ])
            agents['lavila'] = {'model':model, 'tokenizer':tokenizer, 'processor':processor}

        
        #TODO: traicking model and HOI model
        if 'object_track' in self.config['Model']['AtomicAgents']:
            pass
        if 'hoi' in self.config['Model']['AtomicAgents']:
            pass
        if 'image_cap' in self.config['Model']['AtomicAgents']:
            self.img_cap_prompt = self.config['Model']['ImgCaption']['prompt']
            model = Blip2ForConditionalGeneration.from_pretrained(self.config['Model']['ImgCaption']['model_path'])
            agents['image_cap'] = model.to(self.device)
            self.processor = Blip2Processor.from_pretrained(self.config['Model']['ImgCaption']['model_path'])
        if self.disctributed:
            for k in agents.keys():
                if k == 'vicuna':
                    agents[k]['model'] = DistributedDataParallel(agents[k]['model'], device_ids=[self.rank_id], output_device=self.rank_id,gradient_as_bucket_view=True)
                elif k == 'text_encoder':
                    agents[k].bert = DistributedDataParallel(agents[k].bert, device_ids=[self.rank_id], output_device=self.rank_id)
                elif k == 'angle':
                    agents[k].backbone = DistributedDataParallel(agents[k].backbone, device_ids=[self.rank_id], output_device=self.rank_id)
                elif k == 'llama':
                    agents[k]['model'] = DistributedDataParallel(agents[k]['model'], device_ids=[self.rank_id], output_device=self.rank_id,gradient_as_bucket_view=True).module
                elif k == 'lavila':
                    agents[k]['model'] = DistributedDataParallel(agents[k]['model'], device_ids=[self.rank_id], output_device=self.rank_id).module
                else:
                    agents[k] = DistributedDataParallel(agents[k], device_ids=[self.rank_id], output_device=self.rank_id)
            if 'videochat' in agents.keys():
                agents['videochat'] = Captioner(agents['videochat'], self.config['Model']['VideoChat']['generation'], self.device)
        return agents

    def clip_action_prompt(self, vr, clip_index):
        clip, _ = vr.read_clip(clip_index)
        tmpa=[]
        for i in clip:
            img = self.toPIL(i)
            tmpa.append(img)
        clip = self.video_transform(tmpa)
        tc,h,w = clip.shape
        clip = clip.reshape(tc//3 ,3, h, w)
        action_tensor = clip.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        action_cls = action_recognize(self.agents['action_rec'],action_tensor)
        action_prompt = self.action_template.replace('<action_cls>', action_cls)
        return action_prompt, action_cls

    def clip_object_prompt(self, vr, clip_index):
        clip_index = np.unique(clip_index)
        clip_img, imgs = vr.read_clip(clip_index,self.img_transform)
        # _, selected_idx = select_frames_byCos(clip_img, self.agents['clip'], self.clip_processor)
        # clip_index = clip_index[selected_idx]
        imgs = imgs.to(self.device)
        boxes, phrases = get_grounding_output(self.agents['object_det'], imgs, self.object_cls, 
                                                self.config['Model']['ObjectDetector']['box_threshold'], 
                                                self.config['Model']['ObjectDetector']['text_threshold'], distributed=self.disctributed)
        object_prompt = ''
        box_list = {}
        for i, indice in enumerate(clip_index):
            box = boxes[i]
            tmp_box = box.tolist()
            phrase = phrases[i]
            if len(box)>0:
                t = indice
                object_str = 'a/an {} at {},'
                object_info = ''
                box_list[str(t)]= {'boxes':tmp_box,'phrase':phrase}
                for b, p in zip(box,phrase):
                    pos = p.find('(')
                    score = float(p[pos+1:pos+5])
                    p = p[:pos]                    
                    b[:2] -= b[2:] / 2
                    b[2:] += b[:2]
                    b = [round(x.tolist(),2) for x in b]
                    object_info = object_info+object_str.format(p,b)
                tmp_str = self.object_template.replace('<object_info>', object_info)
                tmp_str = tmp_str.replace('<detect_time>','%.2f'%t)
                object_prompt = object_prompt+tmp_str
        return object_prompt, box_list
    
    def clip_img_cap(self, vr, clip_index):
        # import pdb
        # pdb.set_trace()
        clip_index = np.unique(clip_index)
        imgs, _ = vr.read_clip(clip_index)
        # _, selected_idx = select_frames_byCos(clip_img, self.agents['clip'], self.clip_processor)
        # clip_index = clip_index[selected_idx]
        # imgs = imgs[selected_idx]
        captions = {}
        pil_imgs=[]
        for i,img in enumerate(imgs):
            index = clip_index[i]
            img = Image.fromarray(img)
            pil_imgs.append(img)
        inputs = self.processor(images=pil_imgs, text=[self.img_cap_prompt+' Answer:']*len(pil_imgs), return_tensors="pt", padding=True).to(self.device)
        if self.config['Infer']['distributed']:
            caption = self.agents['image_cap'].module.generate(**inputs,
                                                            max_new_tokens=self.config['Model']['ImgCaption']['max_length'], 
                                                            min_length=self.config['Model']['ImgCaption']['min_length'],
                                                            repetition_penalty=self.config['Model']['ImgCaption']['repetition_penalty'])
        else:
            caption = self.agents['image_cap'].generate(**inputs, 
                                                        max_new_tokens=self.config['Model']['ImgCaption']['max_length'], 
                                                        min_length=self.config['Model']['ImgCaption']['min_length'],
                                                        repetition_penalty=self.config['Model']['ImgCaption']['repetition_penalty'])
        caption=self.processor.batch_decode(caption, skip_special_tokens=True)
        for i, cap in enumerate(caption):
            index = clip_index[i]
            captions[str(index)]=cap
        return captions

    def generate_prompt(self,video_path):
        vr = VideoLoader(video_path, self.config['Data']['fps'])
        clip_indexes = vr.get_clip_indexes(self.config['Data']['clip_length'],self.config['Data']['clip_stride'])
        fps = vr.orin_fps
        prompt = ''
        for c in clip_indexes:
            action_prompt, action_cls = self.clip_action_prompt(vr, c)

            step = self.config['Data']['fps']//self.config['Data']['detect_fps']
            index = c[::step]
            object_prompt, box_dict = self.clip_object_prompt(vr,index)
            tmp_str = self.prompt_template.replace('<action_prompt>',action_prompt)
            tmp_str = tmp_str.replace('<object_prompt>',object_prompt)
            prompt = prompt + tmp_str
        return prompt
    
    def clip_prompt(self, vr, clip_index):
        action_prompt, action_cls = self.clip_action_prompt(vr, clip_index)

        step = self.config['Data']['fps']//self.config['Data']['detect_fps']
        index = clip_index[::step]
        object_prompt, box_dict = self.clip_object_prompt(vr,index)
        
        tmp_str = self.prompt_template.replace('<action_prompt>', action_prompt)
        tmp_str = tmp_str.replace('<object_prompt>', object_prompt)
        return tmp_str, action_cls, box_dict
    
    def clip_caption(self, vr, clip_index, info=''):
        start, end = clip_index[0], clip_index[-1]
        # clip_prompt, action_cls, box_dict = self.clip_prompt(vr,clip_index)
        caption_instruct = info+' Please describe the objects and what happened in the video in detail based on the given video information.' # self.caption_template.replace('<video_info',clip_prompt)
        caption = self.agents['videochat'].caption_for_clip(vr, clip_index, caption_instruct, self.disctributed)
        clip_dict = {'s':start.tolist(),'e':end.tolist(), 'cap':caption} #{'start':start, 'end':end, 'action_cls':action_cls,
                    #'objects':box_dict, 'caption':caption}
        return clip_dict, caption

    def clip_caption_lavila(self, vr, clip_indexes):
        caption_list = []
        for c in clip_indexes:
            frames, _ = vr.read_clip(c)
            frames = torch.from_numpy(frames).to(dtype=torch.float32)
            frames = self.agents['lavila']['processor'](frames)
            frames = frames.unsqueeze(0)
            frames = frames.to(self.device)
            image_features = self.agents['lavila']['model'].encode_image(frames)
            generated_text_ids, ppls = self.agents['lavila']['model'].generate(
                image_features,
                self.agents['lavila']['tokenizer'],
                target=None,  # free-form generation
                max_text_length=77,
                top_k=None,
                top_p=0.95,   # nucleus sampling
                num_return_sequences=5,  # number of candidates: 10
                temperature=0.7,
                early_stopping=True,
            )
            max_len = 0
            final_str = ''
            for i in range(len(generated_text_ids)):
                if self.agents['lavila']['tokenizer'].eos_token_id == self.agents['lavila']['tokenizer'].bos_token_id:
                    if self.agents['lavila']['tokenizer'].eos_token_id in generated_text_ids[i][1:].tolist():
                        eos_id = generated_text_ids[i][1:].tolist().index(self.agents['lavila']['tokenizer'].eos_token_id) + 1
                    else:
                        eos_id = len(generated_text_ids[i].tolist()) - 1
                elif self.agents['lavila']['tokenizer'].eos_token_id in generated_text_ids[i].tolist():
                    eos_id = generated_text_ids[i].tolist().index(self.agents['lavila']['tokenizer'].eos_token_id)
                else:
                    eos_id = len(generated_text_ids[i].tolist()) - 1
                generated_text_str = self.agents['lavila']['tokenizer'].tokenizer.decode(generated_text_ids[i][1:eos_id].tolist())
                if max_len < len(generated_text_str):
                    max_len = len(generated_text_str)
                    final_str = generated_text_str
            caption_list.append({'s':c[0].tolist(),'e':c[-1].tolist(), 'cap':final_str})
        return caption_list
    
    def videochat_cap_video(self, vr, clip_index):
        prompt = 'Please describe the objects and what happened in the video in detail.'
        caption = self.agents['videochat'].caption_for_clip(vr, clip_index, prompt, self.disctributed)
        return caption
    
    def videochat_atomic_clipcap(self, vr, action, object_dict, img_cap_dict):
        _, prompt_list = self._prompt_clip_caption(action, object_dict, img_cap_dict)
        caption_list = []
        for i, c in enumerate(action):
            s = int(c['s'])
            e = int(c['e'])
            indexes = np.linspace(s, e-1, 8,endpoint=True,dtype=np.int32)
            _, caption = self.clip_caption(vr, indexes, info=f'Video information: {prompt_list[i]}.')
            clip_cap = {'s': s,'e':e,'cap':caption}
            caption_list.append(clip_cap)
        return caption_list
    
    # def _prompt_clip_caption(self, action, object_dict, img_cap_dict, clip_index,vr):
    #     visual_reduce = self.config['Task']['visual_reduce']
    #     keys = iter(img_cap_dict.keys())
    #     k = next(keys)
    #     frame_num = int(k)
    #     prompt_list = []
    #     prompt_dict_list = []
    #     prompt_prefix = self.config['Model']['Vicuna']['clip_prompt']
    #     for c, indexes in zip(action, clip_index):
    #         prompt_dict = {}
    #         s = int(c['s'])
    #         e = int(c['e'])
    #         if visual_reduce:
    #             clip_img,_ = vr.read_clip(indexes)
    #             # import pdb
    #             # pdb.set_trace()
    #             _, selected_idx = select_frames_byCos(clip_img, self.agents['clip'], self.clip_processor['img_processor'])
    #             key_indexes = indexes[selected_idx]
    #         else:
    #             key_indexes = indexes
    #         action = c['cls']
    #         prompt_dict['action category'] = action
    #         while frame_num>=s and frame_num<= e:
    #             if frame_num in key_indexes:
    #                 img_cap = img_cap_dict[k]
    #                 prompt_dict[f'frame {frame_num}']={'caption':img_cap}
    #                 if k in object_dict.keys():
    #                     object_str_list = []
    #                     object_list = object_dict[k]
    #                     for o,p in zip(object_list['boxes'], object_list['phrase']):
    #                         position = [float('%.2f'%p)for p in o]
    #                         phrase = p.split('(')[0]
    #                         object_str = f'{phrase} at {position}'
    #                         object_str_list.append(object_str)
    #                     prompt_dict[f'frame {frame_num}']['objects']=object_str_list
    #             try:
    #                 k = next(keys)
    #             except Exception as e:
    #                 break
    #             frame_num = int(k)
    #         # prompt_dict['image captions'] = clip_caption_str
    #         # prompt_dict['detected objects'] = clip_object_str
    #         prompt = prompt_prefix+f'User: {prompt_dict}. Assistant: Described in English, this video '
    #         prompt_list.append(prompt)
    #         prompt_dict_list.append(prompt_dict)
    #         with open('./tmp.json','w+') as f:
    #             json.dump(prompt_list,f)
    #     return prompt_list, prompt_dict_list
    
    def convert_prompt_dict(self, prompt_dict_list):
        prompt_list = []
        time_prefix = ['First, ', 'Then, ', 'After that, ', 'Finally, ']
        prompt_prefix = self.config['Model']['Vicuna']['clip_prompt']
        for prompt_dict in prompt_dict_list:
            action = prompt_dict.pop("action category")
            new_prompt_dict = {"action category": action}
            # new_prompt_dict = {}
            caption = ''
            objects = ''
            for i, frames in enumerate(prompt_dict.keys()):
                if i == 0:
                    cur_prefix = time_prefix[0]
                elif i == (len(prompt_dict.keys())-1):
                    cur_prefix = time_prefix[-1]
                else:
                    cur_prefix_idx = np.random.choice([1,2], 1).tolist()
                    cur_prefix = time_prefix[cur_prefix_idx[0]]
                
                caption = caption + cur_prefix + str(prompt_dict[frames]['caption']) + '. '
                if 'objects' in prompt_dict[frames].keys():
                    objects = objects + cur_prefix + str(prompt_dict[frames]['objects']) + '. '
            if not objects == '':
                new_prompt_dict['objects'] = objects
            new_prompt_dict['caption'] = caption
            
            prompt = prompt_prefix+f'User: {new_prompt_dict}. Assistant: this video '
            # prompt = prompt_template.replace('<video info>', f'{new_prompt_dict}')
            prompt_list.append(prompt)
        # import pdb
        # pdb.set_trace()
        return prompt_list
            
    
    def _prompt_clip_caption(self, action, object_dict, img_cap_dict, clip_index,vr):
        def location_size_judge(boxes):
            # boxes(cx,cy,w,h)
            cx, cy, w, h = (b for b in boxes)
            if cy < 0.33:
                loca_str = 'upper'
            elif cy > 0.66:
                loca_str = 'lower'
            else:
                loca_str = ''
            if cx < 0.33:
                loca_str = loca_str + 'left'
            elif cx >0.66:
                loca_str = loca_str + 'right'
            else:
                loca_str = loca_str + 'middle'
            area = w*h
            if area < 0.33:
                size_str = 'small'
            elif area>0.66:
                size_str = 'large'
            else:
                size_str = 'moderate-sized'
            return loca_str, size_str
        
        visual_reduce = self.config['Task']['visual_reduce']
        if not len(img_cap_dict.keys())>0:
            return [], []
        keys = iter(img_cap_dict.keys())
        k = next(keys)
        frame_num = int(k)
        prompt_list = []
        prompt_dict_list = []
        prompt_prefix = self.config['Model']['Vicuna']['clip_prompt']
        
        # -----------
        # total_num = []
        # selected_num = []
        # -----------
        # prompt_cache = []
        for c, indexes in zip(action, clip_index):
            prompt_dict = {}
            s = indexes[0]
            e = indexes[-1]
            if visual_reduce:
                indexes = np.unique(indexes)
                clip_img,_ = vr.read_clip(indexes)
                # import pdb
                # pdb.set_trace()
                _, selected_idx = select_frames_byCos(clip_img, self.agents['clip'], self.clip_processor['img_processor'])
                key_indexes = indexes[selected_idx]
                # ---------
                # total_num.append(len(indexes))
                # selected_num.append(len(key_indexes))
                # ---------
            else:
                key_indexes = np.unique(indexes)
            action = c['cls']
            prompt_dict['action category'] = action
            while frame_num>=s and frame_num<= e:
                if frame_num in key_indexes:
                    img_cap = img_cap_dict[k]
                    prompt_dict[f'frame {frame_num}']={'caption':img_cap}
                    # prompt_dict[f'frame {frame_num}']={}
                    if k in object_dict.keys():
                        object_str_list = []
                        object_list = object_dict[k]
                        for o,p in zip(object_list['boxes'], object_list['phrase']):
                            pos, size_str = location_size_judge(o)
                            phrase = p.split('(')[0]
                            object_str = f'{phrase} at {pos} of the image and is {size_str} in the image'
                            object_str_list.append(object_str)
                        prompt_dict[f'frame {frame_num}']['objects']=object_str_list
                try:
                    k = next(keys)
                except Exception as e:
                    break
                frame_num = int(k)
            # prompt_cache.append(f'Video {len(prompt_cache)}: {prompt_dict}')
            # if len(prompt_cache)==3:
            #     prompt = prompt_prefix+f'User: {prompt_cache}. Assistant: Described in English:'
            #     prompt_cache = []
            
            prompt = prompt_prefix+f'User: {prompt_dict}. Assistant: this video'
            # prompt = prompt_template.replace('<video info>', f'{prompt_dict}')
            prompt_list.append(prompt)
            prompt_dict_list.append(prompt_dict)
            # with open('./tmp.json','w+') as f:
            #     json.dump(prompt_list,f)
        # if len(prompt_cache)>0:
        #     prompt = prompt_prefix+f'User: {prompt_cache}. Assistant: Described in English:'
        #     prompt_list.append(prompt)
        #     prompt_dict_list.append(prompt_dict)
        # ----------
        # return total_num, selected_num
        # ----------
        # prompt_list = self.convert_prompt_dict(prompt_dict_list)
        return prompt_list, prompt_dict_list
    
#     from line_profiler import LineProfiler
#     lp=LineProfiler()
    
#     @lp.profile
    def vicuna_caption_clip(self, action, object_dict, img_cap, clip_index, vr):
        prompt_list, _ = self._prompt_clip_caption(action, object_dict, img_cap, clip_index, vr)
        # -------
        # total_num, selected_num = self._prompt_clip_caption(action, object_dict, img_cap, clip_index, vr)
        # return total_num, selected_num
        # -------
        bs = self.config['Model']['Vicuna']['bs']
        batch_num = len(prompt_list)//bs
        tokenizer = self.agents['vicuna']['tokenizer']
        tokenizer.padding_side='left'
        model = self.agents['vicuna']['model']
        output_list = []
        for i in range(batch_num+1):
            if i < batch_num:
                batch_list = prompt_list[i*bs:(i+1)*bs]
            elif i*bs == len(prompt_list):
                break
            else:
                batch_list = prompt_list[i*bs:]
            batch_input = tokenizer.batch_encode_plus(batch_list,padding=True,return_tensors='pt').to(device=self.device)
            output_tokens = model.generate(**batch_input, max_new_tokens = 128, temperature=0.7, repetition_penalty=1.0, do_sample=False, length_penalty=1.0)
            output = tokenizer.batch_decode(output_tokens,
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                            clean_up_tokenization_spaces=True,)
            output_list.extend(output)
        caption_list = []
        for clip_action,prompt, output in zip(action, prompt_list, output_list):
            new_out = output.split('Assistant: ')[-1]
            clip_cap = {'s': clip_action['s'],'e':clip_action['e'],'cap':new_out}
            caption_list.append(clip_cap)
        
        # import pdb
        # pdb.set_trace()
        # lp.print_stats()
        return caption_list
    
    def llama_caption_clip(self, action, object_dict, img_cap, clip_index, vr):
        prompt_list, _ = self._prompt_clip_caption(action, object_dict, img_cap, clip_index, vr)
        # -------
        # total_num, selected_num = self._prompt_clip_caption(action, object_dict, img_cap, clip_index, vr)
        # return total_num, selected_num
        # -------
        bs = self.config['Model']['LlaMa2']['bs']
        batch_num = len(prompt_list)//bs
        tokenizer = self.agents['llama']['tokenizer']
        tokenizer.padding_side='left'
        model = self.agents['llama']['model']
        output_list = []
        for i in range(batch_num+1):
            if i < batch_num:
                batch_list = prompt_list[i*bs:(i+1)*bs]
            elif i*bs == len(prompt_list):
                break
            else:
                batch_list = prompt_list[i*bs:]
            batch_input = tokenizer.batch_encode_plus(batch_list,padding=True,return_tensors='pt').to(device=self.device)
            output_tokens = model.generate(**batch_input, max_new_tokens = 128, temperature=0.7, repetition_penalty=1.0, do_sample=False, length_penalty=1.0)
            output = tokenizer.batch_decode(output_tokens,
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                            clean_up_tokenization_spaces=True,)
            output_list.extend(output)
        caption_list = []
        for clip_action,prompt, output in zip(action, prompt_list, output_list):
            new_out = output.split('Here is a summary of the video in 50 words: ')[-1]
            clip_cap = {'s': clip_action['s'],'e':clip_action['e'],'cap':new_out}
            caption_list.append(clip_cap)
        
        return caption_list
        
    
    def _reduce_clip(self,clip_cap_dict):
        str_list = []
        for idx, cap_dict in enumerate(clip_cap_dict):
            cap = cap_dict['cap']
            str_list.append(cap)
            import pdb
            pdb.set_trace()
            embed_list = self.agents['text_encoder'].encode(str_list)
    
    def vicuna_caption_video(self, clip_cap_dict, vr, clip_reduce=False):
        fps = vr.vr.get_avg_fps()
        n_frame = vr.n_frame
        clip_cap = ''#{}
        tokenizer = self.agents['vicuna']['tokenizer']
        tokenizer.padding_side='left'
        model = self.agents['vicuna']['model']
        prompt = self.config['Model']['Vicuna']['video_prompt']
        clip_cap_str = ''
        init_queue = Queue(maxsize=1)
        avg_embed = None
        if len(clip_cap_dict)<=1:
            clip_reduce = False
        
        # ------
        # selecte_res = []
        # ------
        for idx, cap_dict in enumerate(clip_cap_dict):
            s = cap_dict['s']
            e = cap_dict['e']
            cap = cap_dict['cap']
            start_rate = s*1.0/n_frame
            if start_rate < 0.25:
                pre_fix = "At the beginning of the video, "
            elif start_rate <0.5:
                pre_fix = "Early in the vide, "
            elif start_rate <0.75:
                pre_fix = "Later in the video, "
            else:
                pre_fix = "At the end of the video, "
            if clip_reduce:
                embed = self.agents['text_encoder'].encode([cap])
                init_queue.put(embed)
                if not init_queue.full():
                    continue
                else:
                    avg_embed = np.concatenate(list(init_queue.queue),0)
                    avg_embed = avg_embed.mean(axis=0)
                    cos = avg_embed.dot(embed[0]) / (np.linalg.norm(avg_embed) * np.linalg.norm(embed[0],axis=0)+1e-10)
                    if cos<0.9:
                        # clip_cap['caption from %.2fs to %.2fs'%(s*1./fps, e*1./fps)]=cap
                        clip_cap = clip_cap + pre_fix + cap + ' '
                        # selecte_res.append(False)
                    # ---------
                    # else:
                    #     selecte_res.append(True)
                    # ---------
                    init_queue.get()
            else:
                # clip_cap['caption from %.2fs to %.2fs'%(s*1./fps, e*1./fps)]=cap
                clip_cap = clip_cap + pre_fix + cap + ' '
                # ------
                # selecte_res.append(False)
                # ------
            # clip_cap_str = clip_cap_str + time_prefix + cap + ' '
        # return selecte_res
        # input = prompt.replace('<video info>', f'{clip_cap}')
        input = prompt + f'{clip_cap}. Assistant: Described in English: this video '
        input_tokens = tokenizer.encode(input)
        input_tokens = torch.as_tensor([input_tokens],device='cuda')
        output_tokens = model.generate(input_tokens, max_new_tokens = 256, temperature=0, repetition_penalty=1.0, do_sample=False, length_penalty=1.0, use_cache=True)
        output = tokenizer.batch_decode(output_tokens,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,)
        new_out = output[0].split('Assistant: Described in English: ')[-1]
        return input, new_out
    
    def llama_caption_video(self, clip_cap_dict, vr, clip_reduce=False):
        fps = vr.vr.get_avg_fps()
        n_frame = vr.n_frame
        clip_cap = ''#{}
        tokenizer = self.agents['llama']['tokenizer']
        tokenizer.padding_side='left'
        model = self.agents['llama']['model']
        prompt = prompt_template
        clip_cap_str = ''
        init_queue = Queue(maxsize=35)
        avg_embed = None
        if len(clip_cap_dict)<=35:
            clip_reduce = False
        
        # ------
        # selecte_res = []
        # ------
        for idx, cap_dict in enumerate(clip_cap_dict):
            s = cap_dict['s']
            e = cap_dict['e']
            cap = cap_dict['cap']
            start_rate = s*1.0/n_frame
            if start_rate < 0.25:
                pre_fix = "At the beginning of the video, "
            elif start_rate <0.5:
                pre_fix = "Early in the vide, "
            elif start_rate <0.75:
                pre_fix = "Later in the video, "
            else:
                pre_fix = "At the end of the video, "
            if clip_reduce:
                embed = self.agents['text_encoder'].encode([cap])
                init_queue.put(embed)
                if not init_queue.full():
                    continue
                else:
                    avg_embed = np.concatenate(list(init_queue.queue),0)
                    avg_embed = avg_embed.mean(axis=0)
                    cos = avg_embed.dot(embed[0]) / (np.linalg.norm(avg_embed) * np.linalg.norm(embed[0],axis=0)+1e-10)
                    if cos<0.9:
                        # clip_cap['caption from %.2fs to %.2fs'%(s*1./fps, e*1./fps)]=cap
                        clip_cap = clip_cap + pre_fix + cap + ' '
                        # selecte_res.append(False)
                    # ---------
                    # else:
                    #     selecte_res.append(True)
                    # ---------
                    init_queue.get()
            else:
                # clip_cap['caption from %.2fs to %.2fs'%(s*1./fps, e*1./fps)]=cap
                clip_cap = clip_cap + pre_fix + cap + ' '
                # ------
                # selecte_res.append(False)
                # ------
            # clip_cap_str = clip_cap_str + time_prefix + cap + ' '
        # return selecte_res
        input = prompt+str(clip_cap)+'. Assistant: Summarized in English, this video '
        input_tokens = tokenizer.encode(input)
        input_tokens = torch.as_tensor([input_tokens],device='cuda')
        output_tokens = model.generate(input_tokens, max_new_tokens = 256, temperature=0, repetition_penalty=1.0, do_sample=False, length_penalty=1.0, use_cache=True)
        output = tokenizer.batch_decode(output_tokens,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,)
        new_out = output[0].split('Assistant: Summarized in English, this video ')[-1]
        return input, new_out



if __name__ == "__main__":
    # get_grounding_output()
    config = yaml.load(open('./config/config.yaml','r'), yaml.FullLoader)
    video_dir = config['Data']['video_dir']
    video_file = os.listdir(video_dir)
    dataset = VideoDataset(video_dir)
    prompter = Prompter(config)
    for f in video_file:
        video_path = os.path.join(video_dir,f)
        # prompt = prompter.generate_prompt(video_path)
        vr = VideoLoader(video_path, config['Data']['fps'])
        clip_indexes = vr.get_clip_indexes(config['Data']['clip_length'],config['Data']['clip_stride'])
        fps = vr.orin_fps
        prompt = ''
        for c in clip_indexes:
            prompter.clip_caption(vr, c)

                
