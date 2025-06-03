from PIL import Image

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from enum import auto, Enum

import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from utils.transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
from utils.easydict import EasyDict

def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class Captioner:
    def __init__(self, model, generation_config, device='cuda:0'):
        self.device = device
        self.model = model
        self.generation_config = generation_config
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        
        self.transform = T.Compose([
            GroupScale(int(224), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(224),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def caption_for_clip(self, vr, clip_index, caption_instruct, distributed=False):
        self.conv = EasyDict({
                                "system": "",
                                "roles": ("Human", "Assistant"),
                                "messages": [],
                                "sep": "###"
                            })
        vid_chat, msg = self.load_video(vr, clip_index)
        TC, H, W = vid_chat.shape
        image = vid_chat.reshape(1, TC//3, 3, H, W).to(self.device)
        if distributed:
             image_emb, _ = self.model.module.encode_img(image)
        else:
            image_emb, _ = self.model.encode_img(image)
        img_list = [image_emb]
        self.conv.messages.append([
            self.conv.roles[0], 
            f"<Video><VideoHere></Video> {msg}\n"
        ])
        self.conv.messages.append([self.conv.roles[0], caption_instruct])
        embs = self.get_context_emb(self.conv, img_list, distributed)
        if distributed:
            outputs = self.model.module.llama_model.generate(
                inputs_embeds=embs,
                max_new_tokens=self.generation_config['max_new_tokens'],
                stopping_criteria=self.stopping_criteria,
                num_beams=self.generation_config['num_beams'],
                do_sample=True,
                min_length=self.generation_config['min_length'],
                top_p=self.generation_config['top_p'],
                repetition_penalty=self.generation_config['repetition_penalty'],
                length_penalty=self.generation_config['length_penalty'],
                temperature=self.generation_config['temperature'],
            )
        else:
            outputs = self.model.llama_model.generate(
                inputs_embeds=embs,
                max_new_tokens=self.generation_config['max_new_tokens'],
                stopping_criteria=self.stopping_criteria,
                num_beams=self.generation_config['num_beams'],
                do_sample=True,
                min_length=self.generation_config['min_length'],
                top_p=self.generation_config['top_p'],
                repetition_penalty=self.generation_config['repetition_penalty'],
                length_penalty=self.generation_config['length_penalty'],
                temperature=self.generation_config['temperature'],
            )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
        if distributed:
            output_text = self.model.module.llama_tokenizer.decode(output_token, add_special_tokens=False)
        else:
            output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        return output_text


    def ask(self,text,conv):
        conv.messages.append([conv.roles[0], text + '\n'])
        return conv

    def answer(self, conv,  img_list, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0):
        conv.messages.append([conv.roles[1], None])
        embs = self.get_context_emb(conv, img_list)
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy(), conv
        
    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(self, vr, clip_index):
        torch_imgs_224,_ = vr.read_clip(clip_index)
        tmpa=[]
        toPIL = T.ToPILImage()
        for i in torch_imgs_224:
            img = toPIL(i)
            tmpa.append(img)
        torch_imgs_224 = self.transform(tmpa)
        fps = float(vr.orin_fps)
        sec = ", ".join([str(round(f / fps, 1)) for f in clip_index])
        # " " should be added in the start and end
        msg = f"The video contains {len(clip_index)} frames sampled at {sec} seconds."
        return torch_imgs_224, msg

    def upload_video(self, image, conv, img_list, num_segments):
        if isinstance(image, str):  # is a image path
            vid_chat, msg = self.load_video(image, num_segments=num_segments, return_msg=True)
            TC, H, W = vid_chat.shape
            image = vid_chat.reshape(1, TC//3, 3, H, W).to(self.device)

        else:
            raise NotImplementedError
        print("Input video shape:", vid_chat.shape)
        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)
        conv.messages.append([
            conv.roles[0], 
            f"<Video><VideoHere></Video> {msg}\n"
        ])
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg, img_list, conv
    
    def upload_img(self, image, conv, img_list):
        img = image#Image.open(image)#.convert('RGB')
        transform = T.Compose(
            [
                T.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        img = transform(img).unsqueeze(0).unsqueeze(0).cuda()
        image_emb, _ = self.model.encode_img(img)
        img_list.append(image_emb)
        conv.messages.append([
            conv.roles[0],
            f"<Image><ImageHere></Image>\n"
        ])
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg,img_list, conv

    def get_context_emb(self, conv, img_list, distributed=False):
        prompt = get_prompt(conv)
        #print(prompt)
        if '<VideoHere>' in prompt:
            prompt_segs = prompt.split('<VideoHere>')
        else:
            prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of visual placeholders and videos."
        if distributed:
            seg_tokens = [
                self.model.module.llama_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            seg_embs = [self.model.module.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        else:
            seg_tokens = [
                self.model.llama_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
