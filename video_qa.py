import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.distributed import init_process_group, get_rank
import yaml
import pdb
import json
from tqdm import tqdm
from data import *
from utils.register import _dict as DataCls
from text2vec import SentenceModel, cos_sim
from model.vicuna.model.model_adapter import load_vicuna
from match.angle_emb import AnglE, Prompts
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

import pdb

B_TOK, E_TOK = '<s>', '</s>'
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# short_answer_prompt_template = f'{B_TOK}{B_INST} {B_SYS}You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. When User provides a question and a long answer to the question, you should summarize the long answer into 1 word.{E_SYS} {E_INST}'\
#     f' Sure, I can help you with that! {E_TOK} {B_TOK} '\
#     f'{B_INST}Question: what is leaking some fluid? Long answer: Based on the information provided, it appears that the car engine is leaking some fluid.{E_INST} Short Answer in 1 word: engine'\
#     f'{B_INST}Question: who shoots a gun in a movie? Long answer: Based on the information provided, the answer to the question "who shoots a gun in a movie?" is the man in the final scene of the third video clip. He is holding a gun to his head while the other person holds a knife to their face, and they are eating noodles and soup together.{E_INST} Short Answer in 1 word: man'\
#     f'{B_INST}Question: <question> Long answer: <long answer>{E_INST} Short Answer in 1 word: '
# short_answer_prompt_template = f'{B_TOK}{B_INST} {B_SYS}You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. When User provides a question and a long answer to the question, you should summarize the long answer into 1 or 2 word.{E_SYS} {E_INST}'\
#     f' Sure, I can help you with that! {E_TOK} {B_TOK} '\
#     f'{B_INST}Question: what is the color of the pants of a person kneeling on one knee Long answer: The color of the pants of the person kneeling on one knee in the video is black.{E_INST} Short Answer in 1 or 2 word: black'\
#     f'{B_INST}Question: what is the person in black doing Long answer: The person in black is holding a soccer ball and standing next to the person playing soccer on the beach.{E_INST} Short Answer in 1 or 2 word: beach soccer'\
#     f'{B_INST}Question: is the person in white indoors Long answer: No, the person in white is not indoors. The video frames show them standing next to birds and other objects outdoors, with trees, grass, and a river in the background.{E_INST} Short Answer in 1 or 2 word: no'\
#     f'{B_INST}Question: is the person in white outdoors Long answer: Yes, the person in white is outdoors in all the frames.{E_INST} Short Answer in 1 or 2 word: yes'\
#     f'{B_INST}Question: <question> Long answer: <long answer>{E_INST} Short Answer in 1 or 2 word: '
question_answer_template = f'{B_TOK}{B_INST} {B_SYS}You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. The User will provide information about the content in a video and ask you a question about this video. You should answer the question winthin 100 word based on the information provided by the user.{E_SYS} {E_INST}'\
    f' Sure, I can help you with that! {E_TOK} {B_TOK} {B_INST}<video info> Answer the following question within 100 word base on the infomation above: <question>{E_INST} Short Answer in 100 word without any explaination: '
choice_qa_template = f"{B_TOK}{B_INST} {B_SYS}You are a helpful expert in first person view video analysis based on given video information.{E_SYS} {E_INST}"\
    f" Sure, I can help you with that! {E_TOK} {B_TOK} {B_INST}Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. If you are not sure, answer with the most likely answer. You are given some language descriptions of a first person view video. Here are the descriptions:\n<video info>\n Here is the question: <question>?\n Here are the choices:\n (A): <option0>\n (B): <option1>\n (C): <option2>\n (D): <option3>\n (E): <option4>.\n You must provide a single-letter answer (A, B, C, D, E). If none of above, answer with the most likely answer. You CANNOT answer 'None of above'."\
    f"{E_INST} The single-letter answer is: "


BOS = '<|begin_of_text|>'
SHI = '<|start_header_id|>'
EHI = '<|end_header_id|>'
EOT = '<|eot_id|>'
llama_3_qa_template = f'{BOS}{SHI}You are a helpful expert in first person view video analysis based on given video information.{EHI}{EOT}\
    {SHI}user{EHI}<video info> Answer the following question within 100 word base on the infomation above: <question>? {EOT}\
    {SHI}assistant{EHI}Short Answer in 100 word without any explaination: '

llama3_choice_qa_template = f'{BOS}{SHI}You are a helpful expert in first person view video analysis based on given video information.{EHI}{EOT}\
{SHI}user{EHI}Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, \
and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. \
If you are not sure, answer with the most likely answer. You are given some language descriptions of a first person view video. \
Here are the descriptions:\n<video info>\n\
Here is the question: <question>?\n\
Here are the choices:\n (A): <option0>\n (B): <option1>\n (C): <option2>\n (D): <option3>\n (E): <option4>.\n {EOT}\
{SHI}assistant{EHI}The single-letter answer is: '
# ---------prompt example ---------
# '''
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>

# Cutting Knowledge Date: December 2023
# Today Date: 23 July 2024

# You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

# What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
def tensor_similarity(query_embeddings,
                corpus_embeddings,
                score_function=cos_sim, device = 'cpu'):

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    query_embeddings = query_embeddings.to(device)
    corpus_embeddings = corpus_embeddings.to(device)
    sim = score_function(query_embeddings, corpus_embeddings)
    return sim

def text_encode(questions, clip_cap_list, text_encoder):
    bs = 16
    q_episodes = len(questions)//bs
    c_episodes = len(clip_cap_list)//bs
    q_embedding, c_embedding = None, None
    
    questions = [{'text':t} for t in questions]
    clip_cap_list = [{'text':t} for t in clip_cap_list]
    for i in range(q_episodes+1):
        if i < q_episodes:
            batch_q = questions[i*bs:(i+1)*bs]
        elif q_episodes*bs == len(questions):
            break
        else:
            batch_q = questions[i*bs:]
        if q_embedding is None:
            q_embedding = text_encoder.encode(batch_q)
        else:
            q_embedding = np.concatenate((q_embedding,text_encoder.encode(batch_q)))
            
    for i in range(c_episodes+1):
        if i < c_episodes:
            batch_c = clip_cap_list[i*bs:(i+1)*bs]
        elif c_episodes*bs == len(clip_cap_list):
            break
        else:
            batch_c = clip_cap_list[i*bs:]
        if c_embedding is None:
            c_embedding = text_encoder.encode(batch_c)
        else:
            c_embedding = np.concatenate((c_embedding, text_encoder.encode(batch_c)))
    return q_embedding, c_embedding

def cap_filter(q_embedding, c_embedding, topk = 5):    
    sim_matrix = tensor_similarity(q_embedding, c_embedding).cpu().numpy()
    sort_index = np.argsort(sim_matrix,axis=-1)
    selected_index = sort_index[:,-topk:]
    return selected_index
    
def choice_video_qa(vid, question, tgt_answer, text_encoder, model, tokenizer, config):
    dataset_name = config['Data']['set_name']
    clip_cap_dir = config['Data']['data_paths'][dataset_name]['clip_cap']
    video_cap_dir = config['Data']['data_paths'][dataset_name]['video_cap']
    text_embed_dir = config['Data']['data_paths'][dataset_name]['text_emb']
    embed_path = os.path.join(text_embed_dir, vid+'.npz')
    
    # tgt_answer_list = [a[0].item() for a in tgt_answer]
    question_list = [q[0][0] for q in question]
    options = []
    for q in question:
        options.append([o[0] for o in q[1:]])
    
    file_name = vid + '.json'
    if os.path.exists(os.path.join(clip_cap_dir,file_name)):
        with open(os.path.join(clip_cap_dir,file_name), 'r') as f:
            clip_cap_dict = json.load(f)
    else:
        return None
    if os.path.exists(os.path.join(video_cap_dir,vid + '.txt')):
        with open(os.path.join(video_cap_dir,vid + '.txt'), 'r') as f:
            video_cap = ''.join(f.readlines())
    else:
        return None
        
    clip_cap_list = [l['cap'] for l in clip_cap_dict]
    
    if os.path.exists(embed_path):
        embed_dict = np.load(embed_path, allow_pickle=True)['arr_0'].tolist()
        q_embedding, c_embedding = embed_dict['q_embedding'], embed_dict['c_embedding']
    else:
        q_embedding, c_embedding = text_encode(question_list, clip_cap_list, text_encoder)
        if text_embed_dir:
            np.savez(embed_path, {'q_embedding': q_embedding, 'c_embedding':c_embedding})
    # return None
    selected_index = cap_filter(q_embedding, c_embedding, topk=config['Infer']['topk'])
    clip_cap_list = np.array(clip_cap_list)
    clip_cap_filtered = clip_cap_list[selected_index].tolist()
    
    prompt_prefix = llama3_choice_qa_template
    video_info = {'video info': video_cap}
    prompt_list = []
    for i,q in enumerate(question_list):
        vid_q = {'Question':q}
        clip_caption = {'clip captions': clip_cap_filtered[i]}
        prompt = prompt_prefix.replace('<video info>', f'{video_info} {clip_caption}')
        prompt = prompt.replace('<question>', q)
        for idx, opt in enumerate(options[i]):
            prompt = prompt.replace(f'<option{idx}>', opt)
        prompt_list.append(prompt)
    bs = config['Model']['bs']
    batch_num = len(prompt_list)//bs
    tokenizer.padding_side='left'
    
    output_list = []
    for i in range(batch_num+1):
        if i < batch_num:
            batch_list = prompt_list[i*bs:(i+1)*bs]
        elif i*bs == len(prompt_list):
            break
        else:
            batch_list = prompt_list[i*bs:]
        batch_input = tokenizer.batch_encode_plus(batch_list,padding=True,return_tensors='pt').to(device=model.device)
        output_tokens = model.generate(**batch_input, max_new_tokens = 200, temperature=0.7, repetition_penalty=1.0, do_sample=False, length_penalty=1.0)
        output = tokenizer.batch_decode(output_tokens,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,)
        output_list.extend(output)
        
    answer_list = []
    for output, q in zip(output_list, question_list):
        pred = output.split("The single-letter answer is: ")[-1]
        pred = pred.replace('\n','')
        # pred = int(pred[0])
        answer_list.append({'question':q, 'pred': pred})
    return answer_list


def video_qa(vid, question, tgt_answer, text_encoder, model, tokenizer, config):
    dataset_name = config['Data']['set_name']
    clip_cap_dir = config['Data']['data_paths'][dataset_name]['clip_cap']
    video_cap_dir = config['Data']['data_paths'][dataset_name]['video_cap']
    text_embed_dir = config['Data']['data_paths'][dataset_name]['text_emb']
    use_options = config['Task']['options']
    embed_path = os.path.join(text_embed_dir, vid+'.npz')
    
    # tgt_answer_list = [a[0].item() for a in tgt_answer]
    question_list = [q[0][0] for q in question]
    if use_options:
        options = []
        for q in question:
            options.append([o[0] for o in q[1:]])
    
    file_name = vid + '.json'
    if os.path.exists(os.path.join(clip_cap_dir,file_name)):
        with open(os.path.join(clip_cap_dir,file_name), 'r') as f:
            clip_cap_dict = json.load(f)
    else:
        return None
    if os.path.exists(os.path.join(video_cap_dir,vid + '.txt')):
        with open(os.path.join(video_cap_dir,vid + '.txt'), 'r') as f:
            video_cap = ''.join(f.readlines())
    else:
        return None
        
    clip_cap_list = [l['cap'] for l in clip_cap_dict]
    
    if os.path.exists(embed_path):
        embed_dict = np.load(embed_path, allow_pickle=True)['arr_0'].tolist()
        q_embedding, c_embedding = embed_dict['q_embedding'], embed_dict['c_embedding']
    else:
        q_embedding, c_embedding = text_encode(question_list, clip_cap_list, text_encoder)
        if text_embed_dir:
            np.savez(embed_path, {'q_embedding': q_embedding, 'c_embedding':c_embedding})
    # return None
    selected_index = cap_filter(q_embedding, c_embedding, topk=config['Infer']['topk'])
    clip_cap_list = np.array(clip_cap_list)
    clip_cap_filtered = clip_cap_list[selected_index].tolist()
    
    prompt_prefix = llama_3_qa_template#question_answer_template
    video_info = {'video info': video_cap}
    prompt_list = []
    for i,q in enumerate(question_list):
        vid_q = {'Question':q}
        clip_caption = {'clip captions': clip_cap_filtered[i]}
        prompt = prompt_prefix.replace('<video info>', f'{video_info} {clip_caption}')
        prompt = prompt.replace('<question>', q)
        prompt_list.append(prompt)
    bs = config['Model']['bs']
    batch_num = len(prompt_list)//bs
    tokenizer.padding_side='left'
    
    output_list = []
    for i in range(batch_num+1):
        if i < batch_num:
            batch_list = prompt_list[i*bs:(i+1)*bs]
        elif i*bs == len(prompt_list):
            break
        else:
            batch_list = prompt_list[i*bs:]
        batch_input = tokenizer.batch_encode_plus(batch_list,padding=True,return_tensors='pt').to(device=model.device)
        output_tokens = model.generate(**batch_input, max_new_tokens = 200, temperature=0.7, repetition_penalty=1.0, do_sample=False, length_penalty=1.0)
        output = tokenizer.batch_decode(output_tokens,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,)
        output_list.extend(output)
        
    answer_list = []
    for output, q, a in zip(output_list, question_list, options):
        pred = output.split("Short Answer in 100 word without any explaination: ")[-1]
        pred = pred.replace('\n','')
        answer_list.append({'question':q, 'answer': a, 'pred': pred})
    return answer_list

def short_answer(vid, result_dir, model, tokenizer):
    # 'User: Question: what did the video regarding the program set in? Long answer:  The video is set in a computer screen. Assistant: Short answer: computer '\
    # 'User: Question: what is a person in a brown blazer blue shirt and glasses doing? Long answer: The person in a brown blazer, blue shirt, and glasses is sitting in a chair and talking to the camera on his phone. Assistant: Short answer: talk '\
    # 'User: Question: who is speaking about the relevance of disorders? Long answer: The man in glasses. Assistant: Short answer: man '\
    # 'User: Question: {} Long answer: {} Assistant: Short answer: '
    prompt_template = short_answer_prompt_template
    if os.path.exists(os.path.join(result_dir, vid+'.json')):
        with open(os.path.join(result_dir, vid+'.json'), 'r') as f:
            result_list = json.load(f)
            f.close()
    else:
        return
    prompt_list = []
    for l in result_list:
        q = l['question']
        a = l['pred']
        prompt_str = prompt_template.replace('<question>', q).replace('<long answer>', a)
        prompt_list.append(prompt_str)
    bs = config['Model']['bs']
    batch_num = len(prompt_list)//bs
    tokenizer.padding_side='left'
    
    output_list = []
    for i in range(batch_num+1):
        if i < batch_num:
            batch_list = prompt_list[i*bs:(i+1)*bs]
        elif i*bs == len(prompt_list):
            break
        else:
            batch_list = prompt_list[i*bs:]
        batch_input = tokenizer.batch_encode_plus(batch_list,padding=True,return_tensors='pt').to(device=model.device)
        output_tokens = model.generate(**batch_input, max_new_tokens = 10, temperature=0.7, repetition_penalty=1.0, do_sample=False, length_penalty=1.0)
        output = tokenizer.batch_decode(output_tokens,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,)
        output_list.extend(output)
        
    for i, (output, result) in enumerate(zip(output_list, result_list)):
        pred = output.split("Short Answer in 1 or 2 word: ")[-1]
        pred = pred.replace('\n','').replace('.','').replace(' ','')
        result_list[i]['short_answer'] = pred
    with open(os.path.join(result_dir, vid+'.json'), 'w') as f:
        json.dump(result_list, f)
        f.close()
    

def main(config):
    rank_id = get_rank() if config['Infer']['distributed'] else 0
    device = torch.device('cuda', rank_id)
    torch.cuda.set_device(rank_id)
    dataset_name = config['Data']['set_name']
    tag = config['Task']['tag']
    result_dir = os.path.join(config['Task']['result_dir'], f'{dataset_name}_{tag}')#'./result/youcook2/videochat_cap'
    text_embed_dir = config['Data']['data_paths'][dataset_name]['text_emb']
    if rank_id==0:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if text_embed_dir and not os.path.exists(text_embed_dir):
            os.makedirs(text_embed_dir)
    dataset = DataCls[dataset_name](config['Data']['data_paths'][dataset_name]['file'],
                    config['Data']['data_paths'][dataset_name]['video_dir'],
                    result_dir)
    sampler = DistributedSampler(dataset, shuffle=False) if config['Infer']['distributed'] else None
    data_loader = DataLoader(dataset, 1, shuffle=False, sampler=sampler)
    if 'sentence' in config['Model']['models']:
        text_encoder = SentenceModel(config['Model']['SentenceEncoder']['model_path'], device=device)
    elif 'angle' in config['Model']['models']:
        text_encoder = AnglE.from_pretrained(config['Model']['Angle']['pretrianed_weights'],
                                        pretrained_lora_path=config['Model']['Angle']['model_path'])
        text_encoder.set_prompt(prompt=Prompts.A)
        text_encoder.backbone.to(device)
    else:
        text_encoder=None
    if 'vicuna' in config['Model']['models']:
        model, tokenizer = load_vicuna(config['Model']['Vicuna']['model_path'])
    elif 'llama2' in config['Model']['models']:
        model = AutoModelForCausalLM.from_pretrained(config['Model']['LlaMa2']['model_path'],torch_dtype=torch.float16)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(config['Model']['LlaMa2']['model_path'])
        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
            )
    elif 'llama3' in config['Model']['models']:
        model = AutoModelForCausalLM.from_pretrained(config['Model']['LlaMa3']['model_path'],torch_dtype=torch.float16)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(config['Model']['LlaMa3']['model_path'])
        tokenizer.add_special_tokens(
            {
                "pad_token": "<|finetune_right_pad_id|>",
            }
            )
    else:
        model, tokenizer = None, None
    
    if model:
        model = model.to(device)
        if config['Infer']['distributed']:
            model = DistributedDataParallel(model, device_ids=[rank_id], output_device=rank_id,gradient_as_bucket_view=True).module
    
    for vid, question, answer in tqdm(data_loader, desc=f'{dataset_name}_{tag}'):
        # answer_list = video_qa(vid[0], question, answer, text_encoder, model, tokenizer, config)
        # if answer_list:
        #     with open(os.path.join(result_dir, vid[0]+'.json'), 'w+') as f:
        #         json.dump(answer_list,f)
        answer_list = choice_video_qa(vid[0], question, answer, text_encoder, model, tokenizer, config)
        if answer_list:
            with open(os.path.join(result_dir, vid[0]+'.json'), 'w+') as f:
                json.dump(answer_list,f)
        # short_answer(vid[0], result_dir, model, tokenizer)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config_qa.yaml')
    parser.add_argument('--local_rank', type=int, default=None)
    args = parser.parse_args()
    
    config = yaml.load(open(args.config,'r'), yaml.FullLoader)
    if config['Infer']['distributed']:
        init_process_group('nccl')
    main(config)