import json
import os
import numpy as np
import torch
from text2vec import SentenceModel, cos_sim
from match.angle_emb import AnglE, Prompts
import argparse
from tqdm import tqdm


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

def answer_number(q_embedding, c_embedding):    
    sim_matrix = tensor_similarity(q_embedding, c_embedding).cpu().numpy()
    answer = np.argmax(sim_matrix, axis=-1)
    return answer

def exact_match(result_dir): 
    file_list = os.listdir(result_dir)
    with open('./data/next_qa/val.json', 'r') as f:
        val_data = json.load(f)
        f.close()
    type_list = {'C':[],'T':[],'D':[]}
    pred = []
    answer = []
    for file in tqdm(file_list, total=len(file_list)):
        vid = file.split('.')[0]
        q_list = []
        for item in val_data:
            if str(item['video']) == vid:
                q_list.append(item)
        with open(result_dir + file, 'r') as f:
            data = json.load(f)
            for item in data:
                
                q = item['question']
                pred_num = -1
                if 'A' in item['pred']:
                    pred_num = 0
                elif 'B' in item['pred']:
                    pred_num = 1
                elif 'C' in item['pred']:
                    pred_num = 2
                elif 'D' in item['pred']:
                    pred_num = 3
                elif 'E' in item['pred']:
                    pred_num = 4
                else:
                    pred_num = 4
                    print(f'Error: {item}')
                pred.append(pred_num)
                answer.append(item['answer'])
                for q_item in q_list:
                    if q_item['question'] == q:
                        if pred_num == q_item['answer']:
                            type_list[q_item['type'][0]].append(1)
                        else:
                            type_list[q_item['type'][0]].append(0)
    pred = np.array(pred)
    answer = np.array(answer)
    accu = np.sum(pred == answer) / len(answer)
    print(f'{np.sum(pred == answer)} / {len(answer)} = {np.sum(pred == answer) / len(answer)}')

    for k in type_list.keys():
        print(f'{k}: {np.sum(type_list[k])} / {len(type_list[k])} = {np.sum(type_list[k]) / len(type_list[k])}')
    return accu

def embed_match(result_dir, text_encoder): 
    file_list = os.listdir(result_dir)
    pred = []
    answer = []
    for file in tqdm(file_list, total=len(file_list)):
        vid = file.split('.')[0]
        q_list = []
        with open(result_dir + file, 'r') as f:
            data = json.load(f)
            for item in data:
                pred_embed = text_encoder.encode([{'text':item['pred']}])
                options = [{'text':o} for o in item['options']]
                option_embed = text_encoder.encode(options)
                pred_num = answer_number(pred_embed, option_embed)
                pred.append(pred_num[0])
                answer.append(item['answer'])
    pred = np.array(pred)
    answer = np.array(answer)
    accu = np.sum(pred == answer) / len(answer)
    print(f'{np.sum(pred == answer)} / {len(answer)} = {np.sum(pred == answer) / len(answer)}')
    return accu




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama2_path', type=str, default='/chenyaofo/hf_models/Llama-2-13b-hf')
    parser.add_argument('--angle_path', type=str, default='/chenyaofo/hf_models/angle-llama-13b-nli')
    parser.add_argument('--result_dir', type=str, default='./result/qa/rebuttal_exp/next_qa_llama3/')
    parser.add_argument('--match_func', type=str, default='exact')
    args = parser.parse_args()
    if args.match_func == 'exact':
        accu = exact_match(args.result_dir)
    elif args.match_func == 'embed':
        text_encoder = AnglE.from_pretrained(args.llama2_path,
                                        pretrained_lora_path=args.angle_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        text_encoder.set_prompt(prompt=Prompts.A)
        text_encoder.backbone.to(device)
        accu = embed_match(args.result_dir, text_encoder)