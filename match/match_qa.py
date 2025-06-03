from text2vec import SentenceModel, semantic_search, cos_sim
import os
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from typing import List
from tqdm import tqdm
from angle_emb import AnglE, Prompts
import json
import argparse
import pickle as pkl
import pdb
from torch.distributed import init_process_group, get_rank
# from torch.nn.parallel.distributed import DistributedDataParallel
# init_process_group('nccl')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='msrvtt')
parser.add_argument('--result_dir', type=str, default = './result/qa/msrvtt_qa_/')
# parser.add_argument('--clip_cap', type=str, default = './result/msrvtt/clip_cap_interpretable_all_visual_reduce/')
parser.add_argument('--local_rank', type=int, default=None)
args = parser.parse_args()

rank_id = get_rank() if args.local_rank else 0
device = torch.device('cuda', rank_id)
torch.cuda.set_device(rank_id)
embeder = AnglE.from_pretrained('/youzeng/pretrained_weights/Llama-2-13b-hf', pretrained_lora_path='/youzeng/pretrained_weights/angle-llama-13b-nli')
embeder.set_prompt(prompt=Prompts.A)
# import pdb
# pdb.set_trace()
embeder.backbone.to(device)

# def cos_sim(a: Tensor, b: Tensor):
#     """
#     Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
#     :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
#     """
#     if not isinstance(a, torch.Tensor):
#         a = torch.tensor(a)

#     if not isinstance(b, torch.Tensor):
#         b = torch.tensor(b)

#     if len(a.shape) == 1:
#         a = a.unsqueeze(0)

#     if len(b.shape) == 1:
#         b = b.unsqueeze(0)

#     a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
#     b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
#     return torch.mm(a_norm, b_norm.transpose(0, 1))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def tensor_similarity(query_embeddings: Tensor,
                corpus_embeddings: Tensor,
                score_function=cos_sim):

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
    return torch.diag(sim)

def compute_metrics(x):
    sx = np.argsort(-x, axis=1)
    d = np.arange(0, len(sx))
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


# embeder = SentenceModel('/youzeng/pretrained_weights/bert-base-nli-mean-tokens',device='cuda')
# embeder = SentenceModel("/youzeng/pretrained_weights/bert-base-nli-mean-tokens", device=device)
data_file_map = {'msrvtt': './data/qa/msrvtt/test_qa_re.json', 'anet':'./result/anet/groundtruth', 'didemo':'./result/didemo/groundtruth'}
data_file = data_file_map[args.dataset]
with open(data_file, 'r') as f:
    anno_list = json.load(f)
    f.close()
if args.dataset == 'msrvtt':
    data_list = [d['clip_name'] for d in anno_list]
elif args.dataset == 'anet':
    pass
    
result_dir = args.result_dir

# tmp_dir = cap_dir
# hard_dir = './tmp/hard_video_cap'
# hard_vids = [l.split('.')[0] for l in os.listdir(hard_dir)]

gt_str = []
cap_str = []
view_list = []
bs = 128
gt_embedding=None
cap_embedding = None
template = 'The answer to the question {} is: {}'
view_template = 'Question: {} Answer: {} pred: {}'
for vid in data_list:
    result_file = os.path.join(result_dir, vid+'.json')
    if os.path.exists(result_file):
        with open(result_file,'r') as f:
            result_list = json.load(f)
        for result in result_list:
            q = result['question']
            p = result['pred']
            t= result['answer']
            tgt_str = template.format(q,t)
            pred_str = template.format(q,p)
            view_str = view_template.format(q,t,p)
            gt_str.append(tgt_str)
            cap_str.append(pred_str)
            view_list.append(view_str)
    else:
        print(result_dir,result_file)
        # cap_str.append('')
        # cap_num.append(1)
gt_episodes = len(gt_str)//bs
cap_episodes = len(cap_str)//bs
gt_str = [{'text':t} for t in gt_str]
cap_str = [{'text':t} for t in cap_str]
for i in tqdm(range(gt_episodes+1),total=gt_episodes+1):
    if i < gt_episodes:
        batch_gt = gt_str[i*bs:(i+1)*bs]
        batch_cap = cap_str[i*bs:(i+1)*bs]
    else:
        batch_gt = gt_str[i*bs:]
        batch_cap = cap_str[i*bs:]
    if gt_embedding is None:
        gt_embedding = embeder.encode(batch_gt)
    else:
        gt_embedding = np.concatenate((gt_embedding,embeder.encode(batch_gt)))
    if cap_embedding is None:
        cap_embedding = embeder.encode(batch_cap)
    else:
        cap_embedding = np.concatenate((cap_embedding,embeder.encode(batch_cap)))
        
# for i in tqdm(range(cap_episodes+1),total=cap_episodes+1):
#     if i < cap_episodes:
#         batch_cap = cap_str[i*bs:(i+1)*bs]
#     else:
#         batch_cap = cap_str[i*bs:]
#     if cap_embedding is None:
#         cap_embedding = embeder.encode(batch_cap)
#     else:
#         cap_embedding = np.concatenate((cap_embedding,embeder.encode(batch_cap)))
# cap_vid = []
# for fp in os.listdir(cap_dir):
#     vid = fp.split('.')[0]
#     cap_vid.append(vid)
#     f = open(os.path.join(cap_dir,fp),'r')
#     cap_str.append(''.join(f.readlines()))
#     if len(cap_str) == bs:
#         if cap_embedding is None:
#            cap_embedding = embeder.encode(cap_str)
#         else:
#             cap_embedding = np.concatenate((cap_embedding,embeder.encode(cap_str)))
#         cap_str.clear()

sim_matrix = tensor_similarity(gt_embedding, cap_embedding).cpu().numpy()
with open('./tmp/msrvtt_qallama2.pkl','wb') as f:
    pkl.dump({'sim':sim_matrix, 'qa_pair': view_list}, f)
# pdb.set_trace()
index = sim_matrix>0.7
accu = np.sum(index)/sim_matrix.shape[0]
view_list = np.array(view_list)
accu_view = view_list[index]


# hits = semantic_search(gt_embedding, cap_embedding, top_k=10)
# hits = semantic_search(cap_embedding, gt_embedding, top_k=10)
# top1_num = 0
# top_5_num = 0
# top_10_num = 0

# for i, hit_res in enumerate(hits):
#     top_10_hits = [cap_vid[f['corpus_id']] for f in hit_res]
#     # top_10_hits = [gt_vid[f['corpus_id']] for f in hit_res]
#     top_10_score = [f['score'] for f in hit_res]
#     query_id = gt_vid[i]
#     # query_id = cap_vid[i]
#     if query_id in top_10_hits[:1]:
#         top1_num += 1
#     if query_id in top_10_hits[:5]:
#         top_5_num += 1
#     if query_id in top_10_hits[:10]:
#         top_10_num += 1
# print('text2video:')
# # print('video2text:')
# print('top1:{}'.format(top1_num/len(cap_embedding)))
# print('top5:{}'.format(top_5_num/len(cap_embedding)))
# print('top10:{}'.format(top_10_num/len(cap_embedding)))
