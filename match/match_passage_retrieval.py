from text2vec import SentenceModel, semantic_search, cos_sim
import os
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from typing import List
from tqdm import tqdm
from angle_emb import AnglE, Prompts

from transformers import AutoConfig, AutoTokenizer
from tevatron.modeling import DenseModel
import torch
import numpy as np

# embeder = AnglE.from_pretrained('/youzeng/pretrained_weights/Llama-2-13b-hf', pretrained_lora_path='/youzeng/pretrained_weights/angle-llama-13b-nli')
# embeder.set_prompt(prompt=Prompts.A)
# embeder.cuda()

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
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
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
    # query_norm = torch.linalg.norm(query_embeddings,dim=1)
    # corpus_norm = torch.linalg.norm(corpus_embeddings, dim=1)
    # sim = query_norm.unsqueeze(1)@corpus_norm.unsqueeze(0)*sim
    return sim

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
config = AutoConfig.from_pretrained(
	'/mnt/cephfs/home/youzeng/pretrained_weights/co-condenser-marco-retriever',
	num_labels=1
)
tokenizer = AutoTokenizer.from_pretrained(
	'/mnt/cephfs/home/youzeng/pretrained_weights/co-condenser-marco-retriever'
)

embeder = DenseModel.load(
	model_name_or_path='/mnt/cephfs/home/youzeng/pretrained_weights/co-condenser-marco-retriever',
	config=config,
)

embeder = embeder.to(device)
embeder.eval()

gt_dir = './result/anet/groundtruth'
cap_dir = './result/anet/video_cap/'

gt_str = []
cap_str = []
bs = 32
gt_embedding=None
cap_embedding = None
gt_vid = []
for fp in os.listdir(gt_dir):
    vid = fp.split('.')[0]
    gt_vid.append(vid)
    f = open(os.path.join(gt_dir,fp),'r')
    gt_str.append(''.join(f.readlines()))
    f.close()
    if os.path.exists(os.path.join(cap_dir,fp)):
        f = open(os.path.join(cap_dir,fp),'r')
        cap_str.append(''.join(f.readlines()))
        f.close()
    else:
        cap_str.append('')
episodes = len(gt_str)//bs

# gt_str = [{'text':t} for t in gt_str]
# cap_str = [{'text':t} for t in cap_str]
for i in tqdm(range(episodes+1),total=episodes+1):
    if i < episodes:
        batch_gt = gt_str[i*bs:(i+1)*bs]
        batch_cap = cap_str[i*bs:(i+1)*bs]
    else:
        batch_gt = gt_str[i*bs:]
        batch_cap = cap_str[i*bs:]
    batch_gt = tokenizer.batch_encode_plus(batch_gt, add_special_tokens=False, return_tensors='pt',padding=True).to(device)
    batch_cap = tokenizer.batch_encode_plus(batch_cap, add_special_tokens=False, return_tensors='pt',padding=True).to(device)
    if gt_embedding is None:
        gt_embedding = embeder(query=batch_gt).q_reps.cpu().detach().numpy()
    else:
        gt_embedding = np.concatenate((gt_embedding,embeder(query=batch_gt).q_reps.cpu().detach().numpy()))
    if cap_embedding is None:
        cap_embedding = embeder(passage=batch_cap).p_reps.cpu().detach().numpy()
    else:
        cap_embedding = np.concatenate((cap_embedding,embeder(passage=batch_cap).p_reps.cpu().detach().numpy()))
    # break
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

tv_metrics = compute_metrics(sim_matrix)
vt_metrics = compute_metrics(sim_matrix.T)
print('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

print("Text-to-Video:")
print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
print("Video-to-Text:")
print('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))


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