from text2vec import SentenceModel, semantic_search, cos_sim
import os
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from angle_emb import AnglE, Prompts
import json
import argparse

#CUDA_VISIBLE_DEVICES=0 python match/match_clip_story.py --dataset anet --video_cap ./result/rebuttal_exp/anet/video_cap_woAction --clip_cap ./result/rebuttal_exp/anet/clip_cap_woAction

from torch.distributed import init_process_group, get_rank
# from torch.nn.parallel.distributed import DistributedDataParallel
# init_process_group('nccl')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='msrvtt')
parser.add_argument('--video_cap', type=str, default = './result/msrvtt/video_cap_llama2/')
parser.add_argument('--clip_cap', type=str, default = './result/msrvtt/clip_cap_llama2/')
parser.add_argument('--bs', type=int, default = 128)
parser.add_argument('--local_rank', type=int, default=None)
args = parser.parse_args()

rank_id = get_rank() if args.local_rank else 0
device = torch.device('cuda', rank_id)
torch.cuda.set_device(rank_id)
embeder = AnglE.from_pretrained('/chenyaofo/hf_models/Llama-2-13b-hf', pretrained_lora_path='/chenyaofo/hf_models/angle-llama-13b-nli')
embeder.set_prompt(prompt=Prompts.A)
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
    return sim

def compute_metrics(x, gt_vid):
        
    sx = np.argsort(-x, axis=1)
    d = np.arange(0, len(sx))
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    anno_file = './data/captions/val_1_act_new.json'
    with open(anno_file, 'r') as f:
        anno = json.load(f)
    act_list = []
    for v in gt_vid:
        vid = 'v_'+v
        if vid in anno and 'actions' in anno[vid]:
            act = anno[vid]['actions']
            act_list.append(act)
        else:
            act_list.append('Unk')
    unique_act = list(set(act_list))
    act_list = np.array(act_list)
    
    cls_metrics = {}
    for act in unique_act:
        ind_act = ind [act_list==act]
        metrics={}
        metrics['R1'] = float(np.sum(ind_act == 0)) * 100 / len(ind_act)
        metrics['R5'] = float(np.sum(ind_act < 5)) * 100 / len(ind_act)
        metrics['R10'] = float(np.sum(ind_act < 10)) * 100 / len(ind_act)
        metrics['MR'] = np.median(ind_act) + 1
        metrics["MedianR"] = metrics['MR']
        metrics["MeanR"] = np.mean(ind_act) + 1
        metrics["cols"] = [int(i) for i in list(ind_act)]
        cls_metrics[act] = metrics
    return cls_metrics


# embeder = SentenceModel('/youzeng/pretrained_weights/bert-base-nli-mean-tokens',device='cuda')
# embeder = SentenceModel("/youzeng/pretrained_weights/bert-base-nli-mean-tokens", device=device)
gt_dir_map = {'msrvtt': './result/msrvtt/groundtruth', 'anet':'./result/anet/groundtruth', 'didemo':'./result/didemo/groundtruth'}

gt_dir = gt_dir_map[args.dataset]
cap_dir = args.video_cap
clip_cap_dir = args.clip_cap

# tmp_dir = cap_dir
# hard_dir = './tmp/hard_video_cap'
# hard_vids = [l.split('.')[0] for l in os.listdir(hard_dir)]

gt_str = []
cap_str = []
cap_num = []
bs = args.bs
gt_embedding=None
cap_embedding = None
gt_vid = []
for fp in os.listdir(gt_dir):
    vid = fp.split('.')[0]
    gt_vid.append(vid)
    f = open(os.path.join(gt_dir,fp),'r',encoding='utf8')
    gt_des = ''.join(f.readlines())
    gt_str.append(gt_des)
    f.close()
    if os.path.exists(os.path.join(cap_dir,fp)):
        f = open(os.path.join(cap_dir,fp),'r',encoding='utf8')
        cap_descript = ''.join(f.readlines())
        f.close()
        cap_sentences = [cap_descript]
        if os.path.exists(os.path.join(clip_cap_dir,vid+'.json')):
            with open(os.path.join(clip_cap_dir,vid+'.json'),'r') as f:
                clip_cap_list = json.load(f)
                for c in clip_cap_list:
                    cap_sentences.append(c['cap'])
        else:
            cap_sentences.append('')
        cap_str.extend(cap_sentences)
        cap_num.append(len(cap_sentences))
        
        # gt_str.append(gt_des)
    else:
        cap_str.append('')
        cap_num.append(1)
gt_episodes = len(gt_str)//bs
cap_episodes = len(cap_str)//bs
gt_str = [{'text':t} for t in gt_str]
cap_str = [{'text':t} for t in cap_str]
for i in tqdm(range(gt_episodes+1),total=gt_episodes+1):
    if i < gt_episodes:
        batch_gt = gt_str[i*bs:(i+1)*bs]
    else:
        batch_gt = gt_str[i*bs:]
    if gt_embedding is None:
        gt_embedding = embeder.encode(batch_gt)
    else:
        gt_embedding = np.concatenate((gt_embedding,embeder.encode(batch_gt)))
        
for i in tqdm(range(cap_episodes+1),total=cap_episodes+1):
    if i < cap_episodes:
        batch_cap = cap_str[i*bs:(i+1)*bs]
    else:
        batch_cap = cap_str[i*bs:]
    if cap_embedding is None:
        cap_embedding = embeder.encode(batch_cap)
    else:
        cap_embedding = np.concatenate((cap_embedding,embeder.encode(batch_cap)))

sim_matrix = tensor_similarity(gt_embedding, cap_embedding).cpu().numpy()
cap_cumnum = np.cumsum(cap_num)
cap_cumnum = np.concatenate((np.array([0]),cap_cumnum))

new_sim_matrix = np.zeros((len(gt_str), len(cap_cumnum)-1))
for i in tqdm(range(len(gt_str))):
    for j in range(1,len(cap_cumnum)):
        tm_matrix = sim_matrix[i,cap_cumnum[j-1]:cap_cumnum[j]]
        new_sim_matrix[i,j-1] = tm_matrix.max()
tv_metrics = compute_metrics(new_sim_matrix, gt_vid)
# vt_metrics = compute_metrics(new_sim_matrix.T)
with open('./cls_retrieval.json', 'w+') as f:
    json.dump(tv_metrics, f)

# with open('./log.txt','w') as f:
#     f.write('\t Length-T: {}, Length-V:{}'.format(len(new_sim_matrix), len(new_sim_matrix[0]))+'\n')
#     f.write("Text-to-Video:"+'\n')
#     f.write('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR'])+'\n')
#     f.write("Video-to-Text:"+'\n')
#     f.write('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR'])+'\n')
# print('\t Length-T: {}, Length-V:{}'.format(len(new_sim_matrix), len(new_sim_matrix[0])))

# print("Text-to-Video:")
# print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
# print("Video-to-Text:")
# print('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))


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
