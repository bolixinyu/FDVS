import numpy as np
from text2vec import SentenceModel, semantic_search, cos_sim
import os
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from angle_emb import AnglE, Prompts
import json
import argparse
from torch.distributed import init_process_group, get_rank

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='charades')
parser.add_argument('--video_cap', type=str, default = './result/charades/video_cap')
parser.add_argument('--clip_cap', type=str, default = './result/charades/clip_cap')
parser.add_argument('--bs', type=int, default = 64)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--local_rank', type=int, default=None)
args = parser.parse_args()

rank_id = get_rank() if args.local_rank else args.gpu
device = torch.device('cuda', rank_id)
torch.cuda.set_device(rank_id)
embeder = AnglE.from_pretrained('/chenyaofo/hf_models/Llama-2-13b-hf', pretrained_lora_path='/chenyaofo/hf_models/angle-llama-13b-nli')
embeder.set_prompt(prompt=Prompts.A)
embeder.backbone.to(device)

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

def ap_score(sorted_labels):
    nr_relevant = len([x for x in sorted_labels if x > 0])
    if nr_relevant == 0:
        return 0.0

    length = len(sorted_labels)
    ap = 0.0
    rel = 0

    for i in range(length):
        lab = sorted_labels[i]
        if lab >= 1:
            rel += 1
            ap += float(rel) / (i + 1.0)
    ap /= nr_relevant
    return ap

def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt

def eval_q2m(scores, q2m_gts):
    n_q, n_m = scores.shape

    gt_ranks = np.zeros((n_q,), np.int32)
    aps = np.zeros(n_q)
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = np.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(np.where(gt_ranks <= 100)[0]) / n_q
    medr = np.median(gt_ranks)
    meanr = gt_ranks.mean()

    return (r1, r5, r10, r100, medr, meanr)

def t2v_map(c2i, t2v_gts):
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0]*len(d_i)

        x = t2v_gts[i][0]
        labels[x] = 1

        sorted_labels = [labels[x] for x in np.argsort(d_i)]

        current_score = ap_score(sorted_labels)
        perf_list.append(current_score)
    return np.mean(perf_list)


def cal_perf(t2v_all_errors, t2v_gt):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr) = eval_q2m(t2v_all_errors, t2v_gt)
    t2v_map_score = t2v_map(t2v_all_errors, t2v_gt)


    print(" * Text to Video:")
    print(" * r_1_5_10_100, medr, meanr: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_r100, 1)]))
    print(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1)))
    print(" * mAP: {}".format(round(t2v_map_score, 4)))
    print(" * "+'-'*10)

    return (t2v_r1, t2v_r5, t2v_r10,t2v_r100, t2v_medr, t2v_meanr, t2v_map_score)


def eval_epoch(model, val_video_dataset, val_text_dataset, opt):
    model.eval()
    logger.info("Computing scores")

    context_info = compute_context_info(model, val_video_dataset, opt)
    query_context_scores, global_query_context_scores, score_sum, query_metas = compute_query2ctx_info(model,
                                                                                                        val_text_dataset,
                                                                                                        opt,
                                                                                                        context_info)
    video_metas = context_info['video_metas']

    v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
    print('clip_scale_scores:')
    cal_perf(-1 * query_context_scores, t2v_gt)
    print('frame_scale_scores:')
    cal_perf(-1 * global_query_context_scores, t2v_gt)
    print('score_sum:')
    t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = cal_perf(-1 * score_sum, t2v_gt)
    currscore = 0
    currscore += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

    return currscore

# embeder = SentenceModel('/youzeng/pretrained_weights/bert-base-nli-mean-tokens',device='cuda')
gt_file_map = {'anet':'./data/captions/val_1.json', 'charades': './data/charades/charades_test.json'}

gt_file = gt_file_map[args.dataset]
cap_dir = args.video_cap
clip_cap_dir = args.clip_cap

with open(gt_file, 'r') as f:
    annos = json.load(f)

# tmp_dir = cap_dir
# hard_dir = './tmp/hard_video_cap'
# hard_vids = [l.split('.')[0] for l in os.listdir(hard_dir)]

gt_str = []
cap_str = []
cap_num = []
bs = args.bs
gt_embedding=None
cap_embedding = None
query_vid = []
video_vid = []
for vid in annos.keys():
    sentence_list = annos[vid]['sentences']
    gt_str.extend(sentence_list)
    if args.dataset == 'anet':
        vid = vid[-11:]
    fp = vid + '.txt'
    query_vid.extend([vid]*len(sentence_list))
    if os.path.exists(os.path.join(cap_dir,fp)):
        
        # gt_vid.append(vid)
        # f = open(os.path.join(gt_dir,fp),'r',encoding='utf8')
        # gt_des = ''.join(f.readlines())
        # gt_str.append(gt_des)
        # f.close()
        # if vid in hard_vids:
        #     cap_dir = hard_dir
        # else:
        #     cap_dir = tmp_dir
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
        video_vid.append(vid)
    else:
        # continue
        cap_str.append('')
        cap_num.append(1)
        video_vid.append(vid)
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
cap_cumnum = np.cumsum(cap_num)
cap_cumnum = np.concatenate((np.array([0]),cap_cumnum))

new_sim_matrix = np.zeros((len(gt_str), len(cap_cumnum)-1))
for i in tqdm(range(len(gt_str))):
    for j in range(1,len(cap_cumnum)):
        tm_matrix = sim_matrix[i,cap_cumnum[j-1]:cap_cumnum[j]]
        new_sim_matrix[i,j-1] = tm_matrix.max()

v2t_gt, t2v_gt = get_gt(video_vid, query_vid)
t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = cal_perf(-1*new_sim_matrix, t2v_gt)