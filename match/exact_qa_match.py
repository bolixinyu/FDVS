import json
import os
from argparse import ArgumentParser
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='anet')
parser.add_argument('--result_dir', type=str, default='./result/qa/anet_qa_llama2')
args = parser.parse_args()

file_map = {'msrvtt': './data/qa/msrvtt/test_qa_re.json', 'anet': 'data/qa/anet/val_qa_pair.json'}
data_path = file_map[args.dataset]

with open(data_path,'r') as f:
	anno_list = json.load(f)
	f.close()
if args.dataset == 'msrvtt':
	data_list = [d['clip_name'] for d in anno_list]
else:
	data_list = list(anno_list.keys())

p_num = 0
total_num = 0
for vid in tqdm(data_list):
	if os.path.exists(os.path.join(args.result_dir, vid+'.json')):
		with open(os.path.join(args.result_dir, vid+'.json'), 'r') as f:
			results = json.load(f)
		for i,r in  enumerate(results):
			rq = r['question']
			rt = str.lower(r['answer'])
			rp = str.lower(r['short_answer'].split('shortanswerin1word:')[-1])
			results[i]['short_answer'] = rp
			if rt == rp:
				p_num+=1
			total_num+=1
		with open(os.path.join(args.result_dir, vid+'.json'), 'w+') as f:
			json.dump(results, f)
	else:
		continue

print(p_num*1.0/total_num, total_num)
		