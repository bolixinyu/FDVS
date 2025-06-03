from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from tqdm import tqdm
import os
import json
from argparse import ArgumentParser
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.distributed import init_process_group, get_rank
from data import *
from utils.register import _dict as DataCls

B_TOK, E_TOK = '<s>', '</s>'
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='anet')
parser.add_argument('--result_dir', type=str, default='./result/qa/anet_qa_llama2')
parser.add_argument('--output_dir', type=str, default='./result/qa/judge')
parser.add_argument('--dist', action='store_true', default=False)
args = parser.parse_args()
if args.dist:
	init_process_group('nccl')
file_map = {'msrvtt_qa': './data/qa/msrvtt/test_qa_re.json', 'anet_qa': 'data/qa/anet/val_qa_pair.json'}
dataset_map = {
	'anet_qa':{'file': './data/qa/anet/val_qa_pair.json', 'video_dir': '/youzeng/datasets/anet_videos/'},
	'msrvtt_qa': {'file': './data/qa/msrvtt/test_qa_re.json', 'video_dir': '/youzeng/datasets/msrvtt/MSRVTT/videos/all'}
}
data_path = file_map[args.dataset]

with open(data_path,'r') as f:
	anno_list = json.load(f)
	f.close()
	
judge_template = f"{B_TOK}{B_INST} {B_SYS}You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "\
	"Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"\
	"------"\
	"##INSTRUCTIONS: "\
	"- Focus on the meaningful match between the predicted answer and the correct answer.\n"\
	"- Consider synonyms or paraphrases as valid matches.\n"\
	f"- Evaluate the correctness of the prediction compared to the answer.{E_SYS} {E_INST}"\
    f" Sure, I can help you with that! {E_TOK} {B_TOK} {B_INST}Please evaluate the following video-based question-answer pair:\n\n"\
	f"Question: <question>\n"\
	f"Correct Answer: <answer>\n"\
	f"Predicted Answer: <prediction>\n\n"\
	"Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "\
	"Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."\
	"DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "\
	"For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."\
	f"{E_INST} response in Python dictionary string:"

rank_id = get_rank() if args.dist else 0
device = torch.device('cuda', rank_id)
torch.cuda.set_device(rank_id)
dataset_name = args.dataset

output_dir = os.path.join(args.output_dir,dataset_name)
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
dataset = DataCls[dataset_name](dataset_map[dataset_name]['file'],
				dataset_map[dataset_name]['video_dir'],
				None)
sampler = DistributedSampler(dataset, shuffle=False) if args.dist else None
data_loader = DataLoader(dataset, 1, shuffle=False, sampler=sampler)

model = LlamaForCausalLM.from_pretrained('/youzeng/pretrained_weights/Llama-2-13b-chat-hf',torch_dtype=torch.float16)
model.eval()
tokenizer = LlamaTokenizer.from_pretrained('/youzeng/pretrained_weights/Llama-2-13b-chat-hf')
tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
            )
model.to(device)
tokenizer.padding_side='left'
if args.dist:
    model = DistributedDataParallel(model, device_ids=[rank_id], output_device=rank_id,gradient_as_bucket_view=True).module

for vid, question, answer in tqdm(data_loader, desc=f'{dataset_name}'):
	vid = vid[0]
	dict_list = []
	if os.path.exists(os.path.join(args.result_dir, vid+'.json')):
		with open(os.path.join(args.result_dir, vid+'.json'), 'r') as f:
			results = json.load(f)
		for i,r in  enumerate(results):
			rq = r['question']
			rt = str.lower(r['answer'])
			rp = str.lower(r['pred'])
			prompts = judge_template.replace('<question>', rq)
			prompts = prompts.replace('<answer>', rt)
			prompts = prompts.replace('<prediction>', rp)
			batch_input = tokenizer.batch_encode_plus([prompts],padding=True,return_tensors='pt').to(device=model.device)
			output_tokens = model.generate(**batch_input, max_new_tokens = 20, temperature=0.7, repetition_penalty=1.0, do_sample=False, length_penalty=1.0)
			output = tokenizer.batch_decode(output_tokens,
							skip_special_tokens=True,
							spaces_between_special_tokens=False,
							clean_up_tokenization_spaces=True,)
			result = output[0].split('response in Python dictionary string:')[-1]
			result = result.replace('\n','')
			dict_list.append(result)
		with open(os.path.join(output_dir,vid+'.json'), 'w+') as f:
			json.dump(dict_list,f)