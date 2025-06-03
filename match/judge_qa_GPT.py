from text2vec import SentenceModel, semantic_search, cos_sim
import os
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from typing import List
from tqdm import tqdm
from angle_emb import AnglE, Prompts
import argparse
import openai
import json
import pdb

# 目前需要设置代理才可以访问 api
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='msrvtt')
parser.add_argument('--result_dir', type=str, default = './result/qa/msrvtt_qa_/')
# parser.add_argument('--clip_cap', type=str, default = './result/msrvtt/clip_cap_interpretable_all_visual_reduce/')
parser.add_argument('--local_rank', type=int, default=None)
args = parser.parse_args()

# rank_id = get_rank() if args.local_rank else 0
# device = torch.device('cuda', rank_id)
# torch.cuda.set_device(rank_id)
# embeder = AnglE.from_pretrained('/youzeng/pretrained_weights/Llama-2-13b-hf', pretrained_lora_path='/youzeng/pretrained_weights/angle-llama-13b-nli')
# embeder.set_prompt(prompt=Prompts.A)
# # import pdb
# # pdb.set_trace()
# embeder.backbone.to(device)

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

def openai_request_via_proxy(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai.api_key}',
        'OpenAI-Organization': 'org-Q8N4twY2bUObji7HTWYK7ovQ'
    }
    data = {
        "model": "gpt-3.5-turbo-1106",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    # {
    #     'prompt': prompt,
    #     'max_tokens': 1000
    # }

    response = requests.post(url, headers=headers, json=data, proxies=proxies)
    return response.json()

def get_api_key():
    # 可以自己根据自己实际情况实现
    # 以我为例子，我是存在一个 openai_key 文件里，json 格式
    '''
    {"api": "你的 api keys"}
    '''
    openai_key_file = './config/openai-key.json'
    with open(openai_key_file, 'r', encoding='utf-8') as f:
        openai_key = json.loads(f.read())
    return openai_key['api-key']

openai.api_key = get_api_key()

content_list = '['
content_template = "Question: {} # "\
         "Correct Answer: {} # "\
         "Predicted Answer: {}"
prompt_template = "Please evaluate the following video-based question-answer pairs:"
         # "For example, your response should look like this: [{'pred': 'yes', 'score': 4.8}, {'pred': 'no', 'score': 2.0}]."

for vid in tqdm(data_list, total = len(data_list)):
    result_file = os.path.join(result_dir, vid+'.json')
    if os.path.exists(result_file):
        with open(result_file,'r') as f:
            result_list = json.load(f)
        for result in result_list:
            q = result['question']
            p = result['pred']
            t= result['answer']
            content = "Please evaluate the following video-based question-answer pair:\n\n"\
         f"Question: {q}"\
         f"Correct Answer: {t}\n"\
         f"Predicted Answer: {p}\n\n"\
         "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. "\
         "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string."\
         "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
            rsp = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo-1106",
                  messages=[
                        {"role": "system", "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                "-----"
                "##INSTRUCTIONS:\n"
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the predictions compared to the answers."},
                        {"role": "user", "content": content}
                    ]
                )
    else:
        print(result_dir,result_file)

# q = "用python实现：提示手动输入3个不同的3位数区间，输入结束后计算这3个区间的交集，并输出结果区间"
# rsp = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#         {"role": "system", "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
# "-----"
# "##INSTRUCTIONS:\n"
# "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
# "- Consider synonyms or paraphrases as valid matches.\n"
# "- Evaluate the correctness of the predictions compared to the answers."},
#         {"role": "user", "content": "Please evaluate the following video-based question-answer pair:\n\n"
#          "Question: {what are animated buckets of paint doing?}"
#          "Correct Answer: {pose}\n"
#          "Predicted Answer: {Painting}\n\n"
#          "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. "
#          "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string."
#          "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
#         }
#     ]
# )