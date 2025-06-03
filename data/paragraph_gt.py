import os
import json

video_id = ['v_'+f[:11] for f in os.listdir('./result/videochat_cap')]
with open('./data/captions/val_1.json','r') as f:
	annotations = json.load(f)
gt_dir = './data/captions/groundtruth'
if not os.path.exists(gt_dir):
	os.makedirs(gt_dir)
for vid in video_id:
	anno = annotations[vid]['sentences']
	para = ''.join(anno)
	with open(os.path.join(gt_dir, vid[2:13]+'.txt'), 'w', encoding='utf-8') as f:
		f.write(para)
		f.close()
	
