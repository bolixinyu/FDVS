import json
import os
from utils.videoReader import VideoLoader
from tqdm import tqdm
import numpy as np

def video_object_prompt(video_promp_path):
    with open(video_promp_path,'r') as f:
        object_dict = json.load(f)
        f.close()
    sufix = 'there is a/an {} at {} of the frame picture.'
    prompt = ''
    for iframe, boxes in object_dict.items():
        t = float(iframe)
        prompt += 'for %.2fs:'%t
        for box, label in zip(boxes['boxes'],boxes['labels']):
            box = ['%.2f'%b for b in box]
            prompt += sufix.format(label, box)
    return prompt

def clip_object_prompt(video_id, object_dir, clip_timestamp):
    object_path = os.path.join(object_dir,video_id+'.json')
    with open(object_path,'r') as f:
        object_dict = json.load(f)
        f.close()
    object_dict = filter_object(object_dict, 0.5)
    sufix = 'there is a/an {} at {} of the frame picture.'
    prompt = ''
    last_t = clip_timestamp[0]
    detect_times = np.array([t for t in object_dict.keys() if float(t)>clip_timestamp[0] and float(t)<clip_timestamp[1]])
    index = np.linspace(0, len(detect_times), 10).clip(max =len(detect_times)-1)
    index = index.astype(np.int64)
    index = np.unique(index)
    times = detect_times[index]
    for t in times:
        prompt += 'for %.2fs:'%float(t)
        for box, label in zip(object_dict[t]['boxes'],object_dict[t]['labels']):
            box = ['%.2f'%b for b in box]
            prompt += sufix.format(label, box)
    return prompt
    
def video_caption_prompt(caption_dir,video_id):
    train_cap_path = os.path.join(caption_dir,'train.json')
    val_cap_path = os.path.join(caption_dir,'val_1.json')
    val2_cap_path = os.path.join(caption_dir,'val_2.json')
    extend_id = 'v_'+video_id
    with open(train_cap_path,'r') as f:
        train_dict = json.load(f)
        f.close()
    with open(val_cap_path,'r') as f:
        val_dict = json.load(f)
        f.close()
    with open(val2_cap_path,'r') as f:
        val2_dict = json.load(f)
        f.close()
    if extend_id in train_dict:
        cap_dict = train_dict[extend_id]
    elif extend_id in val_dict:
        cap_dict = val_dict[extend_id]
    elif extend_id in val2_dict:
        cap_dict = val2_dict[extend_id]
    
    prompt =''
    caption_prompt = 'the caption for video clip from {}s to {}s is {}'
    for time_stamp, sentence in zip(cap_dict['timestamps'], cap_dict['sentences']):
        prompt = prompt + caption_prompt.format(time_stamp[0], time_stamp[1], sentence)
        prompt = prompt + 'the object information detected in this clip is {}'.format(
            clip_object_prompt(video_id, './resultWithTime',time_stamp))
    return prompt

def filter_object(object_dict, score_threshold):
    new_object_dict = {}
    for t, boxes in object_dict.items():
        new_boxes = []
        new_labels = []
        for box, phrase in zip(boxes['boxes'], boxes['labels']):
            index = phrase.find('(')
            score = float(phrase[index+1:index+5])
            phrase = phrase[:index]
            if score > score_threshold:
                new_boxes.append(box)
                new_labels.append(phrase)
        new_object_dict['%.2f'%float(t)] = {'boxes':new_boxes, 'labels':new_labels}
    return new_object_dict
    
def main():
    root = './result'
    new_root = './resultWithTime'
    if not os.path.exists(new_root):
        os.makedirs(new_root)
    files = os.listdir(root)
    video_dir = '/home/youzeng/dataset/anet_caption/videos/'
    video_file = os.listdir(video_dir)
    video_id = [f.split('.')[0] for f in video_file]
    for file in tqdm(files, total=len(files)):
        try:
            with open(os.path.join(os.path.join(root,file)),'r') as f:
                object_dict = json.load(f)
                f.close()
            vid = file.replace('.json','')
            if vid not in video_id:
                continue
            video_f = video_file[video_id.index(vid)]
            vr = VideoLoader(os.path.join(video_dir,video_f), fps=30)
            new_dict = {}
            for iframe, boxes in object_dict.items():
                t = float(iframe)/vr.orin_fps
                new_dict['%.2f'%t] = boxes
            with open(os.path.join(new_root, file),'w') as f:
                json.dump(new_dict,f)
                f.close()
        except Exception as e:
            print(e)
            print(file)
    return None

if __name__ == '__main__':
    video_caption_prompt('./data/captions', 'ZXEc0cahpuw')