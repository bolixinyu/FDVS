import os
import json
from torch.utils.data import Dataset
from utils.register import Register

@Register.registe('next')
class NextCap(Dataset):
    def __init__(self, data_path, video_dir, result_dir):
        super().__init__()
        self.data_path = data_path
        self.video_dir = video_dir
        self.data = self.set_data(self.data_path, self.video_dir, result_dir)
    
    def set_data(self, data_path, video_dir, result_dir):
        with open(data_path, 'r') as f:
            data_dict = json.load(f)
        data_list = []
        for vid, vp in data_dict.items():
            # file_path = os.path.join(video_dir, vid+'.mp4')
            data_list.append(vid)
        # exist_id = [f[:11]for f in os.listdir('./result')]
        # data_list = [f for f in video_list if f[:11] not in exist_id]
        
        data_list = self.remove_exiting_files(data_list,result_dir)
        path_list = [os.path.join(video_dir, f +'.mp4') for f in data_list]
        # path_list = ['/youzeng/datasets/anet_videos/v__15t4WTR19s.mp4']
        return path_list
    
    def remove_exiting_files(self, data_list, existing_dir):
        existing_vid = [f.split('.')[0] for f in os.listdir(existing_dir)]
        data_list = [f for f in data_list if f not in existing_vid]
        return data_list

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

@Register.registe('next_qa')
class NextQA(Dataset):
    def __init__(self, data_path, video_dir, result_dir):
        super().__init__()
        self.data_path = data_path
        self.video_dir = video_dir
        self.data = self.set_data(self.data_path, self.video_dir, result_dir)
    
    def set_data(self, data_path, video_dir, result_dir):
        with open(data_path, 'r') as f:
            data_dict = json.load(f)
        data_list = {}
        for data in data_dict:
            # file_path = os.path.join(video_dir, vid+'.mp4')
            vid = str(data['video'])
            if vid not in data_list:
                data_list[vid] = {'question':[], 'answer':[]}
            options = [data['a0'], data['a1'], data['a2'], data['a3'], data['a4']]
            question = [data['question']] + options
            answer = data['answer']
            data_list[vid]['question'].append(question)
            data_list[vid]['answer'].append(answer)
        # exist_id = [f[:11]for f in os.listdir('./result')]
        # data_list = [f for f in video_list if f[:11] not in exist_id]
        
        data_list = self.remove_exiting_files(data_list,result_dir)
        # path_list = ['/youzeng/datasets/anet_videos/v__15t4WTR19s.mp4']
        return data_list
    
    def remove_exiting_files(self, data_list, existing_dir):
        existing_vid = [f.split('.')[0] for f in os.listdir(existing_dir)]
        final_data_list = []
        for vid in data_list.keys():
            if vid not in existing_vid:
                final_data_list.append((vid, data_list[vid]['question'], data_list[vid]['answer']))
        return final_data_list

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vid, question_list, answer_list = self.data[idx]
        return vid, question_list, answer_list
    

if __name__ == '__main__':
    dataset = NextCap('/chenyaofo/project/videocap/data/next_qa/val_map.json', '/chenyaofo/datasets/NextVideo/videos', '/chenyaofo/project/videocap/result/next_qa/')
    print(dataset[0])