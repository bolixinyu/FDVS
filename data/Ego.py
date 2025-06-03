import os
import json
from torch.utils.data import Dataset
from utils.register import Register

@Register.registe('ego')
class EGO(Dataset):
    def __init__(self, data_path, video_dir, result_dir) -> None:
        super().__init__()
        self.data_path = data_path
        self.video_dir = video_dir
        self.data_list = self.set_data(self.data_path, result_dir, video_dir)
    
    def set_data(self, data_path, result_dir, video_dir):
        with open(data_path,'r') as f:
            anno_list = json.load(f)
            f.close()
        existing_vid = [f.split('.')[0] for f in os.listdir(result_dir)]
        data_list = []
        for k, v in anno_list.items():
            if k not in existing_vid:
                data_list.append(os.path.join(video_dir,v['video']))

        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = self.data_list[index]
        return data

@Register.registe('ego_qa')
class EGO_QA(Dataset):
    def __init__(self, data_path, video_dir, result_dir) -> None:
        super().__init__()
        self.data_path = data_path
        self.video_dir = video_dir
        self.data_list = self.set_data(self.data_path, result_dir, video_dir)
    
    def set_data(self, data_path, result_dir, video_dir):
        with open(data_path,'r') as f:
            anno_list = json.load(f)
            f.close()
        existing_vid = [f.split('.')[0] for f in os.listdir(result_dir)]
        data_list = []
        for k, v in anno_list.items():
            if k not in existing_vid:
                data_list.append({k:v})

        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = self.data_list[index]
        video_id = list(data.keys())[0]
        question = data[video_id]['question']
        options = data[video_id]['options']
        answer = data[video_id]['answer']
        return video_id, (question, options), answer