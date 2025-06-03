import os
import json
from torch.utils.data import Dataset
from utils.register import Register

@Register.registe('charades')
class Charades(Dataset):
    def __init__(self, data_path, video_dir, result_dir):
        super().__init__()
        self.data_path = data_path
        self.video_dir = video_dir
        self.data = self.set_data(self.data_path, self.video_dir, result_dir)
    
    def set_data(self, data_path, video_dir, result_dir):
        data_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4') or f.endswith('.webm') or f.endswith('.mkv')]
        data_list = self.remove_exiting_files(data_list,result_dir)
        with open(data_path, 'r') as f:
            data_dict = json.load(f)
        vid_val_list = list(data_dict.keys())
        data_list = [f for f in data_list if f.split('.')[0] in vid_val_list]
        # exist_id = [f[:11]for f in os.listdir('./result')]
        # data_list = [f for f in video_list if f[:11] not in exist_id]
        path_list = [os.path.join(self.video_dir,f) for f in data_list]
        # path_list = ['/youzeng/datasets/anet_videos/v__15t4WTR19s.mp4']
        return path_list
    
    def remove_exiting_files(self, data_list, existing_dir):
        existing_vid = [f.split('.')[0] for f in os.listdir(existing_dir)]
        data_list = [f for f in data_list if f.split('.')[0] not in existing_vid]
        return data_list

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]