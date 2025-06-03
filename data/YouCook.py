import os
import json
from torch.utils.data import Dataset
from utils.register import Register

@Register.registe('youcook')
class YouCook(Dataset):
    def __init__(self, data_path, video_dir, result_dir) -> None:
        super().__init__()
        self.data_path = data_path
        self.video_dir = video_dir
        self.data_list = self.set_data(self.data_path, result_dir, video_dir)
    
    def set_data(self, data_path, result_dir, video_dir):
        with open(data_path,'r') as f:
            data_dict = json.load(f)['database']
            data_list = [vid for vid in data_dict.keys() if data_dict[vid]['subset']=='validation']
            f.close()
        # import pdb
        # pdb.set_trace()
        existing_vid = [f.split('.')[0] for f in os.listdir(result_dir)]
        data_list = [f for f in data_list if f not in existing_vid]

        file_list = [os.path.join(video_dir,f) for f in os.listdir(video_dir) if f.split('.')[0] in data_list]

        return file_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
