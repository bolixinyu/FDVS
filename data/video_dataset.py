import os
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = self.set_data(self.data_path)
    
    def set_data(self, data_path):
        data_list = [f for f in os.listdir(data_path) if f.endswith('.mp4') or f.endswith('.webm') or f.endswith('.mkv')]
        # exist_id = [f[:11]for f in os.listdir('./result')]
        # data_list = [f for f in video_list if f[:11] not in exist_id]
        path_list = [os.path.join(self.data_path,f) for f in data_list]
        return path_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]