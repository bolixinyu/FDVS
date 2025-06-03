import os
import json
from torch.utils.data import Dataset
from utils.register import Register

@Register.registe('msrvtt')
class MSRVTT(Dataset):
    def __init__(self, data_path, video_dir, result_dir) -> None:
        super().__init__()
        self.data_path = data_path
        self.video_dir = video_dir
        self.data_list = self.set_data(self.data_path, result_dir, video_dir)
    
    def set_data(self, data_path, result_dir, video_dir):
        with open(data_path,'r') as f:
            anno_list = json.load(f)
            f.close()
        data_list = [d['clip_name'] for d in anno_list]
        existing_vid = [f.split('.')[0] for f in os.listdir(result_dir)]
        data_list = [f for f in data_list if f not in existing_vid]

        file_list = [os.path.join(video_dir,f) for f in os.listdir(video_dir) if f.split('.')[0] in data_list]

        return file_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]

@Register.registe('msrvtt_qa')
class MSRVTT_QA(Dataset):
    def __init__(self, data_path, video_dir, result_dir) -> None:
        super().__init__()
        self.data_path = data_path
        self.video_dir = video_dir
        self.data_list = self.set_data(self.data_path, result_dir, video_dir)
    
    def set_data(self, data_path, result_dir, video_dir):
        with open(data_path,'r') as f:
            anno_list = json.load(f)
            f.close()
        data_list = [d['clip_name'] for d in anno_list]
        existing_vid = [f.split('.')[0] for f in os.listdir(result_dir)]
        data_list = [f for f in anno_list] # if f['clip_name'] not in existing_vid]

        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = self.data_list[index]
        video_id = data['clip_name']
        question = data['question']
        answer = data['answer']
        return video_id, question, answer


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import pdb
    dataset = MSRVTT_QA('/youzeng/project/videocap/data/qa/msrvtt/test_qa_re.json',
                       '/youzeng/datasets/msrvtt/MSRVTT/videos/all',
                       './')
    dataloader = DataLoader(dataset,1, shuffle=False)
    
    for vid,question in dataloader:
        pdb.set_trace()