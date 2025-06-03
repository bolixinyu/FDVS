import torch
from torchvision import transforms as T
from model.load_internvideo import load_intern_action, transform_action,kinetics_classnames
from utils.videoReader import VideoLoader
import os
import torch.nn.functional as F

def action_recognize(model, action_tensor):
    with torch.no_grad():
        prediction = model(action_tensor)
        prediction = F.softmax(prediction, dim=1).flatten()
        prediction = kinetics_classnames[str(int(prediction.argmax()))]
    return prediction

if __name__ == "__main__":
    video_dir = '/home/youzeng/dataset/anet_caption/videos/'
    video_file = os.listdir(video_dir)
    device = torch.device('cuda')
    inter_action = load_intern_action(device)
    transforms = transform_action()
    toPIL = T.ToPILImage()
    for f in video_file:
        video_path = os.path.join(video_dir,f)
        vr = VideoLoader(video_path, 2)
        clip_indexes = vr.get_clip_indexes(8,8)
        for c in clip_indexes:
            clip, _ = vr.read_clip(c)
            tmpa=[]
            for i in clip:
                img = toPIL(i)
                tmpa.append(img)
            action_tensor = transforms(tmpa)
            tc,h,w = action_tensor.shape
            action_tensor = action_tensor.reshape(1, tc//3 ,3, h, w).permute(0, 2, 1, 3, 4).to(device)
            with torch.no_grad():
                prediction = inter_action(action_tensor)
                prediction = F.softmax(prediction, dim=1).flatten()
                prediction = kinetics_classnames[str(int(prediction.argmax()))]
            
