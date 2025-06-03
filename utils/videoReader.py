from decord import VideoReader, cpu, gpu
import numpy as np
import math
import torch
from PIL import Image
# from utils.WDistance import SinkhornDistance
# wd = SinkhornDistance()
def get_thum(image, size=(224, 224), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image

def select_frames_byCos(img_clip, img_encoder, processor):
    bs = 8
    num_batch = math.ceil(len(img_clip)/bs)
    feature_list = []
    for i in range(num_batch):
        img_clip_batch = img_clip[i*bs:(i+1)*bs]
        processed_tensor = processor(images=img_clip_batch, return_tensors="pt")
        processed_tensor['pixel_values'] = processed_tensor['pixel_values'].to(device='cuda')
        img_matrix = img_encoder.get_image_features(**processed_tensor).cpu().detach().numpy()
        feature_list.append(img_matrix)
    img_matrix = np.concatenate(feature_list, axis=0)
    consin = img_matrix[0].dot(img_matrix.T) / (np.linalg.norm(img_matrix[0]) * np.linalg.norm(img_matrix.T,axis=0)+1e-10)
    cos_mean = consin.mean()
    consin[0]=-1e10
    selected_idx = consin<cos_mean
    # selected_idx[0] = 
    selected_imgs = img_clip[selected_idx]
    # vectors = []
    # norms = []
    # for image in images:
    #     vector = []
    #     for pixel_tuple in image.getdata():
    #         vector.append(np.average(pixel_tuple))
    #     vectors.append(vector)
    #     # linalg=linear（线性）+algebra（代数），norm则表示范数
    #     # 求图片的范数
    #     norms.append(np.linalg.norm(vector, 2))
    # a, b = vectors
    # a_norm, b_norm = norms
    # # dot返回的是点积，对二维数组（矩阵）进行计算
    # res = np.dot(a / a_norm, b / b_norm)
    return selected_imgs, selected_idx

class VideoLoader:
    def __init__(self, video_path, fps:int,use_gpu=False, new_width=340, new_height=256, keep_ori_ratio=True):
        self.fps=fps
        self.use_gpu = use_gpu
        self.width = new_width
        self.height = new_height
        self.keep_ori_ratio = keep_ori_ratio
        self.video_path = video_path
        self.vr=self._get_frame_stream(self.video_path)
        self.n_frame = len(self.vr) #debug
        self.orin_fps = self.vr.get_avg_fps()
    
    def _get_frame_stream(self,path):
        if self.keep_ori_ratio:
            decoder = VideoReader(path,num_threads=1, ctx=cpu(0))
        else:
            decoder = VideoReader(path,width=self.width, height=self.height,
             num_threads=1, ctx=cpu(0))
        # decoder.seek(0)
        self.height, self.width, _ = decoder[0].shape
        return decoder
    
    def get_clip_indexes(self, len, stride):
        n_frame = math.floor(self.n_frame*(self.fps/self.orin_fps))
        indexes=np.linspace(0, self.n_frame-1, n_frame,endpoint=True,dtype=np.int32)
        indexes = np.unique(indexes)
        indices = np.arange(indexes.shape[0])
        start_indexes = indices[::stride]
        batch_clip_index=[]
        for start in start_indexes:
            if (start+len)<=indexes.shape[0]:
                frame_indexes=indexes[start:(start+len)]
                # patched_clip=vr.get_batch(frame_indexes)
                batch_clip_index.append(frame_indexes)
            # elif (n_frame-start)>=len/2:
            #     frame_indexes = indexes[start:]
            #     batch_clip_index.append(frame_indexes)
                # patched_clip=np.zeros(len, self.height, self.width, 3)
                # tmp_clip=vr.get_batch(frame_indexes)
                # patched_clip[:n_frame-start] = tmp_clip
        return batch_clip_index
    
    def get_clip_indexes_by_kframe(self, length, stride):
        key_indices = self.vr.get_key_indices()
        if 0 not in key_indices:
            key_indices.append(0)
        key_indices.append(self.n_frame)
        clip_index_list = []
        for i in range(len(key_indices)-1):
            s = key_indices[i]
            e = key_indices[i+1]
            clip_indexes = np.linspace(s, e, length,endpoint=False,dtype=np.int32)
            clip_index_list.append(clip_indexes)
        return clip_index_list
    
    def get_all_clip_indexes_by_kframe(self, length, stride):
        key_indices = self.vr.get_key_indices()
        if 0 not in key_indices:
            key_indices.append(0)
        key_indices.append(self.n_frame)
        clip_index_list = []
        for i in range(len(key_indices)-1):
            s = key_indices[i]
            e = key_indices[i+1]
            clip_indexes = np.arange(s, e, dtype=np.int32)
            clip_index_list.append(clip_indexes)
        return clip_index_list
    
    def uniformly_sample_video(self, length):
        indexes=np.linspace(0, self.n_frame-1, length,endpoint=True,dtype=np.int32)
        return indexes
    
    def read_clip(self, clip_index, transforms=None):
        clip = self.vr.get_batch(clip_index).asnumpy()
        # clip = np.zeros((clip_len, self.height, self.width, 3))
        # if len(clip_index)>=clip_len:
        #     clip[:]=self.vr.get_batch(clip_index).asnumpy()
        # elif len(clip_index)<clip_len:
        #     orin_clip = self.vr.get_batch(clip_index).asnumpy()
        #     clip[:len(clip_index)]=orin_clip
        tensor = None
        if transforms is not None:
            tensor = torch.from_numpy(clip)
            tensor = tensor.to(dtype=torch.float32)/255.0
            tensor = tensor.permute(0,3,1,2)
            tensor=transforms(tensor)
        return clip, tensor
            




if __name__ == '__main__':
    reader=VideoLoader('/youzeng/datasets/anet_videos/v___c8enCfzqw.mp4',25)
    clip_indexes = reader.get_clip_indexes(16,16)
    for clip_index in clip_indexes:
        clip,_,_ = reader.read_clip(clip_index, selective=True)