# Towards Long Video Understanding via Fine-detailed Video Story Generation

This repository provides an implementation for paper *Towards Long Video Understanding via Fine-detailed Video Story Generation*, published by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT). The paper is available [here](https://arxiv.org/pdf/2412.06182)

---

Long video understanding has become a critical task in computer vision, driving advancements across numerous applications from surveillance to content retrieval. Existing video understanding methods suffer from two challenges when dealing with long video understanding: intricate long-context relationship modeling and interference from redundancy. To tackle these challenges, we introduce Fine-Detailed Video Story generation (FDVS), which interprets long videos into detailed textual representations. Specifically, to achieve fine-grained modeling of longtemporal content, we propose a Bottom-up Video Interpretation Mechanism that progressively interprets video content from clips to video. To avoid interference from redundant information in videos, we introduce a Semantic Redundancy Reduction mechanism that removes redundancy at both the visual and textual levels. Our method transforms long videos into hierarchical textual representations that contain multi-granularity information of the video. With these representations, FDVS is applicable to various tasks without any fine-tuning. We evaluate the proposed method across eight datasets spanning three tasks. The performance demonstrates the effectiveness and versatility of our method.

## Requirements

Our codebase is implemented based on multiple off-the-shelf models, including [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [AnglE](https://github.com/SeanLee97/AnglE), [InternVideo](https://github.com/OpenGVLab/InternVideo), [Vicuna](https://github.com/lm-sys/FastChat), and 
[BLIP2](https://github.com/salesforce/LAVIS). Please refer to their official repos for installation instructions and model dowload links. 

Clone this repository by:
```bash
git clone https://github.com/bolixinyu/FDVS
```
Install the requirements by:
```bash
cd FDVS
conda env create -f env.yaml
conda activate fdvs
pip install -r requirements.txt
```

## Inference for Fine-Detailed Video Story Generation
To run inference, you should change the model and dataset paths in `./config/config.yaml` and `./config/config.json`. Then you can choose the task `Task.name` in `config.yaml` from [*object_det*, *action_rec*, *image_cap*, *clip_cap*, *video_cap*], dataset `Data.set_name` and the corresponding models `Model.AtomicAgents` (refer to the paper for details). Then you can run the following command:
```bash
bash run.sh
```

## Inference for Video Retrieval
To conduct video retrieval, you can run the following command:
```bash
python match/match_clip_story.py --video_cap ./path/to/videocap --clip_cap ./path/to/clipcap --dataset datasetname
```

## Inference for PRVR
To conduct PRVR, you can run the following command:
```bash
python match/prvr_match.py --video_cap ./path/to/videocap --clip_cap ./path/to/clipcap --dataset datasetname
```

## Inference for VideoQA
To conduct VideoQA, you should modify the paths in `./config/config_qa.yaml`. Then you can run the following command:
```bash
python run_qa.sh
```

## Acknowledgement
We thank the community for their valuable contributions.

## Citation

If you use this code in your research or implementations, please cite the following paper:

```bibtex
@article{you2024towards,
  title={Towards Long Video Understanding via Fine-detailed Video Story Generation},
  author={You, Zeng and Wen, Zhiquan and Chen, Yaofo and Li, Xin and Zeng, Runhao and Wang, Yaowei and Tan, Mingkui},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024}
}
```

