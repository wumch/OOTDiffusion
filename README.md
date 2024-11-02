# OOTDiffusion
This repository is the official implementation of OOTDiffusion

🤗 [Try out OOTDiffusion](https://huggingface.co/spaces/levihsu/OOTDiffusion)

(Thanks to [ZeroGPU](https://huggingface.co/zero-gpu-explorers) for providing A100 GPUs)

<!-- Or [try our own demo](https://ootd.ibot.cn/) on RTX 4090 GPUs -->

> **OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on** [[arXiv paper](https://arxiv.org/abs/2403.01779)]<br>
> [Yuhao Xu](http://levihsu.github.io/), [Tao Gu](https://github.com/T-Gu), [Weifeng Chen](https://github.com/ShineChen1024), [Chengcai Chen](https://www.researchgate.net/profile/Chengcai-Chen)<br>
> Xiao-i Research


Our model checkpoints trained on [VITON-HD](https://github.com/shadow2496/VITON-HD) (half-body) and [Dress Code](https://github.com/aimagelab/dress-code) (full-body) have been released

* 🤗 [Hugging Face link](https://huggingface.co/levihsu/OOTDiffusion) for ***checkpoints*** (ootd, humanparsing, and openpose)
* 📢📢 We support ONNX for [humanparsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) now. Most environmental issues should have been addressed : )
* Please also download [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) into ***checkpoints*** folder
* We've only tested our code and models on Linux (Ubuntu 22.04)

![demo](images/demo.png)&nbsp;
![workflow](images/workflow.png)&nbsp;

## Installation
1. Clone the repository

```sh
git clone https://github.com/levihsu/OOTDiffusion
```

2. Create a conda environment and install the required packages

```sh
conda create -n ootd python==3.10
conda activate ootd
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```

## Inference
1. Half-body model

```sh
cd OOTDiffusion/run
python run_ootd.py --model_path <model-image-path> --cloth_path <cloth-image-path> --scale 2.0 --sample 4
```

2. Full-body model 

> Garment category must be paired: 0 = upperbody; 1 = lowerbody; 2 = dress

```sh
cd OOTDiffusion/run
python run_ootd.py --model_path <model-image-path> --cloth_path <cloth-image-path> --model_type dc --category 2 --scale 2.0 --sample 4
```

## Citation
```
@article{xu2024ootdiffusion,
  title={OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on},
  author={Xu, Yuhao and Gu, Tao and Chen, Weifeng and Chen, Chengcai},
  journal={arXiv preprint arXiv:2403.01779},
  year={2024}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=levihsu/OOTDiffusion&type=Date)](https://star-history.com/#levihsu/OOTDiffusion&Date)

## TODO List
- [x] Paper
- [x] Gradio demo
- [x] Inference code
- [x] Model weights
- [ ] Training code


## 下载预训练模型:
```
pip install torch==2.0.1 deepytorch-inference -f https://aiacc-inference-public-v2.oss-cn-hangzhou.aliyuncs.com/aiacc-inference-torch/stable-diffusion/aiacctorch_stable-diffusion.html

sudo apt install aria2
cd /data/code/models 
 
# 下载 OOTDiffusion 模型:
curl -sL https://sciproxy.com/https://huggingface.co/levihsu/OOTDiffusion | aria2c --continue=true -x 4 -s 4 -k 1M -i -

# 下载 clip-vit-large-patch14 :
curl -sL https://sciproxy.com/https://huggingface.co/openai/clip-vit-large-patch14 | aria2c --continue=true -x 4 -s 4 -k 1M -i -

# 删除原有的 checkpoints:
rm -fr /data/code/OOTDiffusion/checkpoints
# 软链接到 checkpoints:
ln -s /data/code/models/OOTDiffusion/checkpoints /data/code/OOTDiffusion/checkpoints
ln -s /data/code/models/clip-vit-large-patch14 /data/code/OOTDiffusion/checkpoints/clip-vit-large-patch14

cd /data/code/OOTDiffusion/run
time python run_ootd.py --model_path examples/model/model_1.png --cloth_path examples/garment/048769_1.jpg --scale 2.0 --sample 1
```

### bug fix:
如遇报错 cannot import name 'cached_download' from 'huggingface_hub'，需手工从
lib/python3.8/site-packages/diffusers/utils/dynamic_modules_utils.py
删除 cached_download 的 import。

查看 torchvision ~ torch 版本对应关系:
https://pypi.org/project/torchvision/

```
cd /tmp
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh

conda create -n ai python=3.10
```
