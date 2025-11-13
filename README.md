<h1 align="center">MedReasoner: Reinforcement Learning Drives Reasoning Grounding from Clinical Thought to Pixel-Level Precision</h2>

<div align="center">
<a href="https://pris-cv.github.io/MedReasoner.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
<a href="https://arxiv.org/abs/2508.08177"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Paper&color=red&logo=arxiv"></a> &ensp;
<a href=""><img src="https://img.shields.io/static/v1?label=Dataset&message=MedReasoner&color=yellow&logo=huggingface"></a> &ensp;
<a href="https://huggingface.co/collections/zzzyzh/u-mrg"><img src="https://img.shields.io/static/v1?label=Dataset&message=U-MRG&color=yellow&logo=huggingface"></a> &ensp;
</div>

## 🔥 News
MedReasoner has been accepted at AAAI 2026 as a poster!


## 👀 Introduction
<img src=assets/teaser.png width=100% />

**Abstract**: Accurately grounding regions of interest (ROIs) is critical for diagnosis and treatment planning in medical imaging. While multimodal large language models (MLLMs) combine visual perception with natural language, current medical-grounding pipelines still rely on supervised fine-tuning with explicit spatial hints, making them ill-equipped to handle the implicit queries common in clinical practice.
This work makes three core contributions. We first define **Unified Medical Reasoning Grounding (UMRG)**, a novel vision–language task that demands clinical reasoning and pixel-level grounding. Second, we release **U-MRG-14K**, a dataset of 14K samples featuring pixel-level masks alongside implicit clinical queries and reasoning traces, spanning 10 modalities, 15 super-categories, and 108 specific categories. Finally, we introduce **MedReasoner**, a modular framework that distinctly separates reasoning from segmentation: an MLLM reasoner is optimized with reinforcement learning, while a frozen segmentation expert converts spatial prompts into masks, with alignment achieved through format and accuracy rewards.
MedReasoner achieves state-of-the-art performance on U-MRG-14K and demonstrates strong generalization to unseen clinical queries, underscoring the significant promise of reinforcement learning for interpretable medical grounding.

## 💾 Installation
```bash
git clone https://github.com/zzzyzh/MedReasoner.git
cd MedReasoner

conda create -n med_reasoner python=3.10 -y
conda activate med_reasoner
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
pip install "numpy<2.0"

pip install transformers==4.52.4
pip install vllm==0.8.5.post1
pip install flash-attn==2.7.4.post1
pip install deepspeed==0.16.9

pip install llamafactory
pip install -e .
```

## 🚀 Training
```bash
bash scripts/run_rl_lingshu_7b_soft.sh
```

## 🔍 Inference
```bash
bash scripts/merge_model.sh
bash scripts/infer_grounding.sh
```

## Citation
```Text
@article{yan2025medreasoner,
  title={MedReasoner: Reinforcement Learning Drives Reasoning Grounding from Clinical Thought to Pixel-Level Precision},
  author={Yan, Zhonghao and Diao, Muxi and Yang, Yuxuan and Jing, Ruoyan and Xu, Jiayuan and Zhang, Kaizhou and Yang, Lele and Liu, Yanxi and Liang, Kongming and Ma, Zhanyu},
  journal={arXiv preprint arXiv:2508.08177},
  year={2025}
}
```


## Acknowledgements
This code is built on [verl](https://github.com/volcengine/verl) and [Seg-Zero](https://github.com/dvlab-research/Seg-Zero). We thank the authors for sharing their codes.
