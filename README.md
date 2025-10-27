# V-HUB: A VISUAL-CENTRIC HUMOR UNDERSTANDING BENCHMARK FOR VIDEO LLMS

<div style="text-align: center">
  <a href="https://arxiv.org/pdf/2509.25773"><img src="https://img.shields.io/badge/arXiv-2503.23765-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/Foreverskyou/video/tree/main"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" alt="Hugging Face Datasets"></a>
  <a href="https://github.com/MINT-SJTU/STI-Bench"><img src="https://img.shields.io/badge/GitHub-Code-lightgrey" alt="GitHub Repo"></a>
  <a href="https://mint-sjtu.github.io/STI-Bench.io/"><img src="https://img.shields.io/badge/Homepage-STI--Bench-brightgreen" alt="Homepage"></a>
</div>

<p align="center">
    <img src="./figures/teaser.png" width="100%" height="100%">
</p>


## 📐 Dataset Examples

<p align="center">
    <img src="./figures/example.png" width="100%" height="100%">
</p>

## 🔍 Dataset

**License**:
```
V-HUB is only used for academic research. Commercial use in any form is prohibited.
The copyright of all videos belongs to the video owners.
If there is any infringement in V-HUB, please email shi_zpeng@sjtu.edu.cn and we will remove it immediately.
Without prior approval, you cannot distribute, publish, copy, disseminate, or modify V-HUB in whole or in part. 
You must strictly comply with the above restrictions.
```

Please send an email to **shi_zpeng@sjtu.edu.cn**. 🌟

## 🔮 Data Curation and Evaluation Pipeline

<p align="center">
    <img src="./figures/pipline.png" width="100%" height="100%">
</p>

📍 **Downloading**

Use WFDownloader to crawl videos from X. (see [Before filtering](https://huggingface.co/datasets/Foreverskyou/video/tree/main/Before%20filtering))

📍 **Filtering**

After removing duplicate and harmful videos, deploy the Whisper model and only retain videos with less than 10 characters. (see [After filtering](https://huggingface.co/datasets/Foreverskyou/video/tree/main/After%20filtering)).

```bash
python ./filter/extract_speech_text.py
```

📍 **Annotation**

Our annotation platform is Label Studio, please refer to [Annotation_Manual](https://github.com/Foreverskyou/humor_benchmark_evaulation/tree/main/Annotation_Manual) and [Label Studio](https://github.com/HumanSignal/label-studio) for setting up the platform (see [Annotated](https://huggingface.co/datasets/Foreverskyou/video/tree/main/Annotated)).

📍 **Evaluation**: 

```bash
./scripts/Text_Only/example_QA.sh
```
Here we provide example scripts for the three tasks under the three settings: Text-Only, Video-Only, and Video+Audio.

You can specify different tasks, such as: `['QA','explanation','matching']`. And you can also specify different models, for example:`['Qwen2.5-Omni','Qwen2.5-VL','Gemini2.5-flash','GPT-4o','InterVL 3.5','Minicpm 2.6-o','video SALMONN 2']`

## :black_nib: Citation

If you find our work helpful for your research, please consider citing our work. 

```bibtex
@article{shi2025v,
  title={V-HUB: A Visual-Centric Humor Understanding Benchmark for Video LLMs},
  author={Shi, Zhengpeng and Li, Hengli and Zhao, Yanpeng and Zhou, Jianqun and Wang, Yuxuan and Cui, Qinrong and Bi, Wei and Zhu, Songchun and Zhao, Bo and Zheng, Zilong},
  journal={arXiv preprint arXiv:2509.25773},
  year={2025}
}
```
