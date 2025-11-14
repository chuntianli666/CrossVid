# CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models 🎬

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/XXXXX)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-orange.svg)](https://huggingface.co/datasets/Chuntianli/CrossVid)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<img src="assets/logo.png" width="800" alt="CrossVid Logo">

</div>

---

## 🌟 Introduction

**CrossVid** is the first comprehensive benchmark for evaluating **Cross-Video Reasoning (CVR)** in Multimodal Large Language Models (MLLMs). Unlike existing benchmarks focusing on single-video analysis, CrossVid challenges models to simultaneously understand, aggregate, and compare information across multiple videos.

**Key Highlights:**
- 🎯 **First systematic CVR benchmark** with hierarchical task design
- 📊 **9,015 QA pairs** across 5,331 videos from 6 diverse datasets
- 🏗️ **10 specific tasks** spanning 4 dimensions (Comparative, Temporal, Multi-View, Free-Form)
- 🌐 **32 genres** covering real-world scenarios
- ⏱️ **Long-context**: Average 770 seconds per query
- 📝 **Multiple formats**: Single-choice, multiple-choice, and open-ended questions

<div align="center">
<img src="assets/genres.png" width="49%" alt="Genre Distribution">
<img src="assets/tasks.png" width="49%" alt="Task Hierarchy">
</div>

---

## 📢 News

- **[2025-11]** 🎉 CrossVid accepted by **AAAI 2026**!
- **[2025-11]** 📊 Dataset available on [HuggingFace](https://huggingface.co/datasets/Chuntianli/CrossVid).
- **[TODO]** 🔧 Evaluation code coming soon!

---

## 🎯 Benchmark Overview

### Task Dimensions

**📊 Comparative Analysis** - Behavioral Understanding  (BU), Narrative Comprehension (NC), Culinary Comparison (CC), and Procedural Eror Analysis (PEA)

**⏱️ Temporal Understanding** - Plot Inference (PI), Functional Step Alignment (FSA), Procedural Step Sequencing (PSS)

**👁️ Multi-View Reasoning** - Multi-view Spatial Reasoning (MSR) and Multi-view Object Counting (MOC)

**✍️ Free-Form QA** - Comparative Culinary QA (CCQA)

### Data Sources & Statistics

Videos from **6 public datasets**: Animal Kingdom 🦁 | MovieChat-1K 🎬 | YouCook2 👨‍🍳 | VisDrone 🚁 | Charades 🏠 | Assembly101 🔧. 

We thank the creators of these valuable datasets for providing the foundational video resources. 

| Metric | Value | Metric | Value |
|--------|-------|--------|-------|
| 📹 Videos | 5,331 | 🎭 Genres | 32 |
| ❓ QA Pairs | 9,015 | 🎯 Tasks | 10 |
| ⏱️ Avg Video Length | 215s | 📊 Avg Query Duration | 770s |

---

## 📸 Examples

<div align="center">
<img src="assets/examples.png" width="100%" alt="CrossVid Examples">
<p><i>Representative examples showing different cross-video reasoning tasks</i></p>
</div>

---

## 🏗️ Annotation Pipeline

<div align="center">
<img src="assets/pipeline.png" width="100%" alt="Evaluation Pipeline">
</div>

**Process**: Frame Extraction (Qwen2.5-VL-72B) → QA Generation (DeepSeek-R1) → Manual Filtration → Refinement → Quality Control


---

## 🚀 Quick Start

### Installation & Usage
```python
#pip install datasets

from datasets import load_dataset

# Load dataset
dataset = load_dataset("Chuntianli/CrossVid")

# Access sample
sample = dataset[0]
print(sample)
```

### Evaluation is Coming Soon...

---

## 📊 Leaderboard

| Rank | Model | Overall | Comparative | Temporal | Multi-View | Free-Form |
|:----:|:------|:-------:|:-----------:|:--------:|:----------:|:---------:|
| 🥇 | **Gemini-2.5-Pro** | **50.4** | 54.7 | 56.0 | 28.7 | 59.8 |
| 🥈 | GPT-4.1 | 45.2 | 47.6 | 46.7 | 38.4 | 44.6 |
| 🥉 | Doubao-1.5-VL-Pro | 44.3 | 53.8 | 36.1 | 34.7 | 50.1 |
| 4 | GPT-4o | 36.8 | 43.1 | 35.5 | 27.4 | 34.2 |
| 5 | GLM-4.1V-9B-Thinking | 35.1 | 44.7 | 23.1 | 37.8 | 26.9 |
| 6 | Qwen2.5-VL-72B | 34.4 | 42.1 | 29.2 | 23.5 | 41.2 |
| ... | ... | ... | ... | ... | ... | ... |
| - | **Human** | **89.2** | **88.1** | **89.9** | **93.7** | **85.2** |


---

## 📄 License & Contact

<!-- **License**: MIT License - see [LICENSE](LICENSE) -->

**Datasets**: Videos from public datasets - refer to original licenses ([Animal Kingdom](https://github.com/sutdcv/Animal-Kingdom), [MovieChat-1K](https://github.com/rese1f/MovieChat), [YouCook2](http://youcook2.eecs.umich.edu/), [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), [Charades](https://prior.allenai.org/projects/charades), [Assembly101](https://assembly-101.github.io/))

**Contact**: 
- Email: chuntianli666666@gmail.com
- GitHub: [CrossVid](https://github.com/chuntianli666/CrossVid)

**Acknowledgements**: Thanks to dataset authors and our expert annotators.

---

## 📝 Citation
```bibtex
@inproceedings{li2025crossvid,
  title={CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models},
  author={Li, Jingyao and Wang, Jingyun and Tan, Molin and Wang, Haochen and Yan, Cilin and Shi, Likun and Cai, Jiayin and Jiang, Xiaolong and Hu, Yao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

---

<div align="center">

**⭐ Star us on GitHub! ⭐**

[![GitHub Stars](https://img.shields.io/github/stars/chuntianli666/CrossVid?style=social)](https://github.com/chuntianli666/CrossVid)

</div>
