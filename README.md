# AerithVLM: A Hybrid Vision-Language Model for Remote Sensing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

AerithVLM is a modular and compute-efficient Vision-Language Model designed for remote sensing and geospatial analysis. By integrating self-supervised vision encoders (DINOv3) with CLIP-based multimodal alignment, AerithVLM enables robust visual grounding, captioning and geospatial reasoning.

---

## 🌟 Key Features

- **Dual Vision Encoder Architecture**: Combines CLIP ViT-L/14 and DINOv3 ViT-B/16 for robust feature extraction
- **Efficient Alignment Module**: Novel DINOv3-CLIP alignment using lightweight linear projection head
- **LLaMA-2 Integration**: Uses LLaMA-2-7B-Chat with LoRA adapters for efficient fine-tuning
- **Multi-Task Support**: Scene classification, visual grounding, VQA and image captioning
- **Compute-Efficient**: Optimized for training on limited GPU resources (T4/V100)

---



## 🏗️ Architecture

![AerithVLM Architecture](assets/architecture_diagram.png)

AerithVLM consists of three main components:

1. **Vision Encoder**
   - CLIP ViT-L/14 (1024-dim → 768-dim projection)
   - DINOv3 ViT-B/16 (768-dim with linear projection)
   - Element-wise fusion for 768-dim visual embeddings

2. **Modality Alignment Bridge**
   - 6-layer Attention Pooler with 144 learned queries
   - Cross-attention alignment (16 attention heads)
   - Geospatial-aware visual compression

3. **Language Model**
   - LLaMA-2-7B-Chat backbone
   - LoRA adapters for efficient fine-tuning
   - 768-dim to 4096-dim projection layer

---

## 🔧 Installation

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.8 (for GPU support)
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AerithVLM.git
cd AerithVLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```bash
pip install torch torchvision transformers
pip install opencv-python pillow numpy pandas
pip install accelerate bitsandbytes peft
pip install sentencepiece protobuf
```

---

## 📊 Datasets

We use multiple remote sensing datasets for training and evaluation:

### Training Datasets

| Dataset | Task | Samples | Download Link |
|---------|------|---------|---------------|
| UCM | Scene Classification | 2,100 | [Google Drive](https://drive.google.com/your-link) |
| RSVG | Visual Grounding | 1,276 | [Google Drive](https://drive.google.com/your-link) |
| DIOR-RSVG | Visual Grounding | 3,372 | [Google Drive](https://drive.google.com/your-link) |
| RSICap | Image Captioning | 2,585 | [Google Drive](https://drive.google.com/your-link) |
| RSVQA-LR | Visual QA | 572 | [Google Drive](https://drive.google.com/your-link) |
| RSITMD | Scene Classification | 2,632 | [Google Drive](https://drive.google.com/your-link) |

### Evaluation Datasets

| Dataset | Task | Samples | Download Link |
|---------|------|---------|---------------|
| AID | Scene Classification | - | [Google Drive](https://drive.google.com/your-link) |
| WHU-RS19 | Scene Classification | - | [Google Drive](https://drive.google.com/your-link) |
| EuroSAT | Scene Classification | - | [Google Drive](https://drive.google.com/your-link) |
| NWPU | Scene Classification | - | [Google Drive](https://drive.google.com/your-link) |
| fMoW | Scene Classification | - | [Google Drive](https://drive.google.com/your-link) |

### Complete Dataset Package

**All Datasets (Preprocessed)**: [📦 Download from Google Drive](https://drive.google.com/your-complete-dataset-link)

### Dataset Structure

After downloading, organize the datasets as follows:

```
AerithVLM/
├── data/
│   ├── UCM/
│   │   ├── images/
│   │   └── annotations.json
│   ├── RSVG/
│   │   ├── images/
│   │   └── annotations.json
│   ├── DIOR-RSVG/
│   ├── RSICap/
│   ├── RSVQA-LR/
│   └── ...
```

---

## 💾 Model Checkpoints

### Pre-trained Models

| Model | Description | Size | Download Link |
|-------|-------------|------|---------------|
| AerithVLM-Base | Base model with CLIP + DINOv3 fusion | 7.2GB | [Google Drive](https://drive.google.com/your-checkpoint-link) |
| AerithVLM-Finetuned | Fully fine-tuned on RS datasets | 7.5GB | [Google Drive](https://drive.google.com/your-checkpoint-link) |
| DINOv3 Projection Head | Aligned projection weights | 150MB | [Google Drive](https://drive.google.com/your-checkpoint-link) |
| LoRA Adapters | Fine-tuned LoRA weights | 200MB | [Google Drive](https://drive.google.com/your-checkpoint-link) |

### Checkpoint Structure

```
checkpoints/
├── aerithvlm_base/
│   ├── vision_encoder.pth
│   ├── projection_head.pth
│   ├── attention_pooler.pth
│   └── llama_lora_adapter/
├── aerithvlm_finetuned/
│   └── ...
```

---

## 🚀 Usage

### Quick Start

```python
from aerithvlm import AerithVLM
from PIL import Image

# Load model
model = AerithVLM.from_pretrained("checkpoints/aerithvlm_finetuned")
model.eval()

# Load image
image = Image.open("path/to/satellite_image.jpg")

# Scene Classification
prompt = "[CLS] What type of scene is this?"
output = model.generate(image, prompt)
print(f"Scene: {output}")

# Visual Question Answering
prompt = "[VQA] How many buildings are visible in this image?"
output = model.generate(image, prompt)
print(f"Answer: {output}")

# Image Captioning
prompt = "[CAP] Describe this satellite image."
output = model.generate(image, prompt)
print(f"Caption: {output}")
```

### Training

See the Jupyter notebooks in the `notebooks/` directory:

- `01_data_preprocessing.ipynb` - Dataset preparation and preprocessing
- `02_train_projection_head.ipynb` - Training DINOv3-CLIP alignment
- `03_finetune_model.ipynb` - Full model fine-tuning with LoRA
- `04_evaluation.ipynb` - Model evaluation on benchmarks

### Inference Examples

Check out `notebooks/05_inference_examples.ipynb` for detailed examples including:
- Scene classification
- Visual grounding
- Visual question answering
- Image captioning
- Zero-shot evaluation

---

## 📈 Results

### Scene Classification Performance

| Dataset | Accuracy (%) |
|---------|--------------|
| AID | 85.3 |
| WHU-RS19 | 92.1 |
| EuroSAT | 95.7 |
| NWPU | 88.4 |
| UCM | 96.2 |

### Visual Grounding Performance

| Dataset | Accuracy (%) |
|---------|--------------|
| RSVG | 81.7 |
| DIOR-RSVG | 94.4 |

### Visual Question Answering

| Dataset | Accuracy (%) |
|---------|--------------|
| RSVQA-LR | 92.0 |

*For complete results and ablation studies, refer to the thesis document.*

---

## 📝 Citation

If you use AerithVLM in your research, please cite:

```bibtex
@mastersthesis{aerithvlm2025,
  title={AerithVLM: A Hybrid Vision-Language Model for Remote Sensing},
  author={Your Name},
  year={2025},
  school={Your University},
  type={Master's Thesis}
}
```

---

## 🙏 Acknowledgements

This work builds upon several excellent open-source projects:

- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [DINOv3](https://github.com/facebookresearch/dinov2) by Meta AI
- [LLaMA-2](https://github.com/facebookresearch/llama) by Meta AI
- [LHRS-Bot-Nova](https://github.com/NJU-LHRS/LHRS-Bot) for architectural inspiration
- [Transformers](https://github.com/huggingface/transformers) by Hugging Face

Special thanks to my supervisor for guidance and support throughout this research.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or collaboration opportunities, please reach out:

- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)

---

## 🔗 Related Resources

- [Thesis Document](link-to-thesis.pdf)
- [Project Presentation](link-to-presentation.pdf)
- [Demo Video](link-to-demo-video)

---

**Star ⭐ this repository if you find it helpful!**
