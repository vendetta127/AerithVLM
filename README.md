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
```
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.8 (for GPU support)
```

### Setup
```
# Clone the repository
git clone https://github.com/vendetta127/AerithVLM.git
cd AerithVLM

# Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
```
pip install torch torchvision transformers
pip install opencv-python pillow numpy pandas
pip install accelerate bitsandbytes peft
pip install sentencepiece protobuf
```

---
## 📊 Datasets
We use multiple remote sensing datasets for training and evaluation:

### Training Datasets
| Dataset    | Task                | Samples | Download Link |
|------------|---------------------|---------|---------------|
| UCM       | Scene Classification | 2,100  | [Dataset Path](https://paperswithcode.com/dataset/uc-merced-land-use-dataset) |
| RSVG      | Visual Grounding    | 1,276  | [Dataset Path](https://zenodo.org/records/6344334) |
| DIOR-RSVG | Visual Grounding    | 3,372  | [Dataset Path](https://zenodo.org/records/6344334) |
| RSICap    | Image Captioning    | 2,585  | [Dataset Path](dataset/RSICap) |
| RSVQA-LR  | Visual QA           | 572    | [Dataset Path](https://zenodo.org/records/6344334) |
| RSITMD    | Scene Classification | 2,632  | [Dataset Path](https://www.bing.com/search?q=RSITMD+dataset&cvid=3765d53e982043e29868d21bd7ef7785&gs_lcrp=EgRlZGdlKgYIABBFGDkyBggAEEUYOTIGCAEQABhAMgYIAhAAGEDSAQgxODAwajBqNKgCALACAA&FORM=ANAB01&PC=U531) |

### Evaluation Datasets
| Dataset    | Task                | Samples | Download Link |
|------------|---------------------|---------|---------------|
| AID       | Scene Classification | -      | [Google Drive](https://drive.google.com/drive/folders/1jjxiS4QoRHOHupkTPOMf8-YXQSXkVXjL?usp=sharing) |
| WHU-RS19  | Scene Classification | -      | [Google Drive](https://drive.google.com/drive/folders/1jjxiS4QoRHOHupkTPOMf8-YXQSXkVXjL?usp=sharing) |
| SIRI-WHU  | Scene Classification | -      | [Google Drive](https://drive.google.com/drive/folders/1jjxiS4QoRHOHupkTPOMf8-YXQSXkVXjL?usp=sharing) |
| EuroSAT   | Scene Classification | -      | [Google Drive](https://drive.google.com/drive/folders/1jjxiS4QoRHOHupkTPOMf8-YXQSXkVXjL?usp=sharing) |
| NWPU      | Scene Classification | -      | [Google Drive](https://drive.google.com/drive/folders/1jjxiS4QoRHOHupkTPOMf8-YXQSXkVXjL?usp=sharing) |
| METER-ML  | Scene Classification | -      | [Google Drive](https://drive.google.com/drive/folders/1jjxiS4QoRHOHupkTPOMf8-YXQSXkVXjL?usp=sharing) |
| fMoW      | Scene Classification | -      | [Google Drive](https://drive.google.com/drive/folders/1jjxiS4QoRHOHupkTPOMf8-YXQSXkVXjL?usp=sharing) |

### Complete Dataset Package
**All Datasets (Preprocessed)**: [📦 Download from Google Drive](https://drive.google.com/drive/folders/1jjxiS4QoRHOHupkTPOMf8-YXQSXkVXjL?usp=sharing)

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
| Model                 | Description                     | Size  | Download Link |
|-----------------------|---------------------------------|-------|---------------|
| AerithVLM-Base       | Base model with CLIP + DINOv3 fusion | 3.11GB | [Google Drive](https://drive.google.com/file/d/1yRP2gXqlejMOvNiyZGgV19fCZ3Bllnf-/view?usp=drive_link) |
| DINOv3 Projection Head | Aligned projection weights     | 2.5MB | [Google Drive](https://drive.google.com/file/d/14AqDKeL7PvGi5cjnwSFJQoUaZCQDBNNZ/view?usp=drive_link) |
| LoRA Adapters        | Fine-tuned LoRA weights         | 640MB | [Google Drive](https://drive.google.com/file/d/1NnlHb6Xdb8WX00vcyHio-Fv5WQWQZvI8/view?usp=drive_link) |

---
## 📝 Citation
If you use AerithVLM in your research, please cite:
```bibtex
@mastersthesis{aerithvlm2025,
  title={AerithVLM: A Hybrid Vision-Language Model for Remote Sensing},
  author={Muhammad Haris Shahid},
  year={2025},
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
- **Email**: harisshahid127.hr@gmail.com
- **GitHub**: [@vendetta127](https://github.com/vendetta127)
- **LinkedIn**: [Muhammad Haris Shahid](https://www.linkedin.com/in/muhammadharisshahid/)

---
## 🔗 Related Resources
- [Thesis Document](link-to-thesis.pdf)
- [Project Presentation](link-to-presentation.pdf)
- [Demo Video](link-to-demo-video)

---
**Star ⭐ this repository if you find it helpful!**
