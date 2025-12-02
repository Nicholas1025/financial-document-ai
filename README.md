# Financial Document AI

> **AI Framework for Financial Table Understanding from Corporate Annual Reports**

A modular pipeline for detecting, extracting, and validating tabular data from financial documents using state-of-the-art deep learning models.

## 📋 Project Overview

This project implements an end-to-end system for understanding tables in financial documents such as corporate annual reports and SEC filings. The pipeline includes:

- **Table Detection**: Locating table regions within document pages
- **Structure Recognition**: Identifying rows, columns, cells, and headers
- **TEDS Evaluation**: Measuring structural accuracy using Tree-Edit-Distance Similarity

## 🏗️ Architecture

```
Document Source → Input Layer
        ↓
Document Processing (Page Rendering, Layout Extraction, OCR)
        ↓
Table Detection (Microsoft Table Transformer)
        ↓
Structure Reconstruction (Cell Segmentation, Grid Reconstruction, Header ID)
        ↓
Numeric Normalisation → Semantic Mapping → Rule-Based Validation
        ↓
Final Output (Structured Tables + Validation Reports)
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA support (recommended)
- 6GB+ GPU memory

### Setup

```bash
# Clone the repository
git clone https://github.com/Nicholas1025/financial-document-ai.git
cd financial-document-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
financial-document-ai/
├── configs/
│   └── config.yaml          # Dataset paths and model settings
├── modules/
│   ├── data_loaders.py      # Dataset loaders (PubTabNet, FinTabNet, DocLayNet, PubTables-1M)
│   ├── detection.py         # Table detection using Table Transformer
│   ├── structure.py         # Structure recognition (rows, columns, cells)
│   ├── metrics.py           # Evaluation metrics (TEDS, Precision, Recall, F1)
│   └── utils.py             # Utility functions
├── experiments/
│   ├── baseline_doclaynet.py    # DocLayNet detection baseline
│   ├── baseline_pubtables1m.py  # PubTables-1M structure baseline
│   ├── baseline_fintabnet.py    # FinTabNet structure baseline
│   ├── baseline_pubtabnet.py    # PubTabNet TEDS baseline
│   ├── run_all_baselines.py     # Run all experiments
│   └── generate_chapter4_figures.py  # Generate thesis figures
├── outputs/
│   ├── results/             # Experiment JSON results
│   └── figures/             # Generated visualizations
├── data/
│   └── samples/             # Sample test images
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
└── README.md
```

## 📊 Datasets

The system supports four benchmark datasets:

| Dataset | Size | Domain | Task |
|---------|------|--------|------|
| [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) | 568K | Scientific (PubMed) | Structure + Content |
| [PubTables-1M](https://github.com/microsoft/table-transformer) | 1M | Scientific (PubMed) | Detection + Structure |
| [FinTabNet](https://developer.ibm.com/data/fintabnet/) | 113K | Financial (SEC) | Structure Recognition |
| [DocLayNet](https://github.com/DS4SD/DocLayNet) | 81K | Mixed Documents | Layout Detection |

### Dataset Configuration

Update `configs/config.yaml` with your dataset paths:

```yaml
datasets:
  pubtabnet:
    root: "D:/datasets/PubTabNet/pubtabnet/pubtabnet"
    # ...
  fintabnet:
    root: "D:/datasets/FinTabNet_c/FinTabNet.c-Structure/FinTabNet.c-Structure"
    # ...
```

## 🚀 Usage

### Run Individual Baseline Experiments

```bash
# Activate virtual environment
venv\Scripts\activate

# DocLayNet - Table Detection
python experiments/baseline_doclaynet.py --num_samples 500

# PubTables-1M - Structure Recognition
python experiments/baseline_pubtables1m.py --num_samples 500

# FinTabNet - Financial Table Structure
python experiments/baseline_fintabnet.py --num_samples 500

# PubTabNet - TEDS Evaluation
python experiments/baseline_pubtabnet.py --num_samples 500
```

### Run All Baselines

```bash
python experiments/run_all_baselines.py
```

### Generate Chapter 4 Figures

```bash
python experiments/generate_chapter4_figures.py
```

## 📈 Baseline Results (500 samples)

| Dataset | Task | Metric | Score |
|---------|------|--------|-------|
| DocLayNet | Table Detection | F1 @ IoU 0.5 | **74.66%** |
| PubTables-1M | Structure Recognition | Row F1 | **95.24%** |
| PubTables-1M | Structure Recognition | Column F1 | **98.73%** |
| FinTabNet | Structure Recognition | Row F1 | **95.21%** |
| FinTabNet | Structure Recognition | Column F1 | **99.27%** |
| PubTabNet | End-to-End | TEDS (structure) | **48.12%** |
| PubTabNet | End-to-End | TEDS (content) | **16.52%** |

## 🔧 Models Used

- **Table Detection**: [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection)
- **Structure Recognition**: [microsoft/table-transformer-structure-recognition-v1.1-all](https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all)

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | Correctly detected tables / All predicted tables |
| **Recall** | Correctly detected tables / All ground truth tables |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **IoU** | Intersection over Union of bounding boxes |
| **TEDS** | Tree-Edit-Distance Similarity for structure evaluation |

## 🖥️ Hardware Requirements

- **GPU**: NVIDIA GTX 1660 Ti (6GB) or better
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for datasets

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{financial-document-ai,
  author = {Nicholas},
  title = {AI Framework for Financial Table Understanding},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Nicholas1025/financial-document-ai}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft Research for Table Transformer models
- IBM Research for PubTabNet and FinTabNet datasets
- DS4SD for DocLayNet dataset
