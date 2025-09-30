<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->

[![Python][Python-shield]][Python-url]
[![Jupyter][Jupyter-shield]][Jupyter-url]
[![PyTorch][PyTorch-shield]][PyTorch-url]
[![scikit-learn][sklearn-shield]][sklearn-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h1 align="center">JakOlah Classifier</h1>

  <p align="center">
    Machine Learning Pipeline for Smart Waste Classification
    <br />
    CNN Feature Extraction + SVM Classification for Jakarta Waste Management
    <br />
    <br />
    <a href="#usage">View Results</a>
    ·
    <a href="#methodology">Methodology</a>
    ·
    <a href="#performance">Performance</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#methodology">Methodology</a></li>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#dataset">Dataset</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#pipeline-overview">Pipeline Overview</a></li>
    <li><a href="#performance">Performance</a></li>
    <li><a href="#results">Results</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

JakOlah Classifier is a sophisticated machine learning pipeline designed to classify waste images into three categories for Jakarta's waste management system: **Organik** (Organic), **Anorganik** (Inorganic), and **Lainnya** (Others).

**Key Innovation:**

- **Hybrid CNN+SVM Architecture**: Combines the power of pre-trained CNNs for feature extraction with SVM's robust classification capabilities
- **Transfer Learning Approach**: Leverages pre-trained ImageNet models (ResNet50, MobileNetV3) as feature extractors
- **Multiple Kernel Support**: Implements both RBF and Polynomial SVM kernels for optimal performance
- **Production-Ready Pipeline**: Complete end-to-end workflow from preprocessing to model evaluation

**Why This Approach?**

- Pre-trained CNNs provide rich feature representations without requiring large-scale training
- SVM classifiers offer excellent generalization with smaller datasets
- Hybrid approach reduces computational requirements while maintaining high accuracy
- Multiple model variants allow for performance optimization based on deployment constraints

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Methodology

Our classification pipeline implements a **two-stage approach**:

1. **Feature Extraction Stage**: Pre-trained CNN models (frozen weights) extract deep features from preprocessed images
2. **Classification Stage**: SVM models with different kernels classify based on extracted features

**Pipeline Architecture:**

```
Raw Images → Preprocessing → CNN Feature Extractor → SVM Classifier → Predictions
             (Resize,       (ResNet50/MobileNetV3)   (RBF/Polynomial)
              Normalize)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- [![Python][Python-shield]][Python-url] - Core programming language
- [![PyTorch][PyTorch-shield]][PyTorch-url] - Deep learning framework for CNN models
- [![scikit-learn][sklearn-shield]][sklearn-url] - SVM implementation and evaluation metrics
- [![Jupyter][Jupyter-shield]][Jupyter-url] - Interactive development environment
- [![NumPy][NumPy-shield]][NumPy-url] - Numerical computing
- [![Pandas][Pandas-shield]][Pandas-url] - Data manipulation and analysis
- [![Matplotlib][Matplotlib-shield]][Matplotlib-url] - Data visualization
- [![PIL][PIL-shield]][PIL-url] - Image processing

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Dataset

- **Classes**: 3 waste categories (Organik, Anorganik, Lainnya)
- **Split**: 70% Training, 15% Validation, 15% Testing
- **Preprocessing**: Images resized to 224x224, ImageNet normalization applied
- **Format**: Stratified splitting ensures balanced representation across all classes

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

Follow these steps to reproduce the experiments and run the classification pipeline.

### Prerequisites

Ensure you have the following installed:

- **Python 3.8+**
  ```sh
  python --version
  ```
- **Jupyter Notebook or JupyterLab**
  ```sh
  jupyter --version
  ```
- **Git** (for cloning)
  ```sh
  git --version
  ```

### Installation

1. **Clone the repository**

   ```sh
   git clone <repository-url>
   cd "JakOlah Classifier"
   ```

2. **Create virtual environment**

   ```sh
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install required packages**

   ```sh
   pip install torch torchvision torchaudio
   pip install scikit-learn pandas numpy matplotlib seaborn
   pip install pillow jupyter ipykernel
   pip install opencv-python tqdm
   ```

4. **Set up Jupyter kernel**

   ```sh
   python -m ipykernel install --user --name=jakolah-classifier
   ```

5. **Prepare dataset structure**
   ```
   dataset/
   ├── Organik/
   ├── Anorganik/
   └── Lainnya/
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

### Running the Complete Pipeline

Execute the notebooks in order for the full workflow:

1. **Data Preprocessing**

   ```sh
   jupyter notebook 01-preprocessing.ipynb
   ```

   - Data exploration and visualization
   - Image preprocessing and normalization
   - Dataset splitting (train/val/test)
   - Export preprocessed data

2. **Feature Extraction**

   ```sh
   jupyter notebook 02-feature-extraction.ipynb
   ```

   - Load pre-trained CNN models (ResNet50, MobileNetV3)
   - Extract features from preprocessed images
   - Save feature vectors for SVM training

3. **SVM Training**

   ```sh
   jupyter notebook 03-svm-training.ipynb
   ```

   - Train SVM classifiers with different kernels
   - Hyperparameter optimization
   - Model validation and selection
   - Save trained models

4. **Model Evaluation**
   ```sh
   jupyter notebook 04-evaluation.ipynb
   ```
   - Comprehensive performance evaluation
   - Classification reports and confusion matrices
   - Visualization of results

### Quick Start

For a streamlined experience:

```sh
# Run all notebooks in sequence
jupyter nbconvert --execute 01-preprocessing.ipynb
jupyter nbconvert --execute 02-feature-extraction.ipynb
jupyter nbconvert --execute 03-svm-training.ipynb
jupyter nbconvert --execute 04-evaluation.ipynb
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PIPELINE OVERVIEW -->

## Pipeline Overview

### 1. Data Preprocessing (`01-preprocessing.ipynb`)

- **Input**: Raw waste images in class folders
- **Output**: Processed dataset with train/val/test splits
- **Key Steps**:
  - Image resizing to 224×224 pixels
  - ImageNet normalization
  - Stratified dataset splitting
  - Metadata generation

### 2. Feature Extraction (`02-feature-extraction.ipynb`)

- **Input**: Preprocessed images
- **Output**: Feature vectors for each image
- **Key Steps**:
  - Load pre-trained CNN models (frozen weights)
  - Extract features from the last pooling layer
  - Save feature vectors as NumPy arrays

### 3. SVM Training (`03-svm-training.ipynb`)

- **Input**: Feature vectors and labels
- **Output**: Trained SVM models
- **Key Steps**:
  - Feature scaling and normalization
  - Train SVM with RBF and Polynomial kernels
  - Hyperparameter tuning with GridSearchCV
  - Model validation and selection

### 4. Evaluation (`04-evaluation.ipynb`)

- **Input**: Trained models and test data
- **Output**: Performance metrics and visualizations
- **Key Steps**:
  - Test set evaluation
  - Classification reports
  - Confusion matrices
  - Performance comparison

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PERFORMANCE -->

## Performance

### Model Architecture Combinations

| CNN Backbone | SVM Kernel | Accuracy | Training Time |
| ------------ | ---------- | -------- | ------------- |
| ResNet50     | RBF        | 97.9%    | 421.2s        |
| ResNet50     | Polynomial | 98.4%    | 318.5s        |
| MobileNetV3  | RBF        | 98.3%    | 153.1s        |
| MobileNetV3  | Polynomial | 98.7%    | 154.2s        |

### Key Performance Insights

- **Best Overall Performance**: [Model Combination] achieves highest accuracy of XX.X%
- **Computational Efficiency**: MobileNetV3 models provide faster inference with competitive accuracy
- **Robustness**: SVM classifiers show consistent performance across different feature extractors
- **Generalization**: Validation results indicate good generalization to unseen data

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESULTS -->

## Results

### Output Files

After running the complete pipeline, you'll find:

```
Result/
├── 01-Preprocessing-Results.zip      # Preprocessed data and metadata
├── 02-Feature-Extraction-Results.zip # Extracted feature vectors
├── 03-SVM-Training-Results.zip       # Trained models and reports
└── 03-SVM-Training/
    ├── MobileNetV3_poly_model.pkl     # Trained SVM models
    ├── MobileNetV3_rbf_model.pkl
    ├── ResNet50_poly_model.pkl
    ├── ResNet50_rbf_model.pkl
    ├── scalers.pkl                    # Feature scalers
    ├── training_config.json           # Training configuration
    ├── training_results.csv           # Performance metrics
    ├── svm_training_report.md         # Detailed training report
    └── visualizations/
        └── feature_analysis.png       # Feature analysis plots
```

### Model Deployment

The trained models can be loaded and used for inference:

```python
import pickle
import numpy as np

# Load trained model and scaler
with open('Result/03-SVM-Training/ResNet50_rbf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Result/03-SVM-Training/scalers.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions
features = extract_features(image)  # Extract features using CNN
features_scaled = scaler.transform(features.reshape(1, -1))
prediction = model.predict(features_scaled)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

[Python-shield]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch-shield]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[sklearn-shield]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[sklearn-url]: https://scikit-learn.org/
[Jupyter-shield]: https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white
[Jupyter-url]: https://jupyter.org/
[NumPy-shield]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Pandas-shield]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Matplotlib-shield]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
[PIL-shield]: https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white
[PIL-url]: https://pillow.readthedocs.io/
