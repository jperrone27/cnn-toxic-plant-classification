# Deep Learning for Identification of Toxic Plant Species

## Project Overview
This repository contains the source code and research methodology for a deep learning-based image classification system designed to identify toxic plant species native to North America. Originally developed as a research project for **Dr. Hill Zhu’s Deep Learning graduate class at Florida Atlantic University**, this project investigates the efficacy of various Convolutional Neural Network (CNN) architectures in distinguishing between toxic plants and their nontoxic "lookalikes."

The primary goal is to provide a framework for accessible identification technologies to mitigate the roughly 10 million accidental exposure cases to plants like poison ivy and oak each year.

## Key Features
* **Multi-Architecture Evaluation:** Comparative analysis of **ResNet-50**, **Inception V3**, and **VGG-16** models.
* **Transfer Learning & Fine-Tuning:** Leveraging weights pretrained on the ImageNet dataset with custom unfreezing of deep layers to adapt to botanical features.
* **Hardware Acceleration:** Optimized for **NVIDIA CUDA** to leverage GPU parallel processing, significantly reducing training runtimes compared to standard CPU execution.
* **Robust Preprocessing:** Automated pipeline for image resizing, RGB-to-NumPy array conversion, and real-time data augmentation.

## Dataset Description
The model was trained on a curated dataset of **9,952 images** (split 50/50 between toxic and nontoxic classes) sourced via Kaggle from iNaturalist. The dataset covers 10 distinct species:

| **Toxic Species** (Label: 1) | **Nontoxic Lookalikes** (Label: 0) |
| :--- | :--- |
| Eastern Poison Ivy | Fragrant Sumac |
| Western Poison Ivy | Virginia Creeper |
| Eastern Poison Oak | Boxelder |
| Western Poison Oak | Bear Oak |
| Poison Sumac | Jack-in-the-pulpit |

*Note: The dataset is included in `.gitignore` due to its size (10,000+ nested images).*

## Methodology
### 1. Data Augmentation & Preprocessing
To combat overfitting and improve generalization, the system implements:
* Random rotations, zooms, and horizontal/vertical flips.
* Brightness adjustments (0.8 to 1.2 scale).
* Learning rate scheduling to decay the learning rate when validation loss stagnates.

### 2. Model Performance
All models were evaluated over 50 epochs. **ResNet-50** emerged as the top-performing architecture by leveraging "skip connections" to mitigate vanishing gradients.

| Architecture | Training Accuracy | Validation Accuracy |
| :--- | :--- | :--- |
| **ResNet-50** | 96.60% | **84.69%** |
| **Inception V3** | 99.55% | 84.22% |
| **VGG-16** | 86.30% | 82.06% |


## Technologies Used
* **Frameworks:** TensorFlow, Keras
* **Language:** Python
* **Libraries:** NumPy, Scikit-Learn, Matplotlib
* **Hardware:** NVIDIA GeForce GTX 1050 (CUDA-enabled)

## Usage
The current scripts support binary classification. Ensure your local directory reflects the structure expected by the `ImageDataGenerator` in the scripts.

1.  **Preprocessing:** Run `preprocess.py` to handle initial image formatting.
2.  **Training:** Execute the specific model script (e.g., `resnet_train.py`) to begin the fine-tuning process.
3.  **Inference:** Use `predict.py` to run classification on new, unlabeled plant images.
