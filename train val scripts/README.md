# Fire/Non-Fire Image Classification Project

This project implements multiple deep learning models for binary fire/non-fire image classification on the OHID-FF dataset.

## Project Structure

- `config.py`: Project configuration file
- `data_loader.py`: Data loading and preprocessing
- `models.py`: Definition of various neural network models
- `trainer.py`: Training and evaluation logic
- `evaluate_all.py`: Batch evaluation of all models
- `main.py`: Main project entry point
- `prepare_data.py`: Data preparation script
- `requirements.txt`: List of Python dependencies

## Supported Models

1. ResNet18
2. ResNet50
3. VGG16
4. Logistic Regression
5. MobileNetV2
6. DenseNet121
7. ShuffleNetV2
8. InceptionV3

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Run the data preparation script to create directory structure:

```bash
python prepare_data.py
```

Then place your images into the corresponding folders.

### 3. Train Models

```bash
python main.py
```

### 4. View Results

After training is complete, results will be saved in the `results/` directory, including:
- Model comparison CSV file
- Performance visualization charts

## Dataset Information

- Dataset size: 1,197 images (512×512 pixels)
- Class distribution: 647 fire images (54.0%), 550 non-fire images (46.0%)
- Training set: 597 images
- Test set: 600 images

## Experimental Setup

- Each model is trained independently 3 times
- Pretrained weights are used (except for Logistic Regression)
- Stratified random sampling is used to split training and test sets
- Original class proportions are maintained