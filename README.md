Here’s a ready-to-copy **README.md** for your SonoKidney GitHub repo:

````markdown
# SonoKidney AI

SonoKidney AI is a deep learning-based kidney ultrasound image classification system. It uses a custom Convolutional Neural Network (CNN) model to classify kidney ultrasound images into four categories: **Cyst**, **Normal**, **Stone**, and **Tumor**.

The system includes a trained CNN model and a Streamlit-based web interface where users can upload ultrasound images and view prediction results with confidence scores.

---

## Project Overview

Kidney disease is a serious medical condition that may progress without obvious symptoms. Ultrasound imaging is commonly used for kidney screening because it is non-invasive, cost-effective, and radiation-free. However, interpretation may vary depending on the availability and experience of medical specialists.

SonoKidney AI was developed as an academic prototype to demonstrate how deep learning can assist in the automated classification of kidney abnormalities from sonographic images.

---

## Features

- Kidney ultrasound image classification
- Four-class prediction:
  - Cyst
  - Normal
  - Stone
  - Tumor
- Custom CNN architecture
- Image preprocessing and augmentation
- Model training script
- Confusion matrix and classification report generation
- Streamlit web application
- Image upload and prediction interface
- Confidence score display

---

## Model Performance

The trained SonoKidney CNN model achieved the following test results:

| Metric | Value |
|---|---:|
| Accuracy | 94.50% |
| Precision | 94.48% |
| Recall | 94.22% |
| F1-Score | 94% - 95% |

Classification report:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Cyst | 0.96 | 0.95 | 0.96 | 259 |
| Normal | 1.00 | 0.99 | 1.00 | 151 |
| Stone | 0.95 | 0.85 | 0.90 | 149 |
| Tumor | 0.86 | 0.97 | 0.91 | 150 |

---

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Structure

```text
SonoKidney/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── dataset/
│   ├── cyst/
│   ├── normal/
│   ├── stone/
│   └── tumor/
├── models/
│   ├── best_sonokidney_model.keras
│   └── class_indices.json
└── outputs/
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── training_accuracy.png
    ├── training_loss.png
    ├── dataset_standardized/
    └── sonography_final_split/
````

---

## Dataset

The project uses a kidney ultrasound image dataset containing four classes:

* Cyst
* Normal
* Stone
* Tumor

The dataset should be placed inside the `dataset/` folder using the following structure:

```text
dataset/
├── cyst/
├── normal/
├── stone/
└── tumor/
```

Each folder should contain the corresponding ultrasound images.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sonokidney-ai.git
cd sonokidney-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet created, install the dependencies manually:

```bash
pip install tensorflow streamlit pillow numpy matplotlib seaborn scikit-learn
```

---

## Training the Model

To train the CNN model, run:

```bash
python train_model.py
```

The training script will:

* Standardize images to 224 × 224 pixels
* Split the dataset into training, validation, and testing sets
* Train the CNN model
* Save the best model inside the `models/` folder
* Generate classification report and confusion matrix outputs

After training, the following files should be generated:

```text
models/best_sonokidney_model.keras
models/class_indices.json
outputs/classification_report.txt
outputs/confusion_matrix.png
```

---

## Running the Web App

To launch the Streamlit application, run:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

The web app allows users to:

1. Upload a kidney ultrasound image.
2. Preview the uploaded image.
3. Analyze the image using the trained CNN model.
4. View the predicted class and confidence scores.

---

## CNN Architecture

The proposed SonoKidney CNN architecture uses six convolutional blocks followed by a dense classification layer.

Architecture summary:

| Stage        | Layer Description                                    |    Output Size |
| ------------ | ---------------------------------------------------- | -------------: |
| Input        | Kidney ultrasound image                              |  224 × 224 × 3 |
| Conv Block 1 | Conv2D 32 filters, 3×3, stride 2 + BatchNorm + ReLU  | 112 × 112 × 32 |
| Conv Block 2 | Conv2D 32 filters, 3×3, stride 2 + BatchNorm + ReLU  |   56 × 56 × 32 |
| Conv Block 3 | Conv2D 64 filters, 3×3, stride 2 + BatchNorm + ReLU  |   28 × 28 × 64 |
| Conv Block 4 | Conv2D 128 filters, 3×3, stride 2 + BatchNorm + ReLU |  14 × 14 × 128 |
| Conv Block 5 | Conv2D 128 filters, 3×3, stride 2 + BatchNorm + ReLU |    7 × 7 × 128 |
| Conv Block 6 | Conv2D 128 filters, 3×3 + BatchNorm + ReLU           |    7 × 7 × 128 |
| Flatten      | Converts feature map into vector                     | 6,272 features |
| Dense Layer  | Dense 128 + ReLU                                     |    128 neurons |
| Dropout      | Dropout regularization                               |            0.5 |
| Output Layer | Softmax classification                               |      4 classes |

---

## Example Prediction

After uploading an ultrasound image, the system displays:

```text
Predicted Class: CYST
Confidence: 99.9%
```

It also shows confidence scores for all four classes.

---

## Requirements

Example `requirements.txt`:

```text
tensorflow
streamlit
pillow
numpy
matplotlib
seaborn
scikit-learn
```

---

## Important Notes

This project is for **academic and research purposes only**.

SonoKidney AI is **not a replacement for professional medical diagnosis**. Any medical interpretation should be verified by a qualified healthcare professional.

---

## Future Improvements

Possible improvements include:

* Adding more real-world ultrasound images
* Testing pretrained models such as EfficientNet, ResNet, MobileNet, or DenseNet
* Adding Grad-CAM explainability
* Improving the Streamlit user interface
* Adding downloadable diagnostic reports
* Adding user authentication
* Deploying the system online
* Validating the model with clinical experts

---

## Authors

Developed by:

* Gimarino, Althia Grace P.
* Gawan, Kristel Faith

College of Computing Education
University of Mindanao

---

## License

This project is intended for academic use. You may add a license such as MIT License if you want to make the repository open-source.

```
```
