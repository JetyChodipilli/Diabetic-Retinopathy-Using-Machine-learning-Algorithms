# 🧠 Diabetic Retinopathy Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 Overview

This project develops a deep learning-based system for detecting and classifying Diabetic Retinopathy (DR) stages—**No DR, Mild, Moderate, Severe, Proliferative**—using fundus images.

It conducts a comparative study of multiple models:
- CNN
- DenseNet
- VGG16
- VGG19
- ResNetInceptionV2
- Depth-Separable Convolutional Model

The system supports real-time image quality assessment, lesion detection, and DR grading. It uses the **APTOS 2019 dataset** from Kaggle and achieves up to **98% accuracy** with DenseNet.

---

## ✨ Key Features

- 🎯 **Categorical Classification** of DR stages
- 🔍 **Model Comparison** across six architectures
- 🚀 **High Performance**:  
  - DenseNet: 98%  
  - ResNetInceptionV2: 95%  
  - VGG19 (oversampled): 84%  
  - CNN: 74%  
  - VGG16: 72%  
  - Depth-Separable: 60%  
  - Baseline: 52%
- 🩺 **Real-Time Capabilities** for clinical use
- 📊 **Data Handling** with augmentation and oversampling

---

## 🧰 Technologies Used

- **Language**: Python 3.8+
- **Frameworks**: TensorFlow 2.12.0, Keras
- **Libraries**: NumPy, Pandas, OpenCV, Scikit-learn, Matplotlib
- **Tools**: Jupyter Notebook, Google Colab (optional)
- **Datasets**: APTOS 2019 (Kaggle), local dataset, external datasets

---

## 📦 Dataset Details

- **APTOS 2019 (Kaggle)**  
  - 466,247 fundus images from 121,342 patients  
  - Used for training and validation

- **Local Dataset**  
  - 200,136 images from 52,004 patients  
  - Used for evaluation

- **External Datasets**  
  - Three datasets totaling 209,322 images  
  - Used for generalization testing

---

## 🧪 Preprocessing

- Resize images to **224×224 pixels**
- Normalize pixel values
- Apply **data augmentation**: rotation, flipping, zooming
- Use **oversampling** for VGG19 to improve minority class detection

---

## 📊 Model Performance

| Model                  | Accuracy |
|------------------------|----------|
| DenseNet               | 98%      |
| ResNetInceptionV2      | 95%      |
| VGG19 (oversampled)    | 84%      |
| CNN                    | 74%      |
| VGG16                  | 72%      |
| Depth-Separable        | 60%      |
| Baseline               | 52%      |

> **Key Insight**: DenseNet and oversampled VGG19 showed superior performance.

Evaluation metrics include:
- Confusion matrices
- ROC curves
- Precision-recall curves

---

## 📁 Project Structure

```text
diabetic-retinopathy-detection/
├── data/
│   └── aptos2019/              # APTOS 2019 dataset
├── models/
│   ├── cnn.py
│   ├── densenet.py
│   ├── vgg16.py
│   ├── vgg19.py
│   ├── resnet_inception_v2.py
│   └── depth_separable.py
├── notebooks/
│   └── train_evaluate.ipynb
├── utils/
│   ├── preprocess.py
│   └── metrics.py
├── results/
│   ├── evaluation_metrics.csv
│   └── visualizations/
├── assets/
│   └── dr_detection_logo.png
└── README.md
```

---

## 🛠️ Prerequisites

- Python 3.8+
- GPU (recommended: Colab, AWS, or local)
- Kaggle account to download APTOS 2019

Install dependencies:

```bash
pip install tensorflow==2.12.0 keras==2.12.0 numpy==1.23.5 pandas==1.5.3 \
            opencv-python==4.7.0 scikit-learn==1.2.2 matplotlib==3.7.1
```

---

## ⚙️ Setup Instructions

### Clone the Repository

```bash
git clone <repository-url>
cd diabetic-retinopathy-detection
```

### Create `requirements.txt`

```text
tensorflow==2.12.0
keras==2.12.0
numpy==1.23.5
pandas==1.5.3
opencv-python==4.7.0
scikit-learn==1.2.2
matplotlib==3.7.1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download & Prepare Dataset

- Download APTOS 2019 from Kaggle
- Place it in `data/aptos2019/`
- Run preprocessing:

```bash
python utils/preprocess.py
```

---

## 🧪 Train Models

- Open `notebooks/train_evaluate.ipynb` in Jupyter or Colab
- Train each model (CNN, DenseNet, etc.)
- Or run individual scripts:

```bash
python models/densenet.py
```

---

## 📈 Evaluate Models

- Results saved in `results/evaluation_metrics.csv`
- Visualize metrics:

```bash
python utils/metrics.py
```

---

## 🚀 Usage

### Training

- Use `train_evaluate.ipynb` to train and compare models
- Adjust hyperparameters as needed

### Prediction

```python
from models.densenet import DenseNetModel

model = DenseNetModel()
model.load_weights('models/densenet_weights.h5')
prediction = model.predict('path/to/fundus_image.jpg')
print(f"Predicted DR Stage: {prediction}")
```

### Visualization

```bash
python utils/metrics.py
```

Outputs saved in `results/visualizations/`

---

## 🩺 Real-Time Application

Deploy the model for real-time DR detection (additional setup required).

---

## 🖼️ Sample Visualizations

Example confusion matrix for DenseNet model included in `results/visualizations/`.

---

## 🧠 Model Architecture

DenseNet uses **dense connectivity** to enhance feature reuse and reduce parameters, contributing to its high accuracy.

---

## 📝 Notes

- A Kaggle account is required to access the dataset
- Oversampling was key for VGG19 performance
- GPU is highly recommended for training
- DenseNet is preferred for production deployment

---

## 🛠️ Troubleshooting

| Issue               | Solution                                                                 |
|---------------------|--------------------------------------------------------------------------|
| Dataset Errors      | Ensure APTOS 2019 is in `data/aptos2019/`                                |
| Memory Issues       | Reduce batch size or use Colab with GPU                                  |
| Convergence Issues  | Tune learning rate or number of epochs                                   |
| Dependency Conflicts| Use exact versions from `requirements.txt`                               |
| README Wrapping     | Ensure file is named `README.md` and uses LF line endings                |

```bash
sed -i 's/\r$//' README.md
git add README.md
git commit -m "Fix line endings"
git push origin main
```

---

## 🔮 Future Enhancements

- Add ensemble methods
- Integrate more datasets
- Build a web interface
- Optimize for edge devices (quantization/pruning)

---

## 🤝 Contributing

Contributions are welcome! Submit pull requests or open issues for bugs, improvements, or new features.

---

## 📜 License

This project is licensed under the **MIT License**.
