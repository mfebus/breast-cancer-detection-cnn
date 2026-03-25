# breast-cancer-detection-cnn
CNN-based classification model for breast cancer detection, distinguishing between benign and malignant cells with a focus on maximizing recall to reduce false negatives.

## 🧠 Context
Breast cancer is one of the most common and life-threatening cancers worldwide. Early detection is critical for improving survival rates and enabling timely treatment. Traditional diagnosis often relies on manual examination of histopathology images or mammograms, a process that can be time-consuming, subjective, and dependent on clinical expertise.

With increasing demand for scalable and reliable diagnostic tools, there is a growing need for AI-driven solutions that can assist clinicians in identifying cancerous tissue more accurately and efficiently.

---

## 🎯 Objective
To develop a deep learning model capable of classifying breast tissue images as either:

- **Benign**: non-cancerous tissue  
- **Malignant**: cancerous tissue  

The model is designed with a focus on **maximizing recall (sensitivity)** to minimize false negatives, which is critical in medical diagnosis where missed detections can have severe consequences.

---

## 📊 Dataset Overview
- Histopathology image dataset of breast tissue samples  
- Two classes:
  - **Benign (0)**: non-cancerous tissue  
  - **Malignant (1)**: presence of cancerous cells  
- Images may contain noise, variability in staining, and structural differences typical of real-world medical data  

---

## 🛠️ Tools & Technologies
- Python  
- TensorFlow / Keras  
- Convolutional Neural Networks (CNNs)  
- NumPy, Pandas  
- Matplotlib / Seaborn for visualization  
- Jupyter Notebooks (Google Colab)  

---

## 📌 Key Learning Outcomes
- Applied CNN architectures for medical image classification  
- Developed models with a focus on **recall optimization** for clinical relevance  
- Gained hands-on experience working with real-world, high-variability medical imaging data  
- Strengthened understanding of the role of AI in supporting diagnostic decision-making  
