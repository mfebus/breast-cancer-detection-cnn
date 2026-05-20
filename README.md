🔬 Breast Cancer Detection — CNN Classifier
CNN-based classification model for breast cancer detection, distinguishing between benign and malignant histopathology images with a focus on maximizing recall to reduce false negatives.

🧠 Clinical Context
Breast cancer is one of the most common and life-threatening cancers worldwide. Early detection is critical for improving survival rates and enabling timely treatment. Traditional diagnosis relies on manual examination of histopathology images — a process that is time-consuming, subjective, and dependent on clinical expertise.
This project explores how deep learning can assist clinicians by providing automated, scalable classification of breast tissue images, with a deliberate emphasis on sensitivity (recall) — because in oncology, a missed malignant case carries far greater cost than a false alarm.

🎯 Objective
Develop a CNN model to classify breast tissue images as:

Benign (0) — non-cancerous tissue
Malignant (1) — cancerous tissue

Primary metric: Recall for malignant class — minimizing false negatives is the clinical priority.

📊 Dataset
AttributeDetailSourceBreast Histopathology Images — Kaggle (Paul Mooney)Image typeHistopathology patches (50×50px, IDC dataset)ClassesBenign (0), Malignant (1)Input sizeResized to 224×224×3ChallengeClass imbalance, staining variability, real-world noise
Citation:

Cruz-Roa et al. (2014) — PubMed
SPIE Proceedings


🏗️ Model Architecture
Custom CNN built with TensorFlow/Keras:
Input (224×224×3)
→ Conv2D(64, 3×3, ReLU) → MaxPooling → BatchNorm
→ Conv2D(32, 3×3, ReLU) → MaxPooling → BatchNorm
→ Conv2D(32, 3×3, ReLU) → MaxPooling
→ Conv2D(16, 3×3, ReLU)
→ Flatten
→ Dense(64, ReLU) → Dropout(0.25)
→ Dense(32, ReLU) → Dropout(0.25)
→ Dense(32, ReLU)
→ Dense(1, Sigmoid)
Design decisions:

Sigmoid output for binary classification
Dropout layers to reduce overfitting
BatchNormalization for training stability
Padding='same' to preserve spatial dimensions


📈 Results (Baseline)

⚠️ These are baseline results from an initial training run. Model optimization is in progress.

MetricBenign (0)Malignant (1)Precision0.480.50Recall0.840.15F1-Score0.610.23Accuracy0.48
Confusion Matrix:
              Predicted
              Benign  Malignant
Actual Benign  [223      43]
       Malignant [242     43]
Diagnosis: The model shows high recall for benign cases but low recall for malignant cases, indicating class imbalance as the primary challenge. The model is defaulting toward the majority class.

🔧 Planned Improvements

 Address class imbalance via class weighting and/or SMOTE oversampling
 Implement transfer learning (ResNet50 / EfficientNet) for stronger feature extraction
 Add data augmentation (rotation, flipping, zoom) to improve generalization
 Tune decision threshold to optimize recall for malignant class
 Evaluate with AUC-ROC as primary metric


🛠️ Tech Stack
ToolPurposePython 3.xCore languageTensorFlow / KerasModel developmentNumPy / PandasData processingMatplotlib / SeabornVisualizationGoogle ColabTraining environmentscikit-learnEvaluation metrics

📁 Repository Structure
breast-cancer-detection-cnn/
├── notebooks/          # Jupyter notebooks (EDA, training, evaluation)
├── src/                # Python scripts
│   └── app.py
├── data/               # Dataset references (data loaded externally)
├── requirements.txt    # Dependencies
└── README.md

🚀 Getting Started
bash# Clone the repo
git clone https://github.com/mfebus/breast-cancer-detection-cnn.git
cd breast-cancer-detection-cnn

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/
Note: Dataset must be downloaded separately from Kaggle due to size constraints.

👩‍💻 Author
Maria Febus
Master of Applied Data Science — University of Michigan (2026)
Transitioning from Pharma R&D Project Management to Clinical Data Science
GitHub | LinkedIn

This project is part of a portfolio targeting clinical AI applications in oncology and pharma.
