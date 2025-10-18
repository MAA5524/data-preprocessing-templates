# 🧠 Data Preprocessing Templates

A complete collection of **Jupyter Notebook templates** for data preprocessing in **Machine Learning (ML)** and **Deep Learning (DL)** projects.  
This repository provides clean, reusable, and well-documented notebooks for three major data types:

- 🟩 **Tabular Data (Structured)**
- 🟦 **Text Data (NLP)**
- 🟥 **Image Data (Computer Vision)**

Each notebook contains a full end-to-end data preprocessing pipeline — from loading data and performing EDA to feature engineering and model training.

---

## 📂 Repository Structure
```bash
data-preprocessing-templates/
│
├── examples/
    ├── 01_tabular_preprocessing_ex.ipynb
    ├── 02_text_preprocessing_ex.ipynb
    ├── 03_image_preprocessing_ex.ipynb
├── templates/
    ├── 01_tabular_preprocessing.ipynb
    ├── 02_text_preprocessing.ipynb
    ├── 03_image_preprocessing.ipynb
│
├── README.md
└── requirements.txt
```

---

## 🟩 1. Tabular Data Preprocessing

Notebook: `01_tabular_preprocessing.ipynb`

### 🔹 Overview
Covers preprocessing for structured/tabular datasets (e.g., CSV, Excel, SQL data).  
It includes everything from cleaning and visualization to model training.

### 🔸 Steps Included

#### **1. Imports and Settings**
-Load essential Python libraries: pandas, numpy, seaborn, matplotlib.
-Configure display settings for better readability.

#### **2. Load Dataset**
-  Load your CSV dataset into a pandas DataFrame.
  Example:
  ```python
  df = pd.read_csv("data/tabular_data.csv")
  ``` 

#### **3.  Initial Inspection**
- View first rows (`df.head()`)
- data types
- missing values
- duplicates
- summary stats

#### **4. Exploratory Data Analysis (EDA)**
- **Numerical features:** histograms, boxplots, correlations.
- **Categorical features:** count plots.
- **Correlation Heatmap:** detect relationships between numerical columns.
-  **Box Plot:** detect outliers.
-  
#### **5. Missing Value Handling**
Comprehensive handling with **individual functions**:
-  **Drop Columns / Rows** (when missing rate is high/low)
- **Statistical Imputation** (`mean`, `median`, `mode`)
- **Categorical Imputation** (using `mode`)
- **Forward/Backward Fill** for time series
- **KNN Imputer** for feature-correlated filling
> ⚠️ Always perform imputation **after train/test split** to avoid data leakage.

#### **6. Outlier Detection & Treatment**
Two robust methods:
- **IQR Method** — removes extreme values outside interquartile range.
- **Z-Score Method** — filters samples exceeding z-threshold.
 
#### **7. Feature Engineering**
Generate more informative features:
- **Ratio Features** — combine numeric columns meaningfully.
- **Date Extraction** — derive year, month, weekday, etc.


#### **8. Feature Selection**
Reduce dimensionality and improve generalization:
- **SelectKBest (Statistical Tests)** — chi², f_classif.
- **Recursive Feature Elimination (RFE)** — iterative model-based removal.
- **Feature Importance (Tree-based)** — using RandomForest or XGBoost.
  
#### **9. Train-Validation-Test Split**
Split dataset into:
- 64% train
- 16% validation
- 20% test

```python
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2)
```
#### **10. Encoding Categorical Variables**

Different encoding strategies depending on the column type:

| Encoding Type          | Use Case                        | Example             |
| ---------------------- | ------------------------------- | ------------------- |
| **One-Hot Encoding**   | Nominal categorical (unordered) | City, Color         |
| **Label Encoding**     | Binary categorical              | Gender              |
| **Ordinal Encoding**   | Ordered categorical             | Education Level     |
| **Frequency Encoding** | High-cardinality features       | Product ID, Country |

#### **11. Numerical Feature Scaling**
Normalize numeric columns for better model performance:
- **StandardScaler** — mean 0, variance 1 (Gaussian)
- **MinMaxScaler** — [0, 1] normalization
- **RobustScaler** — resistant to outliers
- **Log Transformation** — for skewed distributions

#### **12. Model Training**
End-to-end training examples for common algorithms:
- Logistic Regression
- Decision Tree / Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machine (SVM)
- KNN / Naive Bayes
- Neural Network (MLP)
- Linear & Ridge / Lasso Regression
- PCA for dimensionality reduction
---

## 🟦 2. Text Data Preprocessing (NLP)

Notebook: `02_text_preprocessing.ipynb`

### 🔹 Overview
Covers the full preprocessing workflow for **Natural Language Processing** tasks.

### 🔸 Steps Included

#### **1. Data Loading**
- Import text data (CSV, JSON, or raw text files)

#### **2. Text Cleaning**
- Lowercasing  
- Removing punctuation, numbers, and stopwords  
- Lemmatization & stemming  
- Expanding contractions (“don’t” → “do not”)

#### **3. Tokenization**
- Word-level and sentence-level tokenization  
- Using `nltk`, `spaCy`, or `transformers` tokenizers

#### **4. Text Normalization**
- Removing extra whitespaces  
- Handling emojis and special symbols  
- Dealing with misspellings (TextBlob / SymSpell)

#### **5. Feature Extraction**
- Bag of Words (BoW)  
- TF-IDF  
- Word2Vec / GloVe  
- Sentence Embeddings (BERT, SentenceTransformer)

#### **6. Feature Selection**
- Using chi² or mutual information for feature filtering  
- Dimensionality reduction (TruncatedSVD, PCA for sparse matrices)

#### **7. Model Training**
- Train basic NLP models (Naive Bayes, Logistic Regression, LSTM)  
- Evaluate models using accuracy, F1, confusion matrix  

---

## 🟥 3. Image Data Preprocessing (Computer Vision)

Notebook: `03_image_preprocessing.ipynb`

### 🔹 Overview
Covers preprocessing for image-based datasets before feeding them into CNNs or other models.

### 🔸 Steps Included

#### **1. Image Loading**
- Load from directory or dataset (e.g., CIFAR, ImageNet samples)
- Resize, normalize, and convert color spaces (RGB/Grayscale)

#### **2. Data Cleaning**
- Check for corrupted images  
- Remove duplicates  
- Balance classes (under/oversampling)

#### **3. Image Augmentation**
- Rotation, flipping, shifting, zooming  
- Brightness and contrast adjustments  
- Using `ImageDataGenerator` or `Albumentations`

#### **4. Feature Extraction**
- Pre-trained models (VGG16, ResNet50) for feature extraction  
- Flatten and normalize feature maps

#### **5. Dataset Preparation**
- Split into train/test/validation  
- Apply augmentation only to training data  

#### **6. Model Training**
- Train CNN models (custom or pre-trained)  
- Evaluate with accuracy, precision, recall, and confusion matrix  

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
spacy
tensorflow
torch
albumentations
```

## ⭐ Contribute

If you find this useful:

* Star ⭐ the repo
* Fork and improve it
* Submit pull requests for new techniques (e.g., SMOTE, pipelines, feature drift handling)

---

## 📜 License

MIT License — Free to use and modify with attribution.
