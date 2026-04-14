# Crop Analysis and Prediction

A Streamlit-based AI dashboard for crop recommendation and machine learning pipeline management.

## Features

- **Crop Prediction**: Recommend optimal crops based on soil and weather conditions (N, P, K, Temperature, Humidity, pH, Rainfall)
- **Data Upload**: Import custom CSV datasets for analysis
- **Exploratory Data Analysis**: Summary statistics and correlation heatmaps
- **Data Cleaning**: Handle biologically impossible values and remove outliers (IQR method)
- **Feature Selection**: Variance Threshold and Information Gain methods
- **Model Training**: Logistic Regression and Random Forest classifiers with K-Fold cross-validation
- **Performance Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Feature Importance

## Quick Start

1. Install dependencies:
```bash
pip install streamlit pickle-mixin numpy pandas seaborn matplotlib scikit-learn
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open browser at `http://localhost:8501`

## Usage

1. **Upload Dataset**: Use sidebar to upload a CSV file
2. **Explore Data**: View EDA tab for dataset statistics and correlation
3. **Clean Data**: Handle zeros and remove outliers
4. **Select Features**: Choose feature selection method
5. **Train Model**: Select model type and configure test size
6. **View Performance**: Check accuracy and metrics
7. **Predict Crop**: Use sidebar inputs to get crop recommendations

## Requirements

- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib

## Model

Pre-trained Random Forest classifier (~97% accuracy) for crop prediction.