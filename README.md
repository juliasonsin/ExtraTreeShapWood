# ExtraTreeShapWood
# Wood Anatomy Analysis: Extra-Trees Regressor & SHAP

**Authors:** Julia Sonsin-Oliveira and Deborah Bambil

## Overview
This repository provides a Python script designed to analyze the complex, non-linear relationships between wood anatomical features and macroscopic physical properties. 

This tool was specifically developed to handle the inherent challenges of functional anatomical data. By utilizing an `ExtraTreesRegressor` algorithm coupled with Leave-One-Out Cross-Validation (LOOCV), it predicts physical properties based on anatomical features[cite: 3]. Furthermore, it integrates the `shap` library to calculate SHAP (SHapley Additive exPlanations) values, generating Summary and Bar plots that rank the true impact of each anatomical feature on the model's output[cite: 3]. 

To handle the unit sum constraint of compositional data (e.g., tissue proportions), the script automatically applies the Centred Log-Ratio (CLR) transformation before training the model[cite: 3].

## Dependencies
The script is equipped with an automatic dependency installer[cite: 3]. Upon the first run, it will automatically check for and install the required libraries. The primary dependencies are:
* `pandas` & `numpy` (Data manipulation)[cite: 3]
* `scikit-learn` (Standardization and Extra-Trees model)[cite: 3]
* `scikit-bio` (CLR transformation for compositional data)[cite: 3]
* `shap` (Model interpretability and feature importance)[cite: 3]
* `matplotlib` & `seaborn` (Data visualization)[cite: 3]

## How to Use

1. **Prepare your data:** Ensure your dataset is in `.csv` or `.txt` format. Variables should be in columns and samples (species/specimens) in rows. The target variable (e.g., Wood Density) must be in the last column[cite: 3].
2. **Run the script:** Execute the Python file (`.py`) in your terminal or IDE.
3. **Follow the prompts:** The script features a Universal File Reader. Simply type the name of your file (e.g., `my_data.csv`) when prompted[cite: 3].
4. **Retrieve Results:** The script will automatically generate high-resolution SHAP graphs (`.png`), a detailed textual report (`.txt`) with R² and MSE metrics, and a formatted table (`.csv`) with exact SHAP values in the same directory[cite: 3].

## License
This project is open-source and available for the scientific community. If you utilize this script in your research, please cite the associated publication.
