# Adaptive Anomaly Detection and Prediction using Machine Learning in 5G-enabled IoT Systems

## Overview

This repository contains the source code, implementation files, and supporting materials developed for the master's thesis:

**Adaptive Anomaly Detection and Prediction using Machine Learning in 5G-enabled IoT Systems**

The project investigates anomaly detection in 5G-enabled IoT environments using machine learning and hybrid learning approaches. Multiple classification levels were implemented and evaluated to improve attack detection and analysis.

The implementation includes:

* Static MLP models
* Adaptive MLP models
* Static Hybrid (VAE + XGBoost) models
* Adaptive Hybrid (VAE + XGBoost) models

The experiments were performed using the 5G-NIDD dataset and evaluated across three classification levels:

1. Binary Classification (Benign vs Attack)
2. Attack Category Classification
3. Attack Type Classification

Each model folder contains:

* Source code
* Implementation notebooks
* Individual README file
* Model-specific execution instructions

## Dataset

The dataset is not included in this submission due to file size limitations. It can be downloades from 'https://www.kaggle.com/datasets/humera11/5g-nidd-dataset'

Dataset used:
**5G-NIDD Dataset**

Please place the dataset file (`Combined.csv`) in the required location before executing the notebooks.

## Environment

Recommended environment:

* Python 3.x
* Google Colab
* TensorFlow
* Scikit-learn
* XGBoost
* Pandas
* NumPy

## How to Use

Open the corresponding model folder and follow the instructions provided in its README file.




