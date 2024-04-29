# Socface Project Repository

This repository contains the code for the Socface project, which focuses on predicting gender based on automatic handwriting recognition outputs of French census data from 1836 to 1936. The project aims to digitize and analyze historic census records to create a publicly accessible database, which will facilitate social change studies over a century.

## Project Overview

The Socface project is a collaboration between archivists, demographers, and computer scientists to analyze and digitize handwritten French census data. Using machine learning models, the project performs a binary classification task to predict individual sexes based on transcription outputs from an automatic handwriting recognition engine. A combination of data sparsity, transcription model inaccuracies, and the lack of ground truth makes this task challenging.

## Data Analysis and Engineering

The dataset contains 232 samples, with features derived from census transcriptions. It includes personal names, ages, birth dates, occupations, and relationships. Additionally, a corpus of French first names is used to infer gender based on name frequency. 

## Repository Structure

- `bert_model.py`: Implements the BERT model from Hugging Face for gender prediction.
- `lstm_misspelling.py`: Trains a BiLSTM model to correct misspelled first names.
- `main.py`: Script to launch all models.
- `modelling.py`: Handles data treatment and processing for multiple models.
- `models.py`: Contains scripts for different predictive models.
- `params.py`: Stores fixed parameters for the models.
- `plots.ipynb`: Jupyter notebook for generating plots for visualization and reporting.
- `utils.py`: Includes utility functions used across the project.

## Model Architecture Schema

Please refer to the repository's model architecture schema included in this document for details on the structure and parameters of the models used in this project.

## Data Preprocessing

Data preprocessing involves adding a 'name sex' feature, correcting transcription errors using a BiLSTM model, Python's difflib, or FuzzyWuzzy package, and creating proper string sequences for training. Stop words are treated using the nltk package in Python.

## Binary Classification Models

A range of models are evaluated, including Logistic Regression, Naive Bayes Bernoulli, K-Nearest Neighbors, XGBOOST
