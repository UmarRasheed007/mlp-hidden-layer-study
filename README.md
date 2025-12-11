# Understanding How Hidden Layer Width Influences the Performance of a Multilayer Perceptron (MLP)

## Overview
This repository contains all code, experiments, visualisations, and written materials used to analyse how the width of a hidden layer affects the performance and generalisation behaviour of a Multilayer Perceptron (MLP).  
The project systematically evaluates a range of hidden-layer widths and examines their impact on accuracy, loss dynamics, training stability, and overfitting.

This repository is structured to support reproducibility, transparency, and ease of use for anyone interested in neural-network architecture analysis.

## Key Features
- Full Python implementation for training MLP models with varying widths  
- Automated generation of accuracy curves, loss curves, dataset summary, and final performance figures  
- Multi-page PDF report created programmatically  
- Synthetic dataset with controlled complexity for consistent experimentation  
- Table of contents and written tutorial (DOCX and PDF versions)  
- Easy-to-run code with minimal dependencies  

## Objective
The central goal of this project is to investigate how model capacity controlled here through hidden-layer width affects learning outcomes such as underfitting, overfitting, convergence behaviour, model expressiveness, generalisation capability, and computational cost.

## Experiment Description
All models are trained on a synthetic multi-class dataset generated using sklearn.make_classification.

MLPs were trained with hidden-layer widths:
4, 8, 16, 32, 64, 128

Common parameters:
- Activation: ReLU  
- Optimiser: Adam  
- Learning rate: 0.001  
- Epochs: 40  

## Results Summary
- Small widths (4–16): Underfitting  
- Medium widths (32–64): Best performance  
- Large widths (128+): Overfitting  

## How to Run the Code
```pip install numpy scikit-learn matplotlib```

```python script.py```

Outputs will be saved in mlp_width_outputs/.

## Licence
This project is released under the MIT Licence.

## Author
Umar Rasheed

