# GANIB
Graph synergistic and statistics attention networks with information bottleneck for complex disease-related metabolite prediction


## Overview
![Fig  1](https://github.com/user-attachments/assets/989836a3-bf6d-4856-bc1a-20bd6a15c91b)


## Dependecies
```
- conda=24.4.0
- Python == 3.12
- pytorch == 2.3.0+cu121
- torch_geometric == 2.5.3
- torch_sparse == 0.6.18
- numpy == 1.26.4
- pandas == 2.2.2
- scikit-learn == 1.5.0
- scipy == 1.13.1
- matplotlib == 3.9.0
```


## 🎯 How to run?
```
1. The data1 and data2 folders store the association networks, disease and metabolite networks, and initial characterization data for datasets 1 and 2, respectively.
2. The code folder for implementing the GANIB model, which specifically includes:
  (1) train.py is used to start the GANIB model and set up parameters, implement training and validation, loss function definition, optimizer selection and parameter update.
  (2) model.py is used to build the overall structure of the GANIB model.
  (3) layers.py mainly stores some customized network layers.
  (4) utils.py mainly realizes data loading, evaluation index calculation and plot, etc.
```
