[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

![python: 3.7](https://img.shields.io/badge/python-3.7-blue)

This project will not be developed anymore since October 2022. You can still utilize it for any other projects and to comprehend the paper. For further questions do not hesitate to contact me.

# Risk-Measurement-Framework
For my Masterthesis I develop a framework that measure risks which is called RMF. This framework should measure risks based on ISO 27004. The main goal of this thesis is to measure the extent of damage and the attacker's effort based on risk indicators such as the accuracy, TP, TN, FP, FN, attack time, attacker specificity, attacker's knowledge, attacker's goal, and the computational resources.

The case study is a NN trained for image classification to simulate traffic sign detection for autonomous-driving. The dataset will be manipulated by different backdoor attacks.

# Installation
- The core framework is the [Adversarial-Robustness-Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) (ART).  
- To use ART as a module in your project, you have to install `pip install adversarial-robustness-toolbox`
- For my ML model I use `keras` because the backdoor attacks in the ART only work with NN
- Another important point is that you need `pip install numba==0.54` and `pip install numpy==1.19.5`
- You also need `pip install opencv-python` and `seaborn`

# Abstract

This thesis deals with the field of machine learning, especially security. For this
purpose, the concept of a technical framework will be described, implemented
and evaluated in a case study. Based on ISO 27004, risk indicators are used
to measure and evaluate the extent of damage and the effort required by an
attacker. It is not possible, as assumed, to determine a risk value that represents
the extent of the damage and the effort. Therefore, four different risk matrices
will represent different values and must be interpreted individually.

# Dataset

For my case study I'm using a german traffic sign dataset that is already splittet in training and testing [Kaggle dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/version/1). The dataset is not committed but has no changes from the original dataset from Kaggle.
It uses images from 43 different traffic signs.
