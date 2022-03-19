**Using Python 3.7.***

# Risk-Measurement-Framework
For my Masterthesis program a risk measurement framework called RMF. This framework should meausre risks with the common classic IT security management standard ISO 27000 (family). Different backdoor attacks and measuring the attackers effort by using a threat model is the main goal of this thesis.

The case study is a SVM trained for image classification to simulate traffic sign detection for autonomous-driving. The dataset will be manipulated by different backdoor attacks.

# Installation 💻
- Core Framework is [Adversarial-Robustness-Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox).  
- To use ART as a module in your project, you have to install `pip install adversarial-robustness-toolbox`
- For my ML model I use `sklearn` but you can use whatever you want
- Another important point is that you need `pip install numba==0.54` and `pip install numpy==1.19.5`
- You also need `pip install opencv-python`

# Abstract 🖼️

...

# How to use the framework 🚀

This framework can be used by build in functions into a ML model.

# Dataset 🛑

For my case study I'm using a german traffic sign dataset that is already splittet in training and testing [Kaggle dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/version/1). The dataset is not committed but has no changes from the original dataset from Kaggle.
