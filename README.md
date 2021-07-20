
# Predictive Maintenance Machine Learning

This is my capstone project in the [Machine Learning Engineer](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) and will serve as a demonstration of end-to-end machine learning.

The problem I want to explore is prediction of machine failure from maintenance data.
From the UCI Machine Learning Repository I have chosen to work with the [AI4I 2020 Predictive Maintenance Dataset Data Set](http://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset).
The exploratory data analysis, feature extraction and modeling will be documented in Jupyter notebooks.
All machine learning training will be done on [AWS SageMaker](https://aws.amazon.com/sagemaker/)

The final analysis and insights will be documented in a report.

1. [Exploratory Data Analysis](exploratory-data-analysis.ipynb) - Notebook that explore data set and draws plots and some simple summary statistics.
2. [Feature Engineering](feature_engineering.ipynb) - Notebook that selects features, upsamples using SMOTE and adjust ranges with a Min/Max scalar.
3. [Linear learner baseline](linear-learner-baseline.ipynb) - Notebook trained with AWS SageMaker [Linear Learner Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html)
4. [PyTorch Training Model](pytorch-training-model.ipynb) - Notebook that trains and evaluates simple neural network model.   
    1. [pytorch_model_def.py](pytorch_script/pytorch_model_def.py) - The neural network model.
    2. [train_deploy_pytorch_without_dependencies.py](pytorch_script/train_deploy_pytorch_without_dependencies.py) - SageMaker scripts for training model and inference


[Machine Learning Capstone Proposal](docs/Marcus Holmgren - Machine Learning Capstone Proposal.pdf)

## Tech Stack

Python, NumPy, Pandas, Matplotlib, Seaborn, Jupyter, PyTorch, AWS SageMaker, Imbalanced


## Imbalanced dataset

The data set is highly imbalanced where the feature Machine failure consists of 9661 (0.9661) false values and 339 (0.0339) failures according to the five failure modes. 

I choose to use the [Imbalanced-learn](https://imbalanced-learn.org/stable/) library that provides tools for dealing with imbalanced classes.


## References

* [The 5 Most Useful Techniques to Handle Imbalanced Datasets](https://www.kdnuggets.com/2020/01/5-most-useful-techniques-handle-imbalanced-datasets.html)
