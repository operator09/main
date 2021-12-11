import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import graphviz
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree

df = pd.read_csv("C:\\Users\\operator09\\PycharmProjects\\Main\\TEST\\data\\Breast Cancer Wisconsin (Diagnostic) Data Set.csv")
df = df.drop(['Unnamed: 32', 'id'], axis=1)

df['daignosis'] = df['diagnosis'].map({'M': 1, 'B': 2})
Y = df['diagnosis']
X = df.drop('diagnosis', axis=1)
"""
M = df[df.diagnosis == "M"]
B = df[df.diagnosis == "B"]

plt.scatter(M.radius_mean,M.texture_mean,color='red',label='Malign',alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color='green',label='Benign',alpha=0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

fig, axes = plt.subplots(1,2, figsize=(15,5))
sns.boxplot(ax = axes[0], x=df.diagnosis, y=df['area_mean'],palette='turbo')
axes[0].set_title("Size difference")

sns.boxplot(ax = axes[1],x=df.diagnosis, y=df['perimeter_mean'],palette="PRGn")
axes[1].set_title("Size Difference")

plt.show()
"""
