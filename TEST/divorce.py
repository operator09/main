import os
import numpy as np
import tensorflow as tf
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = pd.read_csv('C:\\Users\\82104\\PycharmProjects\\pythonProject\\TEST\\data\\divorce_data.csv')
data = np.array(data)

print(data)