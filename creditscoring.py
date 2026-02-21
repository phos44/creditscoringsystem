import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import math
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')
df_test = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')
df_train.head()