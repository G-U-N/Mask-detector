'''

使用集成算法实现口罩检测

libraries: sklearn, xgboost

'''

import numpy as np  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.ensemble import AdaBoostClassifier  
import matplotlib.pyplot as plt  
import matplotlib as mpl  
from sklearn import datasets  
import xgboost as xgb  
from sklearn.model_selection import train_test_split


