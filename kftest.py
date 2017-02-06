#kftest.py
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=3)
for train, test in kf.split(X):
    print("%s %s" % (train, test))
print(type(train))