# setup the data for analysis random forest

import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

cwd = os.getcwd() # global variable ick


def crosstab():
	
	# get the columns and clean them up
	col_list = trans_columns()
	
	# concatenate the test and train data sets
	df = get_data(col_list)
	df = drop_columns(df)
	df = create_labels(df)
	labels = get_labels()
	cross, clf, df, result_df = make_preds(df, labels)

	return(cross, clf, df, result_df)


def main():
	cross, clf, df, result_df = crosstab()
	print('done done done')
	# print(c)
	# print(feature_imp)
    
if __name__ == "__main__":
    main()

"""Split the data into training, test, and validation sets."""