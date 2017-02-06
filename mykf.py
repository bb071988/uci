# setup the data for analysis random forest

import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import KFold
from sklearn.crossval import KFold

cwd = os.getcwd() # global variable ick

def fix_col_name(col_split):
        new_string = col_split[1].replace('(','')
        new_string = new_string.replace(')','')
        new_string = new_string.replace(',','-')
        new_string = new_string.replace('BodyBody','Body')
        new_string = new_string.replace('Body','')
        new_string = new_string.replace('Mag','')
        new_string = new_string.replace('mean','Mean')
        new_string = new_string.replace('std','STD')
        return(new_string)
  
def trans_columns():
	###  get's column names from feature_file should be same for train and test data.  Can use same col_list
	col_list =[]
	x = 0
	with open('features.txt','r+') as feature_file:
	    for line in feature_file: # one line is one column
	#         print(line)
	        col_split = line.split() # splits the numbers and characters from the column into seperate fields to work with
	#         print(col_split)
	        new_col = fix_col_name(col_split)
	#         print(new_col)
	        col_list.append(new_col + col_split[0])
	#     col_list = remove_dups(col_list)
	#         break # stop at one record
	return(col_list)


def get_data(col_list):
	# reads file into df.  Modify to pass location for train and test.
	test_file = os.path.normpath(cwd + '/test/X_test.txt')
	dftest = pd.read_table(test_file, header = None, delim_whitespace=True, names = col_list)

	train_file = os.path.normpath(cwd + '/train/X_train.txt')
	dftrain = pd.read_table(train_file, header = None, delim_whitespace=True, names = col_list)
	
	# # df.append(df2)
	df = dftest.append(dftrain)
	
	return(df)
	# return(dftest)


def drop_columns(df):
	drop_list = ['angle','band']
	for col in df.columns.values:
	    for item in drop_list:
	        test = col.find(item) # returns -1 if item not found
	        if test == -1:
	            pass
	            #print("keeping these = find value {} --- finding {} column value {}".format(test,item, col))
	            
	        else:
	            #print("droping these = find value {} --- finding {} column value {}".format(test,item, col))
	            df.drop(col, axis = 1, inplace = True)
	#             new_cols.remove(col) # useful to keep track of new list of columns in new df - could just use df.column.values though
	            break # if its in the drop list drop and stop looking
	return(df)


def create_labels(df):
	file_list = ['/test/y_test.txt','/train/y_train.txt']
	# file_list = ['/test/y_test.txt']
	cat_list = []
	act_num = []
	for file in file_list:
		in_file = os.path.normpath(cwd + file)
		with open(in_file ,'r+') as cat_file:  
			for line in cat_file: # one line is one column
				line = line.strip()
				if '1' in line:
				    activity = 'WALKING'
				elif '2' in line:
				    activity  = 'WALKING_UPSTAIRS'
				elif '3' in line:
				    activity  = 'WALKING_DOWNSTAIRS'
				elif '4' in line:
				    activity  = 'SITTING'
				elif '5' in line:
				    activity  = 'STANDING'
				elif '6' in line:
				    activity  = 'LAYING' 
				else:
				    print("error here")
				    
				cat_list.append(activity)
				act_num.append(line)
	df['activity'] = cat_list # actual activity
	df['act_num'] = act_num
	return(df)

def get_labels():
	label_list = []
	
	with open('new_labels.txt','r+') as new_label_file:
	    
	    for line in new_label_file: # one line is one column
	        line = line.strip()
	        label_list.append(line)
	        #print(cat_list)
	labels = pd.Series(label_list)

	return(labels)

def set_actual(df): # ************ why wasn't train in here too?
	# get unique list of activities
	file = os.path.normpath(cwd + '/test/y_test.txt')
	with open(file ,'r+') as cat_file:
	    num_list = []
	    for line in cat_file: # one line is one column
	        line = line.strip()
	        num_list.append(line)
	        #print(cat_list)
	    
	df['act_num'] = num_list # numerical representation of activity
	return(df)
	
# def make_preds(df, labels):
# 	df = df.drop('activity', axis=1) # axis 1 denotes a column not a row
# 	df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
# 	train, test = df[df['is_train']==True], df[df['is_train']==False]
# # 	print(test.shape, train.shape)
# 	i = np.arange(0,427)
# 	features = df.columns[i] # changed from :4 - many features may be a little bit important
# 	clf = RandomForestClassifier(n_jobs=2,n_estimators=5) # change estimators to 500 for exercise
# 	clf.fit(train[features], train['act_num'])

# 	# Kyle's code starts here
# 	preds = clf.predict(test[features]) #ask kyle about features
# 	s1 = test['act_num']
# 	s2 = pd.Series(preds)
# 	s2.index = np.arange(len(s2)) # This index needs to be reset
# 	s1.index = np.arange(len(s1)) # This one doesn't have to be
# 	result_df = pd.concat([s1, s2], axis=1)
# # 	print(result_df.shape)
# 	result_df.columns = ['actual', 'predicted']
# 	# print(result_df.head())
# 	cross = pd.crosstab(result_df.actual, result_df.predicted)
# 	feature_imp = clf.feature_importances_
# 	return(cross, clf , df, result_df)

def make_preds(df, labels):
	pred_list = []
	df = df.drop('activity', axis=1) # axis 1 denotes a column not a row
	# df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
	# train, test = df[df['is_train']==True], df[df['is_train']==False]
	
	kf = KFold(n_splits=3)
	# train, test = kf.split(df)
	for train, test in kf.split(df):
	# 	print(test.shape, train.shape)
		i = np.arange(0,427)
		features = df.columns[i] # changed from :4 - many features may be a little bit important
		clf = RandomForestClassifier(n_jobs=2,n_estimators=5) # change estimators to 500 for exercise
		clf.fit(train[features], train['act_num'])
	
		# Kyle's code starts here
		preds = clf.predict(test[features]) #ask kyle about features
		s1 = test['act_num']
		s2 = pd.Series(preds)
		s2.index = np.arange(len(s2)) # This index needs to be reset
		s1.index = np.arange(len(s1)) # This one doesn't have to be
		result_df = pd.concat([s1, s2], axis=1)
	# 	print(result_df.shape)
		result_df.columns = ['actual', 'predicted']
		# print(result_df.head())
		cross = pd.crosstab(result_df.actual, result_df.predicted)
		# feature_imp = clf.feature_importances_
		pred_list.append([cross, clf, df, result_df])
		
	return(pred_list)


def crosstab():
	
	# get the columns and clean them up
	col_list = trans_columns()
	
	# concatenate the test and train data sets
	df = get_data(col_list)
	df = drop_columns(df)
	df = create_labels(df)
	labels = get_labels()
	pred_list = make_preds(df, labels)

	return(pred_list)


def main():
	cross, clf, df, result_df = crosstab()
	print('done done done')
	# print(c)
	# print(feature_imp)
    
if __name__ == "__main__":
    main()

"""Split the data into training, test, and validation sets."""