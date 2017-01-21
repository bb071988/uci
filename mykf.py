# setup the data for analysis random forest

import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.preprocessing import OneHotEncoder

cwd = os.getcwd() # global variable ick
# cwd = cwd.replace('/','\\')
# print(cwd)


# clean up column names


    
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
	
	# train_file = os.path.normpath(cwd + '/train/X_train.txt')
	
	# dftrain = pd.read_table(train_file, header = None, delim_whitespace=True, names = col_list)
	
	# # df.append(df2)
	# df = dftest.append(dftrain)
	
	# return(df)
	return(dftest)


def drop_columns(df):
	drop_list = ['angle','band']
	for col in df.columns.values:
	    for item in drop_list:
	        test = col.find(item) # returns -1 if item not found
	        #print("col is {} and item is {} and test is {}".format(col,item, test))
	#         print("test val = {}".format(test))
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
	# file_list = ['/test/y_test.txt','/train/y_train.txt']
	file_list = ['/test/y_test.txt']
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
	        
	  
	    #print(cat_list)
	# print('Length of cat_list is {}'.format(len(cat_list)))	
	# print('Shape of df is {}'.format(df.shape))
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
	    
		# df['label'] = label_list # computed from just the 5 values
	# labels = pd.Series(label_list, index = ['1','2','3','4','5','6']) original modified next line
	labels = pd.Series(label_list)
	# df1.loc[:,'f'] = p.Series(np.random.randn(sLength), index=df1.index)
	# df.loc[:,'label'] = labels
	
	# df['label']=labels
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
	
# def pre_process_labels(df):
# 	rf_enc = OneHotEncoder()
# 	# Supervised transformation based on random forests
# # rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
# # rf_enc = OneHotEncoder()
# # rf_lm = LogisticRegression()
# # rf.fit(X_train, y_train)
# # rf_enc.fit(rf.apply(X_train))
# # rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
	
# 	pass

# 	return(df)

def make_preds(df, labels):
	df = df.drop('activity', axis=1) # axis 1 denotes a column not a row

	df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

	train, test = df[df['is_train']==True], df[df['is_train']==False]
# 	print(test.shape, train.shape)
	features = df.columns[:] # changed from :4 - many features may be a little bit important
	
	clf = RandomForestClassifier(n_jobs=2)

	clf.fit(train[features], train['act_num'])

	# Kyle's code starts here
	preds = clf.predict(test[features]) #ask kyle about features
	
	s1 = test['act_num']
	# s2 = pd.Series(preds)

	# s1 = pd.Series(test['activity'])
	s2 = pd.Series(preds)
	s2.index = np.arange(len(s2)) # This index needs to be reset
	s1.index = np.arange(len(s1)) # This one doesn't have to be
	result_df = pd.concat([s1, s2], axis=1)
# 	print(result_df.shape)
	result_df.columns = ['actual', 'predicted']
	# print(result_df.head())
	cross = pd.crosstab(result_df.actual, result_df.predicted)
	return(cross)


def crosstab():
	
	# get the columns and clean them up
	col_list = trans_columns()
	
	# concatenate the test and train data sets
	df = get_data(col_list)
	df = drop_columns(df)
	df = create_labels(df)
	labels = get_labels()
	# df = set_actual(df) # consolidated into create_labels
	# df = pre_process_labels(df)
	cross = make_preds(df, labels)

	return(cross)


def main():
	c = crosstab()
	print(c)
    
if __name__ == "__main__":
    main()

"""Split the data into training, test, and validation sets."""