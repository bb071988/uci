# setup the data for analysis random forest

import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier


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

	location = r'C:\Users\bob071988\thinkful\ds-new\UCI\test\X_test.txt'

	df = pd.read_table(location, header = None, delim_whitespace=True, names = col_list)
	return(df)


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
	with open(r'C:\Users\bob071988\thinkful\ds-new\UCI\test\y_test.txt','r+') as cat_file:
	    cat_list = []
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
	    #print(cat_list)
	    
	df['activity'] = cat_list # actual activity
	return(df)

def get_labels():
	with open(r'C:\Users\bob071988\thinkful\ds-new\UCI\new_labels.txt','r+') as new_label_file:
	    label_list = []
	    for line in new_label_file: # one line is one column
	        line = line.strip()
	        label_list.append(line)
	        #print(cat_list)
	    
		# df['label'] = label_list # computed from just the 5 values
	labels = pd.Series(label_list, index = ['1','2','3','4','5','6'])
	return(labels)

def set_actual(df):
	# get unique list of activities
	with open(r'C:\Users\bob071988\thinkful\ds-new\UCI\test\y_test.txt','r+') as cat_file:
	    num_list = []
	    for line in cat_file: # one line is one column
	        line = line.strip()
	        num_list.append(line)
	        #print(cat_list)
	    
	df['act_num'] = num_list # numerical representation of activity
	return(df)

def make_preds(df, labels):

	
	# df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
	df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
	# df['species'] = pd.Factor(iris.target, iris.target_names)
	# df['act2'] = pd.Categorical.from_codes(df.act_num, labels)

	train, test = df[df['is_train']==True], df[df['is_train']==False]
	features = df.columns[:4]
	clf = RandomForestClassifier(n_jobs=2)
	y, _ = pd.factorize(train['activity'])
	clf.fit(train[features], y)

	# preds = iris.target_names[clf.predict(test[features])
	# preds = labels[clf.predict(test[features])]

	# pd.crosstab(test['activity'], preds, rownames=['actual'], colnames=['preds'])


	# Kyle's code starts here
	preds = labels[clf.predict(test[features])] #ask kyle about features

	s1 = pd.Series(test['activity'])
	s2 = pd.Series(preds)
	s2.index = np.arange(len(s2)) # This index needs to be reset
	s1.index = np.arange(len(s1)) # This one doesn't have to be
	result_df = pd.concat([s1, s2], axis=1)
	result_df.columns = ['actual', 'predicted']

	cross = pd.crosstab(result_df.actual, result_df.predicted)
	return(cross)


def main():
    col_list = trans_columns()
    df = get_data(col_list)
    df = drop_columns(df)
    df = create_labels(df)
    labels = get_labels()
    df = set_actual(df)
    cross = make_preds(df, labels)

    print(cross)

  
    
if __name__ == "__main__":
    main()

"""Split the data into training, test, and validation sets."""