import pandas as pd 
from sklearn import preprocessing
import numpy as np

le = preprocessing.LabelEncoder()

data = pd.read_csv('YB_balanced_data.csv',low_memory=False)
columns = list(data.columns)
#print("number of features:", len(columns))
object_columns = list(data.select_dtypes(include='object').columns)

#print('object columns name:')

#print(object_columns)

index_list = [data.columns.get_loc(c) for c in object_columns]

#print('object columns:')

#print(index_list)

newdataset = {}

for c in columns:
	if c in object_columns:
		le.fit(np.array(data[c].tolist()).reshape(len(data[c].tolist()),1))
		tmp = le.transform(np.array(data[c].tolist()).reshape(len(data[c].tolist()),1))
		newdataset[c] = tmp
	else:
		newdataset[c] = data[c]


newdataset = pd.DataFrame.from_dict(newdataset)
newdataset.to_csv('YBG_data3.csv')
print(newdataset.shape)

object_columns = list(newdataset.select_dtypes(include='object').columns)
print(object_columns)