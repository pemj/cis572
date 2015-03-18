import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

print("Reading training data")
df = pd.read_csv('../data/train.csv')

def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180

## Add new columns for training data
print("Adding features")
df['EVDtH'] = df.Elevation-df.Vertical_Distance_To_Hydrology
df['EHDtH'] = df.Elevation-df.Horizontal_Distance_To_Hydrology*0.2
df['Highwater'] = df.Vertical_Distance_To_Hydrology < 0
df['Aspect2'] = df.Aspect.map(r)
df.ix[df.Hillshade_3pm==0, 'Hillshade_3pm'] = df.Hillshade_3pm.median()
one_two = df[(df['Cover_Type']<3)] 

features = [col for col in df.columns if col not in ['Cover_Type','Id','is_train']]

X_train = df[features]
y_training = df['Cover_Type']
x_one_two = one_two[features]
y_one_two = one_two['Cover_Type']

## Make GBM
'''
print("Making GBM")
rt = .36
md = 5
ne = 400
mf = None
gbm = GradientBoostingClassifier(learning_rate=rt, max_depth=md, n_estimators=ne, max_features=mf)
gbm.fit(X_train, y_training)

## Make Nearest Neighbor
print("Making random forest")
clf = RandomForestClassifier()
clf.fit(X_train, y_training)

print("Scaling data")
'''
scaler = StandardScaler()
scaled_train = pd.DataFrame(scaler.fit_transform(X_train.astype(np.float)), columns=features)
'''
scaled_train2 = scaled_train.copy()

feature_importances = []
for i in range(0, len(clf.feature_importances_)):
    feature_importances.append((clf.feature_importances_[i] + 0.005)*5)

col_index = 0
for col in features:
    scaled_train2[col] = scaled_train2[col]*feature_importances[col_index]
    col_index += 1

print("Making nearest neighbor")
neighbor = KNeighborsClassifier(n_neighbors=5) #weights="distance"
neighbor_fitted = neighbor.fit(scaled_train2, y_training)
'''
## Make SVM
print("Making SVM")

rbf_svc = SVC(C=1500, kernel="rbf", max_iter=-1) #5.5, 200
svm_fitted = rbf_svc.fit(scaled_train, y_training)

## Make random forest for 1 and 2 differentiation
clf2 = RandomForestClassifier().fit(x_one_two, y_one_two)

## Get test data
print("Reading test data")
with open('output.csv', "w") as outfile:
	outfile.write("Id,Cover_Type\n")
	row_num = 1
	count = 25000
	columns = []
	while(1):
		## Read segment of data
		test = False
		if(len(columns) == 0):
			test = pd.read_csv('../data/test.csv', nrows=count)
			columns = test.columns
		else:
			test = pd.read_csv('../data/test.csv', skiprows=row_num, nrows=count, names=columns)
		print("Row #" + str(row_num) + ", count: "+str(len(test.index))+", first ID: "+str(test.ix[0, 0]))
		row_num+=count
		if(len(test.index) == 0):
			break
		test_ids = test['Id']

		## Add new columns for test data
		test['EVDtH'] = test.Elevation-test.Vertical_Distance_To_Hydrology
		test['EHDtH'] = test.Elevation-test.Horizontal_Distance_To_Hydrology*0.2
		test['Highwater'] = test.Vertical_Distance_To_Hydrology < 0
		test['Aspect2'] = test.Aspect.map(r)
		test.ix[test.Hillshade_3pm==0, 'Hillshade_3pm'] = test.Hillshade_3pm.median()
		X_test = test[features]
		scaled_test = pd.DataFrame(scaler.transform(X_test.astype(np.float)), columns=features)
		'''
		scaled_test2 = scaled_test.copy()
		col_index = 0
		for col in features:
			scaled_test2[col] = scaled_test2[col]*feature_importances[col_index]
			col_index += 1
		'''

		## Make predictions
		#gbm_predict = gbm.predict(X_test)
		#nn_predict = neighbor_fitted.predict(scaled_test2)
		svm_predict = svm_fitted.predict(scaled_test)
		#ot_predict = clf2.predict(X_test)

		## Combine
		voted_predict = []
		for i in range(0,len(svm_predict)):
			'''
			a = np.array([gbm_predict[i], svm_predict[i], nn_predict[i]])
			(values,counts) = np.unique(a,return_counts=True)
			ind=np.argmax(counts)
			ret = values[ind]
			if ret==1 or ret==2:
				ret = ot_predict[i]
			'''
			voted_predict.append(svm_predict[i])

		## Write to output
		for e, val in enumerate(voted_predict):
			outfile.write(str(test_ids[e])+","+str(val)+"\n")
		if(len(test) < count):
			break

