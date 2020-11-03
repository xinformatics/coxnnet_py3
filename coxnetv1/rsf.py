import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from tqdm import tqdm

data = pd.read_csv('/home/xinfy/Desktop/lasurv/project_BC_surv/coxnnet_py3mo/coxnetv1/ssBRCA/datacxph/x.csv')
ytime = pd.read_csv('/home/xinfy/Desktop/lasurv/project_BC_surv/coxnnet_py3mo/coxnetv1/ssBRCA/datacxph/ytime.csv')
ystatus = pd.read_csv('/home/xinfy/Desktop/lasurv/project_BC_surv/coxnnet_py3mo/coxnetv1/ssBRCA/datacxph/ystatus.csv')

y = pd.DataFrame()

y['time'] = ytime['time'].values
y['event'] = ystatus['status'].values

#structured array

y_struc = np.zeros(y.shape[0], dtype={'names':('vital_status','os_time'),'formats':('?','f8')})
y_struc['vital_status'] = y['event'].astype('bool')
y_struc['os_time'] = y['time']

X  = data.copy()

train_scores = []
test_scores = []
for i in tqdm(range(1,11)):
    random_state = i
    X_train, X_test, y_train, y_test = train_test_split(X, y_struc, test_size=0.2,random_state=random_state)
    rsf = RandomSurvivalForest(n_estimators=200,min_samples_split=20,min_samples_leaf=25,max_features="sqrt",n_jobs=-1,random_state=random_state)
    rsf.fit(X_train, y_train)
    train_scores.append(rsf.score(X_train, y_train))
    test_scores.append(rsf.score(X_test, y_test))


print('training',np.asarray(train_scores))
print('training_mean',np.asarray(train_scores).mean(),'training_stdev',np.asarray(train_scores).std())
print('test',np.asarray(test_scores))
print('test_mean',np.asarray(test_scores).mean(),'test_stdev',np.asarray(test_scores).std())