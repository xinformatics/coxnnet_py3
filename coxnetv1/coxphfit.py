from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
import pandas as pd
import numpy as np




data = pd.read_csv('/home/xinfy/Desktop/lasurv/project_BC_surv/coxnnet_py3mo/coxnetv1/ssBRCA/datacxph/x.csv')
ytime = pd.read_csv('/home/xinfy/Desktop/lasurv/project_BC_surv/coxnnet_py3mo/coxnetv1/ssBRCA/datacxph/ytime.csv')
ystatus = pd.read_csv('/home/xinfy/Desktop/lasurv/project_BC_surv/coxnnet_py3mo/coxnetv1/ssBRCA/datacxph/ystatus.csv')


data['time'] = ytime['time'].values
data['event'] = ystatus['status'].values



cph = CoxPHFitter(penalizer=0.05)
# #cph.fit(dataset, duration_col='time', event_col='event', show_progress=True)
scores = k_fold_cross_validation(cph, data, duration_col='time', event_col='event', k=10, scoring_method="concordance_index")

print(np.asarray(scores))

#print(np.asarray(scores).mean(),np.asarray(scores).std())