import pandas as pd

data = pd.read_csv(r"C:\Users\phbuj\Google Drive\Github\PhD_Analysis\DePetrillo.csv")

#%%
data['Trial_type'].loc[data.Trial_type.apply(lambda x: x=='neutral')] = 0.5
data['Trial_type'].loc[data.Trial_type.apply(lambda x: x=='dis')] = 0.333
data['Trial_type'].loc[data.Trial_type.apply(lambda x: x=='adv')] = 0.666

#%% Remake into new table of X and Y based on food and presentation

selector = 'Subject'
# can also be age, sex, cigarette, BMI

splitters = unique_listOfLists(data[['Condition','Reward_type']].values)

food_3d = []; food_2d = [];
money_3d = []; money_2d = [];

#doing in this way in case we need subject-specific analysis
#otherwise would loop on the [Condition, Reward_type]

subTable = []
for n,condition in enumerate(splitters):
    dList = []
    for i in data[selector].unique():
        df = data.loc[data[selector] == i]
        if all(df[['Condition','Reward_type']].values[-1] == condition):
            dList.append(df)
    dList = pd.concat(dList)   
    dList['task'] = np.sum(dList[['Condition','Reward_type']].values[-1])
    subTable.append(dList)    


#%%


import scipy.optimize as opt
from macaque.f_models import trials_2fittable, LL_fit
from scipy.odr import ODR, Model, Data, RealData

to_convert = subTable[0]

dList = []
for to_convert in subTable:
    Y = 1-to_convert['Choice'].values.T
    safe = np.array([0.40] * len(to_convert))
    null = np.array([0] * len(to_convert))
    safe = np.vstack((safe,  np.array([1] * len(to_convert))))
    safe = np.vstack((safe, null, null))
    
    gamble = np.array([0.10] * len(to_convert))
    gamble = np.vstack((gamble, 1- to_convert['Trial_type'].values))
    gamble = np.vstack((gamble, np.array([0.70] * len(to_convert))))
    gamble = np.vstack((gamble, to_convert['Trial_type'].values))
    
    X = np.vstack((safe, gamble)).T
    X = pd.DataFrame(X, columns = [['A_m1', 'A_p1', 'A_m2', 'A_p2', 'B_m1', 'B_p1', 'B_m2', 'B_p2']])
    X = X.astype(float)
    
    MLE = LL_fit(Y, X.astype(float), model = '1logit-2scdf-1prelec', fixedRange=False).fit(disp=False)
    
    dList.append({
                    'condition':to_convert.task.iloc[-1],
                    'nTrials': MLE.nobs,
                    'params': MLE.params,
                    'pvalues': MLE.pvalues,
                    'NM_success': MLE.mle_retvals['converged'],
                    'model_used': MLE.model.model_name,
                    'LL': MLE.llf,
                    'pNames': MLE.model.exog_names,
    #                'Nfeval': MLE.Nfeval,
                    'all_fits': MLE.res_x,
                    'full_model': MLE,
                    'AIC': MLE.aic,
                    'BIC': MLE.bic,
                })
    
    results = pd.DataFrame(dList)
    results =     results[['condition',  'nTrials','model_used', 'pNames', 
                           'params', 'pvalues', 'AIC', 'BIC', 'LL', 
                           'NM_success', 'full_model', 'all_fits']]
    
    if plotit: 
        for n in range(len(results)):
            results.iloc[n].full_model.plot_fullModel()