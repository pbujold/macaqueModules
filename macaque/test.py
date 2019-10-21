# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 21:20:52 2019

@author: phbuj
"""
from macaque.f_models import get_modelLL, trials_2fittable, LL_fit
from macaque.f_probabilityDistortion import plot_MLEfitting
np.warnings.filterwarnings('ignore')

uniqueDays = np.unique(np.hstack((perDaySM.sessionDate.values, 
                                  fractile.sessionDate.values)))

dList = []; pastEffect = []
for date in tqdm(np.sort(uniqueDays), desc=Model):
    tt = Trials.loc[Trials['sessionDate'] == date]
    to_delete = tt.loc[tt.outcomesCount.apply(lambda x: any(np.array(x)>2))].blockNo.unique() 
    tt.drop(tt.loc[np.isin(tt.blockNo.values, to_delete)], inplace=True)
    
    if 'rl' in Model.lower():
        use_Outcomes = True
    X, Y = trials_2fittable(tt, use_Outcomes = use_Outcomes)
    if len(X) < minTrials:
        continue
    MLE = LL_fit(Y, X, model = Model).fit(disp=False)
    #this is wrong! it doesn't necessarily go to zero!
    mag_range = np.array([ min(np.concatenate(np.array(X)[:,[0,4]])),
                          max(np.concatenate(np.array(X)[:,[0,4]])) ])
    dList.append({
        'date':date,
        'nTrials': MLE.nobs,
        'params': MLE.params,
        'pvalues': MLE.pvalues,
        'NM_success': MLE.mle_retvals['converged'],
        'model_used': MLE.model.model_name,
        'LL': MLE.llf,
        'pNames': MLE.model.exog_names,
        'Nfeval': MLE.Nfeval,
        'all_fits': MLE.res_x,
        'full_model': MLE,
        'AIC': MLE.aic,
        'BIC': MLE.bic,
        'trial_index' : tt.loc[tt['sessionDate'] == date].index,
        'mag_range' : mag_range,
        'trials' : [tt.loc[tt['sessionDate'] == date].index]
    })

MLE_fits = MLE_object(pd.DataFrame(dList))

#%%

tt
