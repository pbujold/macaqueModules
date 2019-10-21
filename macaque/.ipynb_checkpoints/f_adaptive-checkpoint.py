# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:45:57 2019

@author: phbuj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
plt.style.use('seaborn-paper')
plt.rcParams['svg.fonttype'] = 'none'
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#plt.rcParams['font.family'] = 'sans-serif'
from macaque.f_toolbox import *
tqdm = ipynb_tqdm()

#%%

def get_avgAdapt(adapt_MLE):
    '''
    '''

#    params[:,-1] = np.round(params[:,-1]  / np.finfo('double').eps, 0)
        
    fig, ax = plt.subplots( 2, len(adapt_MLE), squeeze = False,
                           figsize=( int(np.ceil(len(adapt_MLE)/3))*10 , 6 ))
    
    medians = []
    for i, mle in enumerate(adapt_MLE):
        params = np.vstack(mle.params)
    
        parameters = np.vstack(params).T
        [ ax[0,i].scatter(jitter([n]*len(pp), 0.1), pp, alpha = 0.5) for n,pp in enumerate(parameters) ] 
        ax[0,i].set_xticks(np.arange(len(parameters)))
        ax[0,i].set_xticklabels(mle.pNames.iloc[-1], rotation = 45)
        ax[0,i].set_ylabel('parameter value')
        ax[0,i].grid()
        ax[0,i].scatter(range(len(parameters)), np.median(params, 0), marker = '_', s = 500, color= 'red')
        ax[0,i].scatter(range(len(parameters)), np.mean(params, 0), marker = '_', s = 500, color= 'black')
        squarePlot(ax[0,i])
        parameters = np.vstack(np.log(params)).T
        [ ax[1,i].scatter(jitter([n]*len(pp), 0.1), pp, alpha = 0.5) for n,pp in enumerate(parameters) ] 
        ax[1,i].set_xticks(np.arange(len(parameters)))
        ax[1,i].set_xticklabels(mle.pNames.iloc[-1], rotation = 45)
        ax[1,i].set_ylabel('parameter value')
        ax[1,i].grid()
        ax[1,i].scatter(range(len(parameters)), np.median(np.log(params), 0), marker = '_', s = 500, color= 'red')
        ax[1,i].scatter(range(len(parameters)), np.mean(np.log(params), 0), marker = '_', s = 500, color= 'black')
        squarePlot(ax[1,i])
        medians.append(np.mean(params, 0))
    plt.tight_layout()
    return medians

#%%

def compare_modelPredictions( fixed_MLE, adapt_MLE, Trials, plot_fit = True, nReps = 100 ):     
    '''
    '''
    
#    avg_params = get_avgAdapt(adapt_MLE)
    
    dating = np.unique((fixed_MLE.date.unique(), adapt_MLE.date.unique()))
    fig, ax = plt.subplots( 2, len(dating), squeeze = False, figsize=(int(np.ceil(len(dating)/3))*6 , 6 ))
    r = 0
    for date in tqdm(dating):
        ff = fixed_MLE.loc[fixed_MLE.date == date]
        aa = adapt_MLE.loc[adapt_MLE.date == date]
        
        index = np.hstack(ff.trials)[0]
        tt = Trials.loc[index]
        
        fixedModel = ff.model_used.values[0]
        adaptModel = aa.model_used.values[0]
        
#        if not np.all(ff.full_model.iloc[-1].model.exog == aa.full_model.iloc[-1].model.exog):
#            print('data do not match for: ', date)
        
        ff.full_model.iloc[-1].simulate_CE(tt, n=nReps, ax = ax[0,r], color = 'magenta')
        aa.full_model.iloc[-1].simulate_CE(tt, n=nReps, ax = ax[1,r], color = 'cyan')
        ax[0,r].set_title(ff.date.iloc[-1])
        ax[1,r].set_title(aa.date.iloc[-1])
        r += 1
        
        if plot_fit == True:
            X = ff.full_model.iloc[-1].model.exog
            predicted_mles = predict_adapt2fixed(fixedModel, adaptModel, aa.params.values[-1], tt, nReps = 100)
            plot_predictions(ff.iloc[-1], predicted_mles)
    
    plt.tight_layout()

        
#%%    
def plot_predictions(trueFit, predictedFits):
    '''
    '''
    import itertools
    palette = itertools.cycle(sb.color_palette('colorblind'))
    plotData = np.vstack(( np.linspace(0,1,100), np.ones(100), np.zeros(100),
        np.zeros(100), np.linspace(1,0,100), np.ones(100),
        np.zeros(100), np.zeros(100))).T
    
    fig, ax = plt.subplots( 1, 4, squeeze = False, figsize=(10,5))
    
    mm = trueFit.full_model
    ax[0,1].plot(np.linspace(0,0.5, 100), (mm.utility(np.linspace(0,1, 100)) - min(mm.utility(np.linspace(0,1, 100)))) / (max(mm.utility(np.linspace(0,1, 100))) - min(mm.utility(np.linspace(0,1, 100)))), color = 'k' )
    ax[0,0].plot(np.linspace(-1,1, 100), mm.softmax(plotData), color = 'k')
    ax[0,2].plot(np.linspace(0,1, 100), mm.probability(np.linspace(0,1,100)), color = 'k') 

    mle = predictedFits
    cc = next(palette)
    [ ax[0,1].plot(np.linspace(0,0.5, 100), (mm.utility(np.linspace(0,1, 100)) - min(mm.utility(np.linspace(0,1, 100)))) / (max(mm.utility(np.linspace(0,1, 100))) - min(mm.utility(np.linspace(0,1, 100)))), color = cc, alpha = 0.1  ) for mm in mle.full_model]
    [ ax[0,0].plot(np.linspace(-1,1, 100), mm.softmax(plotData), color = cc, alpha = 0.1) for mm in mle.full_model ]
    [ ax[0,2].plot(np.linspace(0,1, 100), mm.probability(np.linspace(0,1,100)), color = cc, alpha = 0.1) for mm in mle.full_model ]
    parameters = np.vstack(mle.params).T
    [ ax[0,3].scatter(jitter([n]*len(pp), 0.1), pp, alpha = 0.5) for n,pp in enumerate(parameters) ] 
    ax[0,3].set_xticks(np.arange(len(parameters)))
    ax[0,3].set_xticklabels(mle.pNames.iloc[-1])
    ax[0,3].set_ylabel('parameter value')
    [ squarePlot(ax[0,nn]) for nn in range(4) ]
    ax[0,1].set_xticks([0, 0.25, 0.5])
    ax[0,2].set_xticks([0, 0.5, 1.0])
    ax[0,0].axvline(0, color = 'k')
    ax[0,0].set_ylabel(mle.model_used.iloc[-1])

    ax[0,0].set_title('Probability of left choice')
    ax[0,1].set_title('Utility')
    ax[0,2].set_title('Probability distortion')

    ax[0,0].set_xlabel('Δ Value')
    ax[0,1].set_xlabel('Reward magnitude (ml)')
    ax[0,2].set_xlabel('Reward probability')
    plt.tight_layout()

#%%
def predict_adapt2fixed(fixedModel, adaptModel, params, tt, nReps = 100):
    '''
    '''
    from macaque.f_models import get_modelLL, trials_2fittable, LL_fit, define_model
    
    X,Y = trials_2fittable(tt)
    
    model = define_model(adaptModel)
    functions = model[3](params)

    nTrials_past = functions['past_effect']
    pChA = functions['prob_chA'](X)
    
    dList = []
    for n in range(0,nReps):
        randomNo = np.random.rand(len(pChA))
        yy = (randomNo <= pChA) * 1
        
        MLE = LL_fit(yy, X, model = fixedModel).fit(disp=False, callback = False)
        
        dList.append({
                'nTrials': MLE.nobs,
                'params': MLE.params,
                'pvalues': MLE.pvalues,
                'NM_success': MLE.mle_retvals['converged'],
                'model_used': MLE.model.model_name,
                'LL': MLE.llf,
                'pNames': MLE.model.exog_names,
                'all_fits': MLE.res_x,
                'full_model': MLE,
                'AIC': MLE.aic,
                'BIC': MLE.bic,
            })
    
    return pd.DataFrame(dList)

#%%
def compare_fitConditions(all_MLEs, division):
    '''
    '''
    import itertools
    #    fig, ax = plt.subplots( len(all_MLEs), 4, squeeze = False, figsize=(10,15))
    plotData = np.vstack(( np.linspace(0,1,100), np.ones(100), np.zeros(100),
            np.zeros(100), np.linspace(1,0,100), np.ones(100),
            np.zeros(100), np.zeros(100))).T
                          
    if np.size(division[0]) == 1:
        identifiers = np.unique(division)
        stack = False
    else:
        identifiers = unique_listOfLists(division)
        stack = True
    
    for i,MLE in enumerate(all_MLEs):    
        print('failed fits:', MLE.loc[MLE.NM_success == False].date.values)
#        mle = mle.loc[mle.NM_success == True]
        fig, ax = plt.subplots( 1, 5, squeeze = False, figsize=(12,5))
        palette = itertools.cycle(sb.color_palette('colorblind'))
        
        for dd in identifiers:
            if stack == True:
#                 print(len([div == dd for div in division]))
#                 print(len(MLE))
                 mle = MLE.loc[ [all(div == dd) for div in division] ]
            else:
                mle = MLE.loc[division == dd]
            cc = next(palette)
            if 'dynamic' in mle.model_used.iloc[-1]:
                params = np.vstack(mle.params)
#                params[:,-1] = np.round(params[:,-1]  / np.finfo('double').eps, 0)
#                [ ax[0,1].plot(np.linspace(0,0.5, 100), (mm.utility(np.linspace(0,1, 100)) - min(mm.utility(np.linspace(0,1, 100)))) / (max(mm.utility(np.linspace(0,1, 100))) - min(mm.utility(np.linspace(0,1, 100)))), color = cc, alpha = 0.3  ) for mm in mle.full_model]
#                [ ax[0,0].plot(np.linspace(-1,1, 100), mm.softmax(plotData), color = cc, alpha = 0.3) for mm in mle.full_model ]
#                [ ax[0,2].plot(np.linspace(0,1, 100), mm.probability(np.linspace(0,1,100)), color = cc, alpha = 0.3) for mm in mle.full_model ]
                parameters = params.T
                [ ax[0,3].scatter(jitter([n]*len(pp), 0.1), pp, alpha = 0.5, color = cc) for n,pp in enumerate(parameters) ] 
                [ ax[0,4].scatter(jitter([n]*len(pp), 0.1), pp, alpha = 0.5, color = cc) for n,pp in enumerate(np.log(parameters)) ] 
            else:
                [ ax[0,1].plot(np.linspace(0,0.5, 100), (mm.utility(np.linspace(0,1, 100)) - min(mm.utility(np.linspace(0,1, 100)))) / (max(mm.utility(np.linspace(0,1, 100))) - min(mm.utility(np.linspace(0,1, 100)))), color = cc, alpha = 0.3  ) for mm in mle.full_model]
                [ ax[0,0].plot(np.linspace(-1,1, 100), mm.softmax(plotData), color = cc, alpha = 0.3) for mm in mle.full_model ]
                [ ax[0,2].plot(np.linspace(0,1, 100), mm.probability(np.linspace(0,1,100)), color = cc, alpha = 0.3) for mm in mle.full_model ]
                parameters = np.vstack(mle.params).T
                [ ax[0,3].scatter(jitter([n]*len(pp), 0.1), pp, alpha = 0.5, color = cc) for n,pp in enumerate(parameters) ] 
                [ ax[0,4].scatter(jitter([n]*len(pp), 0.1), pp, alpha = 0.5, color = cc) for n,pp in enumerate(np.log(parameters)) ] 
            ax[0,3].set_xticks(np.arange(len(parameters)))
            ax[0,3].set_xticklabels(mle.pNames.iloc[-1], rotation = 45)
            ax[0,3].set_ylabel('parameter value')
            ax[0,4].set_xticks(np.arange(len(parameters)))
            ax[0,4].set_xticklabels(mle.pNames.iloc[-1], rotation = 45)
            ax[0,4].set_ylabel('parameter value')
            [ squarePlot(ax[0,nn]) for nn in range(5) ]
            ax[0,1].set_xticks([0, 0.25, 0.5])
            ax[0,2].set_xticks([0, 0.5, 1.0])
            ax[0,0].axvline(0, color = 'k')
            ax[0,0].set_ylabel(mle.model_used.iloc[-1])
        
            ax[0,0].set_title('Probability of left choice')
            ax[0,1].set_title('Utility')
            ax[0,2].set_title('Probability distortion')
        
            ax[0,0].set_xlabel('Δ Value')
            ax[0,1].set_xlabel('Reward magnitude (ml)')
            ax[0,2].set_xlabel('Reward probability')
            plt.tight_layout()
    return 

#%% 
def get_divisions(softmaxDF, MLEs, type = 'sequence'):
    '''
    '''
    if type.lower() == 'sequence':
        division = [softmaxDF.loc[softmaxDF.sessionDate == dd].seqCode.unique() for dd in np.sort(softmaxDF.sessionDate.unique())]
        division = np.hstack(division)
    elif type.lower() == 'range':
        division = [unique_listOfLists(softmaxDF.loc[softmaxDF.sessionDate == dd].reward_range) for dd in np.sort(softmaxDF.sessionDate.unique())]
        division = flatten(division)
    return np.array(division)[np.isin(np.sort(softmaxDF.sessionDate.unique()), MLEs[-1].date.unique(), )]
    