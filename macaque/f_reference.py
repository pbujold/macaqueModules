# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:48:49 2018

@author: phbuj
"""
import numpy as np
import pandas as pd
from macaque.f_toolbox import *
import matplotlib.pyplot as plt
import seaborn as sb
plt.style.use('seaborn-paper')
plt.rcParams['svg.fonttype'] = 'none'
#plt.rcParams['font.family'] = 'sans-serif'
tqdm = ipynb_tqdm()


#%%
def get_winLose_CE(filteredDF, Trials):
    '''
    Calculate the WSLS logistic regression model from the trials that were used for the reste of the analysis.

    '''
    from macaque.f_choices import get_options
    from macaque.f_psychometrics import get_softmaxData
    from macaque.f_probabilityDistortion import fit_likelihoodModel

    def SRC(params, X, Ypred, Yreal):  #this standardizes the parameters
        return [(params[n] * (np.std(X[:, n]) * (np.std(Ypred) / np.std(Yreal)))
                 / np.std(logit(Ypred))) for n in range(len(params))]

    regTrials = np.sort(
        np.unique(
            np.concatenate([
                np.concatenate(list(val.values()))
                for val in filteredDF.get('trial_index').values
            ])))  #gets the index for all the trials I should use in regression
    sTrials = Trials.loc[regTrials].copy(
    )  #get only trials that were used in the rest of the analysis

    #create new important variablles
    # ----------------------------------------------------------------------------
    sTrials['chG'] = sTrials['ml_received'].astype(int)
    sTrials['g_win'] = sTrials['ml_received'].astype(int)
    sTrials['gEV'] = sTrials['ml_received'].astype(float)
    sTrials['sEV'] = sTrials['ml_received'].astype(float)
    sTrials['pChG'] = sTrials['ml_received'].astype(float)
    sTrials['pG_win'] = sTrials['ml_received'].astype(float)

    gEV = []
    chG = []
    won = []  #will make a list that can then be appended to cTrials as a whole?
    for index, row in sTrials.iterrows():
        if row.outcomesCount[0] == 2:
            if row.gambleChosen == 'A':
                sTrials.at[index, 'chG'] = 1
                if row.ml_received != 0:
                    sTrials.at[index, 'g_win'] = 1
#                    sTrials['g_lose'] = 0
                else:
                    sTrials.at[index, 'g_win'] = 0
#                    sTrials['g_lose'] = 1
            else:
                sTrials.at[index, 'chG'] = 1
            sTrials.at[index, 'gEV'] = row.GA_ev
            sTrials.at[index, 'sEV'] = row.GB_ev
        elif row.outcomesCount[1] == 2:
            if row.gambleChosen == 'B':
                sTrials.at[index, 'chG'] = 1
                if row.ml_received != 0:
                    sTrials.at[index, 'g_win'] = 1
#                    sTrials['g_lose'] = 0
                else:
                    sTrials.at[index, 'g_win'] = 0
#                    sTrials['g_lose'] = 1
            else:
                sTrials.at[index, 'chG'] = 0
            sTrials.at[index, 'gEV'] = row.GB_ev
            sTrials.at[index, 'sEV'] = row.GA_ev
    sTrials['g_lose'] = 1 - sTrials['g_win']

    #--------------------------------------------------------------------------------

    consecutives = sTrials.iloc[np.insert(
        np.diff(sTrials.trialNo.values) == 1, 0, False)].index
    #this does not include beginning ones in a non-consecutive manner
    not_consecutives = sTrials.iloc[np.insert(
        np.diff(sTrials.trialNo.values) > 1, 0, True)].index
    #these trials can't be looked at in terms of what happened before them
    for index in consecutives:
        sTrials.at[index, 'pChG'] = sTrials.loc[index - 1].chG
        sTrials.at[index, 'pG_win'] = sTrials.loc[index - 1].g_win
        sTrials.at[index, 'pG_lose'] = sTrials.loc[index - 1].g_lose
    for index in not_consecutives:
        sTrials.at[index, 'pChG'] = np.nan
        sTrials.at[index, 'pG_win'] = np.nan
        sTrials.at[index, 'pG_lose'] = np.nan

    # ------------------------------------------------------------------------

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 4))
    fig, ax2 = plt.subplots(1, 2, squeeze=False, figsize=(10, 4))

    tt = sTrials.loc[(sTrials.pChG == 1) &
                     (sTrials.trialSequenceMode == 9001)].copy()
    choices_pWin = get_options(
        tt.loc[tt.pG_win == 1],
        mergeBy='all',
        byDates=False,
        mergeSequentials=True)
    softmax_pWin = get_softmaxData(
        choices_pWin, metricType='CE', minSecondaries=4, minChoices=4)
    choices_pLose = get_options(
        tt.loc[tt.pG_win == 0],
        mergeBy='all',
        byDates=False,
        mergeSequentials=True)
    softmax_pLose = get_softmaxData(
        choices_pLose, metricType='CE', minSecondaries=4, minChoices=4)

    NMfit_pWin = fit_likelihoodModel(
        softmax_pWin, Trials, uModel='power', wModel='1-prelec', plotit=False, getError=True)

    ax2[0, 0].plot(
        np.linspace(0, 0.5),
        NMfit_pWin.functions.values[0][1](np.linspace(0, 0.5),
                                          NMfit_pWin.params.values[0][1]),
        color='darkblue',
        lw=2)  #utility param
    ax2[0, 0].plot(np.linspace(0, 0.5), np.linspace(0, 1), '--', color='k')
    ax2[0, 0].grid()

    ax2[0, 1].plot(np.linspace(0, 1),
       NMfit_pWin.functions.values[0][2](np.linspace(0, 1),
       NMfit_pWin.params.values[0][2]), color='darkblue', lw=2)
    ax2[0, 1].plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='k')
    ax2[0, 1].grid()

    upper, lower = np.percentile(np.vstack(NMfit_pWin.bootstrap.values), [5, 95], axis=0)
    bound_upper = NMfit_pWin.functions.values[0][1](np.linspace(0, 0.5),  upper[1])
    bound_lower = NMfit_pWin.functions.values[0][1](np.linspace(0, 0.5),  lower[1])
    ax2[0, 0].fill_between( np.linspace(0, 0.5), bound_lower, bound_upper, color='darkblue', alpha=0.20)

    bound_upper = NMfit_pWin.functions.values[0][2](np.linspace(0, 1),  upper[2])
    bound_lower = NMfit_pWin.functions.values[0][2](np.linspace(0, 1),  lower[2])
    ax2[0, 1].fill_between( np.linspace(0, 1), bound_lower, bound_upper, color='darkblue', alpha=0.20)

    blockWin = np.vstack(NMfit_pWin.bootstrap.values)

    # ----------------------------------------------------------------

    NMfit_pLose = fit_likelihoodModel(
        softmax_pLose, Trials, uModel='power', wModel='1-prelec', plotit=False, getError=True)
    print('Win params: ',  str(NMfit_pWin.params.values))
    print('Lose params: ',  str(NMfit_pLose.params.values))
    ax2[0, 0].plot( np.linspace(0, 0.5),   NMfit_pLose.functions.values[0][1](np.linspace(0, 0.5), NMfit_pLose.params.values[0][1]),
        color='paleturquoise',
        lw=2)  #utility param

    ax2[0, 1].plot(np.linspace(0, 1), NMfit_pLose.functions.values[0][2](np.linspace(0, 1), NMfit_pLose.params.values[0][2]),
        color='paleturquoise',
        lw=2)

    upper, lower = np.percentile(np.vstack(NMfit_pLose.bootstrap.values), [5, 95], axis=0)
    bound_upper = NMfit_pLose.functions.values[0][1](np.linspace(0, 0.5),  upper[1])
    bound_lower = NMfit_pLose.functions.values[0][1](np.linspace(0, 0.5),  lower[1])
    ax2[0, 0].fill_between( np.linspace(0, 0.5), bound_lower, bound_upper, color='paleturquoise', alpha=0.20)

    bound_upper = NMfit_pLose.functions.values[0][2](np.linspace(0, 1),  upper[2])
    bound_lower = NMfit_pLose.functions.values[0][2](np.linspace(0, 1),  lower[2])
    ax2[0, 1].fill_between( np.linspace(0, 1), bound_lower, bound_upper, color='paleturquoise', alpha=0.20)

    blockLose = np.vstack(NMfit_pLose.bootstrap.values)

    softmax_pWin.plot(
        x='primaryEV',
        y='equivalent',
        yerr=np.vstack(softmax_pWin.pSTE.values)[:, 0],
        kind='scatter',
        color='darkblue',
        ax=ax[0, 0],
        grid=True,
        s=50)
    softmax_pLose.plot(
        x='primaryEV',
        y='equivalent',
        yerr=np.vstack(softmax_pLose.pSTE.values)[:, 0],
        kind='scatter',
        color='paleturquoise',
        ax=ax[0, 0],
        grid=True,
        s=50)
    ax[0, 0].legend(['past gamble won', 'past gamble lost'])
    ax[0, 0].plot(np.linspace(0, 0.5), np.linspace(0, 0.5), '--', color='k')

    # ------------------------- Plot the bar graph for the parameter differences

#    import pdb ; pdb.set_trace() #AWESOME WAY TO DEBUG
    fig, ax3 = plt.subplots(1, 2, squeeze=False, figsize=(10, 4))
    lower_BW, upper_BW = np.percentile(np.log(np.vstack(NMfit_pWin.bootstrap[0])), [2.5, 97.5], axis=0)
    lower_BW = np.log(NMfit_pWin.params.values[0]) - lower_BW
    upper_BW = upper_BW - np.log(NMfit_pWin.params.values[0])
    lower_BL, upper_BL = np.percentile(np.log(np.vstack(NMfit_pLose.bootstrap[0])), [2.5, 97.5], axis=0)
    lower_BL = np.log(NMfit_pLose.params.values[0]) - lower_BL
    upper_BL =  upper_BL - np.log(NMfit_pLose.params.values[0])
    ax3[0,0].set_title('Block Sequences')
    ax3[0,0].bar(np.array([1,2,3]), np.log(NMfit_pWin.params.values[0]),  width = 0.35, yerr=np.vstack((lower_BW, upper_BW)), color = 'darkblue')
    ax3[0,0].bar(np.array([1,2,3]) + 0.35, np.log(NMfit_pLose.params.values[0]),  width = 0.35, yerr=np.vstack((lower_BL, upper_BL)), color = 'paleturquoise' )
    ax3[0,0].set_xticks(np.array([1,2,3])+(0.35/2))
    ax3[0,0].set_xticklabels(['softmax', 'utility', 'probability'])
    ax3[0,0].axhline(0)
    ax3[0,0].set_ylim(-1, 3)


    avgSTD = np.std(np.concatenate((np.log(np.vstack(NMfit_pWin.bootstrap[0])), np.log(np.vstack(NMfit_pLose.bootstrap[0])))), axis=0)
    effectSize = np.round( ( np.log(NMfit_pWin.params.values[0]) -  np.log(NMfit_pLose.params.values[0]) ) /  avgSTD, 4)
    print('cohen D: ', effectSize)
    # =========================================================================

    tt = sTrials.loc[(sTrials.pChG == 1) &
                     (sTrials.trialSequenceMode == 9020)].copy()
    choices_pWin = get_options(
        tt.loc[tt.pG_win == 1],
        mergeBy='all',
        byDates=False,
        mergeSequentials=True)
    softmax_pWin = get_softmaxData(
        choices_pWin, metricType='CE', minSecondaries=4, minChoices=4)
    choices_pLose = get_options(
        tt.loc[tt.pG_win == 0],
        mergeBy='all',
        byDates=False,
        mergeSequentials=True)
    softmax_pLose = get_softmaxData(
        choices_pLose, metricType='CE', minSecondaries=4, minChoices=4)

    NMfit_pWin = fit_likelihoodModel(
        softmax_pWin, Trials, uModel='power', wModel='1-prelec', plotit=False, getError=True)
    ax2[0, 0].plot(
        np.linspace(0, 0.5),
        NMfit_pWin.functions.values[0][1](np.linspace(0, 0.5),
                                          NMfit_pWin.params.values[0][1]),
        color='darkred',
        lw=2)  #utility param
    ax2[0, 1].plot(
        np.linspace(0, 1),
        NMfit_pWin.functions.values[0][2](np.linspace(0, 1),
                                          NMfit_pWin.params.values[0][2]),
        color='darkred',
        lw=2)

    upper, lower = np.percentile(np.vstack(NMfit_pWin.bootstrap.values), [5, 95], axis=0)
    bound_upper = NMfit_pWin.functions.values[0][1](np.linspace(0, 0.5),  upper[1])
    bound_lower = NMfit_pWin.functions.values[0][1](np.linspace(0, 0.5),  lower[1])
    ax2[0, 0].fill_between( np.linspace(0, 0.5), bound_lower, bound_upper, color='darkred', alpha=0.20)

    bound_upper = NMfit_pWin.functions.values[0][2](np.linspace(0, 1),  upper[2])
    bound_lower = NMfit_pWin.functions.values[0][2](np.linspace(0, 1),  lower[2])
    ax2[0, 1].fill_between( np.linspace(0, 1), bound_lower, bound_upper, color='darkred', alpha=0.20)

    mixWin = np.vstack(NMfit_pWin.bootstrap.values)

    NMfit_pLose = fit_likelihoodModel(
        softmax_pLose, Trials, uModel='power', wModel='1-prelec', plotit=False, getError=True)
    ax2[0, 0].plot(
        np.linspace(0, 0.5),
        NMfit_pLose.functions.values[0][1](np.linspace(0, 0.5),
                                           NMfit_pLose.params.values[0][1]),
        color='pink',
        lw=2)  #utility param

    upper, lower = np.percentile(np.vstack(NMfit_pLose.bootstrap.values), [5, 95], axis=0)
    bound_upper = NMfit_pLose.functions.values[0][1](np.linspace(0, 0.5),  upper[1])
    bound_lower = NMfit_pLose.functions.values[0][1](np.linspace(0, 0.5),  lower[1])
    ax2[0, 0].fill_between( np.linspace(0, 0.5), bound_lower, bound_upper, color='pink', alpha=0.20)

    bound_upper = NMfit_pLose.functions.values[0][2](np.linspace(0, 1),  upper[2])
    bound_lower = NMfit_pLose.functions.values[0][2](np.linspace(0, 1),  lower[2])
    ax2[0, 1].fill_between( np.linspace(0, 1), bound_lower, bound_upper, color='pink', alpha=0.20)

    mixLose = np.vstack(NMfit_pLose.bootstrap.values)

    # add the bootstrap result to do the t-test
    print('Win params: ',  str(NMfit_pWin.params.values))
    print('Lose params: ',  str(NMfit_pLose.params.values))

    ax2[0, 1].plot(
        np.linspace(0, 1),
        NMfit_pLose.functions.values[0][2](np.linspace(0, 1),
                                           NMfit_pLose.params.values[0][2]),
        color='pink',
        lw=2)

    softmax_pWin.plot(
        x='primaryEV',
        y='equivalent',
        yerr=np.vstack(softmax_pWin.pSTE.values)[:, 0],
        kind='scatter',
        color='darkred',
        ax=ax[0, 1],
        grid=True,
        s=50)
    softmax_pLose.plot(
        x='primaryEV',
        y='equivalent',
        yerr=np.vstack(softmax_pLose.pSTE.values)[:, 0],
        kind='scatter',
        color='pink',
        ax=ax[0, 1],
        grid=True,
        s=50)
    ax[0, 1].legend(['past gamble won', 'past gamble lost'])
    ax[0, 1].plot(np.linspace(0, 0.5), np.linspace(0, 0.5), '--', color='k')

    for axis in ax.reshape(-1):
        x0, x1 = axis.get_xlim()
        y0, y1 = axis.get_ylim()
        axis.set_aspect((x1 - x0) / (y1 - y0))

    for axis in ax2.reshape(-1):
        x0, x1 = axis.get_xlim()
        y0, y1 = axis.get_ylim()
        axis.set_aspect((x1 - x0) / (y1 - y0))

    ax[0, 0].set_ylabel('certainty equivalent')
    ax[0, 1].set_ylabel('certainty equivalent')
    ax[0, 0].set_xlabel('gambleEV')
    ax[0, 1].set_xlabel('gambleEV')
    ax2[0, 0].set_ylabel('utility')
    ax2[0, 1].set_ylabel('probability distortion')
    ax2[0, 0].set_xlabel('magnitude')
    ax2[0, 1].set_xlabel('probability')

    ax[0, 1].set_title('mixed trials')
    ax[0, 0].set_title('repeat trials')

    ax2[0, 1].set_title('probability distortion')
    ax2[0, 0].set_title('utility')

    # --------------------------------------------------------------------------

    lower_BW, upper_BW = np.percentile(np.log(np.vstack(NMfit_pWin.bootstrap[0])), [2.5, 97.5], axis=0)
    lower_BW = np.log(NMfit_pWin.params.values[0]) - lower_BW
    upper_BW = upper_BW - np.log(NMfit_pWin.params.values[0])
    lower_BL, upper_BL = np.percentile(np.log(np.vstack(NMfit_pLose.bootstrap[0])), [2.5, 97.5], axis=0)
    lower_BL = np.log(NMfit_pLose.params.values[0]) - lower_BL
    upper_BL =  upper_BL - np.log(NMfit_pLose.params.values[0])
    ax3[0,1].set_title('Block Sequences')
    ax3[0,1].bar(np.array([1,2,3]), np.log(NMfit_pWin.params.values[0]),  width = 0.35, yerr=np.vstack((lower_BW, upper_BW)), color = 'darkred')
    ax3[0,1].bar(np.array([1,2,3]) + 0.35, np.log(NMfit_pLose.params.values[0]),  width = 0.35, yerr=np.vstack((lower_BL, upper_BL)), color = 'pink' )
    ax3[0,1].set_xticks(np.array([1,2,3])+(0.35/2))
    ax3[0,1].set_xticklabels(['softmax', 'utility', 'probability'])
    ax3[0,1].axhline(0)
    ax3[0,1].set_ylim(-1, 3)


    avgSTD = np.std(np.concatenate((np.log(np.vstack(NMfit_pWin.bootstrap[0])), np.log(np.vstack(NMfit_pLose.bootstrap[0])))), axis=0)
    effectSize = np.round( ( np.log(NMfit_pWin.params.values[0]) -  np.log(NMfit_pLose.params.values[0]) ) /  avgSTD, 4)
    print('cohen D: ', effectSize)
    return blockWin, blockLose, mixWin, mixLose



#%%
def WSLS_model_withSafes(filteredDF, Trials, addConstant=True, model='wsls'):
    '''
    Calculate the WSLS logistic regression model from the trials that were used for the reste of the analysis.

    '''
    from scipy import stats
    import statsmodels.api as sm
    stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
    from statsmodels.graphics.api import abline_plot
    from scipy.special import logit

    def SRC(params, X, Ypred, Yreal):  #this standardizes the parameters
        return [(params[n] * (np.std(X[:, n]) * (np.std(Ypred) / np.std(Yreal)))
                 / np.std(logit(Ypred))) for n in range(len(params))]

    regTrials = np.sort(
        np.unique(
            np.concatenate([
                np.concatenate(list(val.values()))
                for val in filteredDF.get('trial_index').values
            ])))  #gets the index for all the trials I should use in regression
    sTrials = Trials.loc[regTrials].copy(
    )  #get only trials that were used in the rest of the analysis

    #create new important variablles
    # ----------------------------------------------------------------------------
    sTrials['chG'] = sTrials['ml_received'].astype(int)
    sTrials['g_win'] = sTrials['ml_received'].astype(int)
    sTrials['gEV'] = sTrials['ml_received'].astype(float)
    sTrials['sEV'] = sTrials['ml_received'].astype(float)
    sTrials['pChG'] = sTrials['ml_received'].astype(float)
    sTrials['pG_win'] = sTrials['ml_received'].astype(float)

    gEV = []
    chG = []
    won = []  #will make a list that can then be appended to cTrials as a whole?
    for index, row in sTrials.iterrows():
        if row.outcomesCount[0] == 2:
            if row.gambleChosen == 'A':
                sTrials.at[index, 'chG'] = 1
                if row.ml_received != 0:
                    sTrials.at[index, 'g_win'] = 1
#                    sTrials['g_lose'] = 0
                else:
                    sTrials.at[index, 'g_win'] = 0
#                    sTrials['g_lose'] = 1
            else:
                sTrials.at[index, 'chG'] = 1
            sTrials.at[index, 'gEV'] = row.GA_ev
            sTrials.at[index, 'sEV'] = row.GB_ev
        elif row.outcomesCount[1] == 2:
            if row.gambleChosen == 'B':
                sTrials.at[index, 'chG'] = 1
                if row.ml_received != 0:
                    sTrials.at[index, 'g_win'] = 1
#                    sTrials['g_lose'] = 0
                else:
                    sTrials.at[index, 'g_win'] = 0
#                    sTrials['g_lose'] = 1
            else:
                sTrials.at[index, 'chG'] = 0
            sTrials.at[index, 'gEV'] = row.GB_ev
            sTrials.at[index, 'sEV'] = row.GA_ev
    sTrials['g_lose'] = 1 - sTrials['g_win']
    sTrials['chS'] = 1 - sTrials['chG']

    ws_Trials = sTrials
    Params = []
    pVals = []
    sParams = []
    zParams = []
    context = []
    N = 0
    for date in tqdm(sTrials.sessionDate.unique(), desc='Modeling Session'):
        sTrials = ws_Trials.loc[ws_Trials.sessionDate == date]
        consecutives = sTrials.iloc[np.insert(
            np.diff(sTrials.trialNo.values) == 1, 0, False)].index
        #this does not include beginning ones in a non-consecutive manner
        #        xx = sTrials.loc[consecutives] #the trials that are followed
        not_consecutives = sTrials.iloc[np.insert(
            np.diff(sTrials.trialNo.values) > 1, 0, True)].index
        #these trials can't be looked at in terms of what happened before them
        for index in consecutives:
            sTrials.at[index, 'pChG'] = sTrials.loc[index - 1].chG
            sTrials.at[index, 'pG_win'] = sTrials.loc[index - 1].g_win
            sTrials.at[index, 'pG_lose'] = sTrials.loc[index - 1].g_lose
            sTrials.at[index, 'pChS'] = sTrials.loc[index - 1].chS
            sTrials.at[index, 'pChosenEV'] = sTrials.loc[index - 1].chosenEV
        for index in not_consecutives:
            sTrials.at[index, 'pChG'] = np.nan
            sTrials.at[index, 'pG_win'] = np.nan
            sTrials.at[index, 'pG_lose'] = np.nan
            sTrials.at[index, 'pChS'] = np.nan
            sTrials.at[index, 'pChosenEV'] = np.nan
#    sTrials['pChS'] = 1-sTrials['pChG']

#correlate the losestay parameter with the curvature parameter of the probability distortion?

#NOW FROM sTrials MAKE THE APPROPRIATE FUNCTION
#    look at trials where the gamble was previously chosen
#        regressIndex = sTrials.loc[(sTrials.pChG == 1) | (sTrials.pChG == 0)].index
        regressIndex = sTrials.loc[(sTrials.pChG == 1) |
                                   (sTrials.pChS == 1)].index
        pChS = sTrials.loc[regressIndex].pChS.values
        sEV = sTrials.loc[
            regressIndex].sEV.values  #the trials that are followed by a trial before them
        gEV = sTrials.loc[regressIndex].gEV.values
        Y = sTrials.loc[regressIndex].chG.values
        pWin = sTrials.loc[regressIndex].pG_win.values
        #        pWin[pWin == 0] = -1
        pLose = sTrials.loc[regressIndex].pG_lose.values
        pchEV = sTrials.loc[regressIndex].pChosenEV.values

        X = np.vstack((gEV, sEV, pWin, pLose, 1 - pChS, pchEV)).T

        if addConstant:
            X = sm.tools.add_constant(
                X, prepend=True, has_constant='add')  #adds constant for the fit

        results = sm.GLM(Y, X, family=sm.families.Binomial()).fit()
        betas = results.params
        Ps = results.pvalues
        err_y = results.summary2().tables[1].values
        zVal = results.summary2().tables[1].z.values
        success = results.converged

        if any(np.isnan(SRC(betas, X, results.fittedvalues, Y))):
            success = False

        if success:
            N += len(X)
            context.extend(sTrials.trialSequenceMode.unique())
            Params.append(betas)
            pVals.append(Ps)
            #            sParams.append(standardizeBetas(betas, X, Y)) #standardize like Bill and Armin
            sParams.append(SRC(betas, X, results.fittedvalues, Y))
            zParams.append(zVal)

    #no need to plot the intercept as it has been standardized
    if addConstant:
        Params = np.vstack(Params)[:, 1:]
        sParams = np.stack(sParams)[:, 1:]
        zParams = np.stack(zParams)[:, 1:]
        yerr = stats.sem(zParams, axis=0)
        yerr_s = stats.sem(sParams, axis=0)
    else:
        Params = np.vstack(Params)
        sParams = np.stack(sParams)
        zParams = np.stack(zParams)
        yerr = stats.sem(zParams, axis=0)
        yerr_s = stats.sem(sParams, axis=0)

    plt.bar(range(len(Params[1])), sParams.mean(axis=0))
    plt.errorbar(
        range(len(Params[1])),
        sParams.mean(axis=0),
        yerr=yerr_s,
        fmt='none',
        color='k',
        capsize=3)
    t, p = stats.ttest_1samp(sParams, 0.0, axis=0)

    if model.lower() == 'ws':
        plt.xticks(
            range(len(Params[1])), ['gambleEV', 'safeEV', 'pre-gamble won'])
    elif model.lower() == 'ls':
        plt.xticks(
            range(len(Params[1])), ['gambleEV', 'safeEV', 'pre-gamble lost'])
    elif model.lower() == 'wsls':
        plt.xticks(
            range(len(Params[1])),
            ['gambleEV', 'safeEV', 'pre-gamble won', 'pre-gamble lost'])

    plt.axhline(0, color='k')
    gap = 0.2
    for significance, y, x, err in zip(
            p, np.stack(sParams).mean(axis=0), range(0,
                                                     len(betas) + 1), yerr_s):
        if significance < 0.05 and y >= 0:
            plt.plot(x, y + err + gap, marker='*', color='k')
        elif significance < 0.05 and y < 0:
            plt.plot(x, y - err - gap, marker='*', color='k')

    tt, pp = stats.ttest_1samp(zParams, 0.0, axis=0)
    from tabulate import tabulate
    print(
        tabulate([['N:', str(len(zParams))], ['t-values:', str(t)],
                  ['p-values:', str(p)], ['p-values (z):',
                                          str(pp)],
                  ['standardized Params',
                   str(np.stack(sParams).mean(axis=0))],
                  ['nz-scored Params',
                   str(np.stack(zParams).mean(axis=0))]]))
    plt.show()
    print('N = ' + str(N) + ' trials')
    return pVals, sParams


#%%
def WSLS_model(filteredDF, Trials, addConstant=True, model='ws'):
    '''
    Calculate the WSLS logistic regression model from the trials that were used for the reste of the analysis.

    '''


    from scipy import stats
    import statsmodels.api as sm
    stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
    from statsmodels.graphics.api import abline_plot
    from scipy.special import logit

    f_BIC = lambda LL, nP, nT: (2 * -LL) + (nP * np.log(nT))
    def SRC(params, X, Ypred, Yreal):  #this standardizes the parameters
        return [(params[n] * (np.std(X[:, n]) * (np.std(Ypred) / np.std(Yreal)))
                 / np.std(logit(Ypred))) for n in range(len(params))]

    def get_binomialfit(X, Y, mode):
        AIC, BIC, LL = [], [], []
        if mode.lower() == 'logit':
            results = sm.Logit(Y, X).fit()
            if float(results.summary2().tables[0][1][6]):
                AIC = results.aic
                BIC = results.bic
                LL = results.llf
        if mode.lower() == 'glm':
            results = sm.GLM(Y, X, family=sm.families.Binomial()).fit()
            if results.converged:
                AIC = results.aic
                BIC = f_BIC(results.llf, len(X[0]), len(X))
                LL = results.llf
        return AIC, BIC, LL

    sTrials = filteredDF.getTrials(Trials)  #get only trials that were used in the rest of the analysis

    #create new important variablles
    # ----------------------------------------------------------------------------
    sTrials['chG'] = sTrials['ml_received'].astype(int)
    sTrials['g_win'] = sTrials['ml_received'].astype(int)
    sTrials['gEV'] = sTrials['ml_received'].astype(float)
    sTrials['sEV'] = sTrials['ml_received'].astype(float)
    sTrials['pChG'] = sTrials['ml_received'].astype(float)
    sTrials['pG_win'] = sTrials['ml_received'].astype(float)

    gEV = []
    chG = []
    won = []  #will make a list that can then be appended to cTrials as a whole?
    for index, row in sTrials.iterrows():
        if row.outcomesCount[0] == 2:
            if row.gambleChosen == 'A':
                sTrials.at[index, 'chG'] = 1
                if row.ml_received != 0:
                    sTrials.at[index, 'g_win'] = 1
#                    sTrials['g_lose'] = 0
                else:
                    sTrials.at[index, 'g_win'] = 0
#                    sTrials['g_lose'] = 1
            else:
                sTrials.at[index, 'chG'] = 1
            sTrials.at[index, 'gEV'] = row.GA_ev
            sTrials.at[index, 'sEV'] = row.GB_ev
        elif row.outcomesCount[1] == 2:
            if row.gambleChosen == 'B':
                sTrials.at[index, 'chG'] = 1
                if row.ml_received != 0:
                    sTrials.at[index, 'g_win'] = 1
#                    sTrials['g_lose'] = 0
                else:
                    sTrials.at[index, 'g_win'] = 0
#                    sTrials['g_lose'] = 1
            else:
                sTrials.at[index, 'chG'] = 0
            sTrials.at[index, 'gEV'] = row.GB_ev
            sTrials.at[index, 'sEV'] = row.GA_ev
    sTrials['g_lose'] = 1 - sTrials['g_win']

    ws_Trials = sTrials
    Params = []
    pVals = []
    sParams = []
    zParams = []
    context = []
    LL2 = []; BIC2 = []
    LL1 = []; BIC1 = []
    N = 0
    for date in tqdm(sTrials.sessionDate.unique(), desc='Modeling Session'):
        sTrials = ws_Trials.loc[ws_Trials.sessionDate == date]
        consecutives = sTrials.iloc[np.insert(
            np.diff(sTrials.trialNo.values) == 1, 0, False)].index
        #this does not include beginning ones in a non-consecutive manner
        #        xx = sTrials.loc[consecutives] #the trials that are followed
        not_consecutives = sTrials.iloc[np.insert(
            np.diff(sTrials.trialNo.values) > 1, 0, True)].index
        #these trials can't be looked at in terms of what happened before them
        for index in consecutives:
            sTrials.at[index, 'pChG'] = sTrials.loc[index - 1].chG
            sTrials.at[index, 'pG_win'] = sTrials.loc[index - 1].g_win
            sTrials.at[index, 'pG_lose'] = sTrials.loc[index - 1].g_lose
        for index in not_consecutives:
            sTrials.at[index, 'pChG'] = np.nan
            sTrials.at[index, 'pG_win'] = np.nan
            sTrials.at[index, 'pG_lose'] = np.nan
#    sTrials['pChS'] = 1-sTrials['pChG']

#correlate the losestay parameter with the curvature parameter of the probability distortion?

#NOW FROM sTrials MAKE THE APPROPRIATE FUNCTION
#    look at trials where the gamble was previously chosen
#        regressIndex = sTrials.loc[(sTrials.pChG == 1) | (sTrials.pChG == 0)].index
        regressIndex = sTrials.loc[sTrials.pChG == 1].index
        sEV = sTrials.loc[
            regressIndex].sEV.values  #the trials that are followed by a trial before them
        gEV = sTrials.loc[regressIndex].gEV.values
        Y = sTrials.loc[regressIndex].chG.values
        pWin = sTrials.loc[regressIndex].pG_win.values
        #        pWin[pWin == 0] = -1
        pLose = sTrials.loc[regressIndex].pG_lose.values
        risk = []
        for index, row in sTrials.loc[regressIndex].iterrows():
            if row.outcomesCount[0] > 1:
                    risk.extend([np.sqrt(row.gambleA[1] * row.gambleA[3])])
            elif row.outcomesCount[1] > 1:
                    risk.extend([np.sqrt(row.gambleB[1] * row.gambleB[3])])


        if model.lower() == 'ws':
            X = np.vstack((gEV, sEV, pWin)).T
        elif model.lower() == 'ls':
            X = np.vstack((gEV, sEV, pLose)).T
        elif model.lower() == 'wsls':
            X = np.vstack((gEV, sEV, pWin, pLose)).T

        if addConstant:
            X = sm.tools.add_constant(
                X, prepend=True, has_constant='add')  #adds constant for the fit

        # ---------------------------------
        XX = np.stack(
            (np.array(gEV), np.array(sEV), np.array(risk)),
            axis=1)
        XX = sm.tools.add_constant(
            XX, prepend=True, has_constant='add')  #adds constant for the fit
        YY = Y
        aic, bic, ll = get_binomialfit(XX, YY, 'glm')
        LL1.append(ll)
        BIC1.append(f_BIC(ll, np.size(np.vstack(XX), 1)-1, np.size(np.vstack(XX), 0)))
        # -----------------------------------


        results = sm.GLM(Y, X, family=sm.families.Binomial()).fit()
        LL2.append(results.llf)
        BIC2.append(f_BIC(results.llf, np.size(np.vstack(X), 1)-1, np.size(np.vstack(X), 0)))
        betas = results.params
        Ps = results.pvalues
        err_y = results.summary2().tables[1].values
        zVal = results.summary2().tables[1].z.values
        success = results.converged

        if any(np.isnan(SRC(betas, X, results.fittedvalues, Y))):
            success = False

        if success:
            N += len(X)
            context.extend(sTrials.trialSequenceMode.unique())
            Params.append(betas)
            pVals.append(Ps)
            #            sParams.append(standardizeBetas(betas, X, Y)) #standardize like Bill and Armin
            sParams.append(SRC(betas, X, results.fittedvalues, Y))
            zParams.append(zVal)

    #no need to plot the intercept as it has been standardized
    if addConstant:
        Params = np.vstack(Params)[:, 1:]
        sParams = np.stack(sParams)[:, 1:]
        zParams = np.stack(zParams)[:, 1:]
        yerr = stats.sem(zParams, axis=0)
        yerr_s = stats.sem(sParams, axis=0)
    else:
        Params = np.vstack(Params)
        sParams = np.stack(sParams)
        zParams = np.stack(zParams)
        yerr = stats.sem(zParams, axis=0)
        yerr_s = stats.sem(sParams, axis=0)

    plt.bar(range(len(Params[1])), sParams.mean(axis=0))
    plt.errorbar(
        range(len(Params[1])),
        sParams.mean(axis=0),
        yerr=yerr_s,
        fmt='none',
        color='k',
        capsize=3)
    t, p = stats.ttest_1samp(sParams, 0.0, axis=0)

    if model.lower() == 'ws':
        plt.xticks(
            range(len(Params[1])), ['gambleEV', 'safeEV', 'pre-gamble won'])
    elif model.lower() == 'ls':
        plt.xticks(
            range(len(Params[1])), ['gambleEV', 'safeEV', 'pre-gamble lost'])
    elif model.lower() == 'wsls':
        plt.xticks(
            range(len(Params[1])),
            ['gambleEV', 'safeEV', 'pre-gamble won', 'pre-gamble lost'])

    plt.axhline(0, color='k')
    gap = 0.2
    for significance, y, x, err in zip(
            p, np.stack(sParams).mean(axis=0), range(0,
                                                     len(betas) + 1), yerr_s):
        if significance < 0.05 and y >= 0:
            plt.plot(x, y + err + gap, marker='*', color='k')
        elif significance < 0.05 and y < 0:
            plt.plot(x, y - err - gap, marker='*', color='k')

    tt, pp = stats.ttest_1samp(zParams, 0.0, axis=0)
    from tabulate import tabulate
    print(
        tabulate([['N:', str(len(zParams))], ['t-values:', str(t)],
                  ['p-values:', str(p)], ['p-values (z):',
                                          str(pp)],
                  ['standardized Params',
                   str(np.stack(sParams).mean(axis=0))],
                  ['nz-scored Params',
                   str(np.stack(zParams).mean(axis=0))]]))
    plt.show()
    print('N = ' + str(N) + ' trials')


    np.mean(BIC1)
    np.mean(BIC2)
    print('===================================================================================')
    print('T-Test on the log likelihoods of the gev,sev,risk and gev,sev,pWin regression models')
    print('N =', str(len(BIC1)))
    print(stats.ttest_ind(LL1, LL2))
    return pVals, sParams


#%%
def get_trialHistory(filteredDF, Trials):
    '''
    Get all all trials used in the analysis with added past trial information (i.e. trialHistory), as well as the index for the start of each testing session.
    '''
    filteredDF = filteredDF.sort_values(by=['sessionDate', 'division'])
    #what was i doing here with the code?
    dfs = []
    d_ind = []
    for date in tqdm(filteredDF.sessionDate.unique()):
        firsts = []
        tt = [
            np.sort(np.concatenate(list(val.values())))
            for val in filteredDF.loc[filteredDF.sessionDate == date].get(
                'trial_index').values
        ]
        iTrials = np.unique(np.concatenate(tt))
        firsts.extend([iTrials[0]])
        for i in range(1, len(tt)):
            if all(filteredDF.loc[filteredDF.sessionDate==date].iloc[i].division - filteredDF.loc[filteredDF.sessionDate==date].iloc[i-1].division != np.array([0,1]))\
                and tt[i][0] - tt[i-1][-1] > 10:
                firsts.extend([tt[i][0]])

        cTrials = Trials.loc[iTrials].copy()
        cTrials['past_gEV'] = cTrials['time'].astype(list)
        cTrials['past_mEV'] = cTrials['time'].astype(list)
        cTrials['gEV'] = cTrials['GA_ev'].astype(list)

        for ii in iTrials:
            if cTrials.loc[ii].outcomesCount[0] == 2:
                cVal = cTrials.loc[ii].GA_ev
            elif cTrials.loc[ii].outcomesCount[1] == 2:
                cVal = cTrials.loc[ii].GB_ev

            if any(ii == np.array(firsts)):
                cTrials.at[ii, 'past_gEV'] = []
                cTrials.at[ii, 'past_mEV'] = []
                cTrials.at[ii, 'gEV'] = cVal
                d_ind.extend([ii])
                continue

            lowest = np.max(np.array(firsts)[ii - np.array(firsts) > 0])

            if ii - 50 <= lowest:
                ind = lowest
            else:
                ind = ii - 50
                if ind not in iTrials:
                    ind = np.max(iTrials[(iTrials - ind) < 0])
                    if ind <= lowest:
                        ind = lowest
            if ind == ii:
                cTrials.at[ii, 'past_gEV'] = []
                cTrials.at[ii, 'past_mEV'] = []
                cTrials.at[ii, 'gEV'] = cVal
                d_ind.extend([ii])
                continue

            gEV = [
            ]  #will make a list that can then be appended to cTrials as a whole?
            for index, row in cTrials.loc[ind:ii].iterrows():
                if row.outcomesCount[0] == 2:
                    gEV.extend([row.GA_ev])
                elif row.outcomesCount[1] == 2:
                    gEV.extend([row.GB_ev])
            gEV = gEV[::-1][1:]
            meanEV = np.cumsum(gEV) / range(1, len(gEV) + 1)

            cTrials.at[ii, 'past_gEV'] = gEV
            cTrials.at[ii, 'past_mEV'] = meanEV
            cTrials.at[ii, 'gEV'] = cVal
        dfs.append(cTrials)
    past_Trials = pd.concat(dfs)
    return past_Trials, d_ind


#%%
def compare_gambleVariance(filteredDF, Trials, window=20):
    '''
    '''
    from scipy import stats
    import statsmodels.api as sm

    regTrials = np.sort(
        np.unique(
            np.concatenate([
                np.concatenate(list(val.values()))
                for val in filteredDF.get('trial_index').values
            ])))  #gets the index for all the trials I should use in regression
    sTrials = Trials.loc[regTrials].copy()
    ind = sTrials.index
    gambles = []
    safes = []
    context = []
    with tqdm(total=len(sTrials), desc='Fetching Gamble/Safe Data') as pbar:
        for index, row in sTrials.iterrows():
            if row.outcomesCount[0] == 2:
                gambles.extend([row.GA_ev])
                safes.extend([row.GB_ev])
                context.extend([row.trialSequenceMode])
            elif row.outcomesCount[1] == 2:
                gambles.extend([row.GB_ev])
                safes.extend([row.GA_ev])
                context.extend([row.trialSequenceMode])
            pbar.update(1)

    #----------------------------------------------------------------
    #PLOT ALL THE FIGURES RELATING TO PER-TRIAL VARIANCE AND BIN MEANS
    #initially plots the gamble's expected values in two forms (point or line),
    #then a t-test measure of the difference across individual trials (abs))

    seq = np.unique(context)
    fig, ax = plt.subplots(1, 4, squeeze=False, figsize=(15, 4))
    ax2 = ax[0, 0].twiny()
    ax[0, 0].scatter(
        range(len(np.array(gambles)[context == seq[1]])),
        np.array(gambles)[context == seq[1]],
        color='red',
        alpha=0.5)
    ax2.scatter(
        range(len(np.array(gambles)[context == seq[0]])),
        np.array(gambles)[context == seq[0]],
        color='blue',
        alpha=1)
    ax2.set_xticklabels([])
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_ylabel('gamble probability')
    ax[0, 0].set_xlabel('all trials')
    #    plt.plot(range(len(ind)),safes)

    ax3 = ax[0, 1].twiny()
    ax[0, 1].plot(
        range(len(np.array(gambles)[context == seq[1]])),
        np.array(gambles)[context == seq[1]],
        color='red',
        alpha=0.5)
    ax3.plot(
        range(len(np.array(gambles)[context == seq[0]])),
        np.array(gambles)[context == seq[0]],
        color='blue',
        alpha=1)
    ax3.set_xticklabels([])
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_ylabel('gamble probability')
    ax[0, 1].set_xlabel('all trials')
    #    plt.plot(range(len(ind)),safes)

    avgVar_perTrials_m = np.mean(
        np.abs(np.diff(np.array(gambles)[context == seq[1]])))
    avgVar_perTrials_b = np.mean(
        np.abs(np.diff(np.array(gambles)[context == seq[0]])))
    semVar_perTrials_m = np.std(
        np.abs(np.diff(np.array(gambles)[context == seq[1]])))
    semVar_perTrials_b = np.std(
        np.abs(np.diff(np.array(gambles)[context == seq[0]])))
    t, p = stats.ttest_ind(
        np.abs(np.diff(np.array(gambles)[context == seq[1]])),
        np.abs(np.diff(np.array(gambles)[context == seq[0]])),
        equal_var=True)
    ax[0, 2].bar(
        ['mixed', 'blocked'], [avgVar_perTrials_m, avgVar_perTrials_b],
        yerr=[semVar_perTrials_m, semVar_perTrials_b],
        capsize=3,
        color=['red', 'blue'])
    ytop = label_diff(0, 1, 'p=' + str(p), [0, 1], [
        avgVar_perTrials_m + semVar_perTrials_m,
        avgVar_perTrials_b + semVar_perTrials_b
    ], ax[0, 2])
    ax[0, 2].set_ylim(None,
                      ytop + max([semVar_perTrials_m, semVar_perTrials_b]))

    avgVar_perTrials_m = np.mean(
        np.abs(np.diff(np.array(safes)[context == seq[1]])))
    avgVar_perTrials_b = np.mean(
        np.abs(np.diff(np.array(safes)[context == seq[0]])))
    semVar_perTrials_m = np.std(
        np.abs(np.diff(np.array(safes)[context == seq[1]])))
    semVar_perTrials_b = np.std(
        np.abs(np.diff(np.array(safes)[context == seq[0]])))
    t, p = stats.ttest_ind(
        np.abs(np.diff(np.array(safes)[context == seq[1]])),
        np.abs(np.diff(np.array(safes)[context == seq[0]])),
        equal_var=True)
    ax[0, 3].bar(
        ['mixed', 'blocked'], [avgVar_perTrials_m, avgVar_perTrials_b],
        yerr=[semVar_perTrials_m, semVar_perTrials_b],
        capsize=3,
        color=['red', 'blue'])
    ytop = label_diff(0, 1, 'p=' + str(p), [0, 1], [
        avgVar_perTrials_m + semVar_perTrials_m,
        avgVar_perTrials_b + semVar_perTrials_b
    ], ax[0, 3])
    ax[0, 3].set_ylim(None,
                      ytop + max([semVar_perTrials_m, semVar_perTrials_b]))

    # need to do this for safes also!
    #---------------------------------------------Safes
    bottom = 1000
    top = 1200
    fig, ax = plt.subplots(2, 1, squeeze=False)
    ax[1, 0].plot(
        range(bottom, top),
        np.array(gambles)[context == seq[1]][range(bottom, top)],
        color='red')
    ax[0, 0].plot(
        range(bottom, top),
        np.array(gambles)[context == seq[0]][range(bottom, top)],
        color='blue')

    # we then plot the average mean of bins of 24 trials (smallest block size)
    # this is where I should compute a bayesian prior as a reference

    mixed = np.array_split(
        np.array(gambles)[context == seq[1]],
        round(len(np.array(gambles)[context == seq[1]]) / window))
    blocked = np.array_split(
        np.array(gambles)[context == seq[0]],
        round(len(np.array(gambles)[context == seq[0]]) / window))

    print('Means of 24 trial bins')
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(15, 4))
    means = []
    for batch in blocked:
        means.extend([np.mean(batch)])
    ax[0, 0].plot(means, color='blue', alpha=0.5)
    ax[0, 0].set_title("Mean of past" + str(window) +
                       'gamble EVs across all trials')
    ax[0, 1].hist(means, bins=50, color='blue', alpha=0.5)
    ax[0, 1].set_title('Gamble EV means of 24-trial bins across all trials')
    means = []
    for batch in mixed:
        means.extend([np.mean(batch)])
    ax2 = ax[0, 0].twiny()
    ax[0, 0].set_xticklabels([])
    ax2.set_xticklabels([])
    ax2.plot(means, color='red', alpha=0.5)
    ax[0, 1].hist(means, bins=50, color='red', alpha=0.5)


#%%
def multiday_evLR(past_Trials, d_ind, window=10):
    '''
    '''
    import statsmodels.api as sm
    from scipy.special import logit
    from scipy import stats

    def SRC(params, X, Ypred, Yreal):
        return [(params[n] * (np.std(X[:, n]) * (np.std(Ypred) / np.std(Yreal)))
                 / np.std(logit(Ypred))) for n in range(1, len(params))]

    def plot_refRegression(X, Y, row=0, column=0, cols='k'):
        results = sm.GLM(Y, X, family=sm.families.Binomial()).fit()
        #        results = sm.Logit(Y, X).fit()
        betas = results.params
        success = results.converged
        Ps = results.pvalues
        err_y = results.summary2().tables[1].values[:, 1]

        betas = SRC(betas, X, results.fittedvalues, Y)
        ax[row, column].plot(range(0, len(betas)), betas, color=cols)
        ax[row, column].axhline(0, color='k')
        plt.xticks(range(0, len(betas)), ['gEV', 'sEV', 'past gEV'])
        gap = 0.05
        for significance, y, x, err in zip(Ps[1:], betas, range(0, len(betas)),
                                           err_y[1:]):
            if significance < 0.05 and y >= 0:
                ax[row, column].plot(x, y + gap, marker='*', color='k')
            elif significance < 0.05 and y < 0:
                ax[row, column].plot(x, y - gap, marker='*', color='k')

    # ----------------------------------------------
    # From the past trials set up the regression
    past_Trials = past_Trials.drop(d_ind)

    #ONLY TRIALS FROM ONE OR THE OTHER SEQUENCE TYPE
    pastTrials = past_Trials.loc[past_Trials.past_gEV.apply(
        lambda x: len(x) >= window)].copy()

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(7, 5))
    ii = 0
    for seqType, cols in zip(pastTrials.trialSequenceMode.unique(),
                             ['darkblue', 'darkred']):
        sTrials = pastTrials.loc[pastTrials.trialSequenceMode == seqType]
        Params = []
        pVals = []
        sParams = []
        for date in sTrials.sessionDate.unique():
            iTrials = sTrials.loc[sTrials.sessionDate == date]
            Y = []
            safe = []
            for i, row in iTrials.iterrows():
                if row.outcomesCount[0] > 1:
                    safe.extend([row.GB_ev])
                    if row.gambleChosen == 'A':
                        Y.extend([1])
                    else:
                        Y.extend([0])
                elif row.outcomesCount[1] > 1:
                    safe.extend([row.GA_ev])
                    if row.gambleChosen == 'B':
                        Y.extend([1])
                    else:
                        Y.extend([0])

            X = np.vstack(iTrials.past_gEV.apply(lambda x: x[0:window]).values)
            X = np.column_stack((np.array(iTrials.gEV.tolist()), safe, X))
            X = sm.tools.add_constant(
                X, prepend=True, has_constant='add')  #adds constant for the fit
            try:
                results = sm.GLM(Y, X, family=sm.families.Binomial()).fit()
            except:
                continue
            if not results.converged or len(np.unique(
                    iTrials.gEV.tolist())) < 3:
                continue
            if any(np.isnan(SRC(results.params, X, results.fittedvalues, Y))):
                continue
            Params.append(results.params)
            pVals.append(results.pvalues)
            sParams.append(SRC(results.params, X, results.fittedvalues, Y))

        yerr_s = stats.sem(sParams, axis=0)
        np.stack(Params).mean(axis=0)
        plt.errorbar(
            range(1, len(results.params)),
            np.stack(sParams).mean(axis=0),
            yerr=yerr_s,
            fmt='none',
            color='k',
            capsize=3)
        t, p = stats.ttest_1samp(sParams, 0.0, axis=0)
        plt.xticks(
            range(len(results.params)),
            ['intercept', 'gambleEV', 'safeEV', 'variance risk', 'gPos'])
        plt.axhline(0, color='k')
        gap = 0.2
        sigRank = 1
        for significance, y, x, err in zip(
                p,
                np.stack(sParams).mean(axis=0),
                range(1,
                      len(results.params) + 1),
                yerr_s):
            if significance < 0.05 and y >= 0:
                plt.plot(x, y + err + gap, marker='*', color='k')
                sigRank += 1
            elif significance < 0.05 and y < 0:
                plt.plot(x, y - err - gap, marker='*', color='k')
                sigRank += 1
        plt.plot(
            range(1, sigRank),
            np.stack(sParams).mean(axis=0)[0:sigRank - 1],
            color=cols,
            linewidth=3)
        plt.plot(
            range(sigRank - 1, len(results.params)),
            np.stack(sParams).mean(axis=0)[sigRank - 2:],
            color=cols,
            linewidth=1)

    return sParams


#%%
def LogitCompare(softmaxDF,
                 trialHistory,
                 iSessionStart,
                 mode='glm',
                 betaType='std',
                 plotFits=False):
    '''
    '''
    import statsmodels.api as sm
    from scipy import stats
    from scipy.special import logit
    from scipy import stats
    from tabulate import tabulate
    from macaque.f_Rfunctions import oneWay_rmAnova

    # ----------------------------------------

    f_BIC = lambda LL, nP, nT: (-2 * LL) + (nP * np.log(nT))
    f_AIC = lambda LL, nP: (-2 * LL) + (2 * nP)

    def get_binomialfit(X, Y, mode):
        AIC, BIC, LL = [], [], []
        if mode.lower() == 'logit':
            results = sm.Logit(Y, X).fit()
            if float(results.summary2().tables[0][1][6]):
                AIC = results.aic
                BIC = results.bic
                LL = results.llf
        if mode.lower() == 'glm':
            results = sm.GLM(Y, X, family=sm.families.Binomial()).fit()
            if results.converged:
                AIC = results.aic
                BIC = f_BIC(results.llf, len(X[0]), len(X))
                LL = results.llf
        return AIC, BIC, LL

    # ----------------------------------------------
    # From the past trials set up the regression
    trialHistory = trialHistory.drop(iSessionStart)
    #ONLY TRIALS FROM ONE OR THE OTHER SEQUENCE TYPE
    trialHistory = trialHistory.loc[trialHistory.past_gEV.apply(
        lambda x: len(x) >= 1)].copy()

    sTrials = trialHistory
    AICs = []
    BICs = []
    LLs = []
    Dates = []
    n = 0
    for date in sTrials.sessionDate.unique():
        iTrials = sTrials.loc[sTrials.sessionDate == date]
        gEV = []
        sEV = []
        gPos = []
        risk = []
        chG = []
        context = []
        LL = []
        AIC = []
        BIC = []

        for i, row in iTrials.iterrows():
            context.extend([row.trialSequenceMode])
            if row.outcomesCount[0] > 1:
                gEV.extend([row.GA_ev])
                sEV.extend([row.GB_ev])
                gPos.extend([1])
                risk.extend([np.sqrt(row.gambleA[1] * row.gambleA[3])])
                if row.gambleChosen == 'A':
                    chG.extend([1])
                else:
                    chG.extend([0])
            elif row.outcomesCount[1] > 1:
                gEV.extend([row.GB_ev])
                sEV.extend([row.GA_ev])
                gPos.extend([2])
                risk.extend([np.sqrt(row.gambleB[1] * row.gambleB[3])])
                if row.gambleChosen == 'B':
                    chG.extend([1])
                else:
                    chG.extend([0])

        if context != []:
            context = [0 if x < 9020 else 1 for x in context]

        #define X and Y for Binomial regression
        X = np.stack(
            (np.array(gEV), np.array(sEV), np.array(risk), np.array(gPos)),
            axis=1)
        X = sm.tools.add_constant(
            X, prepend=True, has_constant='add')  #adds constant for the fit
        Y = chG

        aic, bic, ll = get_binomialfit(X, Y, mode)
        if aic:
            AIC.extend([aic])
            BIC.extend([bic])
            LL.extend([ll])

        X = np.column_stack((X, iTrials.past_gEV.apply(lambda x: x[0]).values))
        aic, bic, ll = get_binomialfit(X, Y, mode)
        if aic:
            AIC.extend([aic])
            BIC.extend([bic])
            LL.extend([ll])

        X = np.stack(
            (np.array(gEV), np.array(sEV), np.array(gPos),
             iTrials.past_gEV.apply(lambda x: x[0]).values),
            axis=1)
        X = sm.tools.add_constant(
            X, prepend=True, has_constant='add')  #adds constant for the fit
        aic, bic, ll = get_binomialfit(X, Y, mode)
        if aic:
            AIC.extend([aic])
            BIC.extend([bic])
            LL.extend([ll])

        X = np.stack(
            (np.array(gEV), np.array(sEV),
             iTrials.past_gEV.apply(lambda x: x[0]).values),
            axis=1)
        X = sm.tools.add_constant(
            X, prepend=True, has_constant='add')  #adds constant for the fit
        aic, bic, ll = get_binomialfit(X, Y, mode)
        if aic:
            AIC.extend([aic])
            BIC.extend([bic])
            LL.extend([ll])

        if len(AIC) != 4:
            continue
        else:
            AICs.append(AIC)
            BICs.append(BIC)
            LLs.append(LL)
            Dates.extend([n])
            n += 1
    # ------------------------------------------------------------------------

    AICs = np.vstack(AICs)
    BICs = np.vstack(BICs)
    LLs = np.vstack(LLs)

    mAIC = np.mean(AICs, axis=0)
    seAIC = stats.sem(AICs, axis=0)
    mBIC = np.mean(BICs, axis=0)
    seBIC = stats.sem(BICs, axis=0)
    mLL = np.mean(LLs, axis=0)
    seLL = stats.sem(LLs, axis=0)

    models = [['gEV + sEV + risk + pos'], ['gEV + sEV + risk + pos + past'],
              ['gEV + sEV + past + pos'], ['gEV + sEV + past']]

    index = np.arange(3)
    bar_width = 0.2

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.hlines(0, -0.5, max(index) + bar_width * 4)
    for i, label in enumerate(models):
        ax.bar(
            [p + bar_width * i for p in index], [mAIC[i], mBIC[i], mLL[i]],
            bar_width,
            yerr=[seAIC[i], seBIC[i], seLL[i]],
            label=label)
    ax.legend()
    for x, y, s in zip(index, [mAIC[1], mBIC[1], mLL[1]], ['AIC', 'BIC', 'LL']):
        plt.text(
            x + bar_width * 1.5, 0 + -np.sign(y) * (y / y) * 50, s, ha='center')

    comparison = lambda x, y: '>' if x > y else '<'
    print('\n')
    print('AIC scores')
    #use the AIC comparison thing in the ETES paper
    print(
        tabulate([['N:', str(len(AICs)) + ' sessions', '', ''],
                  ['model:', 'risk', '', 'past + risk', '', 'past'], [
                      'mAIC', mAIC[0],
                      comparison(mAIC[0], mAIC[1]), mAIC[1],
                      comparison(mAIC[1], mAIC[2]), mAIC[2]
                  ], [
                      'mBIC', mBIC[0],
                      comparison(mBIC[0], mBIC[1]), mBIC[1],
                      comparison(mBIC[1], mBIC[2]), mBIC[2]
                  ], [
                      'mLL', mLL[0],
                      comparison(mLL[0], mLL[1]), mLL[1],
                      comparison(mLL[1], mLL[2]), mLL[2]
                  ]]))
    ax.axes.get_xaxis().set_visible(False)

    Dates = np.concatenate([[date] * len(models) for date in Dates])
    logisticModels = np.concatenate(models * len(AICs))

    oneWay_rmAnova(np.concatenate(AICs), Dates, logisticModels)


#%%
def plot_blockRegression(filteredDF, specific_G=None):
    '''
    '''
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    def plot_refScatter(X, Y, color='k', row=0, column=0, title=None):
        ax[row, column].scatter(X, Y, c=color)
        ax[row, column].axhline(0, color='k')
        ax[0, 0].axvline(0, color='k')
        ax[row, column].grid(b=True, which='major')
        #    ax[0,0].set_ylim(-0.2,0.2) #y axis length
        #        ax[row,column].set_xlim(-0.45,0.45) #y axis length
        x0, x1 = ax[row, column].get_xlim()
        y0, y1 = ax[row, column].get_ylim()
        ax[row, column].set_aspect((x1 - x0) / (y1 - y0))

        # Note the difference in argument order
        model = sm.OLS(Y, sm.add_constant(X)).fit()
        # Print out the statistics
        betas = model.params
        linReg = lambda x: betas[1] * x + betas[0]
        ax[row, column].plot(
            np.linspace(min(X), max(X)),
            linReg(np.linspace(min(X), max(X))),
            '--',
            color='k')

        ax[row, column].text(
            x1 - (x1 / 2.1),
            y1 - (y1 / 5),
            'R' + '=' + str(round(model.rsquared, 3)),
            style='italic',
            color='k',
            alpha=1)
        ax[row, column].text(
            x1 - (x1 / 2.1),
            y1 - (y1 / 5) * 1.75,
            'p' + '=' + str(round(model.f_pvalue, 3)),
            style='italic',
            color='k',
            alpha=1)

        if title != None:
            ax[row, column].set_title(title)  #this sets the subplot's title

    # -------------------------------------------------------------------------
    import statsmodels.api as sm
    from scipy.special import logit
    import matplotlib.cm as cm

    blockDF = filteredDF.loc[filteredDF.seqCode == 9001].sort_values(
        by=['sessionDate', 'division'])
    CEs = []
    EVs = []
    pEVs = []
    iKeep = []
    for date in blockDF.sessionDate.unique():
        firsts = []
        tt = [
            np.sort(np.concatenate(list(val.values()))) for val in blockDF.loc[
                blockDF.sessionDate == date].get('trial_index').values
        ]
        firsts.extend([0])
        for i in range(1, len(tt)):
            if tt[i][0] - tt[i - 1][-1] > 10:
                firsts.extend([i])
        firsts.extend([len(tt)])

        for n in range(0, len(firsts) - 1):
            iKeep.extend(blockDF.loc[blockDF.sessionDate == date].iloc[firsts[
                n]:firsts[n + 1]].index[1:])
            CEs.extend(blockDF.loc[blockDF.sessionDate == date].iloc[firsts[
                n]:firsts[n + 1]].equivalent[1:].values)
            EVs.extend(blockDF.loc[blockDF.sessionDate == date].iloc[firsts[
                n]:firsts[n + 1]].primaryEV[1:].values)
            pEVs.extend(blockDF.loc[blockDF.sessionDate == date].iloc[firsts[
                n]:firsts[n + 1]].primaryEV[0:-1].values)

    CEs = np.array(CEs)
    EVs = np.array(EVs)
    pEVs = np.array(pEVs)

    mpEVs = []
    mCEs = []
    for ii in iKeep:
        pairing = blockDF.loc[ii].primaryEV
        mpEVs.extend([np.mean(pEVs[EVs == pairing])])
        mCEs.extend(
            [
                np.mean(
                    blockDF.loc[blockDF.primaryEV == pairing].equivalent.values)
            ]
        )  #not the real avg CE - need to make sure I include 0.5s that aren't second

    pEVs = np.array(pEVs)
    EVs = np.array(EVs)
    CEs = np.array(CEs)
    mpEVs = np.array(mpEVs)
    mCEs = np.array(mCEs)

    if specific_G and type(specific_G) != list:
        pEVs = pEVs[EVs == specific_G]
        CEs = CEs[EVs == specific_G]
        mpEVs = mpEVs[EVs == specific_G]
        mCEs = mCEs[EVs == specific_G]
        EVs = EVs[EVs == specific_G]

    try:
        colors = cm.spring(np.linspace(0, 1, int(100 * max(EVs))))
        cols = [colors[int(ev * 100 - 1)] for ev in EVs]
    except:
        cols = 'k'

    fig, ax = plt.subplots(1, 4, squeeze=False, figsize=(20, 5))
    # --------------------------------------------------------------

    plot_refScatter(
        X=pEVs - EVs,
        Y=CEs - EVs,
        color=cols,
        row=0,
        column=0,
        title='pEVs-EVs / CEs-EVs')
    plot_refScatter(
        X=pEVs,
        Y=CEs - EVs,
        color=cols,
        row=0,
        column=1,
        title='pEVs / CEs-EVs')

    #need to remove the mean Ys
    plot_refScatter(
        X=pEVs - mpEVs,
        Y=CEs - mCEs,
        color=cols,
        row=0,
        column=2,
        title='pEVs-mpEVs / CEs-mCEs')
    plot_refScatter(
        X=(pEVs - mpEVs) - EVs,
        Y=CEs - mCEs,
        color=cols,
        row=0,
        column=3,
        title='(pEVs-mpEVs)-EVss / CEs-mCEs')

    fig, ax = plt.subplots(figsize=(0.2, 5))
    cmap = mpl.cm.spring
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='vertical',
        spacing='proportional')
    cb1.set_label('current block probability')
    fig.show()


#%%
