"""
Module for specific probability distortion functions
"""
import numpy as np
import pandas as pd
from macaque.f_toolbox import *
import matplotlib.pyplot as plt
import seaborn as sb
plt.style.use('seaborn-paper')
plt.rcParams['svg.fonttype'] = 'none'
sb.set_palette('colorblind')
#plt.rcParams['font.family'] = 'sans-serif'
tqdm = ipynb_tqdm()


#%%
def softmax_compare(filteredDF, Trials, gamble=0.5):
    from macaque.f_choices import get_options
    from macaque.f_psychometrics import get_softmaxData, plot_softmax
    import pandas as pd

    regTrials = np.sort(
        np.unique(
            np.concatenate([
                np.concatenate(list(val.values()))
                for val in filteredDF.get('trial_index').values
            ])))  #gets the index for all the trials I should use in regression
    sTrials = Trials.loc[regTrials].copy()

    equate = gamble / 2
    dfs = []
    for seqType, cols in zip(sTrials.trialSequenceMode.unique(),
                             ['darkblue', 'darkred']):
        iTrials = sTrials.loc[sTrials.trialSequenceMode == seqType]
        cData = get_options(iTrials, mergeBy='all')
        sm = get_softmaxData(cData, metricType='CE').sort_values('primaryEV')
        dfs.append(sm.loc[sm.primaryEV == equate])
    df = pd.concat(dfs, ignore_index=True)
    plot_softmax(
        df,
        sortBy='primaryEV',
        printRatios=False,
        plot_ci='resampling',
        color=['blue', 'red'])


#    x0,x1 = ax[0,4].get_xlim()
#    y0,y1 = ax[0,4].get_ylim()
#    ax[0,4].set_aspect((x1-x0)/(y1-y0))


#%%
def filter_elicitationSequences(softmaxDF, mode='PD', minProb=3):
    '''
    return only softmax sequences that follow 4 rules
    ---------
    - Gambles need to be 2-outcome
    - m1,m2 need to be 0 and 0.5 respectively
    - p needs to be 0.1:0.9
    - the CE needs to be between 0 and 0.5
    '''
    if mode.lower() == 'pd':
        #clean and filter for 2-outcome gambles of p=0.1:0.9
        filteredDF = softmaxDF.loc[softmaxDF.primary.apply(
            lambda x: len(x) <= 4)].copy()
        filteredDF = filteredDF.loc[filteredDF.primary.apply(
            lambda x: x[0] == 0 and x[-2] == 0.5)]
        filteredDF = filteredDF.loc[filteredDF.primaryEV.apply(lambda x: x in [0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45])]
        if 'equivalent' in filteredDF:
            filteredDF = filteredDF.loc[filteredDF.equivalent.apply(
                lambda x: 0 <= x <= 0.5)]
        #this is to specify that we are interested in specific sequences (in case they were mixed with other things)
#        filteredDF = filteredDF.loc[filteredDF.gList.apply(lambda x: x == "'PT_binaryList_allProbs4.txt'" or x == 'nan')]

        for date in filteredDF.sessionDate.unique():
            if len(
                    unique_listOfLists(
                        filteredDF.loc[filteredDF.sessionDate ==
                                       date].primary.tolist())) < minProb:
                filteredDF.drop(
                    filteredDF.loc[filteredDF.sessionDate == date].index,
                    inplace=True)

    print('Removing', str(len(softmaxDF) - len(filteredDF)),
          'CEs for Probability Distortion Analysis')
    return filteredDF.sort_values(['sessionDate', 'primaryEV'])


#%%
def plot_CEplot(softmaxDF):
    '''
    Plots a combined plot of blocked and mixed CEs, as well as cubic splines fitted to the average data.
    '''
    filteredDF = softmaxDF.copy()
    from scipy import interpolate
    
    filteredDF['diffValue'] = filteredDF['equivalent'] -  filteredDF['primaryEV']

    predicted_Ys = [[0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]]
    mean_Ys = [[0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]]
    shift = -0.01
    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(10, 5))
    for cc, col, context in zip(filteredDF.seqCode.unique(), ['blue', 'red'],
                                ['blocked', 'mixed']):
        #make the numbers the appropriate probabilities
        filteredDF.loc[
            filteredDF.seqCode == cc, ['primaryEV']] = filteredDF.loc[
                filteredDF.seqCode == cc].primaryEV.apply(lambda x: x / 0.5)
        ce_avg = []
        x = np.squeeze(
            filteredDF.sort_values('primaryEV').loc[filteredDF.seqCode == cc,
                                                    ['primaryEV']].values)
        y = np.squeeze(
            filteredDF.sort_values('primaryEV').loc[filteredDF.seqCode == cc,
                                                    ['equivalent']].values)
        for val in np.sort(
                filteredDF.loc[filteredDF.seqCode == cc].primaryEV.unique()):
            ce_avg.extend([
                np.mean(filteredDF.loc[filteredDF.seqCode == cc].loc[
                    filteredDF.loc[filteredDF.seqCode == cc].primaryEV == val]
                        .equivalent.values)
            ])
        mean, bound_upper, bound_lower, yHat = bootstrap_splineCE(
            x, y, method='resampling', n=10000)
        ax[0, 0].plot(np.linspace(0, 1, 100), mean, '-', color='dark' + col)
        ax[0, 1].plot(
            np.linspace(0, 1, 100), mean, '-', color=col, label=context)
        ax[0, 1].plot(
            np.linspace(0, 1, 100), bound_lower, '--', color=col, alpha=0.4)
        ax[0, 1].plot(
            np.linspace(0, 1, 100), bound_upper, '--', color=col, alpha=0.4)
        ax[0, 1].fill_between(
            np.linspace(0, 1, 100),
            bound_lower,
            bound_upper,
            color=col,
            alpha=0.10)
        ax[0, 1].plot(
            np.linspace(0, 1, 1000),
            np.linspace(0, 0.5, 1000),
            color='k',
            linestyle='--')
        ax[0, 1].set_ylim(-0.025, 0.525)  #y axis length
        ax[0, 1].set_xlim(0, 1.0)  #y axis length
        ax[0, 1].set_ylabel('equivalent')  #y axis length
        ax[0, 1].set_xlabel('primaryEV')  #y axis length
        ax[0, 1].grid(b=True, which='major')
        #this sets the aspect ratio to a square
        x0, x1 = ax[0, 1].get_xlim()
        y0, y1 = ax[0, 1].get_ylim()
        ax[0, 1].set_aspect((x1 - x0) / (y1 - y0))
        #        ax[0,1].scatter(np.array([0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45])*2, yHat[1:-1], s=75 )
        ax[0, 1].scatter(
            np.array([0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]) * 2,
            ce_avg,
            s=75)

        #Plot CEs between conditions (same day blocked vs mixed)
        # offset both contexts
        filteredDF.loc[
            filteredDF.seqCode == cc, ['primaryEV']] = filteredDF.loc[
                filteredDF.seqCode == cc].primaryEV.apply(lambda x: x + shift)
        filteredDF.loc[filteredDF.seqCode == cc].plot.scatter(
            x='primaryEV',
            y='equivalent',
            color='dark' + col,
            label=context,
            ax=ax[0, 0])
        ax[0, 0].plot(
            np.linspace(0, 1, 1000),
            np.linspace(0, 0.5, 1000),
            color='k',
            linestyle='--')
        ax[0, 0].set_ylim(-0.025, 0.525)  #y axis length
        ax[0, 0].set_xlim(0, 1.0)  #y axis length
        ax[0, 0].grid(b=True, which='major')
        #this sets the aspect ratio to a square
        x0, x1 = ax[0, 0].get_xlim()
        y0, y1 = ax[0, 0].get_ylim()
        ax[0, 0].set_aspect((x1 - x0) / (y1 - y0))
        shift = -shift

        predicted_Ys.append(yHat[1:-1])
        mean_Ys.append(ce_avg)
        
        ax[0, 2].plot(np.linspace(0, 1, 100), mean - np.linspace(0, 0.5, 100), '-', color='dark' + col)        
        filteredDF.loc[filteredDF.seqCode == cc].plot.scatter(
            x='primaryEV',
            y='diffValue',
            color='dark' + col,
            label=context,
            ax=ax[0, 2])
        ax[0, 2].plot(
            np.linspace(0, 1, 100),
            np.linspace(0, 0, 100),
            color='k',
            linestyle='--')
#         ax[0, 2].set_ylim(-0.025, 0.525)  #y axis length
#         ax[0, 2].set_xlim(0, 1.0)  #y axis length
        ax[0, 2].grid(b=True, which='major')
        #this sets the aspect ratio to a square
        x0, x1 = ax[0, 2].get_xlim()
        y0, y1 = ax[0, 2].get_ylim()
        ax[0, 2].set_aspect((x1 - x0) / (y1 - y0))
        shift = -shift

        predicted_Ys.append(yHat[1:-1])
        mean_Ys.append(ce_avg)
        
    plt.tight_layout()

    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    formula = 'equivalent ~ primaryEV + C(seqCode) + C(seqCode):primaryEV'
    model = ols(formula, data=softmaxDF).fit()
    aov_table = anova_lm(model, typ=2)

    eta_squared(aov_table)
    #    omega_squared(aov_table)
    print(aov_table)
    print('Nb,Nm =', [
        str(len(filteredDF.loc[filteredDF.seqCode == cc]))
        for cc in filteredDF.seqCode.unique()
    ])

    return predicted_Ys, mean_Ys


#%%
#plot the bootstrapped confidence of the spline fit!
def bootstrap_splineCE(xData, yData, method='residuals', n=10000):
    '''
    '''
    from scipy import interpolate

    #-----------------------------------
    def get_CEmean(data):
        x = []
        y = []
        data = np.squeeze(data)
        for Xs in np.unique(data[:, 0]):
            indexer = data[:, 0] == Xs
            x.extend([Xs])
            y.extend([np.mean(data[:, 1][indexer])])
        return np.concatenate([[0], x, [1]]), np.concatenate([[0], y, [0.5]])

    #----------------------------------------------------------
    data = np.array(np.split(np.array([xData, yData]), len(xData), axis=1))
    xMean, yMean = get_CEmean(data)
    xFull = np.linspace(min(xMean), max(xMean), 100)
    fun = interpolate.LSQUnivariateSpline(xMean, yMean, t=[0.5])
    yHat = fun(xMean)
    resid = yHat - yMean

    b1 = []
    for i in range(0, n):
        if method.lower() == 'residuals':
            residBoot = np.random.permutation(resid)
            booty = yHat + residBoot
            bootFun = interpolate.LSQUnivariateSpline(
                xMean, booty, t=[0.5])  #this give same result as matlab softmax
        elif method.lower() == 'resampling':
            xb = np.random.choice(range(len(data)), len(data), replace=True)
            bootSample = np.hstack(data[xb])
            bootSample = bootSample[:, np.argsort(bootSample[0])]
            bootSample = np.array(np.split(bootSample, len(data), axis=1))
            bootx, booty = get_CEmean(bootSample)
            try:
                bootFun = interpolate.LSQUnivariateSpline(
                    bootx, booty,
                    t=[0.5])  #this give same result as matlab softmax
            except:
                continue
        b1.append(bootFun(xFull))

    b1 = np.vstack(b1)
    mean = fun(xFull)
    upper, lower = np.percentile(b1, [5, 95], axis=0)
    return mean, upper, lower, yHat


#%%
def gambleLogit(filteredDF, Trials, mode='glm', betaType='std', plotFits=False):
    '''
    '''
    import statsmodels.api as sm
    from scipy import stats
    stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
    from statsmodels.graphics.api import abline_plot
    from scipy.special import logit

    # ------------------------------------------------------
    def standardizeBetas(params, X, Y):
        '''
        Each regression coefficient was standardized by multiplying the
        raw re- gression coefficient with the ratio of the SD of the
        independent variable corresponding to the coefficient and the SD
        of the dependent variable.
        '''
        return [
            params[n] * (np.std(X[:, n]) / np.std(Y))
            for n in range(1, len(params))
        ]

    # ------------------------------------------------------
    def SRC(params, X, Ypred, Yreal):
        return [(params[n] * (np.std(X[:, n]) * (np.std(Ypred) / np.std(Yreal)))
                 / np.std(logit(Ypred))) for n in range(1, len(params))]

    # -----------------------------------------------

    Params = []
    pVals = []
    sParams = []
    zParams = []
    full_sParams = []
    for date in filteredDF.sessionDate.unique():
        #        if filteredDF.loc[filteredDF.sessionDate == date].seqCode.unique() == np.array([9001]):
        #            continue
        df = filteredDF.loc[filteredDF.sessionDate == date]
        regTrials = np.sort(
            np.unique(
                np.concatenate([
                    np.concatenate(list(val.values()))
                    for val in df.get('trial_index').values
                ]))
        )  #gets the index for all the trials I should use in regression
        sTrials = Trials.loc[regTrials].copy()

        gEV = []
        sEV = []
        gPos = []
        risk = []
        chG = []
        context = []
        for i, row in sTrials.iterrows():
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
        #        GR_int = np.squeeze(GR_int)
        #        X = np.stack((np.array(gEV), np.array(GB_ev), np.array(GR_int), np.array(gPos)), axis=1)
        Y = chG
        X = sm.tools.add_constant(
            X, prepend=True, has_constant='add')  #adds constant for the fit

        if mode.lower() == 'logit':
            results = sm.Logit(Y, X).fit()
            betas = results.params
            success = float(results.summary2().tables[0][1][6])
        if mode.lower() == 'glm':
            #        GLM version of logit regression
            results = sm.GLM(Y, X, family=sm.families.Binomial()).fit()
            betas = results.params
            success = results.converged

        if not success or np.isnan(np.std(logit(results.fittedvalues))):
            print(str(date), 'did not converge')
            continue

#        np.std(logit(results.fittedvalues)):
#            continue
        Params.append(betas)
        pVals.append(results.pvalues)
        sParams.append(standardizeBetas(betas, X,
                                        Y))  #standardize like Bill and Armin
        full_sParams.append(SRC(betas, X, results.fittedvalues, Y))
        zParams.append(results.summary2().tables[1].z.values)

        if plotFits:
            #plot goodness of fit!!!
            nobs = res.nobs
            y = data.endog[:, 0] / data.endog.sum(1)
            yhat = res.mu

            fig, ax = plt.subplots()
            ax.scatter(yhat, y)
            line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
            abline_plot(model_results=line_fit, ax=ax)

            #        plot residual qqplot
            from scipy import stats
            resid = res.resid_deviance.copy()
            resid_std = stats.zscore(resid)
            from statsmodels import graphics
            graphics.gofplots.qqplot(resid, line='r')

    print(np.stack(full_sParams).mean(axis=0))
    #implement a betaType variable that chooses between the z and the s-coefficients
    yerr = stats.sem(zParams, axis=0)
    yerr_s = stats.sem(sParams, axis=0)
    yerr_full = stats.sem(full_sParams, axis=0)
    #sort out the plotting for the parameters
    np.stack(Params).mean(axis=0)
    plt.bar(range(1, len(results.params)), np.stack(full_sParams).mean(axis=0))
    plt.errorbar(
        range(1, len(results.params)),
        np.stack(full_sParams).mean(axis=0),
        yerr=yerr_full,
        fmt='none',
        color='k',
        capsize=3)
    t, p = stats.ttest_1samp(full_sParams, 0.0, axis=0)
    plt.xticks(
        range(len(results.params)),
        ['intercept', 'gambleEV', 'safeEV', 'variance risk', 'gPos'])
    plt.axhline(0, color='k')
    gap = 0.2
    for significance, y, x, err in zip(
            p,
            np.stack(full_sParams).mean(axis=0),
            range(1,
                  len(results.params) + 1),
            yerr_full):
        if significance < 0.05 and y >= 0:
            plt.plot(x, y + err + gap, marker='*', color='k')
        elif significance < 0.05 and y < 0:
            plt.plot(x, y - err - gap, marker='*', color='k')

    tt, pp = stats.ttest_1samp(zParams, 0.0, axis=0)
    from tabulate import tabulate
    print(
        tabulate([['N:', str(len(zParams))], [
            'full_stdParams',
            str(np.stack(full_sParams).mean(axis=0))
        ], ['t-values:', str(t)], ['p-values:', str(p)],
                  ['nz-scored Params',
                   str(np.stack(zParams).mean(axis=0))],
                  ['p-values (z):', str(pp)]]))
    #                    ['standardized Params',  str(np.stack(sParams).mean(axis=0))]]))
    return Params, pVals, sParams, zParams, full_sParams


#%%
def fit_likelihoodModel(filteredDF,
                        Trials,
                        uModel='power',
                        wModel='1-prelec',
                        use_matlab=False,
                        plotit=True,
                        plotFitting=False,
                        plotTQDM=True,
                        getError = False):
    '''
    Nelder-mead search algorithm to simultaneously fit utility, probability, discrete choice, and/or reference curves via maximum loglikelihood fit
    1st) define the U and W function
    2nd) calculate the softmax for every gamble-safe pairing
    3rd) merge all these into the LL functio
    '''
    import numpy as np
    import scipy.optimize as opt
    import pandas as pd
    from scipy.io import savemat
    import time

    np.warnings.filterwarnings('ignore')
    rew_range = unique_listOfLists(
        filteredDF.primary.apply(
            lambda x: [x[0], x[2]]))  # looking for m0 and m1
    mLow = np.min(rew_range)
    mMax = np.max(rew_range)

    #Utility function
    #----------------------------------------
    if uModel.lower() == 'power':
        U = lambda mm, param_u: ((mm - mLow) / (mMax - mLow))**param_u
        p_u = 1
    elif uModel.lower() == 'prelec':
        U = lambda m, param_u: np.exp(-p1 * (-np.log((mm - mLow) / (mMax - mLow)))**param_u)  #prelec
        p_u = 1
    elif uModel.lower() == 'cdf':
        U = lambda x: x  #SORT THIS OUT
        p_u = 1
    elif uModel.lower() == 'loss-prelec':
        U = lambda m, param_u : np.exp(-param_u*(-np.log((mm-mLow)/(mMax-mLow)))**p1p)*int(param_u>0) + np.exp(param_u*(-np.log((mm-mLow)/(mMax-mLow)))**p1n)*int(param_u<0) # [prelec, 2 different p1 for positive or negative p2]
        p_u = 1

    #Probability Distortion
    #----------------------------------------
    if wModel.lower() == '1-prelec':
        W = lambda p, param_w: np.exp(-((-np.log(p))**param_w))
        #prelec 1 param
        p_w = 1
    elif wModel.lower() == '2-prelec':
        W = lambda p, param_w: np.exp(-param_w[1] * (-np.log(p))**param_w[0])
        #prelec 2 params
        p_w = 2
    elif wModel.lower() == 'gonzalez':
        W = lambda p, param_w: (param_w[1] * p**param_w[0]) / (param_w[1] * p**param_w[0] + (1 - p)**param_w[0])
        #gonzalez and wu
        p_w = 2
    elif wModel.lower() == 'tversky':
        W = lambda p, param_w: p**np.array(param_w) / (p**np.array(param_w) + (1 - p)**np.array(param_w))**(1 / np.array(param_w))
        #KandT 1992
        p_w = 1

    #discrete choice
    # -----------------------------------------------------
    def func_Pg(U, W, gamble, safes):
        m1 = gamble[:, 0::2][:, 0]
        m2 = gamble[:, 0::2][:, 1]
        p1 = gamble[:, 1::2][:, 0]
        p2 = gamble[:, 1::2][:, 1]
        Pg = lambda params : 1 / (1 + np.exp( -params[0] * ( (W(p1,params[2:])*U(m1,params[1]) + W(p2,params[2:])*U(m2,params[1])) - U(safes,params[1]))  )  )
        p_c = 1
        return Pg, p_c

    #-------------------------------------------------------------

    def calc_lsq(gambles, func_Pg, U, W, df, result, mLow=0, mMax=0.5):
        probs = [pp[3] for pp in unique_listOfLists(np.squeeze(gambles))]
        mm = np.linspace(mLow, mMax, num=1000)
        CE = []
        Ps = []
        for pp in probs:
            Pg, _ = func_Pg(
                U, W, np.vstack([np.array([0, 1 - pp, 0.5, pp]) for m in mm]),
                mm)
            CE.extend(
                mm[np.abs(Pg(result) - 0.5) == min(np.abs(Pg(result) - 0.5))])
            Ps.extend([pp / 2])

        LSQ = []
        selected = df.primaryEV.values
        #        LSQ = []; selected = filteredDF.loc[filteredDF.sessionDate == day].primaryEV.values
        for pp, ce in zip(Ps, CE):
            LSQ.extend(ce - df.loc[selected == pp].equivalent.values)
        LSQ = sum(np.array(LSQ)**2)
        return LSQ

    #-----------------------------------------------
    if 'sessionDate' not in filteredDF.columns:
        filteredDF['sessionDate'] = 1
        filteredDF['seqCode'] = 9001

    dList = []
    if use_matlab:
        import matlab.engine
        eng = matlab.engine.start_matlab()

    regTrials = np.sort(
        np.unique(
            np.concatenate([
                np.concatenate(list(val.values()))
                for val in filteredDF.get('trial_index').values
            ])))  #gets the index for all the trials I should use in regression
    sTrials = Trials.loc[regTrials].copy(
    )  #get only trials that were used in the rest of the analysis

    for day in tqdm(filteredDF.sessionDate.unique()):
        if len(filteredDF.sessionDate.unique()) == 1:
            trials = sTrials
        else:
            trials = sTrials.loc[sTrials['sessionDate'] == day]
        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]
        safes = np.array(safes)[:, 0]
        gambles = np.vstack(gambles)

        gambleSide = [1 if side[0] == 2 else 2 for side in trials.outcomesCount]
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        context = trials.trialSequenceMode.unique()
        nTrials = len(trials)

        Pg, p_c = func_Pg(U, W, gambles, safes)
        LL = lambda params: sum(chG * np.log(Pg(params))) + sum(chS * np.log(1 - Pg(params)))
        sum_neg_LL = lambda params: -(LL(params))
        x0 = np.ones(p_c + p_u + p_w)

        if use_matlab:
            print(day)
            matStruct = {
                'gmb': gambles,
                'safes': safes,
                'choiceS': chS,
                'choiceG': chG
            }
            savemat('matStruct', matStruct)
            time.sleep(2)
            try:
                matlabResults = eng.choiceData_recoverUW()
            except:
                import pdb
                pdb.set_trace()  #AWESOME WAY TO DEBUG
            params = np.squeeze(matlabResults['pEstimates'])
            fval = np.squeeze(matlabResults['fval'])
            nobs = np.squeeze(matlabResults['nObs'])
            lsq = calc_lsq(
                gambles,
                func_Pg,
                U,
                W,
                filteredDF.loc[filteredDF.sessionDate == day],
                params,
                mLow=mLow,
                mMax=mMax)
            dList.append({
                'date': day,
                'nTrials': sum(nTrials),
                'n_obs': nobs,
                'params': params,
                'NM_success': True,
                'functions': [Pg, U, W],
                'LL': fval,
                'context': np.unique(context)[-1],
                'gambles': gambles,
                'safes': safes,
                'chS': chS,
                'chG': chG,
                'm_range': [mLow, mMax],
                'uFunction': uModel,
                'wFunction': wModel,
                'lsq': lsq
            })
        else:
            #            results = opt.minimize(sum_neg_LL, x0, method='Nelder-Mead', options = {'disp' : False, 'maxfev' : 1e5})
            res_x = []
            results = opt.minimize(
                sum_neg_LL,
                x0,
                method='Nelder-Mead',
                callback=res_x.append,
                options={
                    'maxiter': 1000,
                    'maxfev': 1000
                })
#            import pdb ; pdb.set_trace() #AWESOME WAY TO DEBUG
            Nfeval = [sum_neg_LL(Xs) for Xs in res_x]
            #this doesn't work
            lsq = calc_lsq(
                gambles,
                func_Pg,
                U,
                W,
                filteredDF.loc[filteredDF.sessionDate == day],
                results.x,
                mLow=0,
                mMax=0.5)

            booties = []
            if getError:
                for n in range(2000):
                    xb = np.random.choice(range(len(gambles)), len(gambles), replace=True)
                    Pg_b, p_c = func_Pg(U, W, gambles[xb], safes[xb])
                    LL_b = lambda params: sum(chG[xb] * np.log(Pg_b(params))) + sum(chS[xb] * np.log(1 - Pg_b(params)))
                    sum_neg_LL_b = lambda params: -(LL_b(params))
                    res = opt.minimize(sum_neg_LL_b,x0, method='Nelder-Mead')
                    booties.append(res.x)

            dList.append({
                'date': day,
                'nTrials': nTrials,
                'n_obs': len(safes),
                'params': results.x,
                'NM_success': results.success,
                'functions': [Pg, U, W],
                'LL': sum_neg_LL(results.x),
                'context': np.unique(context)[-1],
                'gambles': [gambles],
                'safes': safes,
                'chS': chS,
                'chG': chG,
                'm_range': [mLow, mMax],
                'uFunction': uModel,
                'wFunction': wModel,
                'lsq': lsq,
                'Nfeval': Nfeval,
                'all_fits': res_x,
                'bootstrap': booties
            })

    if use_matlab:
        eng.quit()
    NM_fit = pd.DataFrame(dList)

    if plotFitting:
        plot_MLEfitting(NM_fit)

    if plotit:
        NM_fit = plot_MLE(NM_fit, Trials, filteredDF, U, W, func_Pg)
#    else:
#        NM_fit = NM_fit.drop(NM_fit.loc[NM_fit.NM_success == False].index) #removes fits that were unsuccessful
    cols = [
        'date', 'nTrials', 'n_obs', 'params', 'NM_success', 'functions', 'LL',
        'lsq', 'context', 'uFunction', 'wFunction', 'gambles', 'safes', 'chS',
        'chG', 'm_range', 'bootstrap'
    ]
    return NM_fit[cols]


#%%
def plot_MLEfitting(NM_fit, plotFittings=True):
    '''
    '''
    import itertools
    
    if plotFittings:
        fig, ax = plt.subplots(
            int(np.ceil(len(NM_fit) / 6)),
            6,
            squeeze=False,
            figsize=(2.5 * int(np.ceil(len(NM_fit) / 10)), 15))
        fig.suptitle(
            'Function Value Estimates (and parameters) during minimization')
        row = 0
        col = 0
        colors = itertools.cycle(sb.color_palette('bright'))
        for n in range(10 * int(np.ceil(len(NM_fit) / 10))):
            if col == 0:
                ax[row, col].set_ylabel('- LL')
            if row == int(np.ceil(len(NM_fit) / 6)) - 1:
                ax[row, col].set_xlabel('runs')
            if n >= len(NM_fit):
                ax[row, col].set_visible(False)
                col += 1
                continue
            ax[row, col].plot(NM_fit.iloc[n]['Nfeval'], linewidth=3, color='k')
            ax[row, col].set_title(str(NM_fit.loc[n]['date']))

            for dd in range(len(NM_fit.loc[n]['params'])):
                cc = next(colors)
                ax2 = ax[row, col].twinx()
                ax2.spines["right"].set_position(("axes", 1 + 0.1 * dd))
                ax2.spines["right"].set_visible(True)

                ax2.plot(
                    np.vstack(NM_fit.loc[n].all_fits)[:, dd], color=cc, alpha=0.4)
                if col == 5:
                    ax2.set_ylabel('param %d' % dd, color=cc)
                ax2.tick_params('y', colors=cc)

            col += 1
            if col > 5:
                row += 1
                col = 0
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    #plot the evolution of the LL and of the parameters

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 4))
    if 'wFunction' in NM_fit.columns:
        plt.suptitle(NM_fit.wFunction.values[0], fontsize=16)
    else:
        plt.suptitle(NM_fit.model_used.values[0], fontsize=16)

    ax[0, 0].plot(NM_fit.LL.values, 'k', linewidth=2)
    plt.grid()
    ax[0, 0].set_ylabel('Negative Log Likelihood')
    #    ax[0,0].xaxis.set_minor_locator(mdates.DayLocator())
    ax[0, 0].set_xticks(range(0, len(NM_fit)))
    ax[0, 0].set_xticklabels(
        NM_fit.date.values, rotation=90, horizontalalignment='center')

    colors = itertools.cycle(sb.color_palette('bright'))
    for dd in range(len(NM_fit.params.values[0])):
        cc = next(colors)
        ax2 = ax[0, 0].twinx()
        ax2.spines["right"].set_position(("axes", 1 + 0.055 * dd))
        ax2.spines["right"].set_visible(True)

        ax2.plot(np.vstack(NM_fit.params.values)[:, dd], color=cc, alpha=0.4)
        ax2.set_ylabel('param %d' % dd, color=cc)
        ax2.tick_params('y', colors=cc)
    plt.show()


#%%
def plot_MLE(NM_fit, Trials, filteredDF, U, W, func_Pg):
    from scipy import stats
    from macaque.f_choices import get_options
    from macaque.f_psychometrics import get_softmaxData
    from macaque.f_Rfunctions import dv3_manova

    softmaxDF = filteredDF.copy()
    print(
        '\n Preparing figures: Parameters Scatter, Bar, Utility, Probability Distortion, Softmax + avgChoices'
    )
    NM_fit_sorted = NM_fit.drop(NM_fit.loc[NM_fit.NM_success == False].index)
    #    import pdb ; pdb.set_trace() #AWESOME WAY TO DEBUG
    x0 = NM_fit_sorted.params.values[0]
    mLow, mMax = NM_fit_sorted.m_range.values[0]
    fig, ax = plt.subplots(1, 5, squeeze=False, figsize=(15, 4))
    fig.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    plt.suptitle(NM_fit_sorted.wFunction.values[0], fontsize=16)

    mean_Ys = []
    for cc, col, context in zip(softmaxDF.seqCode.unique(), ['blue', 'red'],
                                ['blocked', 'mixed']):
        #make the numbers the appropriate probabilities
        softmaxDF.loc[softmaxDF.seqCode == cc, ['primaryEV']] = softmaxDF.loc[
            softmaxDF.seqCode == cc].primaryEV.apply(lambda x: x / 0.5)
        ce_avg = []
        x = np.squeeze(
            softmaxDF.sort_values('primaryEV').loc[softmaxDF.seqCode == cc,
                                                   ['primaryEV']].values)
        y = np.squeeze(
            softmaxDF.sort_values('primaryEV').loc[softmaxDF.seqCode == cc,
                                                   ['equivalent']].values)
        for val in np.sort(
                softmaxDF.loc[softmaxDF.seqCode == cc].primaryEV.unique()):
            ce_avg.extend([
                np.mean(softmaxDF.loc[softmaxDF.seqCode == cc].loc[
                    softmaxDF.loc[softmaxDF.seqCode == cc].primaryEV == val]
                        .equivalent.values)
            ])
        mean_Ys.append(ce_avg)

    #First Subplot
    pC = []
    pU = []
    pW = []
    width = 0.35
    Xs = np.array(range(len(x0)))
    ax[0, 1].axhline(0, color='k')
    for cc, col in zip(NM_fit.context.unique(), ['blue', 'red']):
        width = -(width)
        pC.append(
            np.log(
                np.vstack(NM_fit_sorted.loc[NM_fit_sorted.context == cc]
                          .params.values)[:, 0]))
        pU.append(
            np.log(
                np.vstack(NM_fit_sorted.loc[NM_fit_sorted.context == cc]
                          .params.values)[:, 1]))
        pW.append(
            np.log(
                np.vstack(NM_fit_sorted.loc[NM_fit_sorted.context == cc]
                          .params.values)[:, 2:]))
        yerr = stats.sem(
            np.log(
                np.vstack(NM_fit_sorted.loc[NM_fit_sorted.context == cc]
                          .params.values)),
            axis=0)

        #sort out the plotting for the parameters
        ax[0, 1].bar(
            Xs + width / 2,
            np.log(
                np.vstack(NM_fit_sorted.loc[NM_fit_sorted.context == cc]
                          .params.values)).mean(axis=0),
            width,
            yerr=yerr,
            capsize=3,
            label=str(cc),
            color=col)
        plt.sca(ax[0, 1])
        plt.xticks(Xs + width / 2, ['softmax', 'utility', 'probability'])

    _, sigC = stats.ttest_ind(pC[0], pC[1], equal_var=True)
    _, sigU = stats.ttest_ind(pU[0], pU[1], equal_var=True)
    _, sigW = stats.ttest_ind(pW[0], pW[1], equal_var=True)

    avgSTD = np.std(np.log(np.vstack(NM_fit_sorted.params.values)), axis=0)
    effectSize = np.round((np.mean(np.log(np.vstack(NM_fit_sorted.loc[NM_fit_sorted.context == NM_fit.context.unique()[0]].params.values)), axis=0) - \
                  np.mean(np.log(np.vstack(NM_fit_sorted.loc[NM_fit_sorted.context == NM_fit.context.unique()[1]].params.values)), axis=0)) / \
                  avgSTD, 4)

    col_labels = ['lambda', 'rho', 'alpha']
    if len(Xs) == 4:
        col_labels = ['lambda', 'rho', 'alpha', 'beta']
    row_labels = ['effect size']
    table_vals = [str(s) for s in effectSize]

    fig.subplots_adjust(bottom=0.25)
    plt.table(
        cellText=[table_vals],
        colLabels=col_labels,
        rowLabels=['$\it{d}$' + ':'],
        cellLoc='center',
        bbox=[0, -0.27, 1, 0.15])

    for p, x in zip([sigC, sigU, sigW], Xs):
        if isinstance(p, np.ndarray):
            for pp in p:
                if pp < 0.05:
                    ax[0, 1].scatter(
                        x,
                        np.amax(
                            np.log(np.vstack(
                                NM_fit_sorted.params.values))[:, x]),
                        marker='*',
                        color='k')
                    x += 1
        else:
            if p < 0.05:
                ax[0, 1].scatter(
                    x,
                    np.amax(
                        np.log(np.vstack(NM_fit_sorted.params.values))[:, x]),
                    marker='*',
                    color='k')
    #----------------------------------------------

    #MANOVA on Parameters
    if len(Xs) <= 3:
        print(
            '-------------------------------------------------------------------------------'
        )
        print('IV = context, factor0 = pC, factor1 = pU, factor2 = pW')
        dv3_manova(
            np.concatenate(pC),
            np.concatenate(pU),
            np.concatenate(flatten(pW)),
            IV=NM_fit.context.values)
#    else:
#        dv3_manova(np.concatenate(pC), np.concatenate(pU), np.concatenate(pW[0][:]), IV = NM_fit.context.values )

#----------------------------------------------
    fig_Long, ax_Long = plt.subplots(1, 9, squeeze=False, figsize=(15, 4))
    regTrials = np.sort(
        np.unique(
            np.concatenate([
                np.concatenate(list(val.values()))
                for val in softmaxDF.get('trial_index').values
            ])))  #gets the index for all the trials I should use in regression
    sTrials = Trials.loc[regTrials].copy()

    for n, col in zip(range(0, len(NM_fit.context.unique())), ['blue', 'red']):
        if len(Xs) <= 3:
            #This plots the u/w parameter scatterplot
            ax[0, 0].axhline(0, linestyle='--', color='k')
            ax[0, 0].axvline(0, linestyle='--', color='k')
            ax[0, 0].scatter(pU[n], pW[n], color=col)
            ax[0, 0].set_ylim(-1.05, 1.05)  #y axis length
            ax[0, 0].set_xlim(-1.05, 1.05)  #y axis length
            ax[0, 0].grid(b=True, which='major')

            #draw covariance or confidence interval ellipse
            covarianceEllipse(pU[n], pW[n], ax[0, 0], color=col, draw='CI')

            #this sets the aspect ratio to a square
            x0, x1 = ax[0, 0].get_xlim()
            y0, y1 = ax[0, 0].get_ylim()
            ax[0, 0].set_aspect((x1 - x0) / (y1 - y0))

            gg = np.linspace(0, 1)
            ax[0, 3].plot(
                gg, [W(g, np.mean(np.exp(pW[n]))) for g in gg],
                color=col)  #sort out the plotting for these
            ax[0, 3].plot(gg, gg, color='k', linestyle='--')
            x0, x1 = ax[0, 3].get_xlim()
            y0, y1 = ax[0, 3].get_ylim()
            ax[0, 3].set_aspect((x1 - x0) / (y1 - y0))
            ax[0, 3].text(
                0.6,
                0.2 - (n * 0.1),
                'α' + '=' + str(round(np.mean(np.exp(pW[n])), 4)),
                style='italic',
                color=col,
                alpha=1)

            mm = np.linspace(mLow, mMax, num=1000)
            CE = []
            xVals = np.array(
                [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]) * 2
            for pp in np.linspace(0, 1, num=1000):
                Pg, _ = func_Pg(
                    U, W,
                    np.vstack([np.array([0, 1 - pp, 0.5, pp]) for m in mm]), mm)
                CE.extend(mm[np.abs(Pg([np.mean(np.exp(pC[n])), np.mean(np.exp(pU[n])), np.mean(np.exp(pW[n]))]) -0.5)\
                             == min(np.abs(Pg([np.mean(np.exp(pC[n])), np.mean(np.exp(pU[n])), np.mean(np.exp(pW[n]))]) -0.5))])
            ax[0, 4].plot(
                np.array([0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
                * 2,
                mean_Ys[n],
                'bo ',
                color='dark' + col)
            ax[0, 4].plot(
                np.linspace(0, 1, num=1000), CE, color=col, linestyle='--')

            iTrials = sTrials.loc[sTrials.trialSequenceMode ==
                                  NM_fit.context.unique()[n]]
            cData = get_options(iTrials, mergeBy='all')
            sm = get_softmaxData(
                cData, metricType='CE').sort_values('primaryEV')
            sp = 0
            CEs = []
            for pp in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
                Pg, _ = func_Pg(
                    U, W,
                    np.vstack([np.array([0, 1 - pp, 0.5, pp]) for m in mm]), mm)
                ax_Long[0, sp].plot(
                    mm,
                    1 - Pg([
                        np.mean(np.exp(pC[n])),
                        np.mean(np.exp(pU[n])),
                        np.mean(np.exp(pW[n]))
                    ]),
                    color=col)
                ax_Long[0, sp].text(
                    0.025,
                    0.95 - (n * 0.1),
                    'λ' + '=' + str(round(np.mean(np.exp(pC[n])), 4)),
                    style='italic',
                    color=col,
                    alpha=1)
                ax_Long[0, sp].axhline(0.5, linestyle='-', color='k', alpha=0.5)
                chG = Pg([
                    np.mean(np.exp(pC[n])),
                    np.mean(np.exp(pU[n])),
                    np.mean(np.exp(pW[n]))
                ])
                ax_Long[0,sp].axvline( mm[np.abs(chG -0.5)\
                             == min(np.abs(chG -0.5))],
                            linestyle = '--', color=col, alpha = 0.5)

                row = sm.loc[sm.primaryEV == pp / 2]
                x_mag = [m[0] for m in row.secondary.tolist()[0]]
                chX_freq = np.squeeze(row.freq_sCh.tolist(
                ))  #need the frequency of picking safe, not gamble
                ax_Long[0, sp].plot(x_mag, chX_freq, 'bo ', color='dark' + col)
                Pg, _ = func_Pg(
                    U, W,
                    np.vstack([np.array([0, 1 - pp, 0.5, pp]) for m in x_mag]),
                    x_mag)
                ax_Long[0, sp].plot(
                    x_mag,
                    chX_freq - (1 - Pg([
                        np.mean(np.exp(pC[n])),
                        np.mean(np.exp(pU[n])),
                        np.mean(np.exp(pW[n]))
                    ])) + 0.5,
                    color='dark' + col,
                    linewidth=0.5)
                
                squarePlot(ax_Long[0, sp])
#                 ax_Long[0, sp].set(
#                     adjustable='box-forced',
#                     xlim=[0, 0.5],
#                     ylim=[0, 1.05],
#                     aspect=(0.5 - 0) / (1.05 - 0))
                
                ax_Long[0, sp].set_title('p=' + str(pp))
                sp += 1

        else:
            gg = np.linspace(0, 1)
            ax[0, 3].plot(
                gg, [W(g, np.mean(np.exp(pW[n][:]), axis=0)) for g in gg],
                color=col)  #sort out the plotting for these
            ax[0, 3].plot(gg, gg, color='k', linestyle='--')
            x0, x1 = ax[0, 3].get_xlim()
            y0, y1 = ax[0, 3].get_ylim()
            ax[0, 3].set_aspect((x1 - x0) / (y1 - y0))
            ax[0, 3].text(
                0.6,
                0.2 - (n * 0.1),
                'α' + '=' + str(round(np.mean(np.exp(pW[n][:])), 4)),
                style='italic',
                color=col,
                alpha=1)

            mm = np.linspace(mLow, mMax, num=1000)
            CE = []
            xVals = np.array(
                [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]) * 2
            for pp in np.linspace(0, 1, num=1000):
                Pg, _ = func_Pg(
                    U, W,
                    np.vstack([np.array([0, pp, 0.5, 1 - pp]) for m in mm]), mm)
                CE.extend(mm[np.abs(Pg([ np.mean(np.exp(pC[n])) ,  np.mean(np.exp(pU[n])) ,  np.mean(np.exp(pW[n][:]), axis=0)[0], np.mean(np.exp(pW[n][:]), axis=0)[1]]) -0.5)\
                             == min(np.abs(Pg([ np.mean(np.exp(pC[n])) ,  np.mean(np.exp(pU[n])) ,  np.mean(np.exp(pW[n][:]), axis=0)[0], np.mean(np.exp(pW[n][:]), axis=0)[1]]) -0.5))])
            ax[0, 4].plot(
                np.array([0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
                * 2,
                mean_Ys[n],
                'bo ',
                color='dark' + col)
            ax[0, 4].plot(
                np.linspace(1, 0, num=1000), CE, color=col, linestyle='--')

            iTrials = sTrials.loc[sTrials.trialSequenceMode ==
                                  NM_fit.context.unique()[n]]
            cData = get_options(iTrials, mergeBy='all')
            sm = get_softmaxData(
                cData, metricType='CE').sort_values('primaryEV')
            sp = 0
            CEs = []
            for pp in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
                Pg, _ = func_Pg(
                    U, W,
                    np.vstack([np.array([0, 1 - pp, 0.5, pp]) for m in mm]), mm)
                ax_Long[0, sp].plot(
                    mm,
                    1 - Pg([
                        np.mean(np.exp(pC[n])),
                        np.mean(np.exp(pU[n])),
                        np.mean(np.exp(pW[n][:]), axis=0)[0],
                        np.mean(np.exp(pW[n][:]), axis=0)[1]
                    ]),
                    color=col)
                #                ax[1,sp].plot(mm,1-Pg([ np.mean([np.mean(np.exp(pC[1])),np.mean(np.exp(pC[0]))]),1,1,1]), color='k', linestyle = '--')
                ax_Long[0, sp].text(
                    0.025,
                    0.95 - (n * 0.1),
                    'λ' + '=' + str(round(np.mean(np.exp(pC[n])), 4)),
                    style='italic',
                    color=col,
                    alpha=1)
                ax_Long[0, sp].axhline(0.5, linestyle='-', color='k', alpha=0.5)
                chG = Pg([
                    np.mean(np.exp(pC[n])),
                    np.mean(np.exp(pU[n])),
                    np.mean(np.exp(pW[n][:]), axis=0)[0],
                    np.mean(np.exp(pW[n][:]), axis=0)[1]
                ])
                ax_Long[0,sp].axvline( mm[np.abs(chG -0.5)\
                             == min(np.abs(chG -0.5))],
                            linestyle = '--', color=col, alpha = 0.5)

                row = sm.loc[sm.primaryEV == pp / 2]
                x_mag = [m[0] for m in row.secondary.tolist()[0]]
                chX_freq = np.squeeze(row.freq_sCh.tolist(
                ))  #need the frequency of picking safe, not gamble
                ax_Long[0, sp].plot(x_mag, chX_freq, 'bo ', color='dark' + col)
                Pg, _ = func_Pg(
                    U, W,
                    np.vstack([np.array([0, 1 - pp, 0.5, pp]) for m in x_mag]),
                    x_mag)
                ax_Long[0, sp].plot(
                    x_mag,
                    chX_freq - (1 - Pg([
                        np.mean(np.exp(pC[n])),
                        np.mean(np.exp(pU[n])),
                        np.mean(np.exp(pW[n][:]), axis=0)[0],
                        np.mean(np.exp(pW[n][:]), axis=0)[1]
                    ])) + 0.5,
                    color='dark' + col,
                    linewidth=0.5)
                
                squarePlot(ax_Long[0, sp])
                
#                 ax_Long[0, sp].set(
#                     adjustable='box-forced',
#                     xlim=[0, 0.5],
#                     ylim=[0, 1.05],
#                     aspect=(0.5 - 0) / (1.05 - 0))
                
                sp += 1

        mm = np.linspace(mLow, mMax)
        ax[0, 2].plot(mm, [U(m, np.mean(np.exp(pU[n]))) for m in mm], color=col)
        ax[0, 2].plot(mm, gg, color='k', linestyle='--')
        x0, x1 = ax[0, 2].get_xlim()
        y0, y1 = ax[0, 2].get_ylim()
        ax[0, 2].set_aspect((x1 - x0) / (y1 - y0))
        ax[0, 2].text(
            0.3,
            0.2 - (n * 0.1),
            'ρ' + '=' + str(round(np.mean(np.exp(pU[n])), 4)),
            style='italic',
            color=col,
            alpha=1)

        ax[0, 4].set_ylim(0, 0.5)  #y axis length
        ax[0, 4].set_xlim(0, 1)  #y axis length
        x0, x1 = ax[0, 4].get_xlim()
        y0, y1 = ax[0, 4].get_ylim()
        ax[0, 4].set_aspect((x1 - x0) / (y1 - y0))
        ax[0, 4].grid(b=True, which='major')

    #-------------------------------------------------------------------
    plt.show()

    if len(Xs) <= 3:
        p = sb.JointGrid(x=pU[0], y=pW[0].T[0])
        sb.regplot(
            x=pU[0], y=pW[0].T[0], color='blue', fit_reg=False, ax=p.ax_joint)
        sb.distplot(pU[0], color='blue', ax=p.ax_marg_x)
        sb.distplot(pW[0].T[0], color='blue', vertical=True, ax=p.ax_marg_y)
        sb.regplot(
            x=pU[1], y=pW[1].T[0], color='red', fit_reg=False, ax=p.ax_joint)
        sb.distplot(pU[1], color='red', ax=p.ax_marg_x)
        sb.distplot(pW[1].T[0], color='red', vertical=True, ax=p.ax_marg_y)
        p.ax_joint.axvline(0, linestyle='--', color='k')
        p.ax_joint.axhline(0, linestyle='--', color='k')
        p.ax_joint.grid()
        p.ax_joint.set_xlim(-1.05, 1.05)
        p.ax_joint.set_ylim(-1.05, 1.05)
        p.ax_marg_y.grid('on')
        p.ax_marg_x.grid('on')
        p.ax_joint.set_xlabel('Utility Parameter', fontweight='bold')
        p.ax_joint.set_ylabel('Probability Parameter', fontweight='bold')
        plt.setp(p.ax_marg_x.get_yticklabels(), visible=True)
        plt.setp(p.ax_marg_y.get_xticklabels(), visible=True)

        covarianceEllipse(pU[0], pW[0], ax=p.ax_joint, color='blue', draw='CI')
        covarianceEllipse(pU[1], pW[1], ax=p.ax_joint, color='red', draw='CI')

    return NM_fit_sorted


#%%
def MLE_comparison(filteredDF,
                   Trials,
                   use_matlab=False,
                   plotit=True,
                   plotFitting=False):
    from scipy import stats
    from macaque.f_Rfunctions import oneWay_rmAnova

    BIC = lambda LL, nP, nT: (2 * LL) + (nP * np.log(nT))
    AIC = lambda LL, nP: (2 * LL) + (2 * nP)
    AICc = lambda LL, nP, nT: AIC(LL, nP) + ((2 * nP) * (nP + 1)) / (nT - nP - 1)
    BICs = []
    AICs = []
    AICcs = []
    lsq = []
    dates = []
    function = []
    LLs_all = []
    for wType in ['1-prelec', 'tversky', '2-prelec', 'gonzalez']:
        NM_fit = fit_likelihoodModel(
            filteredDF,
            Trials,
            uModel='power',
            wModel=wType,
            plotit=plotit,
            plotFitting=plotFitting)

        nTs = NM_fit.nTrials.values
        nPs = NM_fit.params.apply(lambda x: len(x)).values
        LLs = NM_fit.LL.values

        lsq.append(NM_fit.lsq.values)
        BICs.append(BIC(LLs, nPs, nTs))
        AICs.append(AIC(LLs, nPs))
        AICcs.append(AICc(LLs, nPs, nTs))
        dates.extend(NM_fit.date.apply(lambda x: x.toordinal()))
        function.extend([wType] * len(NM_fit))
        LLs_all.append(LLs)

    # ----------------------------------
    coloration = ['black', 'teal', 'm', 'orange']
    fig, ax = plt.subplots(4, 1, squeeze=False, figsize=(10, 15))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    for bic, aic, ll, lserr, col in zip(BICs, AICs, LLs_all, lsq, coloration):
        ax[0, 0].plot(bic, color=col, alpha=0.4)
        ax[1, 0].plot(aic, color=col, alpha=0.4)
        ax[2, 0].plot(ll, color=col, alpha=0.4)
        ax[3, 0].plot(lserr, color=col, alpha=0.4)
    ax[0, 0].legend(['1-prelec', 'tversky', '2-prelec', 'gonzalez'])
    ax[0, 0].set_title('BICs')
    ax[1, 0].set_title('AICs')
    ax[2, 0].set_title('LLs')
    ax[3, 0].set_title('LSQerr')
    ax[0, 0].set_xticks(range(0, len(bic)))
    ax[0, 0].set_xticklabels([])
    ax[0, 0].grid(axis='x')
    ax[1, 0].set_xticks(range(0, len(bic)))
    ax[1, 0].set_xticklabels([])
    ax[1, 0].grid(axis='x')
    ax[2, 0].set_xticks(range(0, len(bic)))
    ax[2, 0].set_xticklabels([])
    ax[2, 0].grid(axis='x')
    ax[3, 0].set_xticks(range(0, len(bic)))
    ax[3, 0].grid(axis='x')
    ax[3, 0].set_xticklabels(
        NM_fit.date.values, rotation=90, horizontalalignment='center')
    plt.show()
    # ----------------------------------

    mBIC = []
    mAIC = []
    mAICc = []
    seBIC = []
    seAIC = []
    seAICc = []
    mlsq = []
    selsq = []
    for model in range(len(BICs)):
        mBIC.append(np.mean(BICs[model]))
        seBIC.append(stats.sem(BICs[model]))
        mAIC.append(np.mean(AICs[model]))
        seAIC.append(stats.sem(BICs[model]))
        mAICc.append(np.mean(AICcs[model]))
        seAICc.append(stats.sem(BICs[model]))
        mlsq.append(np.mean(lsq[model]))
        selsq.append(stats.sem(lsq[model]))

    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(8, 3))
    fig.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)

    col = ['gold', 'khaki', 'magenta', 'violet']
    ax[0, 0].bar(
        ['1-prelec', 'tversky', '2-prelec', 'gonzalez'],
        mBIC,
        yerr=seBIC,
        capsize=3,
        color=col)
    ax[0, 0].set_ylim(None, min(mBIC) + 40)  #y axis length
    ax[0, 1].bar(
        ['1-prelec', 'tversky', '2-prelec', 'gonzalez'],
        mAIC,
        yerr=seAIC,
        capsize=3,
        color=col)
    ax[0, 1].set_ylim(None, min(mAIC) + 40)  #y axis length
    ax[0, 2].bar(
        ['1-prelec', 'tversky', '2-prelec', 'gonzalez'],
        mAICc,
        yerr=seAICc,
        capsize=3,
        color=col)
    ax[0, 2].set_ylim(None, max(mAICc) + 40)  #y axis length

    fig.show()
    #    import pdb ; pdb.set_trace() #AWESOME WAY TO DEBUG
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(3, 3))
    fig.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
    col = ['gold', 'khaki', 'magenta', 'violet']
    ax[0, 0].bar(
        ['1-prelec', 'tversky', '2-prelec', 'gonzalez'],
        mlsq,
        yerr=selsq,
        capsize=3,
        color=col)
    fig.show()

    print(
        '-------------------------------------------------------------------------------'
    )
    print('rmANOVA for the BIC scores',
          '\nDV = BIC, ID = date, and IV = function')
    oneWay_rmAnova(np.concatenate(BICs), dates, function)
    print(
        '-------------------------------------------------------------------------------'
    )
    print(
        '-------------------------------------------------------------------------------'
    )
    print('rmANOVA for the leastSquare scores',
          '\nDV = BIC, ID = date, and IV = function')
    oneWay_rmAnova(np.concatenate(lsq), dates, function)
    print(
        '-------------------------------------------------------------------------------'
    )

    ICweights(BICs, ['1-prelec', 'tversky', '2-prelec', 'gonzalez'])


#%%
def covarianceEllipse(xData, yData, ax, color='k', draw='CI'):
    import numpy as np
    from matplotlib.patches import Ellipse

    #-------------------------------------------------
    def cov_ellipse(cov, q=0.95, nsig=None, **kwargs):
        import numpy as np
        from scipy.stats import norm, chi2
        """
        Parameters
        ----------
        cov : (2, 2) array
            Covariance matrix.
        q : float, optional
            Confidence level, should be in (0, 1)
        nsig : int, optional
            Confidence level in unit of standard deviations.
            E.g. 1 stands for 68.3% and 2 stands for 95.4%.

        Returns
        -------
        width, height, rotation :
             The lengths of two axises and the rotation angle in degree
        for the ellipse.
        """

        if q is not None:
            q = np.asarray(q)
        elif nsig is not None:
            q = 2 * norm.cdf(nsig) - 1
        else:
            raise ValueError('One of `q` and `nsig` should be specified.')
        r2 = chi2.ppf(q, 2)

        val, vec = np.linalg.eigh(cov)
        width, height = 2 * np.sqrt(val[:, None] * r2)
        rotation = np.degrees(np.arctan2(*vec[::-1, 0]))
        return width, height, rotation

    #---------------------------

    nstd = 2
    yData = np.squeeze(yData)
    #    import pdb ; pdb.set_trace() #AWESOME WAY TO DEBUG
    cov = np.cov(xData, yData)
    if draw.lower() == 'ci':
        w, h, theta = cov_ellipse(cov, q=0.95)
    elif draw.lower() == 'variance':
        w, h, theta = cov_ellipse(cov, nsig=2)
    ell = Ellipse(
        xy=(np.mean(xData), np.mean(yData)),
        width=w,
        height=h,
        angle=theta,
        color=color)
    ell.set_facecolor('none')
    ax.add_artist(ell)


#    plt.scatter(xData, yData)
#    plt.show()


#%%
def compare_RTs(filteredDF, removeZero=False, error='sem'):
    '''
    '''
    from scipy import stats

    RTs = flatten([list(val.values()) for val in filteredDF.choiceTimes ])
    #find all response times
    sEV = flatten([
        [np.array(y) for y in val.keys()] for val in filteredDF.choiceTimes ])
        #find all secondary EVs
    pEV = flatten([[np.array(ev) for y in val.keys()]  for val, ev in zip(
            filteredDF.choiceTimes, filteredDF.primaryEV) ])
            #find all primary EVs
    context = flatten([[seq for y in val.keys()] for val, seq in zip(
            filteredDF.choiceTimes.values, filteredDF.seqCode) ])
            #find all primary EVs
    chosenEV = flatten( flatten([list(val.values()) for val in filteredDF.chosenEV]))
    date = flatten([[seq.toordinal() for y in val.keys()] for val, seq in zip(
            filteredDF.choiceTimes.values, filteredDF.sessionDate) ])
    date = np.array(date) - date[0]

    #merge all variables os that I get trial-by-trial dep and ind variables
    sEVs = []
    pEVs = []
    contexts = []
    day = []
    for rt, sev, pev, cc, dd in zip(RTs, sEV, pEV, context, date):
        sEVs.extend([sev] * len(rt))
        pEVs.extend([pev] * len(rt))
        contexts.extend([cc] * len(rt))
        day.extend([dd] * len(rt))
    del sEV, pEV, context, date
    deltaEV = np.round(np.array(pEVs) - np.array(sEVs), decimals=2)

    RTs = np.array(flatten(RTs))
    variables = np.array((RTs, deltaEV, np.array(contexts), np.array(chosenEV),
                          day)).T
    aovDF = pd.DataFrame( variables,
                         columns=['RTs', 'deltaEV',
                                  'context', 'chosenEV', 'day'])
    if removeZero == True:
        aovDF.drop(
            index=aovDF.loc[(np.array(chosenEV) > 0.5) |
                            (np.array(chosenEV) < 0.05)].index,
            inplace=True)
    else:
        aovDF.drop(
            index=aovDF.loc[(np.array(chosenEV) > 0.5)].index, inplace=True)

    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    formula = 'RTs ~ C(day) + deltaEV + C(context) + chosenEV + deltaEV:C(context) + deltaEV:chosenEV + C(context):chosenEV + deltaEV:C(context):chosenEV'
    #    formula = 'RTs ~ day + deltaEV + context + chosenEV + deltaEV*context + deltaEV*chosenEV + context*chosenEV + deltaEV*context*chosenEV'
    #    formula = 'RTs ~ deltaEV + day + chosenEV  +  C(context) \
    #    + deltaEV:day + deltaEV:chosenEV + deltaEV:C(context) + day:chosenEV + day:C(context) + C(context):chosenEV\
    #    + deltaEV:day:C(context)  + deltaEV:day:chosenEV + deltaEV:C(context):chosenEV + day:C(context):chosenEV\
    #    + deltaEV:day:C(context):chosenEV'
    model = ols(formula, data=aovDF).fit()
    aov_table = anova_lm(model, typ=3)

    eta_squared(aov_table)
    omega_squared(aov_table)
    print(aov_table)

    # ----------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(10, 4))
    aovDF.groupby(by=['deltaEV', 'context']).mean()['RTs'].unstack().plot(
        color=['blue', 'red'],
        yerr=aovDF.groupby(by=['deltaEV', 'context']).sem()['RTs'].unstack(),
        ax=ax[0, 1],
        grid=True)
    ax[0, 1].legend(['repeat', 'mixed'])
    aovDF.groupby(by=['chosenEV', 'context']).mean()['RTs'].unstack().plot(
        color=['blue', 'red'],
        yerr=aovDF.groupby(by=['chosenEV', 'context']).sem()['RTs'].unstack(),
        ax=ax[0, 2],
        grid=True)
    ax[0, 2].legend(['repeat', 'mixed'])
    aovDF.groupby(by=['context']).mean()['RTs'].plot(
        kind='bar',
        color=['blue', 'red'],
        yerr=aovDF.groupby(by=['context']).sem()['RTs'],
        ax=ax[0, 0],
        grid=False)
    ax[0, 0].set_xticklabels(['repeat', 'mixed'])
    ax[0, 0].set_ylabel('response time')

    print(
        '------------------------------------------------------------------------------------------'
    )
    test = model.t_test_pairwise("C(context)").result_frame[[
        't', 'P>|t|', 'pvalue-hs', 'reject-hs'
    ]]
    print('Pairwise comparison of mean RTs across contexts: ')
    print(model.t_test_pairwise("C(context)").result_frame)
    print(
        '------------------------------------------------------------------------------------------'
    )
    if test['P>|t|'].values < 0.05:
        sigBars = ax[0, 0].get_xticks()
        stars = getAsterisk(test['P>|t|'].values)
        x1, x2 = sigBars[0], sigBars[1]
        y2 = 1.2 * max(aovDF.groupby(by=['context']).mean()['RTs'].values)
        y1 = 0.95 * y2
        ax[0, 0].plot(
            [x1, x1, x2, x2], [y1, y2, y2, y1], linewidth=1.5, color='k')
        ax[0, 0].text(np.mean(sigBars) * 0.8, y2 * 1.01, stars, size=17)
        ax[0, 0].set_ylim(0, y2 * 1.15)

    for axis in ax.reshape(-1):
        x0, x1 = axis.get_xlim()
        y0, y1 = axis.get_ylim()
        axis.set_aspect((x1 - x0) / (y1 - y0))

    # ----------------------------

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 4))
    aovDF.groupby(by=['deltaEV']).mean()['RTs'].plot(
        color=['black'],
        yerr=aovDF.groupby(by=['deltaEV']).sem()['RTs'],
        ax=ax[0, 0],
        grid=True)

    aovDF.groupby(by=['chosenEV']).mean()['RTs'].plot(
        color=['black'],
        yerr=aovDF.groupby(by=['chosenEV']).sem()['RTs'],
        ax=ax[0, 1],
        grid=True)



    ax[0, 0].set_xticklabels(['repeat', 'mixed'])
    ax[0, 0].set_ylabel('response time')


#%%
def compare_filteredRTs(filteredDF, removeZero=False, error='sem'):
    '''
    '''
    from scipy import stats

    RTs = flatten([list(val.values()) for val in filteredDF.filteredRT
                  ])  #find all response times
    sEV = flatten([
        [np.array(y) for y in val.keys()] for val in filteredDF.filteredRT
    ])  #find all secondary EVs
    pEV = flatten([
        [np.array(ev)
         for y in val.keys()]
        for val, ev in zip(filteredDF.filteredRT, filteredDF.primaryEV)
    ])  #find all primary EVs
    context = flatten([
        [seq
         for y in val.keys()]
        for val, seq in zip(filteredDF.filteredRT.values, filteredDF.seqCode)
    ])  #find all primary EVs
    chosenEV = flatten(
        flatten([list(val.values()) for val in filteredDF.chosenEV]))
    date = flatten([[
        seq.toordinal() for y in val.keys()
    ] for val, seq in zip(filteredDF.filteredRT.values, filteredDF.sessionDate)
                   ])
    date = np.array(date) - date[0]

    #merge all variables os that I get trial-by-trial dep and ind variables
    sEVs = []
    pEVs = []
    contexts = []
    day = []
    for rt, sev, pev, cc, dd in zip(RTs, sEV, pEV, context, date):
        sEVs.extend([sev] * len(rt))
        pEVs.extend([pev] * len(rt))
        contexts.extend([cc] * len(rt))
        day.extend([dd] * len(rt))
    del sEV, pEV, context, date
    deltaEV = np.round(np.array(pEVs) - np.array(sEVs), decimals=2)
    #    contexts = ['repeat' if x < 9020 else 'mixed' for x in contexts]
    #    contexts = [contexts]

    RTs = np.array(flatten(RTs))
    variables = np.array((RTs, deltaEV, np.array(contexts), np.array(chosenEV),
                          day)).T
    aovDF = pd.DataFrame(
        variables, columns=['RTs', 'deltaEV', 'context', 'chosenEV', 'day'])
    if removeZero == True:
        aovDF.drop(
            index=aovDF.loc[(np.array(chosenEV) > 0.5) |
                            (np.array(chosenEV) < 0.05)].index,
            inplace=True)
    else:
        aovDF.drop(
            index=aovDF.loc[(np.array(chosenEV) > 0.5)].index, inplace=True)

    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    formula = 'RTs ~ day + deltaEV + C(context) + chosenEV + deltaEV:C(context) + deltaEV:chosenEV + C(context):chosenEV + deltaEV:C(context):chosenEV'
    #    formula = 'RTs ~ day + deltaEV + context + chosenEV + deltaEV*context + deltaEV*chosenEV + context*chosenEV + deltaEV*context*chosenEV'
    #    formula = 'RTs ~ deltaEV + day + chosenEV  +  C(context) \
    #    + deltaEV:day + deltaEV:chosenEV + deltaEV:C(context) + day:chosenEV + day:C(context) + C(context):chosenEV\
    #    + deltaEV:day:C(context)  + deltaEV:day:chosenEV + deltaEV:C(context):chosenEV + day:C(context):chosenEV\
    #    + deltaEV:day:C(context):chosenEV'
    model = ols(formula, data=aovDF).fit()
    aov_table = anova_lm(model, typ=3)

    #    from rpy2.robjects.packages import importr
    #    from rpy2.robjects import Formula
    #    from rpy2.robjects import r
    #
    #    stats = importr('stats')
    #    formula = Formula(formula)
    #    env = formula.environment
    #
    #    from rpy2.robjects import pandas2ri
    #    pandas2ri.activate()
    #    r_df = pandas2ri.py2ri(aovDF)
    #    anova = stats.aov(formula = formula, data = r_df)
    #
    #    print(r.summary(anova))

    eta_squared(aov_table)
    omega_squared(aov_table)
    print(aov_table)

    # ----------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(10, 4))
    aovDF.groupby(by=['deltaEV', 'context']).mean()['RTs'].unstack().plot(
        color=['blue', 'red'],
        yerr=aovDF.groupby(by=['deltaEV', 'context']).sem()['RTs'].unstack(),
        ax=ax[0, 1],
        grid=True)
    ax[0, 1].legend(['repeat', 'mixed'])
    aovDF.groupby(by=['chosenEV', 'context']).mean()['RTs'].unstack().plot(
        color=['blue', 'red'],
        yerr=aovDF.groupby(by=['chosenEV', 'context']).sem()['RTs'].unstack(),
        ax=ax[0, 2],
        grid=True)
    ax[0, 2].legend(['repeat', 'mixed'])
    aovDF.groupby(by=['context']).mean()['RTs'].plot(
        kind='bar',
        color=['blue', 'red'],
        yerr=aovDF.groupby(by=['context']).sem()['RTs'],
        ax=ax[0, 0],
        grid=False)
    ax[0, 0].set_xticklabels(['repeat', 'mixed'])
    ax[0, 0].set_ylabel('response time')

    print(
        '------------------------------------------------------------------------------------------'
    )
    test = model.t_test_pairwise("C(context)").result_frame[[
        't', 'P>|t|', 'pvalue-hs', 'reject-hs'
    ]]
    print('Pairwise comparison of mean RTs across contexts: ')
    print(model.t_test_pairwise("C(context)").result_frame)
    print(
        '------------------------------------------------------------------------------------------'
    )
    if test['P>|t|'].values < 0.05:
        sigBars = ax[0, 0].get_xticks()
        stars = getAsterisk(test['P>|t|'].values)
        x1, x2 = sigBars[0], sigBars[1]
        y2 = 1.2 * max(aovDF.groupby(by=['context']).mean()['RTs'].values)
        y1 = 0.95 * y2
        ax[0, 0].plot(
            [x1, x1, x2, x2], [y1, y2, y2, y1], linewidth=1.5, color='k')
        ax[0, 0].text(np.mean(sigBars) * 0.8, y2 * 1.01, stars, size=17)
        ax[0, 0].set_ylim(0, y2 * 1.15)

    for axis in ax.reshape(-1):
        x0, x1 = axis.get_xlim()
        y0, y1 = axis.get_ylim()
        axis.set_aspect((x1 - x0) / (y1 - y0))


#%%
def get_fit_sideSpecific(filteredDF, Trials):
    '''
    Calculate the WSLS logistic regression model from the trials that were used for the reste of the analysis.

    '''
    from macaque.f_choices import get_options
    from macaque.f_psychometrics import get_softmaxData
    from macaque.f_probabilityDistortion import fit_likelihoodModel

    # -------------------------------------------------------------------------
    #%%
    def plot_LRdifferences(sTrials, title=None, plotTQDM=False):
        leftChoices = get_options(
            sTrials,
            mergeBy='all',
            byDates=False,
            mergeSequentials=True,
            sideSpecific='Left',
            plotTQDM=plotTQDM)
        softmax_left = get_softmaxData(
            leftChoices,
            metricType='CE',
            minSecondaries=4,
            minChoices=4,
            plotTQDM=plotTQDM)
        rightChoices = get_options(
            sTrials,
            mergeBy='all',
            byDates=False,
            mergeSequentials=True,
            sideSpecific='Right',
            plotTQDM=plotTQDM)
        softmax_right = get_softmaxData(
            rightChoices,
            metricType='CE',
            minSecondaries=4,
            minChoices=4,
            plotTQDM=plotTQDM)

        NMfit_left = fit_likelihoodModel(
            softmax_left,
            Trials,
            uModel='power',
            wModel='1-prelec',
            plotit=False,
            plotTQDM=plotTQDM)
        NMfit_right = fit_likelihoodModel(
            softmax_right,
            Trials,
            uModel='power',
            wModel='1-prelec',
            plotit=False,
            plotTQDM=plotTQDM)

        fig, ax2 = plt.subplots(1, 3, squeeze=False, figsize=(10, 4))

        ax2[0, 0].plot(
            np.linspace(0, 0.5),
            NMfit_left.functions.values[0][1](np.linspace(0, 0.5),
                                              NMfit_left.params.values[0][1]),
            color='magenta',
            lw=2)  #utility param
        ax2[0, 0].plot(
            np.linspace(0, 0.5),
            NMfit_right.functions.values[0][1](np.linspace(0, 0.5),
                                               NMfit_right.params.values[0][1]),
            color='y',
            lw=2)  #utility param
        ax2[0, 0].plot(np.linspace(0, 0.5), np.linspace(0, 1), '--', color='k')
        ax2[0, 0].grid()
        ax2[0, 1].plot(
            np.linspace(0, 1),
            NMfit_left.functions.values[0][2](np.linspace(0, 1),
                                              NMfit_left.params.values[0][2]),
            color='magenta',
            lw=2)
        ax2[0, 1].plot(
            np.linspace(0, 1),
            NMfit_right.functions.values[0][2](np.linspace(0, 1),
                                               NMfit_right.params.values[0][2]),
            color='y',
            lw=2)
        ax2[0, 1].plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='k')
        ax2[0, 1].grid()

        softmax_left.plot(
            x='primaryEV',
            y='equivalent',
            yerr=np.vstack(softmax_left.pSTE.values)[:, 0],
            kind='scatter',
            color='magenta',
            ax=ax2[0, 2],
            grid=True,
            s=50)
        softmax_right.plot(
            x='primaryEV',
            y='equivalent',
            yerr=np.vstack(softmax_right.pSTE.values)[:, 0],
            kind='scatter',
            color='y',
            ax=ax2[0, 2],
            grid=True,
            s=50)
        ax2[0, 2].legend(['left choices', 'right choices'])
        ax2[0, 2].plot(
            np.linspace(0, 0.5), np.linspace(0, 0.5), '--', color='k')

        for axis in ax2.reshape(-1):
            x0, x1 = axis.get_xlim()
            y0, y1 = axis.get_ylim()
            axis.set_aspect((x1 - x0) / (y1 - y0))

        ax2[0, 2].set_ylabel('certainty equivalent')
        ax2[0, 2].set_xlabel('gambleEV')
        ax2[0, 0].set_ylabel('utility')
        ax2[0, 1].set_ylabel('probability distortion')
        ax2[0, 0].set_xlabel('magnitude')
        ax2[0, 1].set_xlabel('probability')

        ax2[0, 2].set_title('CEs from side-speific gambles')
        ax2[0, 1].set_title('probability distortion')
        ax2[0, 0].set_title('utility')

        if title:
            plt.suptitle(title)

    #%%
    def plot_LRonly(sTrials, title=None, plotTQDM=False):
        leftChoices = get_options(
            sTrials.loc[sTrials.gambleChosen == 'A'],
            mergeBy='all',
            byDates=False,
            mergeSequentials=True,
            plotTQDM=plotTQDM)
        softmax_left = get_softmaxData(
            leftChoices,
            metricType='CE',
            minSecondaries=4,
            minChoices=4,
            plotTQDM=plotTQDM)
        rightChoices = get_options(
            sTrials.loc[sTrials.gambleChosen == 'B'],
            mergeBy='all',
            byDates=False,
            mergeSequentials=True,
            plotTQDM=plotTQDM)
        softmax_right = get_softmaxData(
            rightChoices,
            metricType='CE',
            minSecondaries=4,
            minChoices=4,
            plotTQDM=plotTQDM)

        NMfit_left = fit_likelihoodModel(
            softmax_left,
            Trials,
            uModel='power',
            wModel='1-prelec',
            plotit=False,
            plotTQDM=plotTQDM)
        NMfit_right = fit_likelihoodModel(
            softmax_right,
            Trials,
            uModel='power',
            wModel='1-prelec',
            plotit=False,
            plotTQDM=plotTQDM)

        fig, ax2 = plt.subplots(1, 3, squeeze=False, figsize=(10, 4))

        ax2[0, 0].plot(
            np.linspace(0, 0.5),
            NMfit_left.functions.values[0][1](np.linspace(0, 0.5),
                                              NMfit_left.params.values[0][1]),
            color='magenta',
            lw=2)  #utility param
        ax2[0, 0].plot(
            np.linspace(0, 0.5),
            NMfit_right.functions.values[0][1](np.linspace(0, 0.5),
                                               NMfit_right.params.values[0][1]),
            color='y',
            lw=2)  #utility param
        ax2[0, 0].plot(np.linspace(0, 0.5), np.linspace(0, 1), '--', color='k')
        ax2[0, 0].grid()
        ax2[0, 1].plot(
            np.linspace(0, 1),
            NMfit_left.functions.values[0][2](np.linspace(0, 1),
                                              NMfit_left.params.values[0][2]),
            color='magenta',
            lw=2)
        ax2[0, 1].plot(
            np.linspace(0, 1),
            NMfit_right.functions.values[0][2](np.linspace(0, 1),
                                               NMfit_right.params.values[0][2]),
            color='y',
            lw=2)
        ax2[0, 1].plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='k')
        ax2[0, 1].grid()

        softmax_left.plot(
            x='primaryEV',
            y='equivalent',
            yerr=np.vstack(softmax_left.pSTE.values)[:, 0],
            kind='scatter',
            color='magenta',
            ax=ax2[0, 2],
            grid=True,
            s=50)
        softmax_right.plot(
            x='primaryEV',
            y='equivalent',
            yerr=np.vstack(softmax_right.pSTE.values)[:, 0],
            kind='scatter',
            color='y',
            ax=ax2[0, 2],
            grid=True,
            s=50)
        ax2[0, 2].legend(['left choices', 'right choices'])
        ax2[0, 2].plot(
            np.linspace(0, 0.5), np.linspace(0, 0.5), '--', color='k')

        for axis in ax2.reshape(-1):
            x0, x1 = axis.get_xlim()
            y0, y1 = axis.get_ylim()
            axis.set_aspect((x1 - x0) / (y1 - y0))

        ax2[0, 2].set_ylabel('certainty equivalent')
        ax2[0, 2].set_xlabel('gambleEV')
        ax2[0, 0].set_ylabel('utility')
        ax2[0, 1].set_ylabel('probability distortion')
        ax2[0, 0].set_xlabel('magnitude')
        ax2[0, 1].set_xlabel('probability')

        ax2[0, 2].set_title('LR-only CEs')
        ax2[0, 1].set_title('probability distortion')
        ax2[0, 0].set_title('utility')

        if title:
            plt.suptitle(title)

    #%%

    regTrials = np.sort(
        np.unique(
            np.concatenate([
                np.concatenate(list(val.values()))
                for val in filteredDF.get('trial_index').values
            ])))  #gets the index for all the trials I should use in regression
    sTrials = Trials.loc[regTrials].copy(
    )  #get only trials that were used in the rest of the analysis

    tt = sTrials.loc[sTrials.trialSequenceMode == 9001].copy()
    plot_LRdifferences(tt, title='Repeat Condition')
    tt = sTrials.loc[sTrials.trialSequenceMode == 9020].copy()
    plot_LRdifferences(tt, title='Mixed Condition')
    plot_LRdifferences(sTrials, title='Both Conditions')

    # -------------------------------------------------------------------------

    tt = sTrials.loc[sTrials.trialSequenceMode == 9001].copy()
    plot_LRonly(tt, title='Repeat Condition')
    tt = sTrials.loc[sTrials.trialSequenceMode == 9020].copy()
    plot_LRonly(tt, title='Mixed Condition')
    plot_LRonly(sTrials, title='Both Conditions')

    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(10, 4))

    tt = sTrials.loc[sTrials.trialSequenceMode == 9001].copy()
    allChoices = get_options(
        tt, mergeBy='all', byDates=False, mergeSequentials=True)
    softmax_all = get_softmaxData(
        allChoices, metricType='CE', minSecondaries=4, minChoices=4)
    NMfit_all = fit_likelihoodModel(
        softmax_all, Trials, uModel='power', wModel='1-prelec', plotit=False)
    ax[0, 0].plot(
        np.linspace(0, 0.5),
        NMfit_all.functions.values[0][1](np.linspace(0, 0.5),
                                         NMfit_all.params.values[0][1]),
        color='blue',
        lw=2)  #utility param
    ax[0, 0].plot(np.linspace(0, 0.5), np.linspace(0, 1), '--', color='k')
    ax[0, 0].grid()
    ax[0, 1].plot(
        np.linspace(0, 1),
        NMfit_all.functions.values[0][2](np.linspace(0, 1),
                                         NMfit_all.params.values[0][2]),
        color='blue',
        lw=2)
    ax[0, 1].plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='k')
    ax[0, 1].grid()
    softmax_all.plot(
        x='primaryEV',
        y='equivalent',
        yerr=np.vstack(softmax_all.pSTE.values)[:, 0],
        kind='scatter',
        color='blue',
        ax=ax[0, 2],
        grid=True,
        s=50)
    ax[0, 2].legend(['repeat', 'mixed'])
    ax[0, 2].plot(np.linspace(0, 0.5), np.linspace(0, 0.5), '--', color='k')

    tt = sTrials.loc[sTrials.trialSequenceMode == 9020].copy()
    allChoices = get_options(
        tt, mergeBy='all', byDates=False, mergeSequentials=True)
    softmax_all = get_softmaxData(
        allChoices, metricType='CE', minSecondaries=4, minChoices=4)
    NMfit_all = fit_likelihoodModel(
        softmax_all, Trials, uModel='power', wModel='1-prelec', plotit=False)
    ax[0, 0].plot(
        np.linspace(0, 0.5),
        NMfit_all.functions.values[0][1](np.linspace(0, 0.5),
                                         NMfit_all.params.values[0][1]),
        color='red',
        lw=2)  #utility param
    ax[0, 1].plot(
        np.linspace(0, 1),
        NMfit_all.functions.values[0][2](np.linspace(0, 1),
                                         NMfit_all.params.values[0][2]),
        color='red',
        lw=2)
    softmax_all.plot(
        x='primaryEV',
        y='equivalent',
        yerr=np.vstack(softmax_all.pSTE.values)[:, 0],
        kind='scatter',
        color='red',
        ax=ax[0, 2],
        grid=True,
        s=50)

    for axis in ax.reshape(-1):
        x0, x1 = axis.get_xlim()
        y0, y1 = axis.get_ylim()
        axis.set_aspect((x1 - x0) / (y1 - y0))

    ax[0, 2].set_ylabel('certainty equivalent')
    ax[0, 2].set_xlabel('gambleEV')
    ax[0, 0].set_ylabel('utility')
    ax[0, 1].set_ylabel('probability distortion')
    ax[0, 0].set_xlabel('magnitude')
    ax[0, 1].set_xlabel('probability')

    ax[0, 2].set_title('LR certainty equivalents')
    ax[0, 1].set_title('probability distortion')
    ax[0, 0].set_title('utility')


#%%
def fit_parametricModel(Trials,
                        Model,
                        plotit=False,
                        plotTQDM=True,
                        plotFitting=False):
    '''
    Nelder-mead search algorithm to simultaneously fit utility, probability, discrete choice, and/or reference curves via maximum loglikelihood fit

    1st) define the U and W function
    2nd) calculate the softmax for every gamble-safe pairing
    3rd) merge all these into the LL functio
    '''
    import numpy as np
    import scipy.optimize as opt
    import pandas as pd
    from scipy.io import savemat
    import time
    from macaque.f_models import get_modelLL

    np.warnings.filterwarnings('ignore')
    mLow = 0  #min(np.concatenate(Trials[['GA_ev','GB_ev']].values))
    mMax = 0.5  #max(np.concatenate(Trials[['GA_ev','GB_ev']].values))
    #%% ----------------------------------------------------------------------------------

    dList = []
    resList = []
    for date in tqdm( Trials.sessionDate.unique(), desc=Model, disable=not plotTQDM):
        tt = Trials.loc[Trials['sessionDate'] == date]
        LL, x0, pNames = get_modelLL(tt, Model)
        sum_neg_LL = lambda params: -(LL(params))

        #        if bounds:
        #            results = opt.minimize(sum_neg_LL, x0, method='TNC', bounds = bounds, options = {'disp' : False, 'maxfev' : 1e5}) #,o
        #        else:
        res_x = []

        results = opt.minimize( sum_neg_LL, x0,
                                method='Nelder-Mead',
                                callback=res_x.append,
                                options={
                                        'disp': False,
                                        'maxfev': 1e5
                                        })  #,options={'maxiter':1000, 'maxfev':1000})
        Nfeval = [sum_neg_LL(Xs) for Xs in res_x]

        nTrials = len(tt)
        context = tt.trialSequenceMode.unique()
        dList.append({
            'date': date,
            'nTrials': nTrials,
            'params': results.x,
            'NM_success': results.success,
            'model_used': Model,
            'LL': sum_neg_LL(results.x),
            'context': np.unique(context)[-1],
            'm_range': [mLow, mMax],
            'pNames': pNames,
            'Nfeval': Nfeval,
            'all_fits': res_x
        })

    NM_fit = pd.DataFrame(dList)

    if plotit:
        parameters = np.vstack(NM_fit.params.values)
        fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(12, 3))
        fig.suptitle(Model)
        for pp in range(np.size(parameters, axis=1)):
            params = parameters[:, pp]
            ax[0, 0].plot(range(len(NM_fit.date.values)), params)
        ax[0, 0].legend(pNames)
        ax[0, 0].get_xaxis().set_visible(False)
        #        plt.xticks(None)

        for pp in range(np.size(parameters, axis=1)):
            params = parameters[:, pp]
            ax[1, 0].plot(range(len(NM_fit.date.values)), np.log(params))
        plt.xticks(
            range(len(NM_fit.date.values)),
            NM_fit.date.values,
            rotation='vertical')
        ax[1, 0].hlines(
            xmin=0, xmax=len(NM_fit.date.values), y=0, linestyles='--')

        for i, pp in zip(range(len(NM_fit.date.values)), NM_fit.NM_success):
            if pp == False:
                #                plt.axvline(i, color = 'r')
                ax[0, 0].axvline(i, color='r')
                ax[1, 0].axvline(i, color='r')
        plt.show()

    if plotFitting:
        plot_MLEfitting(NM_fit, plotFittings=False)

    return NM_fit


#%%
def multiModel_LL(filteredDF, Trials, Models = None, plotit=False, plotFitting=True):
    '''
    '''
    from scipy import stats
    from macaque.f_Rfunctions import oneWay_rmAnova
    import matplotlib.cm as cm

    def compareFits(MLE, ax, logged=False):
        from scipy import stats
        MLE = MLE.drop(MLE.loc[MLE.NM_success == False].index)
        width = 0.35
        Xs = np.array(range(len(MLE.params.values[0])))
        ax.axhline(0, color='k')
        xTicks = MLE.pNames.values[0]
        for cc, col in zip(MLE.context.unique(), ['blue', 'red']):
            width = -(width)
            mle = MLE.loc[MLE.context == cc]

            fittedPs = np.vstack(mle.params.values)
            Xs = np.array(range(len(fittedPs[0])))
            if logged:
                fittedPs[:, -2:] = np.log(fittedPs[:, -2:])
                fittedPs[:, 0] = np.log(fittedPs[:, 0])
            Y = np.mean(fittedPs, axis=0)
            Yerr = stats.sem(fittedPs, axis=0)
            mLL = np.mean(mle.LL)
            LLerr = stats.sem(mle.LL)

            #sort out the plotting for the parameters
            ax.bar(
                Xs + width / 2,
                Y,
                width,
                yerr=Yerr,
                capsize=3,
                label=str(cc),
                color=col)
            plt.xticks(Xs, xTicks)
        return

    def manova(MLE):
        parameters = np.vstack(MLE.params)
        if len(parameters[0]) == 3:
            from macaque.f_Rfunctions import dv3_manova
            dv3_manova(
                parameters[:, 0],
                parameters[:, 1],
                parameters[:, 2],
                IV=MLE.context.values)
        elif len(parameters[0]) == 2:
            from macaque.f_Rfunctions import dv2_manova
            dv2_manova(
                parameters[:, 0], parameters[:, 1], IV=MLE.context.values)
        elif len(parameters[0]) == 4:
            from macaque.f_Rfunctions import dv4_manova
            dv4_manova(
                parameters[:, 0],
                parameters[:, 1],
                parameters[:, 2],
                parameters[:, 3],
                IV=MLE.context.values)

    #-----------------------------------------------------------------


    sTrials = filteredDF.getTrials(Trials)


    BIC = lambda LL, nP, nT: (2 * LL) + (nP * np.log(nT))
    AIC = lambda LL, nP: (2 * LL) + (2 * nP)
    AICc = lambda LL, nP, nT: AIC(LL, nP) + ((2 * nP) * (nP + 1)) / (nT - nP - 1)

    # this is how the function will have to be for the comparison in the future
    #    Models = ['ev', 'ev-gstate', 'prospect','prospectside','WSLSsimple','WSLSlearn', 'RL',
    #              'RLgstate', 'ev-wsls', 'ev-wsls-complex' , 'attention-simple'  , 'attention-pt', 'pt-gstate' ]
    if Models is None:
        Models = [
            'ev_attention', 'prospect_3param_attention', 'ev_withwsls_2param',
            'ev_withwsls_4param', 'prospect_3param', 'prospect_3param_side',
            'wsls_simple', 'wsls_dynamic', 'rl', 'rl_pastgamble', 'ev', 'ev+wsls',
            'prospect+wsls'
        ]

    BICs = []
    AICs = []
    AICcs = []
    dates = []
    fitModel = []
    mleList = []
    LLs_all = []
    for i, wType in enumerate(Models):
        MLE_fit = fit_parametricModel(
            sTrials, Model=wType, plotit=plotit, plotFitting=plotFitting)
        manova(MLE_fit)
#        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(4, 3))
#        if 'wsls' in wType.lower() or 'rl' in wType.lower(
#        ) or 'ev' in wType.lower():
#            logged = False
#            ax[0, 0].axhline(0.5, color='k', linestyle='--')
#        else:
#            logged = True
#        compareFits(MLE_fit, ax[0, 0], logged=logged)

        nTs = MLE_fit.nTrials.values
        nPs = MLE_fit.params.apply(lambda x: len(x)).values
        LLs = MLE_fit.LL.values

        BICs.append(BIC(LLs, nPs, nTs))
        AICs.append(AIC(LLs, nPs))
        AICcs.append(AICc(LLs, nPs, nTs))
        dates.extend(MLE_fit.date.apply(lambda x: x.toordinal()))
        fitModel.extend([wType] * len(MLE_fit))
        LLs_all.append(LLs)

        mleList.append(MLE_fit)
#    statsmodels.multivariate.manova.MANOVA()

#    print('ttest for WSLSsimple: param1,param2')
#    stats.ttest_1samp(np.vstack(mleList[2].params.values)[:, 0], 0.5)
#    stats.ttest_1samp(np.vstack(mleList[2].params.values)[:, 1], 0.5)
#
#    print('ttest for WSLSlearn: param1,param2')
#    stats.ttest_1samp(np.vstack(mleList[3].params.values)[:, 0], 0.5)
#    stats.ttest_1samp(np.vstack(mleList[3].params.values)[:, 1], 0.5)

    # ----------------------------------
    coloration = cm.rainbow(np.linspace(0, 1, len(Models)))
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    for bic, aic, ll, col in zip(BICs, AICs, LLs_all, coloration):
#        ax[0, 0].plot(bic, color=col, alpha=0.4)
#        ax[1, 0].plot(aic, color=col, alpha=0.4)
        ax[0, 0].plot(ll, color=col, alpha=0.4)
#    ax[0, 0].legend(Models)
#    ax[0, 0].set_title('BICs')
#    ax[1, 0].set_title('AICs')
    ax[0, 0].set_title('LLs')
#    ax[0, 0].set_xticks(range(0, len(bic)))
#    ax[0, 0].set_xticklabels([])
#    ax[0, 0].grid(axis='x')
#    ax[1, 0].set_xticks(range(0, len(bic)))
#    ax[1, 0].set_xticklabels([])
#    ax[1, 0].grid(axis='x')
    ax[0, 0].set_xticks(range(0, len(bic)))
    ax[0, 0].grid(axis='x')
    ax[0, 0].set_xticklabels(
        MLE_fit.date.values, rotation=90, horizontalalignment='center')
    # ----------------------------------

    #%% Comparing the models Directly

    mBIC = []
    mAIC = []
    mAICc = []
    seBIC = []
    seAIC = []
    seAICc = []
    for model in range(len(BICs)):
        mBIC.append(np.mean(BICs[model]))
        seBIC.append(stats.sem(BICs[model]))
        mAIC.append(np.mean(AICs[model]))
        seAIC.append(stats.sem(BICs[model]))
        mAICc.append(np.mean(AICcs[model]))
        seAICc.append(stats.sem(BICs[model]))

    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 3))
    #    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
    #                wspace=0.3, hspace=0.3)

    col = ['gold', 'khaki', 'magenta', 'violet']
    ax[0, 0].bar(Models, mBIC, yerr=seBIC, capsize=3, color=col)
    ax[0, 0].set_title('BIC')
    ax[0, 0].set_xticklabels(Models, rotation=90, horizontalalignment='center')
    ax[0, 1].bar(Models, mAIC, yerr=seAIC, capsize=3, color=col)
    ax[0, 1].set_title('AIC')
    ax[0, 1].set_xticklabels(Models, rotation=90, horizontalalignment='center')
    ax[0, 2].bar(Models, mAICc, yerr=seAICc, capsize=3, color=col)
    ax[0, 2].set_title('AICc')
    ax[0, 2].set_xticklabels(Models, rotation=90, horizontalalignment='center')
    fig.show()

    #do the ANOVA for the BIC results
    #    import pdb ; pdb.set_trace() #AWESOME WAY TO DEBUG
    print('\n ANOVA results for BICs')

    print(
        '-------------------------------------------------------------------------------------------------------------------------------------------'
    )
    print('rmANOVA for the BIC scores',
          '\nDV = BIC, ID = date, and IV = function')
    oneWay_rmAnova(np.concatenate(BICs), dates, fitModel)
    print(
        '-------------------------------------------------------------------------------'
    )

    for ll in mleList:
        print('\n ------------------------------------------------------------- \n' )
        print(ll.model_used.unique()[0])
        print(ll.params.mean())
#        print(ll.params.sem())

    print('\n ------------------------------------------------------------- ')
    print(mBIC)

    return mleList


#%%
def wsls_compare(filteredDF, Trials, plotit=False):
    '''
    '''
    from scipy import stats

    #-----------------------------------------------------------------

    regTrials = np.sort(
        np.unique(
            np.concatenate([
                np.concatenate(list(val.values()))
                for val in filteredDF.get('trial_index').values
            ])))  #gets the index for all the trials I should use in regression
    sTrials = Trials.loc[regTrials].copy(
    )  #get only trials that were used in the rest of the analysis

    # this is how the function will have to be for the comparison in the future
    #    Models = ['prospect','prospectside','WSLSsimple','WSLSlearn', 'RL', 'RLgstate', 'hybrid', 'hybrid_simple']

    BICs = []
    AICs = []
    AICcs = []
    dates = []
    fitModel = []
    mleList = []

    MLE_fit = fit_parametricModel(
        sTrials,
        Model='ev_withwsls_2param',
        plotit=plotit,
        plotTQDM=True,
        plotFitting=False)

    MLE = MLE_fit.drop(MLE_fit.loc[MLE_fit.NM_success == False].index)
    Xs = np.array(range(len(MLE.params.values[0])))

    xTicks = MLE.pNames.values[0]

    fittedPs = np.vstack(MLE.params.values)
    Xs = np.array(range(len(fittedPs[0])))
    Y = np.mean(fittedPs, axis=0)
    Yerr = stats.sem(fittedPs, axis=0)
    mLL = np.mean(MLE.LL)
    LLerr = stats.sem(MLE.LL)

    #sort out the plotting for the parameters
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(4, 3))
    ax[0, 0].bar(
        Xs,
        Y,
        yerr=Yerr,
        capsize=3,
        label='ev_withwsls_2param',
        color='k',
        alpha=0.5)
    ax[0, 0].axhline(0.5, color='k')
    plt.xticks(Xs, xTicks)

    print('\nParameter T-Test \n \
-----------------------------------------------------------------------')
    T, P = stats.ttest_1samp(fittedPs[:, 0], 0.5)
    print('Win-Stay:  t-value: %0.3f ; p-value: %0.3f ; N: %d' %
          (T, P, len(fittedPs)))
    T, P = stats.ttest_1samp(fittedPs[:, 1], 0.5)
    print('Lose-Shift:  t-value: %0.3f ; p-value: %0.3f ; N: %d' %
          (T, P, len(fittedPs)))

    allPs = fittedPs

    #%%

    import statsmodels.api as sm
    stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
    from statsmodels.graphics.api import abline_plot
    from scipy.special import logit

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
                else:
                    sTrials.at[index, 'g_win'] = 0
            else:
                sTrials.at[index, 'chG'] = 1
            sTrials.at[index, 'gEV'] = row.GA_ev
            sTrials.at[index, 'sEV'] = row.GB_ev
        elif row.outcomesCount[1] == 2:
            if row.gambleChosen == 'B':
                sTrials.at[index, 'chG'] = 1
                if row.ml_received != 0:
                    sTrials.at[index, 'g_win'] = 1
                else:
                    sTrials.at[index, 'g_win'] = 0
            else:
                sTrials.at[index, 'chG'] = 0
            sTrials.at[index, 'gEV'] = row.GB_ev
            sTrials.at[index, 'sEV'] = row.GA_ev
    sTrials['g_lose'] = 1 - sTrials['g_win']

    MLE_fit = []
    for date in tqdm(sTrials.sessionDate.unique(), desc='fit days'):
        ws_Trials = sTrials.loc[sTrials.sessionDate == date]
        consecutives = ws_Trials.iloc[np.insert(
            np.diff(ws_Trials.trialNo.values) == 1, 0, False)].index
        #this does not include beginning ones in a non-consecutive manner
        #        xx = ws_Trials.loc[consecutives] #the trials that are followed
        not_consecutives = ws_Trials.iloc[np.insert(
            np.diff(ws_Trials.trialNo.values) > 1, 0, True)].index
        #these trials can't be looked at in terms of what happened before them
        for index in consecutives:
            ws_Trials.at[index, 'pChG'] = ws_Trials.loc[index - 1].chG
            ws_Trials.at[index, 'pG_win'] = ws_Trials.loc[index - 1].g_win
            ws_Trials.at[index, 'pG_lose'] = ws_Trials.loc[index - 1].g_lose
        for index in not_consecutives:
            ws_Trials.at[index, 'pChG'] = np.nan
            ws_Trials.at[index, 'pG_win'] = np.nan
            ws_Trials.at[index, 'pG_lose'] = np.nan

        woutTrials = ws_Trials.loc[ws_Trials.pChG == 1]
        df = fit_parametricModel(
            woutTrials,
            Model='ev_withwsls_2param',
            plotit=plotit,
            plotTQDM=False,
            plotFitting=False)
        MLE_fit.extend([df])

    MLE_fit = pd.concat(MLE_fit, ignore_index=True)
    Xs = np.array(range(len(MLE_fit.params.values[0])))

    xTicks = MLE_fit.pNames.values[0]

    fittedPs = np.vstack(MLE_fit.params.values)
    Xs = np.array(range(len(fittedPs[0])))
    Y = np.mean(fittedPs, axis=0)
    Yerr = stats.sem(fittedPs, axis=0)
    mLL = np.mean(MLE_fit.LL)
    LLerr = stats.sem(MLE_fit.LL)

    #sort out the plotting for the parameters
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(4, 3))
    ax[0, 0].bar(
        Xs,
        Y,
        yerr=Yerr,
        capsize=3,
        label='ev_withwsls_2param',
        color='k',
        alpha=0.5)
    ax[0, 0].axhline(0.5, color='k')
    plt.xticks(Xs, xTicks)

    print('\nParameter T-Test \n \
-----------------------------------------------------------------------')
    T, P = stats.ttest_1samp(fittedPs[:, 0], 0.5)
    print('Win-Stay:  t-value: %0.3f ; p-value: %0.3f ; N: %d' %
          (T, P, len(fittedPs)))
    T, P = stats.ttest_1samp(fittedPs[:, 1], 0.5)
    print('Lose-Shift:  t-value: %0.3f ; p-value: %0.3f ; N: %d' %
          (T, P, len(fittedPs)))

    selectPs = fittedPs

    # -----------------------------------------------------------------------------------------

    print('\nAre parameters the same if select trials? \n \
-----------------------------------------------------------------------')

    from macaque.f_Rfunctions import dv2_manova
    IV = np.ravel([[0] * len(allPs), [1] * len(selectPs)])
    parameters = np.vstack((allPs, selectPs))
    dv2_manova(parameters[:, 0], parameters[:, 1], IV=IV)


#%%
def model_checks(filteredDF, Trials, plotit=False):
    '''
    '''
    from scipy import stats

    #-----------------------------------------------------------------

    regTrials = np.sort(
        np.unique(
            np.concatenate([
                np.concatenate(list(val.values()))
                for val in filteredDF.get('trial_index').values
            ])))  #gets the index for all the trials I should use in regression
    sTrials = Trials.loc[regTrials].copy(
    )  #get only trials that were used in the rest of the analysis

    # this is how the function will have to be for the comparison in the future
    #    Models = ['prospect','prospectside','WSLSsimple','WSLSlearn', 'RL', 'RLgstate', 'hybrid', 'hybrid_simple']

    MLE_fit = fit_parametricModel(
        sTrials, Model='prospect_3param_side', plotit=plotit, plotFitting=True)

    MLE = MLE_fit.drop(MLE_fit.loc[MLE_fit.NM_success == False].index)
    Xs = np.array(range(len(MLE.params.values[0])))

    fittedPs = np.vstack(MLE.params.values)
    Y = np.mean(fittedPs, axis=0)
    Yerr = stats.sem(fittedPs, axis=0)
    parameters = MLE.pNames.values[0]

    print('\nParameter T-Test \n \
          -----------------------------------------------------------------------'
         )
    T, P = stats.ttest_1samp(fittedPs[:, 1], 0)
    print(
        'Side-Bias across all sessions:  t-value: %0.3f ; p-value: %0.3f ; N: %d'
        % (T, P, len(fittedPs)))

    # ----------------------------------------------------------------------------

    MLE_fit = fit_parametricModel(
        sTrials,
        Model='prospect_3param_attention',
        plotit=plotit,
        plotTQDM=True,
        plotFitting=False)

    MLE = MLE_fit.drop(MLE_fit.loc[MLE_fit.NM_success == False].index)
    Xs = np.array(range(len(MLE.params.values[0])))

    fittedPs = np.vstack(MLE.params.values)
    Y = np.mean(fittedPs, axis=0)
    Yerr = stats.sem(fittedPs, axis=0)
    parameters = MLE.pNames.values[0]

    print('\nAttention T-Test \n \
          -----------------------------------------------------------------------'
         )
    blocked = np.vstack(MLE.loc[MLE['context'] == 9001].params.values)
    mixed = np.vstack(MLE.loc[MLE['context'] == 9020].params.values)
    T, P = stats.ttest_ind(blocked[:, -1], mixed[:, -1])
    print(
        'Side-Bias across all sessions:  t-value: %0.3f ; p-value: %0.3f ; N: %d'
        % (T, P, len(fittedPs)))
    from macaque.f_Rfunctions import dv4_manova
    dv4_manova(
        fittedPs[:, 0],
        fittedPs[:, 1],
        fittedPs[:, 2],
        fittedPs[:, 3],
        IV=MLE.context.values)

    x0 = MLE.params.values[0]
    mLow, mMax = MLE.m_range.values[0]
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(4, 3))
    plt.suptitle(MLE.model_used.values[0], fontsize=16)

    #First Subplot
    pC = []
    pU = []
    pW = []
    pA = []
    width = 0.35
    Xs = np.array(range(len(x0)))
    ax[0, 0].axhline(0, color='k')
    for cc, col in zip(MLE.context.unique(), ['blue', 'red']):
        width = -(width)
        pC.append(
            np.log(np.vstack(MLE.loc[MLE.context == cc].params.values)[:, 0]))
        pU.append(
            np.log(np.vstack(MLE.loc[MLE.context == cc].params.values)[:, 1]))
        pW.append(
            np.log(np.vstack(MLE.loc[MLE.context == cc].params.values)[:, 2]))
        pA.append(
            np.log(np.vstack(MLE.loc[MLE.context == cc].params.values)[:, 2]))
        yerr = stats.sem(
            np.log(np.vstack(MLE.loc[MLE.context == cc].params.values)), axis=0)

        #sort out the plotting for the parameters
        ax[0, 0].bar(
            Xs + width / 2,
            np.log(np.vstack(
                MLE.loc[MLE.context == cc].params.values)).mean(axis=0),
            width,
            yerr=yerr,
            capsize=3,
            label=str(cc),
            color=col)


#        plt.sca(ax[0, 0])
    ax[0, 0].set_xticks(Xs + width / 2)
    ax[0, 0].set_xticklabels(
        parameters, rotation=0, horizontalalignment='center')

    _, sigC = stats.ttest_ind(pC[0], pC[1], equal_var=True)
    _, sigU = stats.ttest_ind(pU[0], pU[1], equal_var=True)
    _, sigW = stats.ttest_ind(pW[0], pW[1], equal_var=True)
    _, sigA = stats.ttest_ind(pA[0], pA[1], equal_var=True)

    avgSTD = np.std(np.log(np.vstack(MLE.params.values)), axis=0)
    effectSize = np.round((np.mean(np.log(np.vstack(MLE.loc[MLE.context == MLE.context.unique()[0]].params.values)), axis=0) - \
                  np.mean(np.log(np.vstack(MLE.loc[MLE.context == MLE.context.unique()[1]].params.values)), axis=0)) / \
                  avgSTD, 4)

    col_labels = parameters
    row_labels = ['effect size']

    for p, x in zip([sigC, sigU, sigW, sigA], Xs):
        if isinstance(p, np.ndarray):
            if pp < 0.05:
                ax[0, 0].scatter(
                    x,
                    np.amax(np.log(np.vstack(MLE.params.values))[:, x]),
                    marker='*',
                    color='k')
                x += 1
        else:
            if p < 0.05:
                if np.amax(np.log(np.vstack(MLE.params.values))[:, x]) < 0:
                    ax[0, 0].scatter(
                        x,
                        -np.amax(np.log(np.vstack(MLE.params.values))[:, x]),
                        marker='*',
                        color='k')
                else:
                    ax[0, 0].scatter(
                        x,
                        np.amax(np.log(np.vstack(MLE.params.values))[:, x]),
                        marker='*',
                        color='k')

    print('Is attention different from 0.5: ')
    print(' ----------------------------------------------------------- ')
    t1, sig1 = stats.ttest_1samp(pA[0], 0.5)
    t2, sig2 = stats.ttest_1samp(pA[1], 0.5)
    print('Attention block:  t-value: %0.3f ; p-value: %0.3f ; N: %d' %
          (t1, sig1, len(pA[0])))
    print('Attention mixed:  t-value: %0.3f ; p-value: %0.3f ; N: %d' %
          (t2, sig2, len(pA[1])))


def model_predictions(filteredDF, mle_list):
    return

    mle = mle_list.loc[mle_list['model_used'] == model]
    for date in tqdm(filteredDF.sessionDate.unique(), desc='MLE predictions:'):
        CEs = filteredDF.loc[filteredDF['sessionDate'] == date][
            'equivalent']  #this gives me the 'real' CEs
        ll_model = mle.loc[mle.sessionDate == date]

def invert_CEplot(softmaxDF):
    filteredDF = softmaxDF.copy()
    from scipy import interpolate

    predicted_Ys = [[0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]]
    mean_Ys = [[0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]]
    shift = -0.01
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 5))
    for cc, col, context in zip(filteredDF.seqCode.unique(), ['blue', 'red'],
                                ['blocked', 'mixed']):
        #make the numbers the appropriate probabilities
        filteredDF.loc[filteredDF.seqCode == cc, ['primaryEV']] = filteredDF.loc[ filteredDF.seqCode == cc].primaryEV.apply(lambda x: x / 0.5)
        ce_avg = []
        x = np.squeeze( filteredDF.sort_values('primaryEV').loc[filteredDF.seqCode == cc,
                                                    ['primaryEV']].values)
        y = np.squeeze( filteredDF.sort_values('primaryEV').loc[filteredDF.seqCode == cc,
                                                    ['equivalent']].values)
        for val in np.sort(
                filteredDF.loc[filteredDF.seqCode == cc].primaryEV.unique()):
            ce_avg.extend([
                np.mean(filteredDF.loc[filteredDF.seqCode == cc].loc[
                    filteredDF.loc[filteredDF.seqCode == cc].primaryEV == val]
                        .equivalent.values)
            ])
        mean, bound_upper, bound_lower, yHat = bootstrap_splineCE(
            x, y, method='resampling', n=10000)
        ax[0, 0].plot( mean,np.linspace(0, 1, 100), '-', color='dark' + col)

        #Plot CEs between conditions (same day blocked vs mixed)
        # offset both contexts
        filteredDF.loc[
            filteredDF.seqCode == cc, ['primaryEV']] = filteredDF.loc[
                filteredDF.seqCode == cc].primaryEV.apply(lambda x: x + shift)
        filteredDF.loc[filteredDF.seqCode == cc].plot.scatter(
            y='primaryEV',
            x='equivalent',
            color='dark' + col,
            label=context,
            ax=ax[0, 0])
        ax[0, 0].plot(
            np.linspace(0, 0.5, 1000),
            np.linspace(0, 1, 1000),
            color='k',
            linestyle='--')
#        ax[0, 0].set_ylim(-0.025, 0.525)  #y axis length
#        ax[0, 0].set_xlim(0, 1.0)  #y axis length
        ax[0, 0].grid(b=True, which='major')
        #this sets the aspect ratio to a square
        x0, x1 = ax[0, 0].get_xlim()
        y0, y1 = ax[0, 0].get_ylim()
        ax[0, 0].set_aspect((x1 - x0) / (y1 - y0))
        shift = -shift

        predicted_Ys.append(yHat[1:-1])
        mean_Ys.append(ce_avg)
    plt.plot()
    return predicted_Ys, mean_Ys
