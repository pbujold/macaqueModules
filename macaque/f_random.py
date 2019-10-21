# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:39:59 2018

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
def get_finalStatistics(dList, correlations1, correlations2, minTrials = 40):
    '''
    '''
    #need to make this work via a collection of lists
    dataList = []
    df = correlations1.loc[correlations1.riskless_params.apply(lambda x: len(x) > 0)]
    df2 = correlations2.loc[correlations2.riskless_params.apply(lambda x: len(x) > 0)]
    riskless = np.vstack(flatten(df2.riskless_params.values))
    riskless = np.vstack((riskless.T, np.ones(len(riskless)))).T
    legend = []; colours = []
    if len(riskless) != 0:
        dataList.append(riskless)
        legend.extend(['riskless'])
        colours.extend(['blue'])
    risky_PT = np.vstack(df2.risky_params.values)
    risky_PT = risky_PT
    if len(risky_PT) != 0:
        dataList.append(risky_PT)
        legend.extend(['risky_pt'])
        colours.extend(['red'])
    risky_EUT = np.vstack(df.risky_params.values)
    risky_EUT = np.vstack((risky_EUT.T, np.ones(len(riskless)))).T
    if len(risky_EUT) != 0:
        dataList.append(risky_EUT)
        legend.extend(['risky_eut'])
        colours.extend(['red'])
    if len(dList.loc[dList.gambleCE_success.values]) != 0:
        gambleCEs = np.vstack(dList.loc[(dList.gambleCE_success.values) & (dList.nTrials >= minTrials)].gambleCE_params.values)
        gambleCEs = gambleCEs
#        gambleCEs = gambleCEs.loc[gambleCEs.nTrials >= minTrials]
        dataList.append(gambleCEs)
        legend.extend(['gambleCEs'])
        colours.extend(['green'])
    
    #%%
    #post-hoc before 
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    independent = flatten([len(dd) * [rr] for dd, rr, in zip(dataList, [0,1,2,3])])
    dependent = flatten(dataList)
    
    data = pd.DataFrame(np.vstack((independent, np.vstack(dependent).T))).T
    data.columns = ['sequence_type', 'noise', 'side', 'u_temp', 'u_inf', 'w_temp']

    #%% Correlation plots
    
    def grayify_cmap(cmap):
        """Return a grayscale version of the colormap"""
        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))
        
        # convert RGBA to perceived greyscale luminance
        # cf. http://alienryderflex.com/hsp.html
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]
        
        return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)
    
    index1 = np.isin(data.sequence_type.values, [0])
    index2 = np.isin(data.sequence_type.values, [1])
    xx = data.loc[index1][['u_temp', 'u_inf']]
    xx['risky_temp' ]  = data.loc[index2]['u_temp'].values
    xx['risky_inf' ]  = data.loc[index2]['u_inf'].values
    corr = xx.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(3, 3))
    # Generate a custom diverging colormap
#    cmap = sb.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sb.heatmap(corr, annot=True, mask=mask, cmap=grayify_cmap("RdGy"), vmax=1, vmin=-1, center=0,
                square=True, linewidths=.8)
    
    print(calculate_pvalues(xx))

    #%%
#    manova
    from macaque.f_Rfunctions import dv5_manova
    mix = [0,1,2,3]
    a = [[mix[i], mix[ii]] for i,ii in zip(np.random.randint(0,4, 100),np.random.randint(0,4, 100))]
    comparisons = unique_listOfLists(np.sort(a))
    for cc in comparisons:
        if cc[0] == cc[1]:
            continue
        else:
            names = np.array([0,1,2,3])[cc]
            index = np.isin(data.sequence_type.values, names)
            data_arr = np.vstack(data.values)[index,:]
            print('==========================================================')
            print('==========================================================')
            print(np.array(legend)[cc])
            print('------------------------------------------------------------')
#            if np.size(data_arr,1)
            dv5_manova( data_arr[:,1],  data_arr[:,2],  data_arr[:,3], data_arr[:,4], data_arr[:,5], IV=data_arr[:,0])
            

#%%
def behavioural_area(fractile, Behaviour):
    '''
    '''
    dates_in_third = np.intersect1d(Behaviour.sessionDate.unique(), fractile.sessionDate.unique())
    area_fractile = []; area_behaviour = []

    for date in tqdm(dates_in_third, desc = 'fitting fractiles'):
        ff = fractile.loc[fractile.sessionDate == date]
        rr = Behaviour.loc[Behaviour.sessionDate == date]
        
        u1= ff.utility
        u1= np.concatenate(([0],u1,[1]))
        x1 = ff.equivalent
        x1= np.concatenate(([0],x1,[0.5]))
        metric1 = u1 - x1
        
        u2 = rr.utility_real
        x2 = rr.midpoint
        metric2 = u2 - x2
        
        area_fractile.extend([np.mean(metric1)])
        area_behaviour.extend([np.mean(metric2)])
        
    plt.scatter([1] * len(area_fractile), area_fractile)
    plt.scatter([0] * len(area_behaviour), area_behaviour)
        
#%%
def classify_parameters(correlations, binary_MLE):
    '''
    '''
    from macaque.f_models import define_model
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import itertools
    import statsmodels.api as sm
    from scipy import stats
    
    position = np.argmax(['u_' in pname for pname in correlations.pNames[0][0]])
    long = np.sum(['u_' in pname for pname in correlations.pNames.iloc[-1][0]])
    
    risky = np.vstack(correlations.risky_params.values)[:,position:position+long]
    riskless = np.vstack(flatten(correlations.riskless_params.values))[:,position:position+long]
    risky2 = np.vstack(correlations.risky_params.values)[:,position:position+long]
    riskless2 = np.vstack(flatten(correlations.riskless_params.values))[:,position:position+long]

    risky2[:,-2] = np.log(risky[:,-2])
    riskless2[:,-2] = np.log(riskless[:,-2])

#    random = [parameters[:,position:position+long] for parameters in random]
    legend = ['mle', 'behaviour']
    
    binary = np.vstack(binary_MLE.params.values)[:,position:position+long]
    binary2 = np.vstack(binary_MLE.params.values)[:,position:position+long]
#    binary[:,-1] = np.log(binary[:,-1])
    binary2[:,-2] = np.log(binary[:,-2])

    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(10,6))
    i= 0 

#            index = np.hstack([len(rr) * [0]] + [len(ff) * [1]] + [len(bb) * [2]])
#            data = np.vstack((np.vstack((ff,rr,bb)).T, index)).T
    index = np.hstack([len(riskless) * [0]] + [len(risky) * [1]])
    data = np.vstack((np.vstack((riskless2,risky2)).T, index)).T
    palette = itertools.cycle(sb.color_palette('colorblind'))
    for condition in np.unique(data[:,-1]):
        cc = next(palette)
        ii = data[:,-1] == condition
        ax[0,0].scatter(data[ii,0], data[ii,1], color = cc)
        
    ax[0,0].legend(['riskless','risky'])
    for r1,r2 in zip(riskless2,risky2):
        ax[0,0].plot([r1[0], r2[0]], [r1[1], r2[1]] , '--', color='k', alpha = 0.25)
    
    y = data[:,-1]
    X = data[:,:-1]
    Xc = sm.add_constant(X)
    
    try:
        f = 'condition ~  p1 + p2 + p1:p2'
        logitfit = smf.logit(formula = str(f), data = pd.DataFrame(data, columns=['p1', 'p2', 'condition'])).fit()
        print(logitfit.summary())
    except:
        model =sm.Logit(y, X)
        result = model.fit()
        print(result.get_margeff().summary())
        print(result.summary())
    squarePlot(ax[0,0])
    ax[0,0].set_xlabel('temperature')
    ax[0,0].set_ylabel('inflection')
    ax[0,0].grid()

    
    #%%
    
    ff = define_model(correlations.model_used.iloc[-1])
    p0 = ff[1]
    utility = lambda pp: ff[-1](p0)['empty functions']['utility'](np.linspace(0,1,100), pp)
    
    palette = itertools.cycle(sb.color_palette('colorblind'))
    
    index = np.hstack([len(riskless) * [0]] + [len(risky) * [1]])
    data = np.vstack((np.vstack((riskless,risky)).T, index)).T
    palette = itertools.cycle(sb.color_palette('colorblind'))
    area = []; i=0
    for condition in np.unique(data[:,-1]):
        cc = next(palette)
        ii = data[:,-1] == condition
        aa = [np.sum(utility(pp))/100 for pp in data[ii,:-1]]
        ax[0,1].scatter([i] * len(aa), aa, color=cc)
        area.append(aa)
        i+=1
        
    area = np.vstack(area).T
    
    for nn in area:
        plt.plot([0,1], nn, color='k', alpha = 0.5)
    ax[0,1].set_xticks([0,1], ['riskless', 'risky'])
    ax[0,1].set_ylim([0.2,1])
    ax[0,1].set_xlim([-0.25, 1.25])
    squarePlot(ax[0,1])

    print('========================================')
    print( 't-test: ', stats.ttest_rel(area[:,0], area[:,1]))
    
    #%%
    
    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(10,6))
    i= 0 

    index = np.hstack([len(riskless) * [0]] + [len(risky) * [1]] + [len(binary) * [2]])
    data = np.vstack((np.vstack((riskless2,risky2,binary2)).T, index)).T
    
    palette = itertools.cycle(sb.color_palette('colorblind'))
    for condition in np.unique(data[:,-1]):
        cc = next(palette)
        ii = data[:,-1] == condition
        ax[0,0].scatter(data[ii,0], data[ii,1], color = cc)
    
    
    import statsmodels.formula.api as smf
    f = 'condition ~ p1 + p2 + p1:p2'
    logitfit = smf.mnlogit(formula = str(f), data = pd.DataFrame(data, columns=['p1', 'p2', 'condition'])).fit()
    print(logitfit.summary())
    
    squarePlot(ax[0,0])
    ax[0,0].set_xlabel('temperature')
    ax[0,0].set_ylabel('inflection')
    ax[0,0].grid()
    ax[0,0].legend(['riskless','risky','binary'])
    
    #%%
    
    ff = define_model(correlations.model_used.iloc[-1])
    p0 = ff[1]
    utility = lambda pp: ff[-1](p0)['empty functions']['utility'](np.linspace(0,1,100), pp)
    
    palette = itertools.cycle(sb.color_palette('colorblind'))
    
    index = np.hstack([len(riskless) * [0]] + [len(risky) * [1]] + [len(binary) * [2]])
    data = np.vstack((np.vstack((riskless,risky,binary)).T, index)).T
    palette = itertools.cycle(sb.color_palette('colorblind'))
    area = []; mArea = []; err = []; i=0
    for condition in np.unique(data[:,-1]):
        color = next(palette)
        ii = data[:,-1] == condition
        aa = [np.sum(utility(pp))/100 for pp in data[ii,:-1]]
        mArea.extend([np.mean(aa)])
        err.extend([stats.sem(aa)])
        area.append(aa)
        ax[0,1].scatter(jitter([i] * len(aa), 0.05), aa, color=color)
        i+=1

    ax[0,1].errorbar([0,1,2], mArea, yerr = err, color='k')
    
    ax[0,1].set_ylim([0,1])
    ax[0,1].set_xlim([-0.5, 2.5])

    squarePlot(ax[0,1])
    ax[0,1].set_xticks([0,1,2])
    ax[0,1].set_xticklabels(['riskless', 'risky','binary'])
    
    
#    print('========================================')
#    print( 't-test: ', stats.ttest_rel(area[:,0], area[:,1]))
    
#%%
def compare_RTs(trials, gambleCEs, Behaviour, fractile):
    '''
    '''
    import seaborn as sns
    from macaque.f_models import get_modelLL, trials_2fittable, LL_fit
    
    
    tt1 = Behaviour.getTrials(trials)
    tt1['type'] = [0] * len(tt1)
    tt2 = fractile.getTrials(trials)
    tt2['type'] = [1] * len(tt2)
    tt3 = gambleCEs.getTrials(trials)
    tt3['type'] = [2] * len(tt3)
    
    
    df = pd.concat((tt1,tt2,tt3))

    df['diff'] = df['GA_ev'] - df['GB_ev']
    df['stakes'] = df['GA_ev'] + df['GB_ev']
    
    
    sns.pairplot(df, vars=["choiceTime",'ml_drank',"diff",'stakes'], hue='type')
    plt.show()
    sns.violinplot(x=df['type'], y=df['choiceTime'], palette="husl")    
    plt.show()
    sns.lmplot(x='stakes', y='choiceTime', hue='type', data=df, scatter_kws={'alpha':0.2})


#    g = sb.pairplot(df, hue='range', size=2.5, plot_kws=dict(s=80, edgecolor="white", linewidth=2.5, alpha=0.3))
    

#%%
def correlate_dailyParameters(correlations):
    '''
    '''
    from macaque.f_models import define_model
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import itertools
    import statsmodels.api as sm
    
#    fractile_MLE
    
    #mle, legend, ranges, c_specific = extract_parameters(fractile_MLE, dataType = 'mle', minTrials = 40, revertLast = revertLast)
    #behaviour, legend, ranges, c_specific = extract_parameters(fractile_MLE, dataType = 'behaviour', minTrials = 40, revertLast = revertLast)
    
    
    position = np.argmax(['u_' in pname for pname in correlations.pNames[0][0]])
    long = np.sum(['u_' in pname for pname in correlations.pNames.iloc[-1][0]])
    
    risky = np.vstack(correlations.risky_params.values)[:,position:position+long]
    riskless = np.vstack(flatten(correlations.riskless_params.values))[:,position:position+long]

    risky_b = np.vstack(correlations.params_fractile.values)
    riskless_b = np.vstack(correlations.params_random.values)
    fractile = [risky, risky_b]
    random = [riskless, riskless_b]
#    random = [parameters[:,position:position+long] for parameters in random]
    legend = ['mle', 'behaviour']
    
    palette = itertools.cycle(sb.color_palette('colorblind'))
    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(10,6))
    fig2, ax2 = plt.subplots( 1, 2, squeeze = False, figsize=(10,6))
    i = 0
    for mm, bb, rr in zip(fractile, random, legend):
        if i == 0:
            color = next(palette)
            #mm = np.log(mm); bb = np.log(bb)
            print(' REGRESSION dataType: ', rr, '  ============================================= ' )
            ax[0,0].scatter(np.log(bb[:,0]), np.log(mm[:,0]), color = color)
            x = np.log(bb[:,0]); y = np.log(mm[:,0])
            x = sm.add_constant(x, prepend=True)
            mod = sm.OLS(y, x).fit()
            r1 = np.sqrt(mod.rsquared)
            print(' parameter temperature: --------------- ' )
            print('slope: ', mod.params[-1], '; p-val: ', mod.pvalues[-1], '; R^2: ' ,mod.rsquared)
            # -------------------------------------------------------------------------------------
            ax[0,1].scatter(bb[:,-1], mm[:,-1], color = color)
            x = bb[:,-1]; y = mm[:,-1]
            x = sm.add_constant(x, prepend=True)
            mod = sm.OLS(y, x).fit()
            print(' parameter height: --------------- ' )
            print('slope: ', mod.params[-1], '; p-val: ', mod.pvalues[-1], '; R^2: ' ,mod.rsquared)
            r2 = np.sqrt(mod.rsquared)
            # -------------------------------------------------------------------------------------

            corr = np.vstack(([1, r1], [[r2,1]]))
            mask = np.zeros_like(corr, dtype=np.bool)
            f, ax2 = plt.subplots(figsize=(11, 9))
            cmap = sb.diverging_palette(220, 10, as_cmap=True)
            sb.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax2)
            print(' 0 = temperature and 1 = height' )
        i+=1
    ax[0,0].legend(legend)
        
    bb_mags = np.log(np.vstack(random))
    mm_mags = np.log(np.vstack(fractile))
    x = bb_mags[:,0]; y = mm_mags[:,0]
    x = sm.add_constant(x, prepend=True)
    mod = sm.OLS(y, x).fit()
    print('General correlation of temperature: ======================================================')
    print(mod.summary())
    x = np.vstack(random)[:,1]; y = np.vstack(fractile)[:,1]
    x = sm.add_constant(x, prepend=True)
    mod = sm.OLS(y, x).fit()
    print('General correlation of height: ======================================================')
    print(mod.summary())
    
    ax[0,0].plot(np.linspace(min(bb_mags[:,0]), max(bb_mags[:,0])), np.linspace(min(bb_mags[:,0]),max(bb_mags[:,0])), '--', color = 'k')
    ax[0,1].plot(np.linspace(min(np.vstack(random)[:,-1]), max(np.vstack(random)[:,-1])), np.linspace(min(np.vstack(random)[:,-1]),max(np.vstack(random)[:,-1])), '--', color = 'k')
    ax[0,0].grid(); ax[0,1].grid()
    ax[0,0].set_ylabel('log temperature risky'); ax[0,0].set_xlabel('log temperature riskless')
    ax[0,1].set_ylabel('height risky'); ax[0,1].set_xlabel('height riskless')
    squarePlot(ax[0,0]); squarePlot(ax[0,1])
    
    #%%
    ff = define_model(correlations.model_used.iloc[-1])
    p0 = ff[1]
    utility = lambda pp: ff[-1](p0)['empty functions']['utility'](np.linspace(0,1,100), pp)
    all_bbs = []; all_mms = []
    
    palette = itertools.cycle(sb.color_palette('colorblind'))
    fig, ax = plt.subplots( 1, 1, squeeze = False, figsize=(6,6))
    for mm, bb, rr in zip(fractile, random, legend):
        color = next(palette)
        print('dataType: ', rr, '  ============================================= ' )
        bb_area = [np.sum(utility(pp))/100 for pp in bb]
        mm_area = [np.sum(utility(pp))/100 for pp in mm]
        ax[0,0].scatter(bb_area,mm_area, color = color)
        x = bb_area; y = mm_area
        x = sm.add_constant(x, prepend=True)
        mod = sm.OLS(y, x).fit()
        print(' parameter temperature: --------------- ' )
        print('slope: ', mod.params[-1], '; p-val: ', mod.pvalues[-1], '; R^2: ' ,mod.rsquared)
        # -------------------------------------------------------------------------------------
        all_bbs.extend(bb_area)
        all_mms.extend(mm_area)
       
    ax[0,0].legend(legend); ax[0,0].grid()          
    ax[0,0].plot(np.linspace(0,1), np.linspace(0,1), color = 'k')
    squarePlot(ax[0,0])
    
    x = all_bbs; y = all_mms
    x = sm.add_constant(x, prepend=True)
    mod = sm.OLS(y, x).fit()
    print('General correlation area under the curve: ======================================================')
    print(mod.summary())
    ax[0,0].plot(np.linspace(0, 1),
                  (np.linspace(0,1) * mod.params[-1]) + mod.params[0] , '--', color = 'k' )
    ax[0,0].set_ylabel('Area under curve risky'); ax[0,0].set_xlabel('Area under curve riskless')

    print(' =========================================================================')
    print(' =========================================================================')

#%%
def plot_fromLists(all_MLEs):
    '''
    '''
    import itertools
#    fig, ax = plt.subplots( len(all_MLEs), 4, squeeze = False, figsize=(10,15))
    palette = itertools.cycle(sb.color_palette('colorblind'))
    plotData = np.vstack(( np.linspace(0,1,100), np.ones(100), np.zeros(100),
            np.zeros(100), np.linspace(1,0,100), np.ones(100),
            np.zeros(100), np.zeros(100))).T
    
    for i,mle in enumerate(all_MLEs):
        
        print('failed fits:', mle.loc[mle.NM_success == False].date.values)
        mle = mle.loc[mle.NM_success == True]
        
        fig, ax = plt.subplots( 1, 4, squeeze = False, figsize=(10,5))
        cc = next(palette)
        [ ax[0,1].plot(np.linspace(0,0.5, 100), (mm.utility(np.linspace(0,1, 100)) - min(mm.utility(np.linspace(0,1, 100)))) / (max(mm.utility(np.linspace(0,1, 100))) - min(mm.utility(np.linspace(0,1, 100)))), color = cc, alpha = 0.3  ) for mm in mle.full_model]
        [ ax[0,0].plot(np.linspace(-1,1, 100), mm.softmax(plotData), color = cc, alpha = 0.3) for mm in mle.full_model ]
        [ ax[0,2].plot(np.linspace(0,1, 100), mm.probability(np.linspace(0,1,100)), color = cc, alpha = 0.3) for mm in mle.full_model ]
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
def compare_utilities(correlations, dataType = 'mle'):
    import statsmodels.api as sm
#    run a logistic regression on choosing left / right
    
    if dataType.lower() == 'mle':
        randomParams = np.vstack(flatten(correlations.riskless_params.values))[:,2:4]
        fractileParams = np.vstack(correlations.risky_params.values)[:,2:4]
    if dataType.lower() == 'behaviour':
        randomParams = np.vstack(correlations.params_random.values)
        fractileParams = np.vstack(correlations.params_fractile.values)
    
    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(10,6))
    ax[0,0].set_ylabel('temperature fractile'); ax[0,0].set_xlabel('temperature random')
    ax[0,1].set_ylabel('height fractile'); ax[0,1].set_xlabel('height random')
    
    x = randomParams[:,0]; y = fractileParams[:,0]
    x = sm.add_constant(x, prepend=True)
    mod = sm.OLS(y, x).fit()
    print(' parameter temperature: --------------- ' )
    print('slope: ', mod.params[-1], '; p-val: ', mod.pvalues[-1], '; R^2: ' ,mod.rsquared)
    # -------------------------------------------------------------------------------------
    ax[0,0].text( max(randomParams[:,0])*0.65, max(fractileParams[:,0])*0.9, 'R = ' + str(np.round(np.sqrt(mod.rsquared), 3)))
    ax[0,0].text(max(randomParams[:,0])*0.65, max(fractileParams[:,0])*0.85,'p = ' + str(np.round(mod.f_pvalue, 3)))   
    sb.regplot(x=randomParams[:,0], y=fractileParams[:,0], ax=ax[0,0], color = 'k')
    
    x = randomParams[:,1]; y = fractileParams[:,1]
    x = sm.add_constant(x, prepend=True)
    mod = sm.OLS(y, x).fit()
    print(' parameter height: --------------- ' )
    print('slope: ', mod.params[-1], '; p-val: ', mod.pvalues[-1], '; R^2: ' ,mod.rsquared)
    sb.regplot(randomParams[:,1], fractileParams[:,1], ax=ax[0,1],color = 'k')
    ax[0,1].text( max(randomParams[:,1])*0.65, max(fractileParams[:,1])*0.9, 'R = ' + str(np.round(np.sqrt(mod.rsquared), 3)))
    ax[0,1].text(max(randomParams[:,1])*0.65, max(fractileParams[:,1])*0.85,'p = ' + str(np.round(mod.f_pvalue, 3)))
    
    squarePlot(ax[0,0]); squarePlot(ax[0,1])
    plt.tight_layout()
    
    #%%
    ff = define_model(fractile_MLE.model_used.iloc[-1])
    p0 = ff[1]
    utility = lambda pp: ff[-1](p0)['empty functions']['utility'](np.linspace(0,1,100), pp)
    all_bbs = []; all_mms = []
    

    fig, ax = plt.subplots( 1, 1, squeeze = False, figsize=(6,6))

    print('Range: ', rr, '  ============================================= ' )
    bb_area = [np.sum(utility(pp))/100 for pp in bb]
    mm_area = [np.sum(utility(pp))/100 for pp in mm]
    ax[0,0].scatter(bb_area,mm_area, color = color)
    x = bb_area; y = mm_area
    x = sm.add_constant(x, prepend=True)
    mod = sm.OLS(y, x).fit()
    print(' parameter temperature: --------------- ' )
    print('slope: ', mod.params[-1], '; p-val: ', mod.pvalues[-1], '; R^2: ' ,mod.rsquared)
    # -------------------------------------------------------------------------------------
    all_bbs.extend(bb_area)
    all_mms.extend(mm_area)
   
    ax[0,0].legend(legend); ax[0,0].grid()          
    ax[0,0].plot(np.linspace(0,1), np.linspace(0,1), color = 'k')
    squarePlot(ax[0,0])
    
    x = all_bbs; y = all_mms
    x = sm.add_constant(x, prepend=True)
    mod = sm.OLS(y, x).fit()
    print('General correlation area under the curve: ======================================================')
    print(mod.summary())
    ax[0,0].plot(np.linspace(0, 1),
                  (np.linspace(0,1) * mod.params[-1]) + mod.params[0] , '--', color = 'k' )
    ax[0,0].set_ylabel('Area under curve MLE'); ax[0,0].set_xlabel('Area under curve Fractile')

    print(' =========================================================================')
    print(' =========================================================================')
#%%
def correlate_sameDay(correlations):
    return

#%%
def compare_predictionAccuracy():
    return

#%%
  
def winStay_fractile(fractile, Trials):
    '''
    '''
    from macaque.f_models import define_model
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import itertools
    from macaque.f_models import trials_2fittable
    from scipy.stats import sem
    import statsmodels.api as sm
    
    tt = fractile.getTrials(Trials)
    fig, ax = plt.subplots( 1, 1, squeeze = False, figsize=(10,6))
    palette = itertools.cycle(sb.color_palette('colorblind'))
    
    cc = next(palette)
    allData= []
    for date in tqdm(fractile.sessionDate.values , desc='gathering daily trial data'):    
        X, Y = trials_2fittable(tt.loc[tt['sessionDate'] == date], use_Outcomes=True)
            
        outcomes = X.outcomes.values[:-1]
        switch = np.abs(np.diff(Y))
        dataset = []
        for reward in np.unique(outcomes):
            index = (outcomes == reward)
            if sum(index) > 10:
                dataset.append([reward, 1-np.mean(switch[index])])
            
        dataset = np.vstack(dataset)
        ax[0,0].scatter(dataset[:,0], dataset[:,-1], alpha = 0.2, color = cc)
#        ax[0,0].plot(dataset[:,0], dataset[:,-1], alpha = 0.2, color = cc)
        allData.append(dataset)
    
    ax[0,0].axhline(0.5, color = 'k')
    allData = np.vstack(allData); datapoints = []
    for reward in np.unique(allData[:,0]):
        index = (allData[:,0] == reward)
        datapoints.append([reward, np.mean(allData[:,1][index]), sem(allData[:,1][index])])
    datapoints = np.vstack(datapoints)
#        ax[0,i].plot(datapoints[:,0], datapoints[:,1], color = cc)
    ax[0,0].set_ylabel('proportion of side switches (0 no switch)')
    ax[0,0].set_xlabel('past outcome EV')
    ax[0,0].set_ylim([0,1])
    squarePlot(ax[0,0])
    
    mod = sm.OLS(allData[:,1], sm.add_constant(allData[:,0])).fit()
#    print('Range: ', rr, '===================================')
    print(mod.summary())
    ax[0,0].plot(np.linspace(min(allData[:,0]), max(allData[:,0])),
              (np.linspace(min(allData[:,0]), max(allData[:,0])) * mod.params[-1]) + mod.params[0] ,'--', color = cc )
        
    plt.tight_layout()
    
    #%%
    fig, ax = plt.subplots( 1, 1, squeeze = False, figsize=(10,6))
    palette = itertools.cycle(sb.color_palette('colorblind'))

    cc = next(palette)
    allData= []
    
    #THIS IS A BAD MEASURE ACTUALLY
    
    for date in tqdm(fractile.sessionDate.values , desc='gathering daily trial data'):    
            X, Y = trials_2fittable(tt.loc[tt['sessionDate'] == date], use_Outcomes=True)
            
            gA = np.vstack(X[[ 'A_m1',  'A_p1',  'A_m2',  'A_p2' ]].values)
            gB = np.vstack(X[[ 'B_m1',  'B_p1',  'B_m2',  'B_p2' ]].values)
            chA = Y
            outcomes = X.outcomes.values
            
            ggA = np.array([not all(left[-2:]) == 0 for left in gA])
            ggB = np.array([not all(right[-2:]) == 0 for right in gB])
            
            a_is_gamble_is_chosen = (ggA == True) & (chA == 1)
            b_is_gamble_is_chosen = (ggB == True) & (chA == 0)

            where = (a_is_gamble_is_chosen == True) | (b_is_gamble_is_chosen == True) 
            switch = np.abs(np.diff(where))
            where = where[:-1]
            outcomes = outcomes[:-1][where]
            switch = switch[where]
            
            dataset = []
            for reward in np.unique(outcomes):
                index = (outcomes == reward)
                if sum(index) > 5:
                    dataset.append([reward, 1-np.mean(switch[index])])
            
            dataset = np.vstack(dataset)
            ax[0,0].scatter(dataset[:,0], dataset[:,-1], alpha = 0.2, color = cc)
            allData.append(dataset)
    
    ax[0,0].axhline(0.5, color = 'k')
    allData = np.vstack(allData); datapoints = []
    for reward in np.unique(allData[:,0]):
        index = (allData[:,0] == reward)
        datapoints.append([reward, np.mean(allData[:,1][index]), sem(allData[:,1][index])])
    datapoints = np.vstack(datapoints)
    
    ax[0,0].set_ylabel('proportion of gamble/safe switches (0 no switch)')
    ax[0,0].set_xlabel('past outcome EV')
    ax[0,0].set_ylim([0,1])
    squarePlot(ax[0,0])
    
    mod = sm.OLS(allData[:,1], sm.add_constant(allData[:,0])).fit()
#    print('Range: ', rr, '===================================')
    print(mod.summary())
    ax[0,0].plot(np.linspace(min(allData[:,0]), max(allData[:,0])),
              (np.linspace(min(allData[:,0]), max(allData[:,0])) * mod.params[-1]) + mod.params[0] ,'--', color = cc )
    
    plt.tight_layout()

#%%
def plot_overlappingSoftmaxes(fractile, perDaySM, Behaviour):
    '''
    '''
    from scipy.stats import sem
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import itertools
    from macaque.f_uncertainty import bootstrap_sample, bootstrap_function
    
    fig, ax = plt.subplots(1,len(fractile.utility.unique()), squeeze = False, figsize=(12,6))
    fig2, ax2 = plt.subplots(1,1, squeeze = False, figsize=(12,6))
    palette = itertools.cycle(sb.color_palette('colorblind'))
    all_fits= []; cc=[]
    for i,util in enumerate(fractile.utility.unique()):
        color = next(palette)
        cc.append(color)
        df = fractile.loc[fractile.utility == util]
        function = df.func.iloc[-1]
        softmax = lambda params: function(np.linspace(-0.5, 0.5, 100), 0, params)
        fits = np.vstack(df.pFit.values)[:,-1]
        mean, lower, upper = bootstrap_function(softmax, fits, 'median')
        
        [ax[0,i].plot(np.linspace(-0.5,0.5, 100), softmax(ff), color = color) for ff in fits]
        ax2[0,0].plot(np.linspace(-0.5,0.5, 100) ,mean, color = color) 
        ax2[0,0].fill_between(np.linspace(-0.5,0.5, 100), y1=lower, y2=upper, alpha=0.1, color=color)
    
        all_fits.append(fits)
            
    ax2[0,0].legend(fractile.utility.unique())
    ax2[0,0].set_xlim(-0.2, 0.2)
    ax2[0,0].grid()
    
#    add_subplot_axes[ax2[0,0], rect = [0.2,0.2,0.7,0.7]]
    
    rescale = lambda x: 1/x
    
    a = plt.axes([.65, .2, .2, .3], facecolor='w')
    plt.bar(np.arange(len(all_fits)), [np.mean(rescale(ff)) for ff in all_fits], color = cc, yerr= [sem(rescale(ff)) for ff in all_fits])
    plt.xlabel('utility level')
    plt.xticks([])
    plt.ylabel('softmax slope')

    #%%
    
    pdsm = perDaySM.loc[np.isin(perDaySM.sessionDate, Behaviour.date)]
    fig, ax = plt.subplots(1,len(pdsm.midpoint.unique()), squeeze = False, figsize=(12,6))
    fig2, ax2 = plt.subplots(1,1, squeeze = False, figsize=(12,6))
    palette = itertools.cycle(sb.color_palette('colorblind'))
    all_fits= []; cc=[]
    for i,util in enumerate(pdsm.midpoint.unique()):
        color = next(palette)
        cc.append(color)
        df = pdsm.loc[pdsm.midpoint == util]
#        function = df.func.iloc[-1]
#        softmax = lambda params: function(np.linspace(-0.5, 0.5, 100), 0, params)
        fits = np.hstack(df.temperature.values)
        mean, lower, upper = bootstrap_function(softmax, fits, 'median')
        
        [ax[0,i].plot(np.linspace(-0.5,0.5, 100), softmax(ff), color = color) for ff in fits]
        ax2[0,0].plot(np.linspace(-0.5,0.5, 100) ,mean, color = color) 
        ax2[0,0].fill_between(np.linspace(-0.5,0.5, 100), y1=lower, y2=upper, alpha=0.1, color=color)
    
        all_fits.append(fits)
            
    ax2[0,0].legend(pdsm.midpoint.unique())
    ax2[0,0].set_xlim(-0.2, 0.2)
    ax2[0,0].grid()
    
#    add_subplot_axes[ax2[0,0], rect = [0.2,0.2,0.7,0.7]]
    
    rescale = lambda x: 1/x
    
    a = plt.axes([.65, .2, .2, .3], facecolor='w')
    plt.bar(np.arange(len(all_fits)), [np.mean(rescale(ff)) for ff in all_fits], color = cc, yerr= [sem(rescale(ff)) for ff in all_fits])
    plt.xlabel('utility level')
    plt.xticks(np.arange(len(all_fits)))
    plt.ylabel('softmax slope')
    ax = plt.gca()
    ax.set_xticklabels(pdsm.midpoint.unique())

#%%
def plot_ratioBar(perDaySM):
    '''
    '''
    from scipy.stats import sem
    
    ratios = perDaySM.freq_sCh.values
    gaps = perDaySM.gap.values
    midpoints = perDaySM.midpoint.values
    
    fig, ax = plt.subplots(1,3, squeeze = False, figsize=(12,6))
    Y = []
    for i,gg in enumerate([0.02, 0.04, 0.06]):
        choice_rr = np.array([np.array(sCh)[np.array(point) == gg] for point, sCh in zip(gaps, ratios)])
        where = np.array([False if np.size(cc) == 0 else True for cc in choice_rr])
        data = np.vstack((np.hstack((choice_rr[where])), midpoints[where])).T
        
        y = []; x = []
        for option in np.unique(data[:,-1]):
            where_2 = (data[:,-1] == option)
            data_mini = data[where_2,:]
            y.append([np.mean(data_mini[:,0]), np.std(data_mini[:,0]), sem(data_mini[:,0])])
            x.extend([option])
        x=np.hstack(x); y=np.vstack(y)
        
        ax[0,i].bar(x-0.02, y[:,0], width = 0.02, yerr=y[:,-1])
        ax[0,i].bar(x, 1-y[:,0], width = 0.02, yerr=y[:,-1])
        plot_utility = np.cumsum(y[:,0]) - 0.5

        plot_utility = (plot_utility - min(plot_utility)) / (max(plot_utility) - min(plot_utility))
        ax[0,i].plot(x-0.02/2, plot_utility, '-bo', color='blue')
        ax[0,i].plot(np.linspace(min(x-0.02/2), max(x-0.02/2)), np.linspace(min(plot_utility), max(plot_utility)), '--', color='k')
        ax[0,i].axhline(0.5)
        squarePlot(ax[0,i])
        ax[0,i].legend(['choose high','choose low'])
        Y.append(y)
    
    fig, ax = plt.subplots(1,3, squeeze = False, figsize=(8,6))
    for i,y in enumerate(Y):
        ax[0,0].bar((x*10)+(i/10), y[:,0], yerr=y[:,-1], width=0.1)
        ax[0,0].bar((x*10)+(i/10), 1-y[:,0], yerr=y[:,-1], width=0.1, color='k', alpha=0.5)
        ax[0,0].plot((x*10)+(i/10), y[:,0], '--')
        
        ax[0,1].plot((x*10), np.gradient(y[:,0]))
        ax[0,1].axhline(0, color='k')

        util = np.cumsum(y[:,0])
        util = (util - min(util)) / (max(util)-min(util))
        ax[0,2].plot((x*10), util, '-o')
        
    ax[0,0].axhline(0.5, color = 'k')
    ax[0,0].set_xticks((x*10) + 0.1); ax[0,0].set_xticklabels(x)
    ax[0,0].legend([0.02, 0.04, 0.06])
    ax[0,1].set_xticks((x*10) + 0.1); ax[0,1].set_xticklabels(x)
    ax[0,1].legend([0.02, 0.04, 0.06])
    
    ax[0,2].set_xticks((x*10) + 0.1); ax[0,2].set_xticklabels(x)
    ax[0,2].legend([0.02, 0.04, 0.06])
    ax[0,i].plot(np.linspace(min((x*10)), max((x*10))), np.linspace(0,1), '--', color='k')

    ax[0,0].set_ylim([0.5, 1.0])
    squarePlot(ax[0,0]) ; squarePlot(ax[0,1]); squarePlot(ax[0,2])       

    
#%%
def fit_runningTrials(perDaySM, fractile, Trials, Model, minTrials = 100):
    
    from macaque.f_models import get_modelLL, trials_2fittable, LL_fit
    from macaque.f_probabilityDistortion import plot_MLEfitting
    np.warnings.filterwarnings('ignore')

    use_Outcomes = False
    if type(Model) == list:
        MLE_fits = []
        for model in Model:
            MLE_fits.append(fit_runningTrials(perDaySM, fractile, Trials, Model=model, minTrials = 100))
        return MLE_fits
    else:
    
        uniqueDays = np.intersect1d(perDaySM.sessionDate.values, 
                                          fractile.sessionDate.values)
        dList = []; pastEffect = []
        for date in tqdm(np.sort(uniqueDays), desc=Model):
            tt = Trials.loc[Trials['sessionDate'] == date]
            to_delete = tt.loc[tt.outcomesCount.apply(lambda x: any(np.array(x)>2))].blockNo.unique() 
            tt.drop(tt.loc[np.isin(tt.blockNo.values, to_delete)].index, inplace=True)
            
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
        return MLE_fits

#%%
def compare_correlations(correlation):
    '''
    '''
    not_empties = [False if len(xx)==0 else True for xx in correlation.riskless_params.values]
    correlation = correlation.loc[not_empties]
    
    from scipy import stats
    fig, ax = plt.subplots(1,2, squeeze = False, figsize=(6,3))
#    plt.errorbar()
    xx = [1,2]
    yy = np.hstack((correlation.behaviour_RMSerr.mean(), correlation.fit_RMSerr.mean()))
    yyerr =  np.hstack((correlation.behaviour_RMSerr.sem(), correlation.fit_RMSerr.sem()))
    ax[0,0].bar(xx, yy, yerr= yyerr)
    ax[0,0].set_ylim(0,1)
    ax[0,0].axhline(0)
    ax[0,0].set_xticks(xx)
    ax[0,0].set_xticklabels(
        ['err behaviour', 'err MLE'], rotation=45, horizontalalignment='center')
    
    #parameter comparisons
    try:
        riskless = np.vstack(correlation.riskless_params.apply(lambda x: x[0]))
    except:
        riskless = np.vstack(correlation.riskless_params.values)
    risky = np.vstack(correlation.risky_params.values)[:, :4]
    ax[0,1].bar(range(len(riskless[-1])), np.mean(riskless - risky, 0), yerr = stats.sem(riskless - risky, 0))
    ax[0,1].axhline(0)
    pNames = correlation.pNames.iloc[0][0]
    ax[0,1].set_xticks(range(len(riskless[-1])))
    ax[0,1].set_xticklabels(
        pNames, rotation=45, horizontalalignment='center')

#%%
def plot_randomSoftmax(Behaviour, dateNo = 21):
    sigmoid = lambda x, p2: np.array(1 / (1 + np.exp(-(x - 0) / p2)))
    import scipy.stats as stats
    fig, ax = plt.subplots(1,1, squeeze = False, figsize=(6,3))

    date = Behaviour.sessionDate[dateNo]
    df = Behaviour.loc[Behaviour.sessionDate == date].copy()
    ff1 = sigmoid(np.linspace(0,0.07,100), df.temperature[df.midpoint == 0.3].values)
    ff0 = sigmoid(np.linspace(-0.05,0,100), df.temperature[df.midpoint == 0.3].values)
    ff2 = sigmoid(np.linspace(0.07, 0.1, 100), df.temperature[df.midpoint == 0.3].values)
    xx = df.gap.loc[df.midpoint == 0.3].values[0]
    yy = df.freq_sCh.loc[df.midpoint == 0.3].values[0]
    ax[0,0].plot(np.linspace(0,0.07,100), ff1, color='k', alpha = 0.5)
    ax[0,0].plot(np.linspace(-0.05,0,100), ff0, '--', color='k', alpha = 0.5)
    ax[0,0].plot(np.linspace(0.07, 0.1, 100), ff2, '--', color='k', alpha = 0.5)
    ax[0,0].scatter(xx, yy, color='k')
    ax[0,0].axvline(0)
    ax[0,0].grid(axis='y')
    ax[0,0].scatter(0.03, sigmoid(0.03, df.temperature[df.midpoint == 0.3]), color = 'magenta', s=70 )
    ax[0,0].set_ylabel('Choice Ratio (High Value)')
    ax[0,0].set_xlabel('ΔValue')
    plt.tight_layout()

    point = stats.norm.ppf(sigmoid(0.03, df.temperature[df.midpoint == 0.3]), loc=0, scale=1)

#    fig, ax = plt.subplots(2,1, squeeze = False, figsize=(6,3))
    p = sb.JointGrid(x=np.linspace(0,1,100), y=stats.norm.ppf(np.linspace(0,1,100), loc=0, scale=1))
    sb.lineplot(x=np.linspace(0,1,1000), y=stats.norm.ppf(np.linspace(0,1,1000), loc=0, scale=1), color='black', ax=p.ax_joint)
    sb.lineplot( y=np.linspace(-3,3,1000), x=stats.norm.pdf(np.linspace(-3,3,1000)), color='cyan',
                alpha = 0.5, ax=p.ax_marg_y)
    sb.lineplot( y=np.linspace(-3,3,1000), x=stats.norm.pdf(np.linspace(-3,3,1000), loc=0.61), color='magenta',
                alpha = 0.5, ax=p.ax_marg_y)
    sb.lineplot( x=np.linspace(0,1,1000), y=stats.norm.pdf(np.linspace(-3,3,1000)), color='cyan',
                alpha = 0.5, ax=p.ax_marg_x)
    sb.lineplot( x=np.linspace(0,1,1000), y=stats.norm.pdf(np.linspace(-3,3,1000), loc=0.61), color='magenta',
                alpha = 0.5, ax=p.ax_marg_x)
    sb.scatterplot(sigmoid(0.03, df.temperature[df.midpoint == 0.3]), point, color='magenta', ax=p.ax_joint)
#    ax[0,0].plot(np.linspace(0,1,100), stats.logistic.ppf(np.linspace(0,1,100), loc=0, scale=1))
    p.ax_joint.axvline(0.5, color = 'k')
    p.ax_joint.axhline(0, color = 'k')
#    p.ax_joint.fill_between( np.linspace(0,0,100), stats.norm.pdf(np.linspace(-10,10,100)))
#    plt.setp(p.ax_marg_x.get_yticklabels(), visible=True)
    p.ax_marg_y.yaxis.grid(True)
    plt.setp(p.ax_marg_y.get_yticklabels(), visible=True)
    p.ax_marg_x.set_title('normal distribution')
    p.ax_joint.set_ylabel(r'$Φ^{−1}$')
    p.ax_joint.set_xlabel('Choice Ratio (High Value)')


#%%
def plot_illustration(params):
    '''
    '''
    params = np.array(params)
    from scipy.stats import logistic
    def sCDF(mm, params):
        Xs = np.atleast_1d(mm) #range position
        inflection, temp = params[1], params[0]
        return np.where((inflection > 1 or inflection < 0),
                        [0] * len(Xs),
                        np.ravel([np.where(X<inflection, inflection*((X/inflection)**temp), 1-((1-inflection)*(((1-X)/(1-inflection))**temp))) for X in Xs])
                        )

    U = lambda mm, param_u: sCDF(mm, param_u)
    param_sfm = params[0]
    param_util = params[1:]

    CYAN = '#009FE3'
    MAGENTA = '#E6007E'

    norm1 = 1/np.sum(logistic.pdf(np.linspace(-1.5,1.5,1000), scale=(1/param_sfm), loc=0))
    norm2 = 1/np.sum(logistic.pdf(np.linspace(-1.5,1.5,1000), scale=(1/param_sfm)/np.sqrt(2), loc=-.75))

    fig, ax = plt.subplots(1,3, squeeze = False, figsize=(12,3))
    ax[0,0].fill(np.linspace(-1.25,1.25,1000), logistic.pdf(np.linspace(-1.25,1.25,1000), scale=(1/param_sfm)/np.sqrt(2), loc=-.75) * norm2, color = CYAN, alpha=0.3)
    ax[0,0].fill(np.linspace(-1.25,1.25,1000), logistic.pdf(np.linspace(-1.25,1.25,1000), scale=(1/param_sfm)/np.sqrt(2), loc=.75) * norm2, color = MAGENTA, alpha=0.3)
    ax[0,0].fill(np.linspace(-1.25,1.25,1000), logistic.pdf(np.linspace(-1.25,1.25,1000), scale=(1/param_sfm)/np.sqrt(2), loc=-0.5) * norm2, color = CYAN, alpha=0.2)
    ax[0,0].fill(np.linspace(-1.25,1.25,1000), logistic.pdf(np.linspace(-1.25,1.25,1000), scale=(1/param_sfm)/np.sqrt(2), loc=0.5) * norm2, color = MAGENTA, alpha=0.2)
    ax[0,0].fill(np.linspace(-1.25,1.25,1000), logistic.pdf(np.linspace(-1.25,1.25,1000), scale=(1/param_sfm)/np.sqrt(2), loc=-.25) * norm2, color = CYAN, alpha=0.1)
    ax[0,0].fill(np.linspace(-1.25,1.25,1000), logistic.pdf(np.linspace(-1.25,1.25,1000), scale=(1/param_sfm)/np.sqrt(2), loc=.25) * norm2, color = MAGENTA, alpha=0.1)

    ax[0,0].fill(np.linspace(-1.25,1.25,1000), logistic.pdf(np.linspace(-1.25,1.25,1000), scale=(1/param_sfm), loc=0) * norm1, color = '#6700AA', alpha=0.6)
    ax[0,0].yaxis.set_ticks_position('left')
    ax[0,0].xaxis.set_ticks_position('bottom')
    ax[0,0].spines['right'].set_visible(False)
    ax[0,0].spines['top'].set_visible(False)
#    ax2=ax[0,0].twinx()
#    for m in np.linspace(0,1,6):
#        ax2.scatter(U(np.linspace(m-0.15,m+0.15,5), param_util) - U(np.linspace(m+0.15,m-0.15,5), param_util), [m]*5, color='k')
#    ax2.set_ybound(-0.05, 1.05)
    ax[0,0].set_title('Random Error on Utilities')
    ax[0,0].set_xlabel('noise distribution')
#    ax[0,0].set_ylabel('relative reward value')
#    ax[0,0].spines['top'].set_visible(False)
    ax[0,0].spines['left'].set_visible(False)
#    ax[0,0].spines['right'].set_visible(False)
    ax[0,0].set_yticklabels([])
#     --------------------------------------------------------------------------------------------------
    ax[0,1].plot(np.linspace(-1.25,1.25,1000), logistic.cdf(np.linspace(-1.25,1.25,1000), loc=0, scale=1/param_sfm ), color='#6700AA', linewidth=3)
    ax[0,1].axvline(-0.5, color='k', linestyle='--')
    ax[0,1].axvline(0.5, color='k', linestyle='--')
    i=1/100
    for _ in range(25):
        ax[0,1].fill_between(np.linspace(0,1,100)+i, -0.05, 1.05, color=MAGENTA,alpha=0.01)
        ax[0,1].fill_between(np.linspace(-1,0,100)-i, -0.05, 1.05, color=CYAN,alpha=0.01)
        i+=1/25
#    for m in np.linspace(0,1,6):
#        ax[0,1].scatter(U(np.linspace(m-0.15,m+0.15,5), param_util) - U(np.linspace(m+0.15,m-0.15,5), param_util), [m]*5, color='k')
    ax[0,1].yaxis.set_ticks_position('left')
    ax[0,1].xaxis.set_ticks_position('bottom')
    ax[0,1].spines['top'].set_visible(False)
    ax[0,1].spines['bottom'].set_visible(False)
    ax[0,1].set_xbound(-1, 1)
    ax[0,1].set_ybound(-0.05, 1.05)
#    ax[0,1].set_title('Probability of Choosing Red')
    ax[0,1].set_xlabel('relative value of red option')
    ax[0,1].set_ylabel('probability of choosing red option')

    # --------------------------------------------------------------------------------------------------
    ax[0,2].plot(np.linspace(0,1,100), U(np.linspace(0,1,100),param_util), color = 'k',linewidth = 3)
    ax[0,2].plot(np.linspace(0,1,100),np.linspace(0,1,100), color = 'k', linestyle='--')
#    for m in np.linspace(0,1,6):
#        ax[0,2].scatter([m]*5, U(np.linspace(m-0.15,m+0.15,5), param_util), color='k')
    ax[0,2].spines['right'].set_visible(False)
    ax[0,2].spines['top'].set_visible(False)

    bound = logistic.ppf(0.95, loc=0, scale=1/param_sfm )
    for _ in range(50):
        ax[0, 2].fill_between( np.linspace(0,1,100), U(np.linspace(0,1,100),param_util)-bound, U(np.linspace(0,1,100),param_util)+bound, color='#6700AA', alpha=0.01)
        bound = bound - (bound/25)
    ax[0,2].set_ybound(-0.05, 1.05)
    ax[0,2].set_xbound(-0.005, 1.05)
    ax[0,2].set_title('Utility of Rewards')
    ax[0,2].set_xlabel('reward magnitude')
    ax[0,2].set_ylabel('utility')
    plt.tight_layout()


#%%
def plot_midpointSoftmax(softmaxDF,
                      minPoints=3,
                      plotit=True):
    '''
    Basically a psychometrics-elicitation function but centered around a midpoint between safes rather than on a gamble or safe versus its secondary options
    '''
    import scipy.optimize as opt
    from scipy import stats
    import matplotlib.cm as cm

    norm = lambda x: (x - min(x)) / (max(x) - min(x))
    sigmoid = lambda x, p2: np.array(1 / (1 + np.exp(-(x - 0) / p2)))

    dfs = []
    i = 0

    # plot the softmax fits between small differences
    for date in softmaxDF.sessionDate.unique():
        df = softmaxDF.loc[softmaxDF.sessionDate == date]
        df = df.loc[~np.isnan(df.temperature)]
        if len(df) < minPoints:
            continue

        fig, ax = plt.subplots(int(np.ceil(len(df)/9)), 9, squeeze = False,
                               figsize=(15, 2*int(np.ceil(len(df)/9))))
        r=0; c=0
        for index,row in df.iterrows():
            ax[r,c].set_title(str(row['midpoint']))
            ax[r,c].plot(np.arange(0,0.1,0.001), sigmoid(np.arange(0,0.1,0.001), row['temperature']), color='blue', linestyle='--')
            ax[r,c].scatter(row['gap'], row['freq_sCh'], color='k')
            ax[r,c].axvline(0.03, color = 'green')
            ax[r,c].grid(True)
            ax[r,c].vlines(0, 0, 1, color='red')
            ax[r,c].text(0.01, 0.1, 'temp:'+str(np.round(row['temperature'], 2)))
            c+=1
            if c>8:
                c=0
                r+=1

        if c < 8 and r == int(np.ceil(len(df)/9)) - 1:
            while c <= 8:
                fig.delaxes(ax[r, c])
                c += 1
        fig.suptitle('sessionDate: ' + str(df['sessionDate'].unique()[0]))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#%%
def predict_certaintyEquivalent(correlations, gambleCEs, best_model):
    '''
    '''
    from scipy import stats
    from collections import Counter
    from macaque.f_models import define_model
    #need to make this work via a collection of lists
    dataList = []
    df = correlations.loc[correlations.riskless_params.apply(lambda x: len(x) > 0)]
    risky = np.vstack(df.risky_params.values)
    riskless = np.vstack(flatten(df.riskless_params.values))
    riskless2 = np.vstack((riskless.T, risky[:,-1])).T
    riskless = np.vstack((riskless.T, np.array(len(riskless)*[1]))).T
    dataList = []
    dataList.append(riskless); dataList.append(riskless2); dataList.append(risky)
    
    #%%

    #utility parameters are 1 and 2
    risklessRange = unique_listOfLists(correlations.riskless_range.values)[0]
    riskyRange = unique_listOfLists(correlations.risky_range.values)[0]
    ranges = np.vstack((risklessRange, riskyRange))
    
    norm = lambda x: (x - riskyRange[0]) / (riskyRange[1] - riskyRange[0])
    randomDFRange = [min(risklessRange), max(risklessRange)]
    startRandom = norm(randomDFRange[0])
    endRandom = norm(randomDFRange[1])    
    
#    meanParams = [np.mean(params,0) for params in dataList]
#    semParams = [[stats.sem(params)] for params in dataList]

    position = np.argmax(['u_' in pname for pname in correlations.pNames.iloc[-1][0]])
    long = np.sum(['u_' in pname for pname in correlations.pNames.iloc[-1][0]])

#    gambleCEs.primary
    ff = define_model(best_model)
    p0 = ff[1]
    utility = ff[-1](p0)['empty functions']['utility']
    value = ff[-1](p0)['empty functions']['value']
    #already with probability distortion!!!

    gg = gambleCEs.loc[gambleCEs.spread == Counter(gambleCEs.spread.values).most_common(1)[0][0]]
    gg = gg.loc[gg.m_range.apply(lambda x: x == [0, 0.5])]
    gg = gg.loc[gg.equivalent.apply(lambda x: riskyRange[0]<x<riskyRange[1])]
    primaries = np.vstack(gg.primary.values)
    print(np.vstack(unique_listOfLists(primaries)))
    primaries[:,0::2] = (primaries[:,0::2] - riskyRange[0]) / (riskyRange[1] - riskyRange[0])
    primaries = np.vstack(unique_listOfLists(primaries))
    EVs = [np.round((p[0]*p[1]) + (p[2] * p[3]),3) for p in primaries]

    CE_riskless = []; CE_risky = []
    from scipy.optimize import minimize
    bnds = [[0, 1.0]]
    
    
#    palette = itertools.cycle(sb.color_palette('colorblind'))
    
    all_CEs = []
    for dd in tqdm(dataList):
#        color = next(palette)
        divided_CEs = []
        for params in dd:
            CEs = []
            vv_dist = value(primaries, params[position:])
            for v1, ev in zip(vv_dist,EVs):
                uu = lambda x: utility(x, params[position:position+long])
                f1 = lambda x: (v1 - uu(x))**2
                CEs.append(np.hstack([minimize(f1, 0.5, method='SLSQP').x, ev])*np.max(ranges))
            divided_CEs.append(CEs)
        all_CEs.append(np.vstack(divided_CEs))
        

    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(6,6))
    ax[0,0].scatter( x= jitter(all_CEs[0][:,1], 0.01), y=all_CEs[0][:,0], color='red', zorder=10, alpha = 0.20)
    ax[0,0].scatter( x= jitter(all_CEs[1][:,1], 0.01), y=all_CEs[1][:,0], color='darkred', zorder=10, alpha = 0.20)
    ax[0,0].scatter( x= jitter(all_CEs[2][:,1], 0.01), y=all_CEs[2][:,0], color='blue', zorder=10, alpha = 0.20)
    ax[0,0].scatter( x= jitter(gg.primaryEV.values, 0.0075), y=gg.equivalent.values, color='g', zorder=10)
    ax[0,0].plot(np.linspace(riskyRange[0]+0.05,riskyRange[1]), np.linspace(riskyRange[0]+0.05,riskyRange[1]), '--', color='k')
    ax[0,0].set_xlabel('gamble EV')
    ax[0,0].set_ylabel('certainty equivalent')
    ax[0,0].grid()
    squarePlot(ax[0,0])
    
    index = np.unique(all_CEs[0][:,1])
    riskless_ce = [np.mean(all_CEs[0][:,0][all_CEs[0][:,1] == ii]) for ii in index]
    risky_ce = [np.mean(all_CEs[2][:,0][all_CEs[2][:,1] == ii]) for ii in index]
    riskless_sem = [stats.sem(all_CEs[0][:,0][all_CEs[0][:,1] == ii]) for ii in index]
    risky_sem = [stats.sem(all_CEs[2][:,0][all_CEs[2][:,1] == ii]) for ii in index]
    riskless_ce2 = [np.mean(all_CEs[1][:,0][all_CEs[1][:,1] == ii]) for ii in index]
    riskless_sem2 = [stats.sem(all_CEs[1][:,0][all_CEs[1][:,1] == ii]) for ii in index]
    
    gg_ce = [np.mean(gg.equivalent.values[gg.primaryEV.values == ii]) for ii in gg.primaryEV.unique()]
    gg_sem = [stats.sem(gg.equivalent.values[gg.primaryEV.values == ii]) for ii in gg.primaryEV.unique()]
      
    ax[0,1].errorbar(index, riskless_ce, yerr = riskless_sem, color = 'red')
    ax[0,1].errorbar(index, riskless_ce2, yerr = riskless_sem2, color = 'darkred')
    ax[0,1].errorbar(index, risky_ce, yerr = risky_sem, color = 'blue')
    ax[0,1].errorbar(gg.primaryEV.unique(), gg_ce, yerr=gg_sem, color = 'green')
    ax[0,1].legend([ 'riskless', 'riskless2','risky', 'equivariant'])
    ax[0,1].set_ylim([0, 0.5]); ax[0,1].set_xlim([riskyRange[0]+0.05,riskyRange[1]])
    ax[0,1].plot(np.linspace(riskyRange[0]+0.05,riskyRange[1]), np.linspace(riskyRange[0]+0.05,riskyRange[1]), '--', color='k', alpha = 0.2, label='_nolegend_')

    squarePlot(ax[0,1])
    plt.tight_layout()    
    
    riskless_old = np.vstack((all_CEs[0].T, [0] * len(all_CEs[0]))).T
    riskless_new = np.vstack((all_CEs[1].T, [0] * len(all_CEs[1]))).T
    risky_new = np.vstack((all_CEs[2].T, [1] * len(all_CEs[2]))).T
    gg_new = np.vstack((gg.equivalent.values, gg.primaryEV.values, [2] * len( gg.equivalent.values)  )).T
    
    print('predicted CEs: ', len(riskless_old))
    print('true CEs: ', len(gg_new))
    
    anovaData = np.vstack((riskless_old, risky_new))
    print('conditions = ', 'riskless_old /' ,  'risky /' )
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    ddf = pd.DataFrame(anovaData, columns=['CE','EV', 'Condition' ])
    formula = 'CE ~ C(EV) + C(Condition) + C(EV):C(Condition)'
    model = ols(formula, data=ddf).fit()
    print(f"Overall model F({model.df_model: .0f}, {model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4}")
    aov_table = anova_lm(model, typ=2)
    eta_squared(aov_table)
#    print(model.summary())
    #    omega_squared(aov_table)
    print(' -------------------------------------------')
    print(aov_table)
    print('Nb,Nm =', [
        str(len(ddf.loc[ddf.Condition == cc]))
        for cc in ddf.Condition.unique()
    ])
    print('\n')
    
    #%%
    anovaData = np.vstack((riskless_new, risky_new))
    print('conditions = ', 'riskless /' ,  'risky /' )
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    ddf = pd.DataFrame(anovaData, columns=['CE','EV', 'Condition' ])
    formula = 'CE ~ C(EV) + C(Condition) + C(EV):C(Condition)'
    model = ols(formula, data=ddf).fit()
    print(f"Overall model F({model.df_model: .0f}, {model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4}")
    aov_table = anova_lm(model, typ=2)
    eta_squared(aov_table)
#    print(model.summary())
    #    omega_squared(aov_table)
    print(' -------------------------------------------')
    print(aov_table)
    print('Nb,Nm =', [
        str(len(ddf.loc[ddf.Condition == cc]))
        for cc in ddf.Condition.unique()
    ])
    print('\n')
    
    #%%    
    print('conditions = ', 'riskless /' ,  'true CEs /' )
    anovaData = np.vstack((riskless_new, gg_new))    
#    print('\n', '\n')
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    ddf = pd.DataFrame(anovaData, columns=['CE','EV', 'Condition' ])
    formula = 'CE ~ C(EV) + C(Condition) + C(EV):C(Condition)'
    model = ols(formula, data=ddf).fit()
    print(f"Overall model F({model.df_model: .0f}, {model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4}")
    aov_table = anova_lm(model, typ=2)
    eta_squared(aov_table)
#    print(model.summary())
    #    omega_squared(aov_table)
    print(' -------------------------------------------')
    print(aov_table)
    print('Nb,Nm =', [
        str(len(ddf.loc[ddf.Condition == cc]))
        for cc in ddf.Condition.unique()
    ])
    print('\n')
#    print(model.t_test_pairwise('EV', method='bonferroni').result_frame)
    #%%
    print('conditions = ', 'risky /' ,  'true CEs /' )
    anovaData = np.vstack((gg_new, risky_new))    
#    print('\n', '\n')
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    ddf = pd.DataFrame(anovaData, columns=['CE','EV', 'Condition' ])
    formula = 'CE ~ C(EV) + C(Condition) + C(EV):C(Condition)'
    model = ols(formula, data=ddf).fit()
    print(f"Overall model F({model.df_model: .0f}, {model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4}")
    aov_table = anova_lm(model, typ=2)
    eta_squared(aov_table)
#    print(model.summary())
    #    omega_squared(aov_table)
    print(' -------------------------------------------')
    print(aov_table)
    print('Nb,Nm =', [
        str(len(ddf.loc[ddf.Condition == cc]))
        for cc in ddf.Condition.unique()
    ])
    print('\n')
#%%
def get_randomUtility(softmaxDF, Trials, minPoints=3, plotit=True):
    '''
    Basically a psychometrics-elicitation function but centered around a midpoint between safes rather than on a gamble or safe versus its secondary options
    '''
    from macaque.f_models import define_model
    import scipy.optimize as opt
    from scipy import stats
    import matplotlib.cm as cm


    sigmoid = lambda x, p2: np.array(1 / (1 + np.exp(-(x - 0) / p2)))
    
#    Model = Model.replace('1prelec', 'ev').replace('-cumulative','')

    #UTILITY FROM MULTIPLE-GAPPED MIDPOINTS
    #-----------------------------------------------------------------------------------

    i = 0
    r=0
    if plotit:
        fig, ax = plt.subplots( 3, len(softmaxDF.sessionDate.unique()), squeeze = False,
                           figsize=( int(np.ceil(len(softmaxDF.sessionDate.unique())/3))*6 , 6 ))
    fig2, ax2 = plt.subplots( 1, 3,  squeeze = False, figsize=( 10,4 ))
    dList = []
    
    ax2[0,0].plot( np.linspace(softmaxDF.midpoint.min(), softmaxDF.midpoint.max(), 100),  np.linspace(softmaxDF.midpoint.min(), softmaxDF.midpoint.max(), 100), '--', color='black')
    
    for date in softmaxDF.sessionDate.unique():
        df = softmaxDF.loc[softmaxDF.sessionDate == date].copy()
        df = df.loc[~np.isnan(df.temperature)]
#        df = df.loc[df.temperature < 10]
        if len(df) < minPoints:
            continue
        
        midpoints = df['midpoint']
        norm = lambda x: (x - min(x)) / (max(x) - min(x))
        norm2 = lambda x: (x * (max(midpoints) - min(midpoints))) + min(midpoints)
        fake_gap = 0.03  #in the middle
        randomUtils = []
        sCh = np.array([sigmoid(fake_gap, param) for param in df.temperature.values])
#        params = df.temperature.values #this is basically the spread of the gaussians
        randomUtils = np.cumsum(stats.norm.ppf(sCh, loc=0, scale=1))  #inverse cummulative distribution function
#        randomUtils = np.cumsum(sCh) 
        randomUtils = norm(randomUtils)
        df['utility'] = randomUtils
        df['utility_real'] = norm2(randomUtils)
        df['avg_ratio'] = sCh
        
        df['cumulative_utility'] = norm(np.cumsum(sCh - 0.5))

        chTimes = np.array([ np.concatenate(list(val.values())) for val in df.get('choiceTimes').values ])
        mTimes = [np.mean(chTimes[x]) for x in range(len(chTimes))]
        df['mTimes'] = mTimes
        stdTimes = [np.std(chTimes[x]) for x in range(len(chTimes))]
        df['stdTimes'] = stdTimes
#        chTimes2 = np.array([ val for val in df.get('choiceTimes').values  ])
#        choiceMidpoints = [ [middie]*len(chTimes2[ii]) for ii,middie in enumerate(midpoints)]
        dList.append(df)
        
        ax2[0,0].plot(midpoints, norm2(randomUtils), '-o', color = 'k', alpha = 0.3)
        ax2[0,1].plot(midpoints, sCh, '--o', color='k', alpha = 0.3)
        ax2[0,2].scatter(norm2(randomUtils), mTimes, color='blue', alpha = 0.3)

        if plotit:

            ax[0,r].plot(midpoints, randomUtils)

            ax[0,r].plot(np.linspace(min(midpoints), max(midpoints), 100),  np.linspace(0, 1, 100), '--', color='black')
            #plot points
            ax[0,r].scatter(midpoints, randomUtils, color='k')
            ax[0,r].grid()
            #plot the pattern of choice percentages and softmax temperatures
            ax[1,r].plot(midpoints, sCh, color='k')
    #        ax2 = ax[r,1].twinx()
            ax[1,r].plot(midpoints, mTimes, color='r')
            ax[1,r].grid()
            #plot cummulatetive probabilities (-50%)

            ax[2,r].errorbar(randomUtils,mTimes,linestyle = '--',  color='black', yerr=stdTimes)
    #        ax[r,2].errorbar(midpoints,  stdTimes)

    #        fun, params = function( midpoints, norm(np.cumsum(np.array(sCh) - 0.5)))  #fit the sCDF curve
    #        ax[r,2].plot( np.linspace(min(midpoints), max(midpoints), 100), fun( np.linspace( min(midpoints), max(midpoints), 100)))
            ax[0,r].set_ylabel('random utility')
            ax[0,r].set_title(str(np.unique(df.sessionDate.values)[0]))
            ax[0,r].set_xlabel('midpoint')
            ax[1,r].set_title('choice probability')
    #        ax[r,1].set_xlabel('prob choose higher')
            ax[1,r].legend(['pChH', 'smTemp'])
            ax[1,r].set_xlabel('midpoint magnitude')
            ax[2,r].set_title('fitted rU (DCM)')
            ax[2,r].set_xlabel('utility')
            ax[2,r].set_ylabel('response time')
            ax[2,r].grid()
            
            [ squarePlot(ax[nn,r]) for nn in range(3) ]
            
            r+=1

    if plotit:
        fig.suptitle('Utilities from midpoints:')
        fig.tight_layout()

    dList = pd.concat(dList)
    dList.groupby('midpoint').mean()['utility'].plot(style='-bo', ax=ax2[0,0], color='red')
    dList.groupby('midpoint').mean()['avg_ratio'].plot(style='-bo', ax=ax2[0,1], color='red')
    
    ax2[0,0].set_xlim(0,0.5)
    ax2[0,0].set_xticks([0,0.1, 0.2, 0.3, 0.4, 0.5])
    ax2[0,1].set_xlim(0,0.5)
    ax2[0,1].set_xticks([0,0.1, 0.2, 0.3, 0.4, 0.5])
    ax2[0,0].set_title('random utility')
    ax2[0,0].set_xlabel('midpoint')
    ax2[0,1].set_title('choice probability')
    ax2[0,1].set_xlabel('midpoint magnitude')
    ax2[0,1].set_ylabel('high choice ratio')
    ax2[0,2].set_title('fitted rU (DCM)')
    ax2[0,2].set_xlabel('utility')
    ax2[0,2].set_ylabel('response time')
    ax2[0,2].grid()
    
    squarePlot(ax2[0,0])
    squarePlot(ax2[0,1])
    squarePlot(ax2[0,2])
    
    fig2.tight_layout()
        
    y = dList.temperature.values
    x = dList.midpoint.values
    
    fig3, ax3 = plt.subplots( 1, 1,  squeeze = False, figsize=( 4,4 ))
    ax3[0,0].scatter(jitter(x, 0.002), y, color='k', alpha=0.5)
    dList.groupby('midpoint').mean()['temperature'].plot(style='-bo', ax=ax3[0,0], color='red')
    ax3[0,0].set_xlim([0,0.5])
    ax3[0,0].set_xlabel('midpoint')
    ax3[0,0].set_ylabel('1 / softmax temperature')
    squarePlot(ax3[0,0])
    ax3[0,0].grid()
    
    return dList


#%%
def LL_randomUtility(midpointDF, Trials, Model='random-power', minTrials = 100, fixedRange = False):
    '''
    '''
    from macaque.f_models import get_modelLL, trials_2fittable, LL_fit
    from macaque.f_probabilityDistortion import plot_MLEfitting
    np.warnings.filterwarnings('ignore')

    cols = ['date', 'nTrials', 'NM_success', 'params', 'model_used', 'AIC', 'BIC', 'LL',
            'pNames', 'pvalues', 'full_model', 'all_fits', 'mag_range', 'trials']
    use_Outcomes = False

    if type(Model) == list:
        MLE_fits = []
        for model in Model:
            MLE_fits.append(LL_randomUtility(midpointDF, Trials, Model=model, minTrials = minTrials, fixedRange=fixedRange))
        return MLE_fits
    else:
        tt = midpointDF.getTrials(Trials)
        dList = []; pastEffect = []
        for date in tqdm(np.sort(midpointDF.sessionDate.unique()), desc=Model):
            
            if 'dynamic' in Model:
                use_Outcomes = True
                
            if date == 0:
                X, Y = trials_2fittable(tt, use_Outcomes=use_Outcomes)
            else:
                X, Y = trials_2fittable(tt.loc[tt['sessionDate'] == date], use_Outcomes = use_Outcomes)
            if len(X) < minTrials:
                continue
            MLE = LL_fit(Y, X, model = Model, fixedRange = fixedRange).fit(disp=False, callback=False)

            #this is wrong! it doesn't necessarily go to zero!
            mag_range = np.array([ min(np.concatenate(np.array(X)[:,[0,4]])),
                                  max(np.concatenate(np.array(X)[:,[0,4]])) ])
            if fixedRange == True:
                mag_range = np.array([0, 0.5])

            dList.append({
                'date':date,
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
                'trial_index' : tt.loc[tt['sessionDate'] == date].index,
                'mag_range' : mag_range,
                'trials' : [tt.loc[tt['sessionDate'] == date].index]
            })

        MLE_fits = MLE_object(pd.DataFrame(dList))
        if Model.lower() == 'dynamic-value' or Model.lower() == 'random-rl':
            MLE_fits['past_that_matters'] = MLE_fits.full_model.apply(lambda x: x.model_parts['past_effect']).values
            cols = cols + ['past_that_matters']
        plot_MLEfitting(MLE_fits, plotFittings=False)
    return MLE_fits[cols]

#%%
def compare_MLEresults(MLE_list, comparison='BIC', selection = 'mean'):
    '''
    '''
    from macaque.f_Rfunctions import oneWay_rmAnova
    import statsmodels.api as sm
    from scipy import stats
    
    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((1, 3), (0, 2))

    Xs = []; BIC = []; AIC = []; LL = []
    realXs = []; dates=[]; names=[]; functions=[]
    for i,mle in enumerate(MLE_list):
        BIC.extend(mle.BIC.values)
        AIC.extend(mle.AIC.values)
        LL.append(mle.LL.values)
        Xs.extend(np.random.normal(i, 0.08, size=len(mle)))
        realXs.extend([i]*len(mle))
        dates.append(mle.date.values)
        names.extend(mle.model_used.unique())
        functions.extend(mle.model_used.values)

    if comparison.lower() == 'bic':
        results = BIC
    elif comparison.lower() == 'aic':
        results = AIC
    else:
        results == LL
    sb.boxplot(realXs, results, ax=ax2, color='white', saturation=1, width=0.5)
    plt.setp(ax2.artists, edgecolor = 'k', facecolor='w')
    plt.setp(ax2.lines, color='k')
    ax2.scatter(Xs, results, color='k', alpha=0.2)
    ax2.set_xticklabels(names, rotation=45)
    ax2.set_ylabel(comparison + ' score')
    ax2.set_xlabel('model tested')

    #now lets see if we can plot the differences with the control
    control = [i for i, mle in enumerate(MLE_list) if mle.model_used.iloc[-1] == 'control-random']
    if len(control) > 0:
        LLzero = MLE_list[control[0]].LL.values
        ax1.set_ylabel('-LL - controlLL')
    else:
        LLzero = 0
        ax1.set_ylabel('-LL')

    for ll, dd in zip(LL, dates):
        ax1.plot(ll-LLzero)
    ax1.legend(names)
    ax1.set_xticks(range(0, len(dd)))
    ax1.set_xticklabels(dd, rotation=45)
#    ax1.grid(axis='y')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlabel('session dates')

    plt.tight_layout()

    print('\n----------------------------------------------------------')
    rmAnova = oneWay_rmAnova(results, [x.toordinal() for x in np.concatenate(dates)], functions)

    data = []
    for ff in np.unique(functions):
        where = np.array(functions) == ff
        data.append(np.array(results)[where])
    data = np.array(data)
    
    print('\n================================================================================')
    print(stats.friedmanchisquare(*[data[x] for x in np.arange(data.shape[0])]))
    print('\n')
    post = sm.stats.multicomp.MultiComparison(results, functions)
    print(post.allpairtest(stats.wilcoxon, method = 'holm')[0])

    results = pd.DataFrame(columns=names)
    for i,mle in enumerate(MLE_list):
        if comparison.lower() == 'bic':
            results[mle.model_used.unique()[0]] = mle.BIC.values
        if comparison.lower() == 'aic':
            results[mle.model_used.unique()[0]] = mle.AIC.values
    print(results.describe().loc[['count', '50%', 'mean', 'std']])
#    print(rmAnova)
#    return rmAnova
    if selection.lower() == 'mean':
        model_name = results.describe().loc[[ 'mean' ]].T['mean'].argmin()
        if 'random' in model_name:
            return model_name.replace('ev', 'power-cumulative')
        else:
            return model_name
    elif selection.lower() == 'median':
        model_name = results.describe().loc[[ '50%' ]].T['50%'].argmin()
        if 'ev' in model_name.split()[2]:
            return model_name.replace('power', 'ev').replace('-cumulative','')
        else:
            return model_name
        

#%%
def compare_utilities(correlations):
    '''
    '''
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    df = correlations.loc[correlations.riskless_params.apply(lambda x: len(x) > 0)]
    riskless = np.vstack(flatten(df.riskless_params.values))
    risky = np.vstack(df.risky_params.values)
    risky = risky[:,:np.size(riskless, 1)]

    from macaque.f_Rfunctions import oneWay_rmAnova
    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax3 = plt.subplot2grid((1, 3), (0, 2))

    bpl = ax1.boxplot(riskless[:,2:], positions= np.arange(3, np.size(riskless, 1)+1) *2.0-0.4 -1, sym='', widths=0.6)
    bpr = ax1.boxplot(risky[:,2:], positions= np.arange(3, np.size(risky, 1)+1) *2.0+0.4 -1, sym='', widths=0.6)
    set_box_color(bpl, 'black')
    set_box_color(bpr, 'black')

    risklessX = np.ravel(np.tile(range(3, np.size(riskless, 1)+1), (len(riskless), 1)))
    risklessX = np.random.normal(risklessX, 0.04, size=len(risklessX))
    ax1.scatter(risklessX*2.0-0.4-1, np.ravel(riskless[:,2:]), c='blue', label='riskless', alpha=0.2)
    ax1.scatter(risklessX*2.0+0.4-1, np.ravel(risky[:,2:]), c='red', label='risky', alpha=0.2)
    ax1.axhline(0, alpha=0.3, color='k')
    fig.legend()

    # ------------------------------------------------------------------------
    ax2 = ax1.twinx()
    bpl2 = ax2.boxplot(riskless[:,:2], positions= np.arange(5, 7) *2.0-0.4, sym='', widths=0.6)
    bpr2 = ax2.boxplot(risky[:,:2], positions=np.arange(5, 7) *2.0+0.4, sym='', widths=0.6)
    set_box_color(bpl2, 'black')
    set_box_color(bpr2, 'black')

    risklessX = np.ravel(np.tile(np.arange(5, 7), (len(riskless), 1)))
    risklessX = np.random.normal(risklessX, 0.04, size=len(risklessX))
    ax2.scatter(risklessX*2.0-0.4, np.ravel(riskless[:,:2]), c='blue', label='riskless', alpha=0.2)
    ax2.scatter(risklessX*2.0+0.4, np.ravel(risky[:,:2]), c='red', label='risky', alpha=0.2)

    ax1.set_xticks(np.hstack((np.arange(3, np.size(riskless, 1)+1)-0.5, np.arange(5, 7))) *2)
    ax1.set_xticklabels( np.hstack((df.pNames.iloc[0][0][2:], df.pNames.iloc[0][0][:2])) )
    ax1.set_xbound(3.75,13)
    ax1.set_ylabel('decision parameters')
    ax2.yaxis.set_ticks_position('left')
    ax2.spines['left'].set_position(('data', 8.5))
#    ax2.set_ylabel('softmax parameter')
    fig.suptitle('riskless/risky parameter comparisons')
#    ax1.spines['top'].set_visible(False)
#    ax1.spines['right'].set_visible(False)
#    ax2.spines['top'].set_visible(False)
#    ax2.spines['right'].set_visible(False)

    # -----------------------------------------------------------------------
    from macaque.f_probabilityDistortion import covarianceEllipse

    ax3.scatter(riskless[:,2], riskless[:,3], color='blue')
    covarianceEllipse(riskless[:,2], riskless[:,3], ax3, color='blue', draw='CI')

    ax3.scatter(risky[:,2],risky[:,3], color='red')
    covarianceEllipse(risky[:,2],risky[:,3], ax3, color='red', draw='CI')

    ax3.set_xlabel(df.pNames.iloc[0][0][2])
    ax3.set_ylabel(df.pNames.iloc[0][0][3])
    ax3.grid()
    x0, x1 = ax3.get_xlim()
    y0, y1 = ax3.get_ylim()
    ax3.set_aspect((x1 - x0) / (y1 - y0))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#    manova
    from macaque.f_Rfunctions import dv3_manova
    variables = np.vstack((riskless, risky))
    sorting = ['riskless']*len(riskless) + ['risky']*len(risky)
    dv3_manova( variables[:,0],  variables[:,1],  variables[:,2], IV=np.array(sorting))
    
    #%%
    from scipy import stats
    risky = np.vstack(df.risky_params.values)
    riskless = np.vstack((riskless.T, np.ones(len(riskless)))).T
    dataList = np.array((riskless, risky))
    means =  [np.mean(np.log(data),0) for data in dataList]
    sems = [stats.sem(np.log(data),0) for data in dataList]
    
    fig, ax = plt.subplots( 1, 1, squeeze = False,
                           figsize=(6,3))
    width = 0.25; i=-1
    for mm, ss, cc in zip(means, sems, ['blue','red']):
        ax[0,0].bar(np.array([1,2,3,4, 5])+(width*i), mm, width, yerr = ss, color = cc)
        i+=1
#    
#    means_side =  [np.mean(data[1],0) for data in dataList]
#    sems_side =  [stats.sem(data[1],0) for data in dataList]
#    ax[0,0].bar([2-width, 2], means_side, width, yerr = sems_side, color = ['blue','red'])
#    
    legend = ['riskless', 'risky']
    ax[0,0].set_xticks( np.array([1,2,3,4,5]) - width/2)
    ax[0,0].set_xticklabels( df.pNames.iloc[0][0] )
    ax[0,0].legend(legend)
    ax[0,0].axhline(0, color='k')

#%%
def get_hybridUtilities(equivariant, gambleCEs, trials, model = 'risky-scdf'):
    '''
    '''
    from macaque.f_models import trials_2fittable, LL_fit
    from collections import Counter

    if len(equivariant) == 0:
        dating = np.unique(gambleCEs.sessionDate.unique())
        equivariant = pd.DataFrame(equivariant, columns=gambleCEs.columns)
        gambleCEs = gambleCEs.loc[gambleCEs.spread == Counter(gambleCEs.spread.values).most_common(1)[0][0]]
    elif len(gambleCEs) == 0:
        dating = np.unique(equivariant.sessionDate.unique())
        gambleCEs = pd.DataFrame(gambleCEs, columns=equivariant.columns)
    else:
        gambleCEs = gambleCEs.loc[gambleCEs.spread == Counter(gambleCEs.spread.values).most_common(1)[0][0]]
        dating = np.unique(np.concatenate((equivariant.sessionDate.unique(), gambleCEs.sessionDate.unique())))
    dList = []

    for date in tqdm(dating, desc='Fitting dailies'):
        MLE = []
        MLE2 = []
        if np.isin(date, equivariant.sessionDate):
            tt = equivariant.loc[equivariant.sessionDate.apply(lambda x: x==date)].getTrials(trials)
            if len(tt) > 100:
                print(len(tt))
            X, Y = trials_2fittable(tt, use_Outcomes = False)
            MLE = LL_fit(Y, X, model = model).fit(disp=False)
            success1 = MLE.mle_retvals['converged']
            range1 = equivariant.loc[equivariant.sessionDate.apply(lambda x: x==date)].m_range.iloc[0]
            names =  MLE.model.exog_names
            params1 = MLE.params
        else:
            success1 = False
            range1 = False
            params1 = []

        if np.isin(date, gambleCEs.sessionDate):
            tt = gambleCEs.loc[gambleCEs.sessionDate.apply(lambda x: x==date)].getTrials(trials)
            
    #        if len(tt) < 100:
            X, Y = trials_2fittable(tt, use_Outcomes = False)
            MLE2 = LL_fit(Y, X, model = model).fit(disp=False)
            success2 = MLE2.mle_retvals['converged']
            range2 = gambleCEs.loc[gambleCEs.sessionDate.apply(lambda x: x==date)].m_range.iloc[0]
            names =  MLE2.model.exog_names
            params2 = MLE2.params
        else:
            success2 = False
            range2 = np.nan
            params2 = []

        dList.append({
                'date':date,
                'gambleCE_params': params2,
                'equivariant_params': params1,
                'gambleCE_range': range2,
                'equivariant_range': range1,
                'pNames': names,
                'equivariant_success' : success1,
                'gambleCE_success' : success2,
                'nTrials' : len(tt),
                'model_gambleCE' : MLE2,
                'model_equivariant' : MLE,
                 'trial_index' : tt.index.values})
    dList = pd.DataFrame(dList)
    return dList

#%%
def plot_modelComparisons(MLE_list, correlations, best_model):
    '''
    '''
    from matplotlib.pyplot import cm
    from scipy import stats
    #need to make this work via a collection of lists
    
    colour = ['k', 'silver', 'lightgray', 'dimgray', 'gainsboro']
    fig, ax = plt.subplots( 1, 1, squeeze = False, figsize=(6,6))
    legend = [mle.iloc[0].model_used for mle in MLE_list]
    colour=cm.rainbow(np.linspace(0,1,len(MLE_list)))
    dating = correlations.date.values
    
    for mle, cc in zip(MLE_list, colour):
        mle = mle.loc[np.isin(mle.date.values, dating)]
        mle = mle.loc[mle.NM_success==True]
        if len(mle) == 0:
            uu = np.linspace(0,0,100)
            min_curve = np.linspace(0,0,100)
            max_curve = np.linspace(0,0,100)
            ax[0,0].plot(np.linspace(0,1,100), uu , color=cc, alpha = 0.75)
            ax[0,0].fill_between(np.linspace(0,1,100), y1=min_curve, y2=max_curve,
              facecolor=cc, alpha=0.25, color = cc)
            continue
        
        if mle.iloc[0].model_used == best_model.replace('power', 'ev').replace('-cumulative',''):
            cc = 'black'
            
        if np.sum(['u_' in pname for pname in mle.pNames.iloc[-1]]) == 0 :
            uu = np.linspace(0,1,100)
            min_curve = np.linspace(0,1,100)
            max_curve = np.linspace(0,1,100)
        elif np.sum(['u_' in pname for pname in mle.pNames.iloc[-1]]) == 1 :
            
            position = np.argmax(['u_' in pname for pname in mle.pNames[0]])
            
            pp = np.mean(np.vstack(mle.params.values)[:,position:position+1], 0)
            sem = stats.sem(np.vstack(mle.params.values)[:,position:position], 0)
            function = mle.iloc[0].full_model
            utility = function.model_parts['empty functions']['utility']
            uu = utility(np.linspace(0,1,100), pp)
            
            lower = utility(np.linspace(0,1,100), [pp[0] - sem[0]])
            upper = utility(np.linspace(0,1,100), [pp[0] + sem[0]])
            all_lines = np.vstack((lower,upper))
            min_curve = np.min(all_lines, 0)
            max_curve = np.max(all_lines, 0)
        else:
            position = np.argmax(['u_' in pname for pname in mle.pNames.iloc[-1]])
            
            pp = np.median(np.vstack(mle.params.values)[:,position:position+2], 0)
            
            
            sem = stats.sem(np.vstack(mle.params.values)[:,position:position+2], 0)
            function = mle.iloc[0].full_model
            utility = function.model_parts['empty functions']['utility']
            uu = utility(np.linspace(0,1,100), pp)
            
            lower = utility(np.linspace(0,1,100), [pp[0] - sem[0],
                            pp[1] - sem[1]])
            mid1 = utility(np.linspace(0,1,100), [pp[0] - sem[0],
                        pp[1] + sem[1]])
            mid2 = utility(np.linspace(0,1,100), [pp[0] + sem[0],
                        pp[1] - sem[1]])
            upper = utility(np.linspace(0,1,100), [pp[0] + sem[0],
                        pp[1] + sem[1]])
            all_lines = np.vstack((lower,mid1,mid2,upper))
            min_curve = np.min(all_lines, 0)
            max_curve = np.max(all_lines, 0)
        
        ax[0,0].plot(np.linspace(0,1,100), uu , color=cc, alpha = 0.75)
        ax[0,0].fill_between(np.linspace(0,1,100), y1=min_curve, y2=max_curve,
          facecolor=cc, alpha=0.25, color = cc)
        
    ax[0,0].set_xlabel('reward magnitude')
    ax[0,0].set_ylabel('utility')
    ax[0,0].legend(legend)
    ax[0,0].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,0])
    
#%%
def plot_exampleModels(MLE_list, riskyMLEs, riskyMLEs2, best_model):
    '''
    '''
    from matplotlib.pyplot import cm
    from macaque.f_models import define_model
    model = define_model(best_model)
    p0 = model[1]
    ff = model[3](p0)
    colour=cm.rainbow(np.linspace(0,1,7))
    
    fig, ax = plt.subplots( 1, 1, squeeze = False, figsize=(6,6))
    legend = ['p1=1; p2=1',
              'p1=1; p2=2',
              'p1=2; p2=1',
              'p1=2; p2=2',
              'p1=0.5; p2=1',
              'p1=1; p2=0.5',
              'p1=0.5; p2=0.5']
    
    plt.plot(np.linspace(0,1, 100), ff['empty functions']['utility'](np.linspace(0,1, 100), [1,1]), color = colour[3])
    plt.plot(np.linspace(0,1, 100), ff['empty functions']['utility'](np.linspace(0,1, 100), [1,2]), color = colour[0])
    plt.plot(np.linspace(0,1, 100), ff['empty functions']['utility'](np.linspace(0,1, 100), [2,1]), color = colour[1])
    plt.plot(np.linspace(0,1, 100), ff['empty functions']['utility'](np.linspace(0,1, 100), [2,2]), color = colour[2])
    plt.plot(np.linspace(0,1, 100), ff['empty functions']['utility'](np.linspace(0,1, 100), [0.5,1]), color = colour[4])
    plt.plot(np.linspace(0,1, 100), ff['empty functions']['utility'](np.linspace(0,1, 100), [1,0.5]), color = colour[5])
    plt.plot(np.linspace(0,1, 100), ff['empty functions']['utility'](np.linspace(0,1, 100), [0.5,0.5]), color = colour[6])
    
    plt.plot(np.linspace(0,1), np.linspace(0,1), '--', color = 'k')
    ax[0,0].legend(legend)
    squarePlot(ax[0,0])
    
    # --------------------------------------------------

    ii = [i for i, mle in enumerate(MLE_list) if mle.model_used.iloc[0] == best_model.replace('power-cumulative', 'ev').replace('-cumulative','')][0]
    mle = MLE_list[ii]
    date = mle.loc[mle.BIC == mle.BIC.min()].date.values[0]
    if len(riskyMLEs.loc[riskyMLEs.date == date]) == 0:
        date = mle.date[17]
    model = mle.loc[mle.date == date].full_model.values[0]
    ff, ax = model.plot_fullModel(color='b', return_fig = True)
#    ff = plt.gcf()

    # ----------------------------------------------------
    
    rr = riskyMLEs.loc[riskyMLEs.date == date]
    model = rr.full_model.values[0]
    model.plot_fullModel(fig = ff, ax = ax, color='r', return_fig = False )
    
    # ----------------------------------------------
    
    rr = riskyMLEs2.loc[riskyMLEs2.date == date]
    model = rr.full_model.values[0]
    model.plot_fullModel(fig = ff, ax = ax, color='darkred', return_fig = False )

#%%
def plot_behaviouralUtilities(correlations, best_model):
    from scipy import stats
    from macaque.f_models import define_model
    #need to make this work via a collection of lists
    dataList = []
    df = correlations.loc[correlations.riskless_params.apply(lambda x: len(x) > 0)]
    risky = np.vstack(df.params_fractile.values)
    riskless = np.vstack(df.params_random.values)
    riskless_range = np.median(np.vstack(df.riskless_range.values), 0)
    risky_range = np.median(np.vstack(df.risky_range.values), 0)
    legend = []; colours = []
    if len(riskless) != 0:
        dataList.append(riskless)
        legend.extend(['riskless'])
        colours.extend(['blue'])
    if len(risky) != 0:
        dataList.append(risky)
        legend.extend(['riskY'])
        colours.extend(['red'])

    #%%
    #utility parameters are 1 and 2
    
    from macaque.f_uncertainty import bootstrap_sample, bootstrap_function
    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(10,6))
    ff = define_model(best_model)
    p0 = ff[1]
    utility = ff[-1](p0)['empty functions']['utility']
    
    uu = lambda x : utility(x, np.mean(dataList[1], 0))
    top = uu((riskless_range[1] - risky_range[0]) / (risky_range[1] - risky_range[0]))
    bottom = uu((riskless_range[0] - risky_range[0]) / (risky_range[1] - risky_range[0]))
    
    i = 0
    for params, cc in zip(dataList, colours):
        if 1==0:
            uu = lambda pp: utility(np.linspace(0,1,100), pp)
            mean, lower, upper = bootstrap_function(uu, params, 'mean')
            ax[0,0].plot(np.linspace(riskless_range[0],riskless_range[1],100), 
              ((mean - risky_range[0]) / (risky_range[1] - risky_range[0])), color=cc)
            ax[0,0].fill_between(np.linspace(riskless_range[0],riskless_range[1],100), 
              y1=((lower- risky_range[0]) / (risky_range[1] - risky_range[0])), 
              y2=((upper- risky_range[0]) / (risky_range[1] - risky_range[0])), alpha=0.25, facecolor=cc)
            mean, lower, upper = bootstrap_function(uu, params, 'median')
            ax[0,1].plot(np.linspace(riskless_range[0],riskless_range[1],100), 
              mean, color=cc)
            ax[0,1].fill_between(np.linspace(riskless_range[0],riskless_range[1],100), 
              y1=((lower- risky_range[0]) / (risky_range[1] - risky_range[0])), 
              y2=((upper- risky_range[0]) / (risky_range[1] - risky_range[0])), alpha=0.25, facecolor=cc)
        else:
            uu = lambda pp: utility(np.linspace(0,1,100), pp)
            mean, lower, upper = bootstrap_function(uu, params, 'mean')
            ax[0,0].plot(np.linspace(0,1,100), mean, color=cc)
            ax[0,0].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
            mean, lower, upper = bootstrap_function(uu, params, 'median')
            ax[0,1].plot(np.linspace(0,1,100), mean, color=cc)
            ax[0,1].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
        i+=1

    ax[0,0].set_xlabel('reward magnitude')
    ax[0,0].set_ylabel('mean utility')
    ax[0,0].legend(legend)
    ax[0,0].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,0])
    
    ax[0,1].set_xlabel('reward magnitude')
    ax[0,1].set_ylabel('median utility')
    ax[0,1].legend(legend)
    ax[0,1].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,1])
    
    plt.suptitle('side-by-side utilities')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#%%
def plot_all_avgUtilities(dList, correlations, best_model, minTrials = 40):
    '''
    '''
    from scipy import stats
    from macaque.f_models import define_model
    #need to make this work via a collection of lists
    dataList = []
    df = correlations.loc[correlations.riskless_params.apply(lambda x: len(x) > 0)]
    riskless = np.vstack(flatten(df.riskless_params.values))
    legend = []; colours = []
    if len(riskless) != 0:
        dataList.append(riskless)
        legend.extend(['riskless'])
        colours.extend(['blue'])
    if len(dList.loc[dList.gambleCE_success.values]) != 0:
        gambleCEs = np.vstack(dList.loc[(dList.gambleCE_success.values) & (dList.nTrials >= minTrials)].gambleCE_params.values)
#        gambleCEs = gambleCEs.loc[gambleCEs.nTrials >= minTrials]
        dataList.append(gambleCEs)
        legend.extend(['gambleCEs'])
        colours.extend(['green'])
    if len(dList.loc[dList.equivariant_success.values]) != 0:
        equivariants = np.vstack(dList.loc[dList.equivariant_success.values].equivariant_params.values)
        equivariants = equivariants[:,:np.size(riskless, 1)]
        dataList.append(equivariants)
        legend.extend(['equivariants'])
        colours.extend(['brown'])
    risky = np.vstack(df.risky_params.values)
    if len(risky) != 0:
        dataList.append(risky)
        legend.extend(['riskY'])
        colours.extend(['red'])

    #%%

    from macaque.f_uncertainty import bootstrap_sample, bootstrap_function
    from scipy.stats import norm
    
    fig, ax = plt.subplots( 2, 3, squeeze = False, figsize=(10,6))
    ff = define_model(best_model)
    p0 = ff[1]
    utility = ff[-1](p0)['empty functions']['utility']
    softmax = ff[-1](p0)['empty functions']['pChooseA']
    probability = ff[-1](p0)['empty functions']['probability']
    
    position = np.argmax(['u_' in pname for pname in correlations.pNames.iloc[-1][0]])
    long = np.sum(['u_' in pname for pname in correlations.pNames.iloc[-1][0]])
    

    for params, cc in zip(dataList, colours):
        uu = lambda pp: utility(np.linspace(0,1,100), pp)
        mean, lower, upper = bootstrap_function(uu, params[:,position:position+long], 'mean')
        ax[0,0].plot(np.linspace(0,1,100), mean, color=cc)
        ax[0,0].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
        mean, lower, upper = bootstrap_function(uu,  params[:,position:position+long], 'median')
        ax[1,0].plot(np.linspace(0,1,100), mean, color=cc)
        ax[1,0].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
        
    for params, cc in zip(dataList, colours):
        sm = lambda pp: 1 / (1 + np.exp( -pp[0] * ( (np.linspace(-0.5,0.5,100) - pp[1]) )  ))
        mean, lower, upper = bootstrap_function(sm,  params[:,:position], 'mean')
        ax[0,1].plot(np.linspace(-0.5,0.5,100), mean, color=cc)
        ax[0,1].fill_between(np.linspace(-0.5,0.5,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
        mean, lower, upper = bootstrap_function(sm,  params[:,:position], 'median')
        ax[1,1].plot(np.linspace(-0.5,0.5,100), mean, color=cc)
        ax[1,1].fill_between(np.linspace(-0.5,0.5,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
        
    for params, cc in zip(dataList, colours):
        if np.size(params,1) > 3:
            prob = lambda pp: probability(np.linspace(0,1,100), pp)
            mean, lower, upper = bootstrap_function(prob,  params[:,-1], 'mean')
            ax[0,2].plot(np.linspace(0,1,100), mean, color=cc)
            ax[0,2].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
            mean, lower, upper = bootstrap_function(prob,  params[:,-1], 'median')
            ax[1,2].plot(np.linspace(0,1,100), mean, color=cc)
            ax[1,2].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
        else:
            ax[0,2].plot(np.linspace(0,1,100), np.linspace(0,1,100), color=cc)

    ax[0,0].set_xlabel('reward magnitude')
    ax[0,0].set_ylabel('mean utility')
    ax[0,0].legend(legend)
    squarePlot(ax[0,0])
    
    ax[1,0].set_xlabel('reward magnitude')
    ax[1,0].set_ylabel('median utility')
    ax[1,0].legend(legend)
    squarePlot(ax[1,0])
    
    ax[0,1].set_xlabel('Δ value')
    ax[0,1].axvline(0)
    ax[0,1].set_ylabel('mean pChA')
    ax[0,1].legend(legend)  
    squarePlot(ax[0,1])
    
    ax[1,1].set_xlabel('Δ value')
    ax[1,1].axvline(0)
    ax[1,1].set_ylabel('median pChA')
    ax[1,1].legend(legend)   
    squarePlot(ax[1,1])
    
    ax[0,2].set_xlabel('reward probability')
    ax[0,2].set_ylabel('mean probability distortion')
    ax[0,2].legend(legend)
    ax[0,2].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,2])
    
    ax[1,2].set_xlabel('reward probability')
    ax[1,2].set_ylabel('median probability distortion')
    ax[1,2].legend(legend)
    ax[1,2].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[1,2])
    
    plt.suptitle('fitted functions')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#%%
def plot_avg_functions(correlations, correlations2, best_model, minTrials = 40):
    '''
    '''
    from scipy import stats
    from macaque.f_models import define_model
    from scipy import stats
    from macaque.f_models import define_model
    #need to make this work via a collection of lists
    dataList = []
    df = correlations.loc[correlations.riskless_params.apply(lambda x: len(x) > 0)]
    riskless = np.vstack(flatten(df.riskless_params.values))
    riskless = np.vstack((riskless.T, len(riskless)*[1])).T
    legend = []; colours = []
    if len(riskless) != 0:
        dataList.append(riskless)
        legend.extend(['riskless'])
        colours.extend(['blue'])
    risky = np.vstack(df.risky_params.values)
    risky = risky[:,:]
    if len(risky) != 0:
        dataList.append(risky)
        legend.extend(['riskY'])
        colours.extend(['red'])
    df2 = correlations2.loc[correlations2.riskless_params.apply(lambda x: len(x) > 0)]
    risky2 = np.vstack(df2.risky_params.values)
    risky2 = np.vstack((risky2.T, len(riskless)*[1])).T
    if len(risky) != 0:
        dataList.append(risky2)
        legend.extend(['risky without pDist'])
        colours.extend(['darkred'])

        
    #%%
    from macaque.f_uncertainty import bootstrap_sample, bootstrap_function
    from scipy.stats import norm
    
    fig, ax = plt.subplots( 2, 3, squeeze = False, figsize=(10,6))
    ff = define_model(best_model)
    p0 = ff[1]
    utility = ff[-1](p0)['empty functions']['utility']
    softmax = ff[-1](p0)['empty functions']['pChooseA']
    probability = ff[-1](p0)['empty functions']['probability']

    position = np.argmax(['u_' in pname for pname in correlations.pNames.iloc[-1][0]])
    long = np.sum(['u_' in pname for pname in correlations.pNames.iloc[-1][0]])

    for params, cc in zip(dataList, colours):
        uu = lambda pp: utility(np.linspace(0,1,100), pp)
        mean, lower, upper = bootstrap_function(uu, params[:,position:position+2], 'mean')
        ax[0,0].plot(np.linspace(0,1,100), mean, color=cc)
        ax[0,0].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
        mean, lower, upper = bootstrap_function(uu,  params[:,position:position+2], 'median')
        ax[1,0].plot(np.linspace(0,1,100), mean, color=cc)
        ax[1,0].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
        
    for params, cc in zip(dataList, colours):
        sm = lambda pp: 1 / (1 + np.exp( -pp[0] * ( (np.linspace(-0.5,0.5,100) - pp[1]) )  ))
        mean, lower, upper = bootstrap_function(sm,  params[:,:position], 'mean')
        ax[0,1].plot(np.linspace(-0.5,0.5,100), mean, color=cc)
        ax[0,1].fill_between(np.linspace(-0.5,0.5,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
        mean, lower, upper = bootstrap_function(sm,  params[:,:position], 'median')
        ax[1,1].plot(np.linspace(-0.5,0.5,100), mean, color=cc)
        ax[1,1].fill_between(np.linspace(-0.5,0.5,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
        
    for params, cc in zip(dataList, colours):
        if np.size(params,1) > 3:
            prob = lambda pp: probability(np.linspace(0,1,100), pp)
            mean, lower, upper = bootstrap_function(prob,  params[:,position+long:], 'mean')
            ax[0,2].plot(np.linspace(0,1,100), mean, color=cc)
            ax[0,2].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
            mean, lower, upper = bootstrap_function(prob,  params[:,position+long:], 'median')
            ax[1,2].plot(np.linspace(0,1,100), mean, color=cc)
            ax[1,2].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25, facecolor=cc)
        else:
            ax[0,2].plot(np.linspace(0,1,100), np.linspace(0,1,100), color=cc)

    ax[0,0].set_xlabel('reward magnitude')
    ax[0,0].set_ylabel('mean utility')
    ax[0,0].legend(legend)
    squarePlot(ax[0,0])
    
    ax[1,0].set_xlabel('reward magnitude')
    ax[1,0].set_ylabel('median utility')
    ax[1,0].legend(legend)
    squarePlot(ax[0,1])
    
    ax[0,1].set_xlabel('Δ value')
    ax[0,1].set_ylabel('mean pChA')
    ax[0,1].legend(legend)  
    squarePlot(ax[0,2])
    
    ax[1,1].set_xlabel('Δ value')
    ax[1,1].set_ylabel('median pChA')
    ax[1,1].legend(legend)   
    squarePlot(ax[1,0])
    
    ax[0,2].set_xlabel('reward probability')
    ax[0,2].set_ylabel('mean probability distortion')
    ax[0,2].legend(legend)
    ax[0,2].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[1,1])
    
    ax[1,2].set_xlabel('reward probability')
    ax[1,2].set_ylabel('median probability distortion')
    ax[1,2].legend(legend)
    ax[1,2].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[1,2])
    
    plt.suptitle('fitted functions')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#%%
def plot_allUtilities(dList, correlations, minTrials = 40):
    '''
    '''
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    #need to make this work via a collection of lists
    dataList = []
    df = correlations.loc[correlations.riskless_params.apply(lambda x: len(x) > 0)]
    riskless = np.vstack(flatten(df.riskless_params.values))
    riskless = np.vstack((riskless.T, np.ones(len(riskless)))).T
    legend = []; colours = []
    if len(riskless) != 0:
        dataList.append(riskless)
        legend.extend(['riskless'])
        colours.extend(['blue'])
    risky = np.vstack(df.risky_params.values)
    risky = risky
    if len(risky) != 0:
        dataList.append(risky)
        legend.extend(['riskY'])
        colours.extend(['red'])
    if len(dList.loc[dList.gambleCE_success.values]) != 0:
        gambleCEs = np.vstack(dList.loc[(dList.gambleCE_success.values) & (dList.nTrials >= minTrials)].gambleCE_params.values)
        gambleCEs = gambleCEs
#        gambleCEs = gambleCEs.loc[gambleCEs.nTrials >= minTrials]
        dataList.append(gambleCEs)
        legend.extend(['gambleCEs'])
        colours.extend(['green'])
    if len(dList.loc[dList.equivariant_success.values]) != 0:
        equivariants = np.vstack(dList.loc[dList.equivariant_success.values].equivariant_params.values)
        equivariants = equivariants[:,:np.size(riskless, 1)]
        dataList.append(equivariants)
        legend.extend(['equivariants'])
        colours.extend(['brown'])

    #%%

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax3 = plt.subplot2grid((1, 3), (0, 2))

    gap = -0.9
    ax1=[]

    i = 0
    for _ in range( len(dataList[0][0])):
            if i == 0 :
                ax1.extend([ax]); 
            else:
                ax1.extend([ax.twinx()])
            i+=1

#    ax1.extend([ax.twinx()]); ax1.extend([ax.twinx()])
#    ax1.extend([ax.twinx()])
    for sequenceType, color in zip(dataList, colours):
        i = 0
        for param in range( len(dataList[0][0])):
            risklessX = [i] * len(sequenceType)
            risklessX = np.random.normal(risklessX, 0.015, size=len(risklessX))
            if param == 3 or param==1:
                bp = ax1[param].boxplot(sequenceType[:,param], positions= [i *4.0+gap+0.2], sym='', widths=0.4)
                ax1[param].scatter(risklessX*4.0+gap+0.2, np.ravel(sequenceType[:,param]), c=color, label='riskless', alpha=0.2)
            else:
                bp = ax1[param].boxplot(np.log(sequenceType[:,param]), positions= [i *4.0+gap+0.2], sym='', widths=0.4)
                ax1[param].scatter(risklessX*4.0+gap+0.2, np.log(np.ravel(sequenceType[:,param])), c=color, label='riskless', alpha=0.2)
#       
            set_box_color(bp, 'black')
#
            if gap == -0.9:
                ax1[param].yaxis.set_ticks_position('left')
                ax1[param].spines['left'].set_position(('data', i*4-1.5))
            i+=1
        gap += 0.6

    ax.axhline(0, alpha=0.3, color='k')
    ax.set_xticks( np.arange(len(dataList[0][0])) *4 )
    ax.set_xticklabels(df.pNames.iloc[0][0]) 
    ax.set_xbound(-2,18.5)
#    ax1[1].set_ylabel('decision parameters')
    fig.suptitle('riskless/risky parameter comparisons')

    # -----------------------------------------------------------------------
    from macaque.f_probabilityDistortion import covarianceEllipse

    for sequenceType, color in zip(dataList, ['blue', 'green', 'brown', 'red']):
        ax3.scatter(np.log(sequenceType[:,2]), sequenceType[:,3], color=color)
        covarianceEllipse(np.log(sequenceType[:,2]), sequenceType[:,3], ax3, color=color, draw='CI')

    ax3.set_xlabel(df.pNames.iloc[0][0][2])
    ax3.set_ylabel(df.pNames.iloc[0][0][3])
    ax3.grid()
    ax3.set_xbound(-5, 5)
    x0, x1 = ax3.get_xlim()
    y0, y1 = ax3.get_ylim()
    ax3.set_aspect((x1 - x0) / (y1 - y0))
    ax3.legend(legend)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        #%%
    #Plot these as barplots
    from scipy import stats
    means =  [np.mean(np.log(data),0) for data in dataList]
    sems = [stats.sem(np.log(data),0) for data in dataList]
    
    fig, ax = plt.subplots( 1, 1, squeeze = False,
                           figsize=(6,3))
    width = 0.25; i=-2
    for mm, ss, cc in zip(means, sems, colours):
        ax[0,0].bar(np.array([1,2,3,4,5])+(width*i), mm, width, yerr = ss, color = cc)
        i+=1
        
#    means_side =  [np.mean(data[1],0) for data in dataList]
#    sems_side =  [stats.sem(data[1],0) for data in dataList]
#    ax[0,0].bar(np.array([2-width, 2, 2+width]) - width, means_side, width, yerr = sems_side, color = colours)
 
    ax[0,0].set_xticks( np.array([1,2,3,4, 5]) - width )
    ax[0,0].set_xticklabels( dList.pNames.iloc[-1] )
    ax[0,0].legend(legend)
    ax[0,0].axhline(0, color='k')
    
    #%%

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax3 = plt.subplot2grid((1, 3), (0, 2))

    gap = -0.9
    ax1=[]

    i = 0
    for _ in range( len(dataList[0][0])):
            if i == 0 :
                ax1.extend([ax]); 
            else:
                ax1.extend([ax.twinx()])
            i+=1

#    ax1.extend([ax.twinx()]); ax1.extend([ax.twinx()])
#    ax1.extend([ax.twinx()])
    for sequenceType, color in zip(dataList[:-1], colours[:-1]):
        i = 0
        for param in range( len(dataList[0][0])):
            risklessX = [i] * len(sequenceType)
            risklessX = np.random.normal(risklessX, 0.015, size=len(risklessX))
            if param == 3 or param==1:
                bp = ax1[param].boxplot(sequenceType[:,param], positions= [i *4.0+gap+0.2], sym='', widths=0.4)
                ax1[param].scatter(risklessX*4.0+gap+0.2, np.ravel(sequenceType[:,param]), c=color, label='riskless', alpha=0.2)
            else:
                bp = ax1[param].boxplot(np.log(sequenceType[:,param]), positions= [i *4.0+gap+0.2], sym='', widths=0.4)
                ax1[param].scatter(risklessX*4.0+gap+0.2, np.log(np.ravel(sequenceType[:,param])), c=color, label='riskless', alpha=0.2)
#       
            set_box_color(bp, 'black')
#
            if gap == -0.9:
                ax1[param].yaxis.set_ticks_position('left')
                ax1[param].spines['left'].set_position(('data', i*4-1.5))
            i+=1
        gap += 0.6

    ax.axhline(0, alpha=0.3, color='k')
    ax.set_xticks( np.arange(len(dataList[0][0])) *4 )
    ax.set_xticklabels(df.pNames.iloc[0][0]) 
    ax.set_xbound(-2,18.5)
#    ax1[1].set_ylabel('decision parameters')
    fig.suptitle('riskless/risky parameter comparisons')

    # -----------------------------------------------------------------------
    from macaque.f_probabilityDistortion import covarianceEllipse

    for sequenceType, color in zip(dataList, ['blue', 'green', 'brown', 'red']):
        ax3.scatter(np.log(sequenceType[:,2]), sequenceType[:,3], color=color)
        covarianceEllipse(np.log(sequenceType[:,2]), sequenceType[:,3], ax3, color=color, draw='CI')

    ax3.set_xlabel(df.pNames.iloc[0][0][2])
    ax3.set_ylabel(df.pNames.iloc[0][0][3])
    ax3.grid()
    ax3.set_xbound(-5, 5)
    x0, x1 = ax3.get_xlim()
    y0, y1 = ax3.get_ylim()
    ax3.set_aspect((x1 - x0) / (y1 - y0))
    ax3.legend(legend)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        #%%
    #Plot these as barplots
    from scipy import stats
    means =  [np.median(data,0) for data in dataList]
    sems = [stats.sem(data,0) for data in dataList]
    
    fig, ax = plt.subplots( 1, 1, squeeze = False,
                           figsize=(6,3))
    width = 0.25; i=-2
    for mm, ss, cc in zip(means, sems, colours):
        ax[0,0].bar(np.array([1,2,3,4,5])+(width*i), mm, width, yerr = ss, color = cc)
        i+=1
        
#    means_side =  [np.mean(data[1],0) for data in dataList]
#    sems_side =  [stats.sem(data[1],0) for data in dataList]
#    ax[0,0].bar(np.array([2-width, 2, 2+width]) - width, means_side, width, yerr = sems_side, color = colours)
 
    ax[0,0].set_xticks( np.array([1,2,3,4, 5]) - width )
    ax[0,0].set_xticklabels( dList.pNames.iloc[-1] )
    ax[0,0].legend(legend)
    ax[0,0].axhline(0, color='k')
    
    #%%
    #post-hoc before 
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    independent = flatten([len(dd) * [rr] for dd, rr, in zip(dataList, [1,2,3])])
    dependent = flatten(dataList)
    print('--------------------------------------------------------------')
    print('1 is [0,0.5] \n2 is [0, 1.0] \n3 is [0.5, 1.0]')
    
    data = pd.DataFrame(np.vstack((independent, np.vstack(dependent).T))).T
    data.columns = ['risktype', 'p1', 'p2', 'p3', 'p4', 'p5']
    range_lm = ols('p1 ~ C(risktype)',  data=data).fit()
    print(range_lm.summary2())
    range_lm = ols('p2 ~ C(risktype)',  data=data).fit()
    print(range_lm.summary2())
    range_lm = ols('p3 ~ C(risktype)',  data=data).fit()
    print(range_lm.summary2())
    range_lm = ols('p4 ~ C(risktype)',  data=data).fit()
    print(range_lm.summary2())

    #%%
#    manova
    from macaque.f_Rfunctions import dv4_manova
    IV = np.hstack([[i] * len(data) for i,data in enumerate(dataList)]).T
    dataList2 = np.vstack(dataList)
    dataList2 = np.vstack((dataList2.T, IV)).T
#    dataList[:,:3] = np.log(dataList[:,:3])
    dv4_manova( dataList2[:,0],  dataList2[:,1],  dataList2[:,2], dataList2[:,3], IV=dataList2[:,-1])
    
    print('================================== no binary ===================')
    IV = np.hstack([[i] * len(data) for i,data in enumerate(dataList[:-1])]).T
    dataList2 = np.vstack(dataList[:-1])
    dataList2 = np.vstack((dataList2.T, IV)).T
    dv4_manova( np.log(dataList2[:,0]),  dataList2[:,1],  np.log(dataList2[:,2]), dataList2[:,3], IV=dataList2[:,-1])
  

#%%
def plot_listMLE(MLE_list):
    '''
    '''
    for mle in MLE_list:
        mle.plot_fullModels()
        plt.gcf()
        plt.title(mle.model_used.unique()[0])
    return

#%%
def delta_utilities(MLE_list, Behaviour, best_model = None):
    import scipy.optimize as opt

    if best_model == None:
        mBIC = []; position=[]
        for i,mle in enumerate(MLE_list):
            mBIC.extend([mle.BIC.mean()])
            position.extend([i])

        mBIC = np.array(mBIC); position=np.array(position)
        modelMLE = MLE_list[int(position[mBIC == min(mBIC)])]
    else:
        ii = [i for i, mle in enumerate(MLE_list) if mle.model_used.iloc[0] == best_model.replace('power', 'ev').replace('-cumulative','')][0]
        modelMLE = MLE_list[ii]

    uniqueDates = np.unique(np.concatenate((Behaviour.sessionDate.values, modelMLE.date.values)))
    dating = [date for date in uniqueDates if (np.isin(date, Behaviour.sessionDate.values) and np.isin(date, modelMLE.date.values))]

    fig, ax = plt.subplots( 3, len(dating), squeeze = False,
                           figsize=( int(np.ceil(len(dating)/3))*6 , 6 ))
    r=0; mSE=[]
    for date in dating:
        aggregate = Behaviour.loc[Behaviour.sessionDate == date]
        if aggregate.empty:
            continue
        sChoice = modelMLE.loc[modelMLE.date == date]
        model = sChoice['full_model'].iloc[0]
        mRange = sChoice['mag_range'].iloc[0]
        midpoints = aggregate.midpoint.values
        utils = aggregate.utility.values

        #need to find where the points fits in the range
        norm = lambda x: (x - mRange[0]) / (mRange[1] - mRange[0])
        aggregateRange = mRange
        startUtil = model.utility(norm(aggregateRange[0]))
        endUtil = model.utility(norm(aggregateRange[1]))

        ax[0,r].plot(np.linspace(mRange[0], mRange[1], 100), model.utility(np.linspace(0,1,100)))
        ax[0,r].plot(np.linspace(mRange[0], mRange[1], 100), np.linspace(0, 1, 100), '--', color='k')
        ax[0,r].scatter(midpoints, (utils*(endUtil-startUtil))+startUtil, color='k')
        ax[0,r].set_title(str(date))
        ax[0,r].set_ylabel('utility')
        ax[0,r].set_xlabel('reward magnitude')

        SE = ((utils*(endUtil-startUtil))+startUtil - model.utility(norm(midpoints)))
        mSE.extend([np.mean(SE)])
        ax[1,r].plot(midpoints, SE )
        ax[1,r].axhline(0, linestyle='--', color='k')
        ax[1,r].grid()
        #  ax[1,r].set_title('error')
        ax[1,r].set_ylabel('error')
        ax[1,r].set_xlabel('tested midpoint')
    #
    #    utility = lambda x, p1, p2 : model.model.model_parts([10,p1,p2])['utility'](x)
    #    pFit, pCov = opt.curve_fit( utility, midpoints, (utils*(endUtil-startUtil))+startUtil, p0=[1,0.01])  # , bounds=param_bounds)
    #    ax[r,2].plot(np.linspace(0,1,100), utility(np.linspace(0,1,100), pFit[0], pFit[1]), color='k')
    #    ax[r,2].plot(np.linspace(0,1,100), model.utility(np.linspace(0,1,100)), 'b' )

        middies = np.unique(aggregate.midpoint.values)
        RTs = np.concatenate(aggregate.mTimes.values)
        middies = flatten([[mids]*len(rt) for mids, rt in zip(middies, RTs)])
        RTs = flatten(RTs)
        ax[2,r].plot(middies, RTs, 'bo', alpha=0.1, color='k')

        #        sns.boxplot(middies, RTs, ax=ax[0,2], color='white', saturation=1)
        #        plt.setp(ax[2,r].artists, edgecolor = 'k', facecolor='w')
        #        plt.setp(ax[2,r].lines, color='k')
        ax[2,r].grid()
        ax[2,r].axhline(min(RTs), color='k', linestyle = '--')
        ax[2,r].set_xlabel('reward magnitudes')
        ax[2,r].set_ylabel('response times')
        r+=1
    plt.tight_layout()

#%%
def plotAll_Behaviour(Behaviour, fractile, model = '2logit-2scdf-ev'):
    import seaborn as sns
    import scipy.optimize as opt
    fig, ax = plt.subplots( 1, 4, squeeze = False,  figsize=( 10 ,6 ))
    fig2, ax2 = plt.subplots( 1, 1, squeeze = False,  figsize=( 4 ,4 ))

    mRange = unique_listOfLists(fractile.reward_range)[0]
    f_utilities = np.hstack((fractile.utility.values, [0, 1]))
    mask = [True if np.isin(uu,[1.0, 0, 0.75, 0.875, 0.5, 0.125 , 0.25]) else False for uu in f_utilities]
    f_equivalent = np.hstack((fractile.equivalent.values, [mRange[0], mRange[1]]))[mask]
    f_utilities = f_utilities[mask]
    ax[0,0].plot(f_equivalent, f_utilities, 'bo', color = 'darksalmon', alpha=0.4)
    ax2[0,0].plot(f_equivalent, f_utilities, 'bo', color = 'red', alpha=0.4)
    ax[0,0].set_xlim(0, 0.5)
       
    midpoints = Behaviour.midpoint.values
    utilities = Behaviour.utility.values
    
    from macaque.f_models import define_model
    outputs = define_model(model)
    p0 = outputs[1]
    utility = outputs[-1](p0)['empty functions']['utility']
    p0 = outputs[1][1:-1]
    function = lambda x, p1, p2 : utility(x, [p1,p2])
    ff = None
    try:
#        pFit, pCov = opt.curve_fit( function, xdata= (f_equivalent-mRange[0]) / (mRange[1] - mRange[0]),
#                                   ydata= f_utilities,  p0=[3, 0.75], 
#                                   method='lm', maxfev = 1000)
#        def ff(x):
#                return function(x, pFit[0], pFit[1])
#        ax[0,0].plot(np.linspace(mRange[0], mRange[1], 100),
#          ff(np.linspace(0,1,100)), color = 'darkred', linewidth=5)
        
#        maxie = ff(max(midpoints) / 0.5 )
#        minie = ff(min(midpoints) / 0.5)
        
#        pFit, pCov = opt.curve_fit( function, xdata= (midpoints-mRange[0]) / (mRange[1] - mRange[0]),
#                                   ydata= utilities,  p0=[1, 0.5], 
#                                   method='lm', maxfev = 1000)
#        def ff(x):
#                return function(x, pFit[0], pFit[1])
#        ax[0,3].plot(np.linspace(min(midpoints), max(midpoints) , 100),
#          (ff(np.linspace(0,1,100)) * (maxie - minie)) + minie, color = 'darkblue',
#          linewidth=5)
        
        ax[0,3].plot(midpoints, utilities, 'bo', color = 'lightblue', 
          alpha=0.4)
        ax2[0,0].plot(midpoints, utilities, 'bo', color = 'blue', 
          alpha=0.4)
        ax[0,3].plot(np.linspace(mRange[0], mRange[1]), np.linspace(0, 1), '--', color = 'k')
        
    except:
        print('not normalized')
        ax[0,0].plot(midpoints, utilities, 'bo', color = 'lightblue', alpha=0.2, markersize=10)
        ax[0,0].plot(np.linspace(min(midpoints), max(midpoints)), np.linspace(0, 1), '--', color = 'k')
        mMid = np.unique(midpoints)
        mUtil = [np.mean(utilities[midpoints==midpoint]) for midpoint in mMid]
        ax[0,0].plot(mMid, mUtil, 'bo', color='blue')
    
    ax[0,0].grid()
    ax[0,0].set_xlabel('reward magnitudes')
    ax[0,0].set_ylabel('utilities')
    ax[0,3].grid()
    ax[0,3].set_xlabel('reward magnitudes')
    ax[0,3].set_ylabel('utilities')
    ax[0,0].plot(np.linspace(0, 0.5), np.linspace(0, 1), '--', color = 'k')
    ax2[0,0].grid()
    ax2[0,0].set_xlabel('reward magnitudes')
    ax2[0,0].set_ylabel('utilities')
    ax2[0,0].plot(np.linspace(0,0.5), np.linspace(0,1), '--', 'k')
#    ax2[0,0].plot(np.linspace(0,1), np.linspace(0,1), '--', 'k')
    
        

    #%%
    middies = Behaviour.midpoint.values
    temp = Behaviour.choiceTimes.values
#    temp = [merge_dicts(cc) for cc in temp]
    middies = np.hstack([[mm] * len(tt) for mm, tt in zip(middies, [list(cc.keys()) for cc in temp])])
    
    gaps = (np.hstack([list(cc.keys()) for cc in temp]) - middies)*2
    times = flatten([list(cc.values()) for cc in temp], 1)
    mTimes = [np.mean(x) for x in times]
    
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    normalize = mcolors.Normalize(vmin=gaps.min(), vmax=gaps.max())
    colormap = cm.winter
    
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(gaps)
    plt.colorbar(scalarmappaple, ax=ax[0,1], fraction=0.046, pad=0.04)
    
    scalarMap = cm.ScalarMappable(norm=normalize, cmap=colormap)    
    colorVal = scalarMap.to_rgba(gaps)
    
    ax[0,1].scatter(jitter(middies, 0.005), mTimes, c=colorVal, alpha=0.2)
    ax[0,1].grid()
#    ax[0,1].axhline(min(flatten(times)), color='k', linestyle = '--')
    ax[0,1].set_xlabel('reward magnitudes')
    ax[0,1].set_ylabel('response times')
    
    primaries = np.concatenate([[y]*len(x.keys()) for x, y in zip(fractile.choiceTimes, fractile.primaryEV)])
    primaries = primaries - np.hstack([list(x.keys()) for x in fractile.choiceTimes])
    times =  flatten([list(val.values()) for val in fractile.get('choiceTimes').values ])
    times = np.array([np.mean(x) for x in times])
    utils = np.concatenate([[y]*len(x.keys()) for x, y in zip(fractile.choiceTimes, fractile.utility)])
    
    normalize = mcolors.Normalize(vmin=primaries.min(), vmax=primaries.max())
    colormap = cm.spring
    scalarMap = cm.ScalarMappable(norm=normalize, cmap=colormap)    
    colorVal = scalarMap.to_rgba(primaries)
    
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(primaries)
    plt.colorbar(scalarmappaple, ax=ax[0,2], fraction=0.046, pad=0.04)

    ax[0,2].scatter(jitter(utils, 0.015), times, c=colorVal, alpha=0.2)
    ax[0,2].grid()
#    ax[0,2].axhline(min(np.concatenate([np.concatenate(list(x.values())) for x in fractile.choiceTimes.values])), color='k', linestyle = '--')
    ax[0,2].set_xlabel('reward magnitudes')
    ax[0,2].set_ylabel('response times')
#    ax[0,2].set_ylim(ax[0,1].get_ylim())
    squarePlot(ax[0,0])
    squarePlot(ax[0,1])
    squarePlot(ax[0,2])
    squarePlot(ax[0,3])
    plt.tight_layout()
 
    #%%
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    ddf = pd.DataFrame(np.vstack((middies, mTimes, gaps)).T, columns=['middies','RT', 'gap' ])
    formula = 'RT ~ middies + gap + gap:middies'
    olsmodel = ols(formula, data=ddf).fit()
    aov_table = anova_lm(olsmodel, typ=2)
    eta_squared(aov_table)
    #    omega_squared(aov_table)
    print('\n -------------------------------------------')
    print(aov_table)
    print('Nb,Nm =', [
        str(len(ddf.loc[ddf.middies == cc]))
        for cc in ddf.middies.unique()
    ])
    print('middie =', ddf.middies.unique())
    
    ddf = pd.DataFrame(np.vstack((primaries, times, utils)).T, columns=['primaries','RT', 'utilities'])
    formula = 'RT ~ primaries + C(utilities) +  primaries:C(utilities) '
    olsmodel = ols(formula, data=ddf).fit()
    aov_table = anova_lm(olsmodel, typ=2)
    eta_squared(aov_table)
    #    omega_squared(aov_table)
    print('\n -------------------------------------------')
    print(aov_table)
    print('Nb,Nm =', [
        str(len(ddf.loc[ddf.utilities == cc]))
        for cc in ddf.utilities.unique()
    ])
    print('middie =', ddf.utilities.unique())
#    plt.tight_layout()

    
#%%
def plot_allMidpoints(Behaviour):
    '''
    '''
    import seaborn as sns
    fig, ax = plt.subplots( 1, 2, squeeze = False,  figsize=( 6 ,3 ))

    midpoints = np.concatenate(Behaviour.midpoints.values)
    utilities = np.concatenate(Behaviour.utilities.values)

    ax[0,0].plot(midpoints, utilities, 'bo', color = 'k', alpha=0.2)
    ax[0,0].plot(np.linspace(min(midpoints), max(midpoints)), np.linspace(0, 1), '--', color = 'k')
    ax[0,0].grid()
    ax[0,0].set_xlabel('reward magnitudes')
    ax[0,0].set_ylabel('utilities')

    mMid = np.unique(midpoints)
    mUtil = [np.mean(utilities[midpoints==midpoint]) for midpoint in mMid]

    ax[0,0].plot(mMid, mUtil, 'bo', color='red')

    RT = np.concatenate(Behaviour.mean_chTimes.values)

    sns.boxplot(midpoints, RT, ax=ax[0,1], color='white', saturation=1)
    plt.setp(ax[0,1].artists, edgecolor = 'k', facecolor='w')
    plt.setp(ax[0,1].lines, color='k')
    ax[0,1].grid()
#    ax[0,1].axhline(min(np.concatenate([np.concatenate(Behaviour.choiceTimes.values[x]) for x in range(len(Behaviour))])), color='k', linestyle = '--')
    ax[0,1].set_xlabel('reward magnitudes')
    ax[0,1].set_ylabel('response times')
    plt.tight_layout()

#%%
def get_undefiniedFractiles(riskySM, mismatch = 0.025):
    '''
    Getting the fractile utility points in Ugo
    '''
    from macaque.f_choices import get_options
    from macaque.f_psychometrics import get_softmaxData

    fract_SM = riskySM.loc[riskySM.primary.apply( lambda x: len(x) <= 4)].copy()
    fract_SM = fract_SM.loc[fract_SM.equivalent.apply(lambda x: 0 <= x <= 0.5)]
    # -------------------------------------------------------------------------
    fract_SM = fract_SM.loc[(fract_SM.gList.str.contains('step')) | (fract_SM.seqCode == 9001)]

    for date in tqdm(fract_SM.sessionDate.unique()):
        sm = fract_SM.loc[fract_SM['sessionDate'] == date]
        mRange = np.concatenate(np.vstack(sm.primary.values)[:, 0::2])
        fract_SM.loc[fract_SM.sessionDate == date, 'min_rew'] = min(mRange)
        fract_SM.loc[fract_SM.sessionDate == date, 'max_rew'] = max(mRange)
    fract_SM['reward_range'] = fract_SM[['min_rew', 'max_rew']].values.tolist()
    fract_SM.drop(columns=['min_rew'], inplace=True)
    fract_SM.drop(columns=['max_rew'], inplace=True)

    fractDates = fract_SM.loc[fract_SM.gList.str.contains('step')].sessionDate.unique()
    fract_SM = fract_SM.loc[np.isin(fract_SM.sessionDate, fractDates) ]
    fract_SM.loc[fract_SM.gList.str.contains('step2'), 'fractile'] = 1
    fract_SM.loc[fract_SM.gList.str.contains('step3'), 'fractile'] = 2
    fract_SM.loc[fract_SM.gList.str.contains('util4'), 'fractile'] = 3

    #--------------------------------------------------------------------------

    reward_range=[0, 0.5]
    fract_SM['utility'] = np.nan
    dlist = []
    for date in tqdm(fract_SM.sessionDate.unique(), desc='assigning utility values'):
        sm = fract_SM.loc[fract_SM['sessionDate'] == date]
        topStep = np.nanmax(sm.fractile.values)
        topBlocks = sm.loc[sm.fractile == topStep]  #trials that have a step2
        chains = [0]
        chains.extend([sm.loc[sm.division == block].index.max() for block in topBlocks.division.unique()])
        for i in range(1,len(chains)):
            df = sm.loc[(sm.index <= chains[i]) & (sm.index >= chains[i-1])]
            df['division'] = df.division.values.min()
            for ii, row in df.iterrows():
                if row['fractile'] == 1:
                    if row.primaryEV > np.mean(reward_range):
                        df.loc[ii, 'utility'] = 0.75
                        first_step = row['primary'][0]
                        df.loc[df.index, 'first_step'] = first_step
                    else:
                        df.loc[ii, 'utility'] = 0.25
                        first_step = row['primary'][2]
                        df.loc[df.index, 'first_step'] = first_step
                if row['fractile'] == 2:
                    if row.primaryEV > np.mean(reward_range):
                        df.loc[ii, 'utility'] = 0.875
                    else:
                        df.loc[ii, 'utility'] = 0.125


            ii = df.loc[df.primaryEV == np.mean(reward_range)].index
            step1 = df.loc[(df.primaryEV == np.mean(reward_range)) & (np.abs(df.equivalent.values - first_step) < mismatch)].index
            if not step1.empty:
                df.loc[step1, 'fractile'] = 0
                df.loc[step1, 'utility'] = 0.5

            df.dropna(subset=['utility'], inplace = True)
            dlist.append(df)
    #-----------------------------------------------------------------------------

    fract_SM = pd.concat(dlist)
    cols = [
        'sessionDate', 'reward_range', 'division', 'fractile', 'primary',
        'primaryEV', 'equivalent', 'utility', 'secondary', 'secondaryEV',
        'm_range', 'freq_sCh', 'pFit', 'pSTE', 'no_of_Trials', 'nTrials',
        'primarySide', 'choiceList', 'choiceTimes', 'moveTime', 'trial_index',
        'oClock', 'func', 'metricType', 'seqCode', 'gList', 'first_step' ]

    fract_SM = fract_SM.sort_values(by=['sessionDate', 'division', 'utility'])[cols]
    fract_SM['iTrials'] = [str(np.sort(np.concatenate(list(val.values()))))for val in fract_SM.get('trial_index').values]
    fract_SM.drop_duplicates(
        subset=['iTrials', 'division', 'sessionDate'],
        keep='first',
        inplace=True)
    fract_SM.reset_index(drop=True, inplace=True)
    return fract_SM

#%%
def get_fractile(Trials, undefinedFractiles = False, limited_range = [[0.0, 0.5]]):
    '''
    '''
    from macaque.f_utility import get_fractileUtility
    riskySM = Trials.get_Risky(minSecondaries = 3, minChoices = 4)
    if undefinedFractiles == True:
        fractile = get_undefiniedFractiles(riskySM, 0.025)
    else:
        riskySM = riskySM.loc[riskySM.seqCode == 9050]
        fractile = get_fractileUtility(riskySM)

    # ----------------------------------
    def specificRange(df, rr):
        return df.apply(lambda x: x== rr)

    if limited_range:
        dList = []
        for rr in limited_range:
            dList.append(fractile.loc[specificRange(fractile.reward_range, rr)])
        fractile = pd.concat(dList)
        return fractile
    else:
        return fractile
#%%
def get_fractileComparison(Trials, MLE_list, Behaviour, fractile = None, 
                           best_model = None, minPoints = 3, fixedRange = False):
    '''
    '''
    import scipy.optimize as opt
    from macaque.f_models import trials_2fittable, LL_fit, define_model
    from scipy.stats import linregress
    from scipy.odr import ODR, Model, Data, RealData

#    def sCDF(mm, p1,p2):
#        Xs = np.atleast_1d(mm) #range position
#        inflection, temp = p2,p1
#        if inflection > 1 or inflection < 0:
#            inflection = np.nan
#        return np.array([inflection*((X/inflection)**temp) if X<inflection else 1-((1-inflection)*(((1-X)/(1-inflection))**temp)) for X in Xs])

    if best_model == None:
        mBIC = []; position=[]
        for i,mle in enumerate(MLE_list):
            mBIC.extend([mle.BIC.mean()])
            position.extend([i])

        mBIC = np.array(mBIC); position=np.array(position)
        mle = MLE_list[int(position[mBIC == min(mBIC)])]
    else:
        decisionFunctions = best_model.split('-')
        decisionFunctions[2] = 'ev'; decisionFunctions = decisionFunctions[:3]
        decisionFunctions = '-'.join(decisionFunctions)
        ii = [i for i, mle in enumerate(MLE_list) if mle.model_used.iloc[0] == decisionFunctions][0]
        mle = MLE_list[ii]

    if len(fractile) == 0:
         fractile = get_fractile(Trials)

    #get the dates that are intersects of either of the 3 utility techniques
    dates_in_third = np.intersect1d(Behaviour.sessionDate.unique(), fractile.sessionDate.unique())
    dates_in_both = np.intersect1d(fractile.sessionDate.unique(), mle.date.unique())
    dating = np.unique(np.concatenate((dates_in_third, dates_in_both)))
    if dating.size == 0:
        print('\n ** No matching utility and random utility days found **')
        return [],[]

    tt = fractile.getTrials(Trials) #here I get the trials that form the fractile utility

    fig, ax = plt.subplots( 3, len(dating), squeeze = False,
                           figsize=( int(np.ceil(len(dating)/3))*6 , 6 ))
    fig2, secondAxis = plt.subplots( 1, 2, squeeze = False,
                           figsize=( 8 , 4 ))

    r=0; dList=[]; dList2 = []
    for date in tqdm(dating, desc = 'fitting fractiles'):
        randomDF = Behaviour.loc[Behaviour.sessionDate == date]
        modelDF = mle.loc[mle.date == date]
        fractileDF = fractile.loc[fractile.sessionDate == date]

        mRange = fractileDF.loc[fractileDF.fractile == 0].primary.iloc[0][::2]

        X, Y = trials_2fittable(tt.loc[tt['sessionDate'] == date])
        MLE = LL_fit(Y, X, model = best_model, fixedRange = fixedRange).fit(disp=False)

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
                'mag_range' : mRange
            })

    
        position = np.argmax(['u_' in pname for pname in MLE.model.exog_names])
        long = np.sum(['u_' in pname for pname in MLE.model.exog_names])
        p0 = MLE.model.start_params[position:position+long]
              
        util = lambda params, x : MLE.model_parts['empty functions']['utility'](x, params)
#        utility = ff[-1](p0)['empty functions']['utility']
#        sCDF = lambda x, p1, p2 : utility(x, [p1,p2])
        # fractile first
#        function = lambda x, p1, p2 : sCDF(x, [p1,p2])
        if len(fractileDF) < 3:
            ax[0,r].text(0.1, 0.5, 'little points')
            ax[1,r].text(0.1, 0.5, 'little points')
        
        
        x = []; y = [];
        for block in fractileDF.division.unique():
                yy = fractileDF.loc[fractileDF.division == block].utility.values
                xx = fractileDF.loc[fractileDF.division == block].equivalent.values
                if len(np.unique(yy)) >= minPoints:
                    x.extend(xx)
                    y.extend(yy)

        ax[0,r].scatter(x,y, color='k')
        ax[1,r].scatter(x,y, color='k')
        x = np.array(x) / mRange[1]
        y = np.array(y)
#        y = np.array(np.insert(y, [0, len(y)], [0, 1]).tolist())

#        try:
        data = RealData(x, y)
        model2fit = Model(util)
#        print( MLE.model.start_params )
        odr = ODR(data, model2fit, p0)
#            odr.set_job(fit_type=2)
        output = odr.run()
        pFit = output.beta

##        except:
#            print('1' + str(date))
#            continue
        function = lambda x : util(pFit, x)

        if len(fractileDF)>=3 and len(X) < 100:
            ax[2,r].text(0.1, 0.5, 'Trials count under 100')

        ax[0,r].plot(np.linspace(mRange[0], mRange[1], 100), function(np.linspace(0,1, 100)), color='k', alpha=0.5)
        ax[1,r].plot(np.linspace(mRange[0], mRange[1], 100), function(np.linspace(0,1, 100)), color='k', alpha=0.5)

        ax[0,r].plot(np.linspace(mRange[0], mRange[1], 100), np.linspace(0, 1, 100), '--', color='k')
        ax[1,r].plot(np.linspace(mRange[0], mRange[1], 100), np.linspace(0, 1, 100), '--', color='k')

        norm = lambda x: (x - mRange[0]) / (mRange[1] - mRange[0])

        if not randomDF.empty:
            #random behaviour
            midpoints = randomDF.midpoint.values
            utils = randomDF.utility.values

            randomDFRange = [min(midpoints), max(midpoints)]
            startRandom = norm(randomDFRange[0])
            endRandom = norm(randomDFRange[1])

            ax[1,r].scatter(midpoints, (utils*(endRandom-startRandom))+startRandom, color='blue')

            startRandom = MLE.utility(randomDFRange[0])
            endRandom = MLE.utility(randomDFRange[1]/mRange[1])
            ax[2,r].plot(midpoints, (utils*(endRandom-startRandom))+startRandom, color='blue')

#            try:
            data = RealData(norm(midpoints), utils)
            model2fit = Model(util)
            odr = ODR(data, model2fit, p0)
#            odr.set_job(fit_type=2)
            output = odr.run()
            pFit2 = output.beta
                
#                pFit2, pCov2 = opt.curve_fit( sCDF, midpoints, utils, p0=[1, 0.5], method='lm', maxfev = 1000)
#            except:
#                print('2' + str(date))
#                continue
            function2 = lambda x : util(pFit2, x)
#            function2 = lambda x : sCDF(x, pFit2[0], pFit2[1])
            
        if not modelDF.empty:
            #model
            model = modelDF['full_model'].iloc[0]
            modelRange = modelDF['mag_range'].iloc[0]

            startModel = MLE.utility(modelRange[0])
            endModel = MLE.utility(modelRange[1]/mRange[1])

            ax[2,r].plot(np.linspace(modelRange[0], modelRange[1], 100),
              (model.utility(np.linspace(0,1,100))*(endModel-startModel))+startModel,
              color = 'red')

            ax[0,r].plot(np.linspace(modelRange[0], modelRange[1], 100),
              model.utility(np.linspace(0,1,100)),
              color = 'red')

        ax[0,r].set_title(str(date))
        ax[0,r].set_ylabel('utility')
        ax[0,r].set_xlabel('reward magnitude')
        ax[0,r].grid()

        ax[1,r].set_title(str(date))
        ax[1,r].set_ylabel('utility')
        ax[1,r].set_xlabel('reward magnitude')
        ax[1,r].grid()

        if len(fractileDF) >= 3:
            ax[2,r].plot( np.linspace(mRange[0],mRange[1],100), MLE.utility(np.linspace(0,1,100)), color='k')
        ax[2,r].plot( np.linspace(mRange[0],mRange[1],100), np.linspace(0,1,100), '--', color='k', alpha=0.5)
        r+=1

        #Now plot the collective figure: ---------------------------------------------------------------------
        if  len(fractileDF) >= 3:
            model.utility(np.linspace(0,1,100))
            randomFit =  np.vstack((np.linspace(0,1, 100),
                  model.utility(np.linspace(0,1,100)))).T
            fractileFit = np.vstack((np.linspace(0,1, 100), MLE.utility(np.linspace(0,1, 100)))).T
            fit_Corr = linregress(randomFit[:,1], fractileFit[:,1]).rvalue
            fit_RMSerr = np.sqrt(np.mean((randomFit[:,1] - fractileFit[:,1])**2))

            secondAxis[0,0].plot(randomFit[:,1], fractileFit[:,1], color = 'magenta')
            secondAxis[0,0].plot(randomFit[:,1], randomFit[:,1], '--', alpha = 0.5, color='k')

            startRandom = MLE.utility(randomDFRange[0])
            endRandom = MLE.utility(randomDFRange[1]/mRange[1])
            secondAxis[0,1].plot( (function2(np.linspace(0,1))*(endRandom-startRandom))+startRandom, function(np.linspace(startModel, endModel)), color='cyan')
#            secondAxis[0,1].plot( (utils*(endRandom-startRandom))+startRandom, MLE.utility(midpoints/mRange[1]), color = 'cyan' )
            secondAxis[0,1].plot( (function2(np.linspace(0,1))*(endRandom-startRandom))+startRandom, (function2(np.linspace(0,1))*(endRandom-startRandom))+startRandom,  '--', alpha = 0.5, color='k')
            behave_Corr = linregress((function2(np.linspace(0,1))*(endRandom-startRandom))+startRandom, function(np.linspace(startModel, endModel))).rvalue
            behave_RMSerr = np.sqrt(np.mean((function2(np.linspace(0,1))*(endRandom-startRandom))+startRandom - function(np.linspace(startModel, endModel))**2))

            dList2.append({
                    'date':date,
                    'fit_correlation':fit_Corr,
                    'fit_RMSerr':fit_RMSerr,
                    'behaviour_correlation':behave_Corr,
                    'behaviour_RMSerr':behave_RMSerr,
                    'riskless_params': modelDF.params.values,
                    'risky_params': MLE.params,
                    'riskless_range': modelRange,
                    'risky_range': mRange,
                    'pNames': [MLE.model.exog_names], 
                    'params_fractile': pFit,
                    'params_random': pFit2,
                    'model_used': MLE.model.model_name})

    secondAxis[0,0].set_ylabel('risky utility')
    secondAxis[0,0].set_xlabel('riskless utility')
    secondAxis[0,0].set_title('fit comparisons')
    secondAxis[0,0].grid()
    secondAxis[0,1].set_ylabel('risky utility')
    secondAxis[0,1].set_xlabel('riskless utility')
    secondAxis[0,1].set_title('behaviour comparison')
    secondAxis[0,1].grid()
    fig.tight_layout()

    cols = ['date', 'nTrials', 'NM_success', 'params', 'model_used', 'AIC', 'BIC', 'LL',
            'pNames', 'pvalues', 'full_model', 'Nfeval', 'all_fits', 'mag_range']
    return MLE_object(pd.DataFrame(dList))[cols], pd.DataFrame(dList2)

#%%
def pool_measuredUtilities(Trials, perDaySM, fractile, Behaviour, best_model = 'risky-scdf'):
    '''
    '''
    from macaque.f_models import get_modelLL, trials_2fittable, LL_fit

    tt_riskless = perDaySM.getTrials(Trials)
    X, Y = trials_2fittable(tt_riskless, use_Outcomes = False)
    MLE_riskless = LL_fit(Y, X, model = best_model.replace('power', 'ev').replace('-cumulative','')).fit(disp=False)
    MLE_riskless.plot_fullModel()
    x = [(bb - rr[0])/(rr[1] - rr[0]) for bb, rr in zip(Behaviour.midpoints.values, Behaviour.mag_range.values)]
    x = np.hstack(x)
    y = np.hstack(Behaviour.utilities.values)
    MLE_riskless.plot_utility()
    ax = plt.gca()
    ax.scatter(x,y, alpha = 0.3, color = 'k')

    tt_risky = fractile.getTrials(Trials)
    X, Y = trials_2fittable(tt_risky, use_Outcomes = False)
    MLE_risky = LL_fit(Y, X, model = best_model).fit(disp=False)
    MLE_risky.plot_fullModel()
    x = [(bb - rr[0])/(rr[1] - rr[0]) for bb, rr in zip(fractile.equivalent.values, fractile.reward_range.values)]
    y = fractile.utility.values
    MLE_risky.plot_utility()
    ax = plt.gca()
    ax.scatter(x,y, alpha = 0.3, color = 'k')

#%%
def pool_equivariants(equivariant, gambleCEs, trials,  model = 'risky-scdf', plotFittings= True):
    '''
    '''
    from macaque.f_models import trials_2fittable, LL_fit
    if len(equivariant) != 0:
        for rr in unique_listOfLists(equivariant.m_range.values):
            tt = equivariant.loc[equivariant.m_range.apply(lambda x: x==rr)].getTrials(trials)
            X, Y = trials_2fittable(tt, use_Outcomes = False)
            MLE = LL_fit(Y, X, model = model).fit(disp=False)
            print('-----------------------------------------------------')
            print('range of rewards: ' + str(rr))
            print( np.sort(equivariant.loc[equivariant.m_range.apply(lambda x: x==rr)].sessionDate.apply(lambda x: x.strftime("%Y-%m-%d")).unique()) )
            MLE.plot_fullModel(title = 'equivariant utilities')

        # ----- LOOKING AT GAMBEL CES ALSO ---------------------------------
    if len(gambleCEs) != 0:
        for rr in unique_listOfLists(np.vstack(gambleCEs.m_range.values)):
            tt = gambleCEs.loc[gambleCEs.m_range.apply(lambda x: x==rr)].getTrials(trials)
            X, Y = trials_2fittable(tt, use_Outcomes = False)
            MLE2 = LL_fit(Y, X, model = model).fit(disp=False)
            MLE2.plot_fullModel(title = 'gambleCEs utilities')
            print('reward range for gamble CEs: ' + str(rr))

        #plot the evolution of the LL and of the parameters
        if plotFittings and len(equivariant) != 0:
            fig, ax = plt.subplots( 1, 1, squeeze=False, figsize=(6,3))
            fig.suptitle(
                'Function Value Estimates (and parameters) during minimization')
            colors = ['orange', 'teal', 'm', 'darkgreen']
            ax[0,0].set_ylabel('- LL')
            ax[0,0].set_xlabel('runs')
            ax[0,0].plot(MLE.Nfeval, linewidth=3, color='k')
            for dd in range(len(MLE.params)):
                cc = colors[dd]
                ax2 = ax[0,0].twinx()
                ax2.spines["right"].set_position(("axes", 1 + 0.1 * dd))
                ax2.spines["right"].set_visible(True)
                ax2.plot(np.vstack(MLE.res_x)[:, dd], cc, alpha=0.4)
                ax2.set_ylabel('param ' + MLE.model.exog_names[dd], color=cc)
                ax2.tick_params('y', colors=cc)

        if plotFittings and len(equivariant) != 0:
            fig, ax = plt.subplots( 1, 1, squeeze=False, figsize=(6,3))
            fig.suptitle(
                'Function Value Estimates (and parameters) during minimization')
            ax[0,0].set_ylabel('- LL')
            ax[0,0].set_xlabel('runs')
            ax[0,0].plot(MLE2.Nfeval, linewidth=3, color='k')
            for dd in range(len(MLE2.params)):
                cc = colors[dd]
                ax3 = ax[0,0].twinx()
                ax3.spines["right"].set_position(("axes", 1 + 0.1 * dd))
                ax3.spines["right"].set_visible(True)
                ax3.plot(np.vstack(MLE2.res_x)[:, dd], cc, alpha=0.4)
                ax3.set_ylabel('param ' + MLE2.model.exog_names[dd], color=cc)
                ax3.tick_params('y', colors=cc)

#%%
def get_equivariants(trials, minGambles = 30):
    from macaque.f_choices import get_options, get_psychData
    choices = get_options(trials.loc[trials.trialSequenceMode == 9020],
                          mergeBy = 'block', byDates=True)
    gchoices = get_psychData(choices, metricType='transitivity', transitType='gambles')

    sm = choices.getPsychometrics(metricType = 'ce', minSecondaries = 3, minChoices = 4)

    gchoices['spread_1'] = gchoices.option1.apply(lambda x: np.round(np.diff(x[::2]), 2)[0])
    gchoices['spread_2'] = gchoices.option2.apply(lambda x: np.round(np.diff(x[::2]), 2)[0])

    sm['spread'] = sm.primary.apply(lambda x: np.round(np.diff(x[::2]), 2)[0])
    sm = sm.loc[sm.primary.apply(lambda x: len(x) <= 4)]

    #get equal-spread gambles
    equivariant = gchoices.loc[gchoices['spread_1'] == gchoices['spread_2']]
    equivariant.spread_1.unique() #see the unique spreads

    probabilityMask_1 = equivariant.option1.apply(lambda x: len(np.unique(x[1::2]))==1)
    probabilityMask_2 = equivariant.option2.apply(lambda x: len(np.unique(x[1::2]))==1)
    equivariant = equivariant.loc[ probabilityMask_1 & probabilityMask_2 ]

    equivariant.drop(columns=['spread_2'], inplace=True)

    dList = []; spreading = []
    for spread in equivariant.spread_1.unique():
        df = equivariant.loc[equivariant.spread_1 == spread]
        if len(df) < minGambles:
            continue
        else:
            spreading.extend([spread])
        psycho = df.getPsychometrics(metricType = 'trans', minSecondaries = 1)
        psycho['spread'] = spread
        psycho = sort_byEquivariant(psycho)
        dList.append(psycho)
    if len(dList) != 0:
        dList = pd.concat(dList, ignore_index = True)
        sm = sm.loc[ np.isin(sm.sessionDate, dList.sessionDate.unique()) ]
        sm_spread = sm.loc[np.isin(sm.spread, spreading)]
        date_list = dList.sessionDate.unique()
    else:
        from collections import Counter
        counting = Counter(sm.spread)
        spreading = [key for key, val in counting.items() if val>minGambles]
        sm_spread = sm.loc[np.isin(sm.spread, spreading)]
        date_list = sm_spread.sessionDate.unique()

    for date in date_list:
        if len(dList) != 0:
            secondaries = np.hstack(dList.loc[dList.sessionDate == date].secondaryEV.values)
        else:
            secondaries = np.hstack(sm_spread.loc[sm_spread.sessionDate == date].secondaryEV.values)
        if any(secondaries > 0.5):
            if len(dList) != 0:
                dList.loc[dList.sessionDate == date, 'm_range'] = [[[0, 1.0]] * len(dList.loc[dList.sessionDate == date])]
            if np.isin(date, sm_spread.sessionDate.unique()):
                sm_spread.loc[sm_spread.sessionDate == date, 'm_range'] = [[[0, 1.0]] * len(sm_spread.loc[sm_spread.sessionDate == date])]
        else:
            if len(dList) != 0:
                dList.loc[dList.sessionDate == date, 'm_range'] = [[[0, 0.5]] * len(dList.loc[dList.sessionDate == date])]
            if np.isin(date, sm_spread.sessionDate.unique()):
                sm_spread.loc[sm_spread.sessionDate == date, 'm_range'] = [[[0, 0.5]] * len(sm_spread.loc[sm_spread.sessionDate == date])]

    sm_spread = sm_spread.loc[sm_spread.spread != sm_spread.primary.apply(lambda x: x[2])]
    return dList, sm_spread

#%%
def sort_byEquivariant(softmaxDF):
    '''
    '''
    import scipy.optimize as opt
    from macaque.f_psychometrics import psychometricDF
    np.warnings.filterwarnings('ignore')
    sigmoid = lambda x, p2: np.array(1 / (1 + np.exp(-(x - 0) / p2)))
    newDF = []
    param_bounds = ([0.01], [1])
    for date in tqdm(softmaxDF.sessionDate.unique()):
        df = softmaxDF.loc[softmaxDF.sessionDate == date]
        unique_gambles = np.unique(df.primary)
        for gamble in unique_gambles:
            subdf = df.loc[df.primary.apply(lambda x: x==gamble)]

            sEV = np.squeeze(subdf.secondaryEV.tolist())
            pChooseSecondary = np.squeeze(subdf.freq_sCh.tolist())
            popt, pcov = opt.curve_fit(sigmoid, sEV, pChooseSecondary, p0=[1], method='lm')
#                                       bounds = param_bounds)

            newDF.append(pd.DataFrame({'sessionDate' : date,
                     'primary' : [subdf.primary.values[0]],
                     'primaryEV' : subdf.primaryEV.values[0],
                     'secondary' : [np.squeeze(subdf.secondary.tolist(), 2)],
                     'secondaryEV' : [sEV],
                     'temperature' : popt[0],
                     'temp_error' : pcov[0],
                     'm_range' : [subdf.m_range.min()],
                     'freq_sCh' : [subdf.freq_sCh.tolist()],
                     'no_of_Trials' : [np.squeeze(subdf.no_of_Trials.tolist())],
                     'nTrials' : np.sum(np.squeeze(subdf.no_of_Trials.tolist())),
                     'primarySide' : [merge_dicts(subdf.primarySide.values)],
                     'choiceList' : [merge_dicts(subdf.choiceList.values)],
                     'filteredRT' : [merge_dicts(subdf.filteredRT.values)],
                     'choiceTimes' : [merge_dicts(subdf.choiceTimes.values)],
                     'moveTime' : [merge_dicts(subdf.moveTime.values)],
                     'trial_index' : [merge_dicts(subdf.trial_index.values)],
                     'oClock' : [merge_dicts(subdf.oClock.values)],
                     'metricType' : np.unique(subdf.metricType)[0],
                     'division' : np.unique(subdf.division)[0],
                     'seqCode' : [np.unique(subdf.seqCode)[0]],
                     'gList' : [np.unique(subdf.gList)[0]],
                     'chosenEV' : [merge_dicts(subdf.chosenEV.values)],
                     'spread' : np.unique(subdf.spread.values)[0] }))
    newDF = pd.concat(newDF, ignore_index=True)
    newDF = psychometricDF(newDF.sort_values(by=['sessionDate', 'primaryEV']))
    cols = ['sessionDate', 'primary', 'primaryEV', 'secondary', 'secondaryEV',
       'spread', 'temperature', 'temp_error', 'm_range',  'freq_sCh', 'no_of_Trials',
       'nTrials', 'primarySide', 'choiceList', 'filteredRT', 'choiceTimes',
       'moveTime', 'trial_index', 'oClock', 'metricType', 'division',
       'seqCode', 'gList', 'chosenEV']
    return newDF[cols]

#%%
def get_fakeCorrelation(fract_mle):
    mBIC = []; position=[]
    for i,mle in enumerate(fract_mle):
        mBIC.extend([mle.BIC.mean()])
        position.extend([i])

    mBIC = np.array(mBIC); position=np.array(position)
    mle = fract_mle[int(position[mBIC == min(mBIC)])]

    dList = []
    for date in mle.date.unique():
        df = mle.loc[mle.date == date]
        dList.append({
                'date':date,
                'fit_correlation':np.nan,
                'fit_RMSerr':np.nan,
                'behaviour_correlation':np.nan,
                'behaviour_RMSerr':np.nan,
                'riskless_params': np.nan,
                'risky_params': np.ravel(df.params.tolist()),
                'riskless_range': np.nan,
                'risky_range': np.ravel(df.mag_range.tolist()),
                'pNames': np.ravel(df.pNames.tolist())})
    dList = pd.DataFrame(dList)
    return dList

#%%
def get_spreadSpecific(gambleCEs):
    from macaque.f_models import trials_2fittable, LL_fit
    dList = []
    for spread in gambleCEs.spread.unique():
        df = gambleCEs.loc[gambleCEs.spread == spread]
        tt = df.getTrials(Trials)
        X, Y = trials_2fittable(tt, use_Outcomes = False)
        MLE = LL_fit(Y, X, model = 'risky-scdf').fit(disp=False)
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
                        'spread' : spread,
                        'trials' : [tt.loc[tt['sessionDate'] == date].index]
                    })
        return pd.DataFrame(dList)

#%%
class MLE_object(pd.DataFrame):
    '''
    '''
    @property
    def _constructor(self):
        return MLE_object

    def plot_Softmaxes(self):
        ax = plt.gca()
        plotData = np.vstack(( np.linspace(0,1,100), np.ones(100), np.zeros(100),
            np.zeros(100), np.linspace(1,0,100), np.ones(100),
            np.zeros(100), np.zeros(100))).T
        for xx in self.full_model.apply(lambda x: x.softmax).values:
            ax.plot(plotData[:,0] - plotData[:,-4], xx(plotData), color = 'k', alpha = 0.2)
        params = self.params.mean()
        functions = self.full_model.apply(lambda x: x.model.model_parts).iloc[0](params)
        ax.plot(plotData[:,0] - plotData[:,-4], functions['prob_chA'](plotData), color = 'blueviolet', linewidth = 3, alpha=0.8)
        ax.grid(which='both',axis='x')
        ax.set_title('softmax curve')
        ax.set_xlabel('value difference')
        ax.set_ylabel('pCh left option')

    def plot_Utilities(self):
        ax = plt.gca()
        ranges=[]
        for xx, rr in zip(self.full_model.apply(lambda x: x.utility).values, self.mag_range.values):
            plotData = np.linspace(rr[0],rr[1],100)
            ax.plot(plotData, xx(np.linspace(0,1,100)), color = 'k', alpha = 0.3)
            ranges.append(rr)

        for rr in np.unique(ranges, axis=0):
            subset = self.loc[self.mag_range.apply(lambda x : all(x==rr))]
            params = subset.params.mean()
            functions = subset.full_model.apply(lambda x: x.model.model_parts).iloc[0](params)
            plotData = np.linspace(rr[0],rr[1],100)
            ax.plot(plotData, functions['utility'](np.linspace(0,1,100)), color = 'r', linewidth = 3, alpha=0.8)
            ax.plot(plotData, np.linspace(0,1,100), '--', color = 'k', linewidth = 3)
        plt.grid()
        ax.set_title('utility')
        ax.set_xlabel('reward magnitude')
        ax.set_ylabel('utils')

    def plot_Probabilities(self):
        ax = plt.gca()
        plotData = np.linspace(0,1,100)
        for xx in self.full_model.apply(lambda x: x.probability).values:
            ax.plot(plotData, xx(plotData), color = 'k', alpha = 0.2)
        params = self.params.mean()
        functions = self.full_model.apply(lambda x: x.model.model_parts).iloc[0](params)
        ax.plot( plotData,  functions['probability_distortion'](plotData), color = 'blue', linewidth = 3)
        ax.plot(plotData, plotData, '--', color = 'k', linewidth = 3)

        ax.set_title('probability distortion')
        ax.set_xlabel('reward probability')
        ax.set_ylabel('subjective probability')
        plt.grid()

    def plot_fullModels(self):
        fig, ax = plt.subplots(1,3, squeeze=False, figsize=(12,3))
        plt.sca(ax[0,0])
        self.plot_Softmaxes()
        plt.sca(ax[0,1])
        self.plot_Utilities()
        plt.sca(ax[0,2])
        self.plot_Probabilities()
        plt.tight_layout()
        plt.title(self.model_used.unique()[0])

    def summary_long(self):
        from statsmodels.iolib.summary2 import summary_col
        info_dict={'LL' : lambda x: "{:.2f}".format(x.llf),
               'BIC' : lambda x: "{:.2f}".format(x.bic),
               'No. observations' : lambda x: "{0:d}".format(int(x.nobs))}

        summary = summary_col([x for x in self.full_model.values],
                    stars = True,
                    model_names=self.date.apply(lambda x : x.strftime("%d/%m/%y")).tolist(),
                    info_dict=info_dict)
        print(summary)

    def summary_short(self):
        from tabulate import tabulate
        df = self.copy()
        for i,name in enumerate(self.pNames[0]):
            df[name] = np.vstack(self['params'].values)[:,i]
        cols = np.concatenate((self.pNames[0],[ 'LL','BIC', 'nTrials'] ))
        df = df[cols].describe()
        df.drop('count', inplace=True)
        print('\n' +  tabulate(df, headers='keys', tablefmt="presto", floatfmt=".2f"))

#%%
from types import MethodType
class randomDF(pd.DataFrame):
    @property
    def _constructor(self):
        return randomDF

    def get_Random(self, minPoints = 3, minSecondaries = 3, minChoices = 6, minRatio = 0.5):
        from macaque.f_choices import get_options, get_psychData
        from macaque.f_psychometrics import get_softmaxData
        listChoices = get_options(self.loc[self.trialSequenceMode == 9020], byDates = True)
        rChoices = get_psychData(listChoices, metricType = 'trans', transitType = 'safes')
#        rChoices = rChoices.loc[( rChoices.chose1 / ( rChoices.chose1 + rChoices.chose2 ) ) >= minRatio]        
        filteredDF = get_softmaxData(rChoices, metricType = 'trans', minSecondaries = 0, minChoices = 4)
        return filteredDF
#        multiGAP, uniGAP = f_utility.get_randomUtility(rChoices, minPoints = 3, minSecondaries = 3, minChoices = 6, plotit = True)

    def get_Merged(self, minPoints = 3, minSecondaries = 0, minChoices = 4):
        from macaque.f_choices import get_options, get_psychData
        from macaque.f_psychometrics import get_softmaxData
        listChoices = get_options(self)
        safeChoices = get_psychData(listChoices, metricType = 'trans', transitType = 'safes')
        filteredDF = get_softmaxData(safeChoices, metricType = 'trans', minSecondaries = minSecondaries, minChoices = minChoices)
        return filteredDF

    def get_Risky(self,  minSecondaries = 3, minChoices = 4, mergeBy='block'):
        from macaque.f_choices import get_options, get_psychData
        from macaque.f_psychometrics import get_softmaxData
        fractChoices = get_options(self, mergeBy = mergeBy, byDates = True, mergeSequentials=False)
        filteredDF = get_softmaxData(fractChoices, metricType = 'CE', minSecondaries = minSecondaries, minChoices = minChoices)
        return filteredDF