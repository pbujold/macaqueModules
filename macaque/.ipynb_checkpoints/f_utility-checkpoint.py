"""
Module specifically for the analysis of utility
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
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
plt.rc('errorbar', capsize = 3)

#plt.rcParams['font.family'] = 'sans-serif'
from macaque.f_toolbox import *
tqdm = ipynb_tqdm()

#%%
def compare_rangeAndUtility(Trials, fractile, fractile_MLE, binary = None, binary_MLE = None):
    '''
    '''
#    sb.set_context("paper")
    
    results = [[],[],[]]
    cum_mags = []
    norm = lambda x: (x - min(x)) / (max(x) - min(x))
    if np.size(binary) > 1:
        dating = np.unique(np.hstack((fractile_MLE.date.unique(), binary_MLE.date.unique())))
    else:
        dating = fractile_MLE.date.unique()
    fractile = fractile.loc[np.isin(fractile.sessionDate.values, dating)].copy()
    if np.size(binary) > 1:
        binary = binary.loc[np.isin(binary.sessionDate.values, dating)].copy()
    
    fig, ax = plt.subplots( 1, 1, squeeze = False, figsize=(6,24))
    tt_fractile = fractile.getTrials(Trials)
    if np.size(binary) > 1:
        tt_binary = binary.getTrials(Trials)
    
    if np.size(binary) > 1:
        allTrials = pd.concat((tt_fractile, tt_binary))
    else:
        allTrials = tt_fractile
    allTrials = allTrials.drop_duplicates(['sessionDate','time'])
    
    for i, date in enumerate(np.sort(allTrials.sessionDate.unique())):
        rr_1 = fractile_MLE.loc[fractile_MLE.date == date].mag_range.values
        if np.size(binary) > 1:
            rr_2 = binary_MLE.loc[binary_MLE.date == date].mag_range.values
        else:
            rr_2 = []
        rr = unique_listOfLists(np.hstack((rr_1, rr_2)))[0]
        
        ff = fractile_MLE.loc[fractile_MLE.date == date]
        util = ff.iloc[-1].full_model.model_parts['utility'](np.linspace(0,1))
#        ranging = ff.mag_range.values
        
        df = allTrials.loc[allTrials.sessionDate == date]
        mA = flatten([np.array(options)[0::2] for options in df.gambleA.values])
        mB = flatten([np.array(options)[0::2] for options in df.gambleB.values])
        allMagnitudes = np.hstack((mA,mB))
        mean = np.mean(allMagnitudes, 0)
        std = np.std(allMagnitudes, 0)
        min_m = min(allMagnitudes)
        max_m = max(allMagnitudes)
        cum_mags.extend(allMagnitudes)
        
        inflection = np.linspace(rr[0], rr[1])[np.gradient((util)) == max(np.gradient((util)))]
        cum_mean = np.mean(cum_mags, 0)
        
        ax[0,0].plot(np.linspace(rr[0], rr[1]), (1-util)+i-0.5, color = 'black')
        ax[0,0].plot(np.linspace(rr[0], rr[1]), norm(np.gradient((1-util)))+i-0.5, 
          color = 'darkred', alpha = 0.4)
        ax[0,0].plot([inflection,inflection], [i-.5, i+.5], '--',color = 'darkred')
#        ax[0,0].plot([cum_mean,cum_mean], [i, i+1], '--',color = 'blue')
        ax[0,0].plot(mean, i, marker = 'o', color = 'black')
        ax[0,0].plot(rr, [i,i], '--', color = 'grey')
        ax[0,0].scatter(mean-std,   i, marker = "|", color = 'black')
        ax[0,0].scatter(mean+std, i, marker = "|", color = 'black')
        
#        print(np.linspace(rr[0], rr[1])[np.gradient((util)) == max(np.gradient((util)))] )
        if sum(rr) == 0.5:
            results[0].extend([inflection - mean])
        
        if sum(rr) == 1.0 or sum(rr) == 1.4000000000000001:
            results[1].extend([inflection - mean])
        if sum(rr) == 1.5:
            results[2].extend([inflection - mean])
            
    ax[0,0].set_ylabel('reward magnitude')
    ax[0,0].set_xlabel('testing session')
    plt.gca().invert_yaxis()
#    if np.size(binary) > 1:
#        ax[0,0].set_ylim([-0.05, 1.05])
    results = [np.array(flatten(rr)) for rr in results]
    index = flatten([[i]*len(rr) for i, rr in enumerate(results)])
    results = flatten(results)    
    
    df = pd.DataFrame(np.vstack((results, index)).T, columns = ['diff','range'])
    data = []
    for ff in np.unique(df.range.values):
        where = np.array(df.range.values) == ff
        data.append(df['diff'].values[where])
    data = np.array(data)
    
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.formula.api import glm
    from statsmodels.stats.anova import anova_lm
    
    print('\n=============================')
    print(stats.mstats.kruskalwallis(*[data[x] for x in np.arange(data.shape[0])]))
    print('\n') 
    post = sm.stats.multicomp.MultiComparison(df['diff'], df.range)
    print(post.allpairtest(stats.ranksums, method = 'holm')[0])
    [print(stats.ttest_1samp(data[x], 0)) for x in np.arange(data.shape[0])]
    
    return 

#%%
def LR_lossMetric(fractile_MLE, fractile, Trials):
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
    fig, ax = plt.subplots( 1, 3, squeeze = False, figsize=(10,6))
    palette = itertools.cycle(sb.color_palette('colorblind'))
    for i, rr in enumerate(unique_listOfLists(fractile_MLE.mag_range.values)):
        cc = next(palette)
        df = fractile_MLE.loc[fractile_MLE.mag_range.apply(lambda x: x == rr)]
        allData= []
        for date in tqdm(df.date.values , desc='gathering daily trial data'):    
                X, Y = trials_2fittable(tt.loc[tt['sessionDate'] == date], use_Outcomes=True)
                
                outcomes = X.outcomes.values[:-1]
                switch = np.abs(np.diff(Y))
                dataset = []
                for reward in np.unique(outcomes):
                    index = (outcomes == reward)
                    if sum(index) > 10:
                        dataset.append([reward, np.mean(switch[index])])
                
                dataset = np.vstack(dataset)
                ax[0,i].scatter(dataset[:,0], dataset[:,-1], alpha = 0.2, color = cc)
                allData.append(dataset)
        
        ax[0,i].axhline(0.5, color = 'k')
        allData = np.vstack(allData); datapoints = []
        for reward in np.unique(allData[:,0]):
            index = (allData[:,0] == reward)
            datapoints.append([reward, np.mean(allData[:,1][index]), sem(allData[:,1][index])])
        datapoints = np.vstack(datapoints)
#        ax[0,i].plot(datapoints[:,0], datapoints[:,1], color = cc)
        ax[0,i].set_ylabel('proportion of side switches (0 no switch)')
        ax[0,i].set_xlabel('past outcome EV')
        ax[0,i].set_ylim([0,1])
        squarePlot(ax[0,i])
        
        mod = sm.OLS(allData[:,1], sm.add_constant(allData[:,0])).fit()
        print('Range: ', rr, '===================================')
        print(mod.summary())
        ax[0,i].plot(np.linspace(min(allData[:,0]), max(allData[:,0])),
                  (np.linspace(min(allData[:,0]), max(allData[:,0])) * mod.params[-1]) + mod.params[0] ,'--', color = cc )
        
    plt.tight_layout()
    
    #%%
    fig, ax = plt.subplots( 1, 3, squeeze = False, figsize=(10,6))
    palette = itertools.cycle(sb.color_palette('colorblind'))
    for i, rr in enumerate(unique_listOfLists(fractile_MLE.mag_range.values)):
        cc = next(palette)
        df = fractile_MLE.loc[fractile_MLE.mag_range.apply(lambda x: x == rr)]
        allData= []
        
        for date in tqdm(df.date.values , desc='gathering daily trial data'):    
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
                        dataset.append([reward, np.mean(switch[index])])
                
                dataset = np.vstack(dataset)
                ax[0,i].scatter(dataset[:,0], dataset[:,-1], alpha = 0.2, color = cc)
                allData.append(dataset)
        
        ax[0,i].axhline(0.5, color = 'k')
        allData = np.vstack(allData); datapoints = []
        for reward in np.unique(allData[:,0]):
            index = (allData[:,0] == reward)
            datapoints.append([reward, np.mean(allData[:,1][index]), sem(allData[:,1][index])])
        datapoints = np.vstack(datapoints)
        
        ax[0,i].set_ylabel('proportion of gamble/safe switches (0 no switch)')
        ax[0,i].set_xlabel('past outcome EV')
        ax[0,i].set_ylim([0,1])
        squarePlot(ax[0,i])
        
        mod = sm.OLS(allData[:,1], sm.add_constant(allData[:,0])).fit()
        print('Range: ', rr, '===================================')
        print(mod.summary())
        ax[0,i].plot(np.linspace(min(allData[:,0]), max(allData[:,0])),
                  (np.linspace(min(allData[:,0]), max(allData[:,0])) * mod.params[-1]) + mod.params[0] ,'--', color = cc )
    plt.tight_layout()
        

#%%
def correlate_dailyParameters(fractile_MLE, revertLast = False):
    '''
    '''
    from macaque.f_models import define_model
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import itertools
    import statsmodels.api as sm
    
#    fractile_MLE
    
    mle, legend, ranges, c_specific = extract_parameters(fractile_MLE, dataType = 'mle', minTrials = 40, revertLast = revertLast)
    behaviour, legend, ranges, c_specific = extract_parameters(fractile_MLE, dataType = 'behaviour', minTrials = 40, revertLast = revertLast)
    mle = [parameters[:,2:] for parameters in mle]

    palette = itertools.cycle(sb.color_palette('colorblind'))
    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(10,6))
    for mm, bb, rr, color in zip(mle, behaviour, ranges, c_specific):
        mm = np.log(mm); bb = np.log(bb)
        print(' REGRESSION Range: ', rr, '  ============================================= ' )
        ax[0,0].scatter(bb[:,0], mm[:,0], color = color)
        x = bb[:,0]; y = mm[:,0]
        x = sm.add_constant(x, prepend=True)
        mod = sm.OLS(y, x).fit()
        print(' parameter temperature: --------------- ' )
        print('slope: ', mod.params[-1], '; p-val: ', mod.pvalues[-1], '; R^2: ' ,mod.rsquared)
        # -------------------------------------------------------------------------------------
        ax[0,1].scatter(bb[:,-1], mm[:,-1], color = color)
        x = bb[:,-1]; y = mm[:,-1]
        x = sm.add_constant(x, prepend=True)
        mod = sm.OLS(y, x).fit()
        print(' parameter height: --------------- ' )
        print('slope: ', mod.params[-1], '; p-val: ', mod.pvalues[-1], '; R^2: ' ,mod.rsquared)
        # -------------------------------------------------------------------------------------
        
    bb_mags = np.log(np.vstack(behaviour))
    mm_mags = np.log(np.vstack(mle))
    x = bb_mags[:,0]; y = mm_mags[:,0]
    x = sm.add_constant(x, prepend=True)
    mod = sm.OLS(y, x).fit()
    print('General correlation of temperature: ======================================================')
    print(mod.summary())
    x = bb_mags[:,1]; y = mm_mags[:,1]
    x = sm.add_constant(x, prepend=True)
    mod = sm.OLS(y, x).fit()
    print('General correlation of height: ======================================================')
    print(mod.summary())
    
    ax[0,0].plot(np.linspace(min(bb_mags[:,0]), max(bb_mags[:,0])), np.linspace(min(bb_mags[:,0]),max(bb_mags[:,0])), '--', color = 'k')
    ax[0,1].plot(np.linspace(min(bb_mags[:,-1]), max(bb_mags[:,-1])), np.linspace(min(bb_mags[:,-1]),max(bb_mags[:,-1])), '--', color = 'k')
    ax[0,0].grid(); ax[0,1].grid()
    ax[0,0].set_ylabel('temperature MLE'); ax[0,0].set_xlabel('temperature Fractile')
    ax[0,1].set_ylabel('height MLE'); ax[0,1].set_xlabel('height Fractile')
    squarePlot(ax[0,0]); squarePlot(ax[0,1])
    
    #%%
    ff = define_model(fractile_MLE.model_used.iloc[-1])
    p0 = ff[1]
    utility = lambda pp: ff[-1](p0)['empty functions']['utility'](np.linspace(0,1,100), pp)
    all_bbs = []; all_mms = []
    
    palette = itertools.cycle(sb.color_palette('colorblind'))
    fig, ax = plt.subplots( 1, 1, squeeze = False, figsize=(6,6))
    for mm, bb, rr, color in zip(mle, behaviour, ranges, c_specific):
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
def quantifyAdaptation(MLE_df, dataType='behaviour', revertLast = False):
    '''
    '''
    from scipy import stats
    from macaque.f_models import define_model
    dataList, legend, ranges, c_specific = extract_parameters(MLE_df, dataType = dataType, minTrials = 40, revertLast = revertLast)
    if dataType == 'mle':
        dataList = [ff[:,2:4] for ff in dataList]
        
    #%%
    from macaque.f_uncertainty import bootstrap_sample, bootstrap_function
    import statsmodels.api as sm

    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(10,6))
   
    
    ff = define_model(MLE_df.model_used.iloc[-1])
    p0 = ff[1]
    utility = ff[-1](p0)['empty functions']['utility']
    i = 0; past_integral= 0 
    print('-------------------------------------------------')
    for params, rr, color in zip(dataList, ranges, c_specific):
        uu = lambda pp: utility(np.linspace(0,1,100), pp)
        mean, lower, upper = bootstrap_function(uu, params, 'median')
        print(rr, '/ median = ', np.median(params, 0), ' / CI = ', bootstrap_sample(params, 'median'))
        ax[0,0].plot(np.linspace(0,1,100), mean, color = color)
        ax[0,0].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25, color = color)
        integral = np.sum(mean)    
        quantification = (integral - past_integral) / past_integral
        if past_integral != 0:
            print('relative percent change: ', quantification, 'for range ', rr)
#            dailyDifferences = [np.sum(uu(pp)) - past_integral for pp in params]
#            ax2[0,i].plot(dailyDifferences)
            
#            dailyDifferences = sm.add_constant(dailyDifferences)
#            rlm_model = sm.RLM(np.arange(0, len(dailyDifferences)), dailyDifferences).fit()
#            print(rlm_model.params, rlm_model.pvalues)
#            
            i +=1
        past_integral = integral
        
    ax[0,0].set_xlabel('reward magnitude')
    ax[0,0].set_ylabel('median utility')
    ax[0,0].legend(legend)
    ax[0,0].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,1])
    
    plt.suptitle('side-by-side utilities')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    #%%
    fig2, ax2 = plt.subplots( 1, 3, squeeze = False, figsize=(10,6))
    n=0; print('\n')
    for params, rr in zip(dataList, ranges):
        uu = lambda pp: utility(np.linspace(0,1,100), pp)
        past = 0; y = []
        for i,pp in enumerate(params):
            if past != 0 :
                y.extend([(np.sum(uu(pp)) - past) / past])
            past = np.sum(uu(pp))
            
#        x = np.arange(0, len(y))
#        x = sm.add_constant(x)
#        rlm_model = sm.OLS(y, x).fit()
        t,p = stats.ttest_1samp(y, 0)
        print('adaptation within: ', rr, ' t: ', t, ' p: ', p, ' mean: ', np.mean(y))
        ax2[0,n].plot(y)
        ax2[0,n].axhline(0, color = 'k')
        squarePlot(ax2[0,n])
        n += 1
        
    #%%
    print('\n')
    reference= ranges
    
    index = np.arange(0, len(ranges))[flatten(np.diff(ranges) == 1)]
    if np.size(index) == 0:
        index = [0]
    
    uu = lambda pp: utility(np.linspace(0,1,200), pp)
    mean_large_200, _, _ = bootstrap_function(uu, dataList[index[0]], 'median')
#    mean_large_200 = (mean_large_200  * (np.max(ranges[index]) - np.min(ranges[index]))) + np.min(ranges[index])
#    mean = mean/1.3
    top2 = mean_large_200[-1 + int(((min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:])) / (max(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:]))*200))]
    bottom2 = mean_large_200[0]
    
    uu = lambda pp: utility(np.linspace(0,1,100), pp)
    mean_large_100, _, _ = bootstrap_function(uu, dataList[index[0]], 'median')
#    mean_large_100 = (mean_large_100  * (np.max(ranges[index]) - np.min(ranges[index]))) + np.min(ranges[index])
    
    spreads = np.diff(np.sort(unique_listOfLists(ranges), 0))
    means = np.mean(np.sort(unique_listOfLists(ranges), 0), 1)
    
    for params, rr, color in zip(dataList, ranges, c_specific):
        if np.diff(rr)==max(spreads): #if the parameters are from the full range
            uu = lambda pp: utility(np.linspace(0,1,100), pp)
            mean, lower, upper = bootstrap_function(uu, params, 'median')
            ax[0,1].plot(np.linspace(rr[0],rr[1],100), mean, color=color)
            ax[0,1].fill_between(np.linspace(rr[0],rr[1],100), y1=lower, y2=upper, alpha=0.25, color=color)
        
        elif np.diff(rr)<max(spreads) and np.mean(rr) == min(means):
        # if the parameters are from the low range
            if rr[0] != reference[index].min():
                # for ugo -> if the 0 of full rnage doesnt match low range
                uu = lambda pp: utility(np.linspace(0,1,100), pp)
                mean, lower, upper = bootstrap_function(uu, params, 'median')
                bottom2 = -mean[int(((min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:]) - min(rr)) / (max(rr) - min(rr))*100))]
#                bottom2 = bottom2/1.3
                bottom2 = (bottom2*(top2-0))+0
                ax[0,1].plot(np.linspace(rr[0],rr[1],100), (mean*(top2-bottom2))+bottom2, color=color)
                ax[0,1].fill_between(np.linspace(rr[0],rr[1],100), y1=(lower*(top2-bottom2))+bottom2, y2=(upper*(top2-bottom2))+bottom2, alpha=0.25, color=color)
                
                integral = sum((mean*(top2-bottom2))+bottom2)
                no_adaptation = sum(mean_large_200[:int(((min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:])) / (max(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:]))*200))])
                full_adaptation = sum((mean_large_100*(top2-bottom2))+bottom2) 
                
                max_area = full_adaptation - no_adaptation
                adaptation_percentage = (integral - no_adaptation) / max_area
                ax[0,1].plot(np.linspace(rr[0],rr[1],100), (mean_large_100*(top2-bottom2))+bottom2, '--', color=color)
                
            else:
                uu = lambda pp: utility(np.linspace(0,1,100), pp)
                mean, lower, upper = bootstrap_function(uu, params, 'median')
                
                integral = sum((mean*(top2-bottom2))+bottom2)
                no_adaptation = sum(mean_large_200[:int(((min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:])) / (max(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:]))*200))])
                full_adaptation = sum((mean_large_100*(top2-bottom2))+bottom2) 
                
                max_area = full_adaptation - no_adaptation
                adaptation_percentage = (integral - no_adaptation) / max_area
                print('lower range adaptation percentage: ', adaptation_percentage)
#                print('lower GAC bound: ', adaptation_percentage)
#                  print('upper GAC bound: ', adaptation_percentage)
                
                ax[0,1].plot(np.linspace(rr[0],rr[1],100), (mean*(top2-bottom2))+bottom2, color=color)
                ax[0,1].fill_between(np.linspace(rr[0],rr[1],100), y1=(lower*(top2-bottom2))+bottom2, y2=(upper*(top2-bottom2))+bottom2, alpha=0.25, color=color)   
                ax[0,1].plot(np.linspace(rr[0],rr[1],100), (mean_large_100*(top2-bottom2))+bottom2, '--', color=color)
                
        elif np.diff(rr)<max(spreads) and np.mean(rr) == max(means):
            # if the parameters are from the high range
            uu = lambda pp: utility(np.linspace(0,1,100), pp)
            mean, lower, upper = bootstrap_function(uu, params, 'median')
            ax[0,1].plot(np.linspace(rr[0],rr[1],100), (mean*(1.0-top2))+top2, color=color)
            ax[0,1].fill_between(np.linspace(rr[0],rr[1],100), y1=(lower*(1.0-top2))+top2, y2=(upper*(1.0-top2))+top2, alpha=0.25, color=color)
            ax[0,1].plot(np.linspace(rr[0],rr[1],100),(mean_large_100*(1.0-top2))+top2, '--', color=color)
                
            integral = sum((mean*(1.0-top2))+top2)
            no_adaptation = sum(mean_large_200[int(((min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:])) / (max(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:]))*200)):])
            full_adaptation = sum((mean_large_100*(1.0-top2))+top2)
            
            max_area = full_adaptation - no_adaptation
            adaptation_percentage = (integral - no_adaptation) / max_area
            print('higher range adaptation percentage: ', adaptation_percentage)
    
    ax[0,1].set_xlabel('reward magnitude')
    ax[0,1].set_ylabel('median utility')
    ax[0,1].legend(legend)
    ax[0,1].plot(np.linspace(0,reference[index].max(),100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,1])
    plt.suptitle('overlapping utilities')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


#%%
def compare_reactionTimes(fractile, fractile_MLE, Trials, revertLast = False):
    '''
    '''
    import seaborn as sns
    import scipy.optimize as opt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    fig, ax = plt.subplots( 1, 3, squeeze = False,  figsize=( 6 ,3 ))
    normalize = mcolors.Normalize(vmin=-.8, vmax=0.8)
    colormap = cm.winter
    
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    
    f_data, legend, ranking1, c_specific = extract_parameters(fractile_MLE, dataType = 'behaviour', minTrials = 40, revertLast = revertLast)        
    
    Xs, Ys, Zs, ranges = [[],[],[],[]]
    for i, rr in enumerate(ranking1):
        df = fractile.loc[fractile.reward_range.apply(lambda x: all(x==rr))]
        temp = df.choiceTimes.values
        utilities = df.utility.values
        EVs = df.primaryEV.values
        secondaries = np.array([list(cc.keys()) for cc in temp])
        times = np.array([list(cc.values()) for cc in temp])
        
        for uu in np.unique(utilities):
            where = (utilities == uu)
            ev = EVs[where]
            ss = secondaries[where]
            tt = flatten([[np.mean(pp) for pp in points] for points in times[where]])
            
            z = flatten([m-np.array(n) for m,n in zip(ev, ss)])
            y = tt
            x = [uu] * len(tt)
            
            if uu == 0.5:
                scalarmappaple.set_array(z)
                plt.colorbar(scalarmappaple, ax=ax[0,i], fraction=0.046, pad=0.04, label='gEV - sEV')
            
            ax[0,i].scatter(x-(np.array(z)/7), y, c=z, cmap='winter', alpha=0.2)
            ax[0,i].grid()
        #    ax[0,1].axhline(min(flatten(times)), color='k', linestyle = '--')
            ax[0,i].set_xlabel('reward utility')
            ax[0,i].set_ylabel('response times')
            Xs.extend(x); Ys.extend(y); Zs.extend(z); ranges.extend([i]*len(x))
        
        ax[0,i].set_title(rr)
        squarePlot(ax[0,i])
    plt.tight_layout()

    variables = np.vstack((Ys, Xs, np.abs(Zs), ranges, np.ones(len(Ys)))).T
    df = pd.DataFrame(variables, columns = ['RTs', 'utility_level', 'delta_EV', 'range', 'constant'])
 
    plt.figure()
    sb.boxplot(x='range', y='RTs', hue='utility_level', data=df, palette='winter', showfliers=False).set( xticklabels=ranking1)
    plt.grid()
    
    plt.figure()
    g = sb.lmplot(x='delta_EV', y='RTs', hue='range', data=df, scatter_kws={'alpha':0.2})

    #%%
    import statsmodels.api as sm
    from statsmodels.formula.api import glm
    from statsmodels.stats.anova import anova_lm
    
    from statsmodels import graphics

    g = sb.pairplot(df, hue='range', size=2.5, plot_kws=dict(s=80, edgecolor="white", linewidth=2.5, alpha=0.3))
    for t, l in zip(g._legend.texts, legend): t.set_text(l)

#    plt.legend(legend)
    fig = plt.gcf()
    fig.tight_layout()
    
    formula = 'RTs ~ delta_EV + utility_level + C(range) + delta_EV:utility_level + C(range):utility_level'
    glm_model = glm(formula=formula, data=df, family=sm.families.Gamma(link = sm.genmod.families.links.identity)).fit()
#    glm_model = ols(formula=formula, data=df,  family='Gamma').fit()
#    glm_model = sm.GLM(endog=df.RTs, exog=df[['constant','utility_level','delta_EV','range']], family=sm.families.Poisson()).fit() 
    print(glm_model.summary2())
    graphics.gofplots.qqplot(glm_model.resid_response, line='r')
    print('\n ----------------------------------------------',
          '\n post-hoc (bonferroni) on range term:')
    print(glm_model.t_test_pairwise('C(range)', method='bonferroni').result_frame[['coef','pvalue-bonferroni','reject-bonferroni']])

#    sm.graphics.plot_ccpr_grid(glm_model)
#    ff = plt.gcf()
#    ff.tight_layout()
#    sm.graphics.plot_partregress_grid(glm_model)
    
#    sns.pairplot(df, x_vars=['utility_level', 'delta_EV', 'range'], y_vars='RTs', size=7, aspect=0.7, kind='reg')

#%%
def compare_inflections(fractile_MLE, revertLast = False,  dataType = 'behaviour'):
    '''
    '''
    from scipy import stats
    import scipy.optimize as opt
    import itertools
    import statsmodels.api as sm
    from macaque.f_Rfunctions import dv2_manova
    
    def plotting_iqr(ff):
        med = np.median(ff, 0)
        err = med - np.percentile(ff, [25, 75], 0)
#        upper_err = np.percentile(ff, 75, 0) - med 
        return np.abs(err)
    
    f_data, legend, ranking1, c_specific = extract_parameters(fractile_MLE, dataType = dataType, minTrials = 40, revertLast = revertLast)        
    palette = itertools.cycle(sb.color_palette('colorblind'))
    
    if dataType == 'mle':
        f_data = [ff[:,2:4] for ff in f_data]
    
    fig, ax = plt.subplots( 1, 3, squeeze = False, figsize=(12,6))
    gap = 0
    
    model_parts = fractile_MLE.full_model.iloc[-1]
    model = model_parts.model_parts['empty functions']['utility']
    for ff,rr, color in zip(f_data, ranking1, c_specific):
#        color = next(palette)
        ax[0,0].bar(np.array([0,1])+gap, np.mean(ff, 0), 0.2, yerr = plotting_iqr(ff), color = color)
        print('median parameters: ', np.median(ff), '   ; for range ', rr)
        print('inter quartile rangge:  ' ,  stats.iqr(ff, 0))
        gap += 0.25
     
    ax[0,0].legend(legend)
    ax[0,0].set_xticks([0.25, 1.25])
    ax[0,0].set_xticklabels(['p1','p2'])
    
    #%%
    f_data = np.vstack((np.vstack(f_data).T, np.hstack([len(ff) * [i] for i, ff in enumerate(f_data)]))).T
    manovaData = np.vstack((f_data))   
    print('\n ==============================================')
    print('2-WAY mANOVA ON UTILITY PARAMETERS (METHODS AND RANGE)')
    print('-------------------------------------------- \n')
    dv2_manova(DV1 = np.log(manovaData[:,0]), DV2= np.log(manovaData[:,1]), IV = manovaData[:,-1])
    
    #%%

    cc_ratio = []; inflection = []
    
    ranking = ranking1

    for dataset in tqdm(manovaData, desc='Gathering Inflections'):
            utility = model(np.linspace(0,1,10000), dataset[:2])
            where = np.argmax(np.gradient(utility))
            shape = np.gradient(np.gradient(utility))
            shape[shape<=0] = 0
            shape[shape>0] = 1
                       
            integral = np.sum(utility) / len(utility)
            cc_ratio.extend([integral])
#            quantification = (integral - past_integral) / past_integral
            
            inflection.extend([np.linspace(ranking[int(dataset[2])][0],ranking[int(dataset[2])][1],10000)[where]])

    inflection = np.array(inflection)
    cc_ratio = np.array(cc_ratio)
    mData = [inflection[manovaData[:,-1] == i] for i in np.unique(manovaData[:,-1])]
    index = [manovaData[manovaData[:,-1] == i, -1] for i in np.unique(manovaData[:,-1])]
      
    correlations=[]
    gap = 0;  coloration = 0
    use_range = ranking
        
    palette = itertools.cycle(sb.color_palette('colorblind'))
    past = []
    for mm, ii, rr, color in zip(mData, index, use_range, c_specific):
       
#        color = next(palette)
        dataset = mm
        if np.size(past) != 0 :
            past = (past * (rr[1] - rr[0])) + rr[0]
            ax[0,1].plot( [(ii*0.2) - 0.1, (ii*0.2) + 0.1], 
              [np.median(past), np.median(past)], '--', color='k')
            past = np.array([0 if pp < 0 else pp for pp in past])
            past = np.array([1 if pp > 1 else pp for pp in past])
            t,p = stats.ranksums(dataset, past)
#            t,p = stats.ttest_ind(dataset,past ) 
        else:
            t=np.nan; p=np.nan
        x = np.arange(len(dataset))
        x = sm.add_constant(x, prepend=True)
        mod = sm.OLS(dataset, x).fit()
        correlations.append([mod.params[-1], mod.pvalues[-1], t, p, len(dataset)])
#        ax[0,1].arrow( x = (ii*0.5) + gap, y = np.mean(dataset) / 2, 
#          dx = 0, dy = np.sign(mod.params[-1]) * np.mean(dataset) / 4, 
#          fc="k", ec="k", head_width=0.05, head_length=0.05 )
        ax[0,1].bar( (np.unique(ii)*0.2) , np.median(dataset), 0.2, color=color, alpha = 0.2)
        ax[0,1].scatter( jitter(ii*0.2, 0.02) , dataset, color=color, alpha = 0.75)
        past = (dataset   - rr[0]) / (rr[1] - rr[0])

    correlations = np.vstack((np.array(correlations).T, legend)).T
#    correlations = np.vstack((np.array(correlations).T, flatten([[ll] * len(legend) for ll in ['fractile','binary']]))).T
    print('\n =================================================')
    print('inflection slopes:')
    df = pd.DataFrame(correlations, columns = [['slope','pval', 'past_t', 'past_p', 'N', 'range']])
    print(df)
    print(df[['past_t','past_p']])
    
    ax[0,1].legend(ranking)
    ax[0,1].set_xticks([0, 0.2, 0.4])
    ax[0,1].set_xticklabels(legend)
    
    # NEED TO CHECK IF THESE ARE DIFFERENT TO NO ADAPTATION OR FULL ADAPTATION
    correlations = []
    mData = [cc_ratio[manovaData[:,-1] == i] for i in np.unique(manovaData[:,-1])]
    gap = 0;  coloration = 0
    use_range = ranking
    palette = itertools.cycle(sb.color_palette('colorblind'))
    past = []
    for mm, ii, rr, color in zip(mData, index, use_range, c_specific):
#        color = next(palette)
        dataset = mm
        if np.size(past) != 0 :
            past = (past - rr[0]) / (rr[1] - rr[0])
            ax[0,2].plot( [(ii*0.2) - 0.1, (ii*0.2) + 0.1], 
              [np.median(past), np.median(past)], '--', color='k')
            past = np.array([0 if pp < 0 else pp for pp in past])
            past = np.array([1 if pp > 1 else pp for pp in past])
            t,p = stats.ranksums(dataset, past)
            #t,p = stats.ttest_ind(dataset,past ) 
        else:
            t=np.nan; p=np.nan
        x = np.arange(len(dataset))
        x = sm.add_constant(x, prepend=True)
        mod = sm.OLS(dataset, x).fit()
        correlations.append([mod.params[-1], mod.pvalues[-1], t, p, len(dataset)])
#        ax[0,2].arrow( x = (ii*0.5), y = np.mean(dataset) / 2, 
#          dx = 0, dy = np.sign(mod.params[-1]) * np.mean(dataset) / 4, 
#          fc="k", ec="k", head_width=0.05, head_length=0.05 )
        ax[0,2].bar( (np.unique(ii)*0.2) , np.median(dataset), 0.2, color=color, alpha = 0.2)
        ax[0,2].scatter( jitter(ii*0.2, 0.02) , dataset, color=color, alpha = 0.75)
        past = (dataset  * (rr[1] - rr[0])) + rr[0]

    correlations = np.vstack((np.array(correlations).T, legend)).T
#    correlations = np.vstack((np.array(correlations).T, flatten([[ll] * len(legend) for ll in ['fractile','binary']]))).T
    print('\n =================================================')
    print('area under the curve:')
    df = pd.DataFrame(correlations, columns = [['slope','pval', 'past_t', 'past_p', 'N', 'range']])
    print(df)
    print(df[['past_t','past_p']])

    ax[0,2].legend(ranking)
    ax[0,2].set_xticks([0, 0.2, 0.4])
    ax[0,2].set_xticklabels(legend)
    
    ax[0,1].axhline(0.5, color='k', linestyle='--')
    ax[0,2].axhline(0.5, color='k', linestyle='--')
    
    ax[0,1].set_ylim(-0.05,1.05); ax[0,2].set_ylim(-0.05,1.05)
    ax[0,1].set_ylabel('inflection'); ax[0,2].set_ylabel('area under the curve')
    squarePlot(ax[0,0]); squarePlot(ax[0,1]); squarePlot(ax[0,2])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    #%%
    
    df = pd.DataFrame({'range' : manovaData[:,-1],
                       'inflection' : inflection,
                       'convexity' : cc_ratio})
    
    print('\n ==============================================')
    print('2-WAY ANOVA ON INFLECTION POINT')
    print('-------------------------------------------- \n')
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    formula = 'inflection ~  C(range)'
    model = ols(formula, data=df).fit()
    aov_table = anova_lm(model, typ=2)

    print('-------------------------------------------------')
    eta_squared(aov_table)
    #    omega_squared(aov_table)
    print(aov_table)
    print('-------------------------------------------------')
    print('Nb,Nm = \n', str(df.range.value_counts()))  
    print('\n', model.summary2())
    print('Bonferonni P is: ', str(0.05/len(model.params)))
    
    data = []
    for ff in np.unique(df.range.values):
        where = np.array(df.range.values) == ff
        data.append(np.array(df.inflection.values)[where])
    data = np.array(data)
    
    print('\n=============================')
    print(stats.mstats.kruskalwallis(*[data[x] for x in np.arange(data.shape[0])]))
    print('\n') 
    post = sm.stats.multicomp.MultiComparison(df.inflection, df.range)
    print(post.allpairtest(stats.ranksums, method = 'holm')[0])
    
    print('\n ==============================================')
    print('2-WAY ANOVA ON AREA UNDER THE CURVE')
    print('-------------------------------------------- \n')
    formula = 'convexity ~ C(range)'
    model = ols(formula, data=df).fit()
    aov_table = anova_lm(model, typ=2)

    print('-------------------------------------------------')
    eta_squared(aov_table)
    #    omega_squared(aov_table)
    print(aov_table)
    print('-------------------------------------------------')
    print('Nb,Nm = \n', str(df.range.value_counts()))  
    print('\n', model.summary2())
    print('Bonferonni P is: ', str(0.05/len(model.params)))
    
    data = []
    for ff in np.unique(df.range.values):
        where = np.array(df.range.values) == ff
        data.append(np.array(df.convexity.values)[where])
    data = np.array(data)
    
    print('\n=============================')
    print(stats.mstats.kruskalwallis(*[data[x] for x in np.arange(data.shape[0])]))
    print('\n') 
    post = sm.stats.multicomp.MultiComparison(df.convexity, df.range)
    print(post.allpairtest(stats.ranksums, method = 'holm')[0])



#%%
def plot_rangeSchema(Trials, fractile, fractile_MLE, binary = None, binary_MLE = None):
    '''
    '''
    if np.size(binary) > 1:
        dating = np.unique(np.hstack((fractile_MLE.date.unique(), binary_MLE.date.unique())))
    else:
        dating = fractile_MLE.date.unique()
    fractile = fractile.loc[np.isin(fractile.sessionDate.values, dating)].copy()
    if np.size(binary) > 1:
        binary = binary.loc[np.isin(binary.sessionDate.values, dating)].copy()
    
    fig, ax = plt.subplots( 1, 1, squeeze = False, figsize=(12,6))
    tt_fractile = fractile.getTrials(Trials)
    if np.size(binary) > 1:
        tt_binary = binary.getTrials(Trials)
    
    if np.size(binary) > 1:
        allTrials = pd.concat((tt_fractile, tt_binary))
    else:
        allTrials = tt_fractile
    allTrials = allTrials.drop_duplicates(['sessionDate','time'])
    
    for i, date in enumerate(np.sort(allTrials.sessionDate.unique())):
        rr_1 = fractile_MLE.loc[fractile_MLE.date == date].mag_range.values
        if np.size(binary) > 1:
            rr_2 = binary_MLE.loc[binary_MLE.date == date].mag_range.values
        else:
            rr_2 = []
        rr = unique_listOfLists(np.hstack((rr_1, rr_2)))[0]
        
        df = allTrials.loc[allTrials.sessionDate == date]
        mA = flatten([np.array(options)[0::2] for options in df.gambleA.values])
        mB = flatten([np.array(options)[0::2] for options in df.gambleB.values])
        allMagnitudes = np.hstack((mA,mB))
        mean = np.mean(allMagnitudes, 0)
        std = np.std(allMagnitudes, 0)
        min_m = min(allMagnitudes)
        max_m = max(allMagnitudes)
        
        ax[0,0].plot(i, mean, marker = 'o', color = 'black')
        ax[0,0].plot([i,i], rr, color = 'grey')
        ax[0,0].scatter( i, mean-std,  marker = "_", color = 'black')
        ax[0,0].scatter( i, mean+std, marker = "_", color = 'black')
        
    ax[0,0].set_ylabel('reward magnitude')
    ax[0,0].set_xlabel('testing session')
    if np.size(binary) > 1:
        ax[0,0].set_ylim([-0.05, 1.05])
    
    dates=[]
    fig, ax = plt.subplots( 1, 1, squeeze = False, figsize=(12,6))
    for i,date in enumerate(np.sort(Trials.sessionDate.unique())):
#        df = Trials.loc[Trials.sessionDate == date]
        dates.append(date)
        if np.isin(date, dating):
            df = allTrials.loc[allTrials.sessionDate == date]
            mA = flatten([np.array(options)[0::2] for options in df.gambleA.values])
            mB = flatten([np.array(options)[0::2] for options in df.gambleB.values])
            allMagnitudes = np.hstack((mA,mB))
            mean = np.mean(allMagnitudes, 0)
            std = np.std(allMagnitudes, 0)
            min_m = min(allMagnitudes)
            max_m = max(allMagnitudes)
            
            ax[0,0].errorbar(i, mean, fmt = 'o', yerr=std, color = 'black')
            ax[0,0].scatter( i, min_m,  marker = "_", color = 'red')
            ax[0,0].scatter( i, max_m, marker = "_", color = 'red')
        else:
            color = 'blue'
            ax[0,0].plot([i,i], [0,1], color = 'blue', linestyle = '--')
            
        
    ax[0,0].set_ylabel('reward magnitude')
    ax[0,0].set_xlabel('testing session')
    if np.size(binary) > 1:
        ax[0,0].set_ylim([-0.05, 1.05])
        
    print(dates)
#%%
def plot_multipleLinReg(fractile_MLE, binary_MLE):
    '''
    '''
    from macaque.f_uncertainty import bootstrap_sample, bootstrap_function
    from scipy.stats import sem
    import statsmodels.api as sm
    import itertools
    
    d2 = lambda f_x : np.gradient(np.gradient(f_x))
    d1 = lambda f_x : np.gradient(f_x)
    
    RRA = lambda x :  - ( np.gradient(np.gradient(x)) / np.gradient(x) )
    binary_MLE = binary_MLE.loc[binary_MLE.NM_success == True]
    fractile_MLE = fractile_MLE.loc[fractile_MLE.NM_success == True]

    fractile_MLE.params
    fitting = flatten(fractile_MLE['behavioural_fit'].values)
    function = lambda x, params : fractile_MLE.full_model.values[0].model_parts['empty functions']['utility'](x, params)

    dataList, legend, ranking, c_specific = extract_parameters(fractile_MLE, dataType = 'behaviour', minTrials = 40, revertLast = False)

    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(12,6))
    palette = itertools.cycle(sb.color_palette('colorblind'))
    comparison= []
    for rr, cc in zip(ranking, palette):
        df = binary_MLE.loc[binary_MLE.mag_range.apply(lambda x : all(x==rr))]
        params_range = []
        for datapoints in df.behavioural_data.values:
            xx = (datapoints[0][0] * (max(rr) - min(rr))) + min(rr)
            xFit = xx[1:-1]
            if len(xFit) >= 3:
                yFit = datapoints[0][1][1:-1]
                xFit = sm.add_constant(xFit, prepend=True)
                mod = sm.OLS(yFit,xFit).fit()
                ax[0,0].plot(np.linspace(min(xx), max(xx)),
                  (np.linspace(min(xx), max(xx)) * mod.params[-1]) + mod.params[0] , color = cc )
                ax[0,0].scatter(xx, datapoints[0][1], color = cc, alpha = 0.2)
                params_range.append(mod.params)
        ax[0,1].hist(np.vstack(params_range)[:,0], alpha = 0.3, color = cc)
        comparison.append(np.vstack(params_range))
    ax[0,0].axhline(0, color='k')

    inflection=[]
    for daily_params, rr in tqdm(zip(dataList, ranking), desc='Gathering Inflections'):
        flex = []
        for dataset in daily_params:
            utility = function(np.linspace(0,1,10000), dataset)
            where = np.argmax(np.gradient(utility))
            shape = np.gradient(np.gradient(utility))
            shape[shape<=0] = 0
            shape[shape>0] = 1
            flex.extend([np.linspace(min(rr),max(rr),10000)[where]])
        ax[0,1].hist(flex, alpha = 0.3, color = cc)
        inflection.append(flex)

    for ii, cc in zip(inflection, comparison):
        t,p = stats.ttest_ind(ii, cc[:,0])
        print('t-value: ', t)
        print('p-value: ', p)
    
   
#%%
def plot_ArrowPlatt(fractile_MLE, binary_MLE, dataType = 'behaviour', revertLast = False):
    '''
    Compute Arrow-Platt index of relative risk aversion 
    '''
    from macaque.f_uncertainty import bootstrap_sample, bootstrap_function
    from scipy.stats import sem
    import statsmodels.api as sm
    import itertools
    d2 = lambda f_x : np.gradient(np.gradient(f_x))
    d1 = lambda f_x : np.gradient(f_x)
    
    RRA = lambda x :  - ( np.gradient(np.gradient(x)) / np.gradient(x) )
    binary_MLE = binary_MLE.loc[binary_MLE.NM_success == True]
    fractile_MLE = fractile_MLE.loc[fractile_MLE.NM_success == True]

    x = [(option[0]  * (mRange[1] - mRange[0])) + mRange[0] for option, mRange in zip(flatten(binary_MLE.behavioural_data.values, 1), binary_MLE.mag_range.values)]
    mRange = np.mean(np.vstack(binary_MLE.mag_range.values), 1)
    y = [option[1] for option in flatten(binary_MLE.behavioural_data.values, 1)]
    
    fractile_MLE.params
    fitting = flatten(fractile_MLE['behavioural_fit'].values)
    function = lambda x, params : fractile_MLE.full_model.values[0].model_parts['empty functions']['utility'](x, params)

    dataList, legend, ranking, c_specific = extract_parameters(fractile_MLE, dataType = dataType, minTrials = 40, revertLast = revertLast)
    if dataType == 'mle':
        dataList = [ff[:,2:4] for ff in dataList]
        
    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(12,6))
    fig4, ax4 = plt.subplots( 1, 2, squeeze = False, figsize=(12,6))
    fig5, ax5 = plt.subplots( 1, 1, squeeze = False, figsize=(6,6))
    ax2=ax[0,0].twinx()
    ax3=ax[0,1].twinx()
#    palette = itertools.cycle(sb.color_palette('colorblind'))
    legend = []; lines = []
    all_data_x = []; all_data_y = []
    for params, rr, cc in zip(dataList, ranking, c_specific):
            df = binary_MLE.loc[binary_MLE.mag_range.apply(lambda x: all(x == rr))]
            x = np.hstack([(option[0]  * (rr[1] - rr[0])) + rr[0] for option in flatten(df.behavioural_data.values, 1)])
            y = np.hstack([option[1] for option in flatten(df.behavioural_data.values, 1)])
            mean_points, error, X = np.vstack([[np.mean(y[x==Xs]), sem(y[x==Xs]), Xs] for Xs in np.unique(x)]).T
            ax2.errorbar((X - rr[0])/(rr[1]-rr[0]), mean_points, fmt = 'o', yerr=error, markersize=10, color = cc)
            ax3.errorbar(X, mean_points, fmt = 'o', yerr=error, markersize=10, color = cc)
            
            uu = lambda pp: d2(function(np.linspace(0,1,1000), pp))
            mean, lower, upper = bootstrap_function(uu, params, 'median')
            ax[0,0].plot(np.linspace(0,1,1000), mean, '--', color = cc)
            ax[0,0].fill_between(np.linspace(0,1,1000), y1=lower, y2=upper, alpha=0.2, color = cc)
            ax[0,1].plot(np.linspace(rr[0],rr[1],1000), mean, '--', color = cc)
            ax[0,1].fill_between(np.linspace(rr[0],rr[1],1000), y1=lower, y2=upper, alpha=0.2, color = cc)
            
            uu = lambda pp: d1(function(np.linspace(0,1,1000), pp))
#            uu = lambda pp: function(np.linspace(0,1,1000), pp)
            inf = []
            for pp in params:
                mean=uu(pp)
                where = (mean == max(mean))  
                inf.extend([np.linspace(0,1,1000)[where]])
            inflection = np.median(inf)
            
#            where_min = (lower == max(lower))
#            inflection_min = np.linspace(0,1,1000)[where_min]
#            where_max = (upper == max(upper))
#            inflection_max = np.linspace(0,1,1000)[where_max]
#            
#            ax4[0,1].errorbar((X - rr[0])/(rr[1]-rr[0]), mean_points, fmt = 'o', yerr=error, markersize=10, color = cc)
            ax4[0,1].axvline((inflection * (rr[1] - rr[0])) + rr[0] , linestyle = '--', color = cc)
#            ax4[0,1].axvline(inflection_min, linestyle = '--', color = cc, alpha = 0.5)
#            ax4[0,1].axvline(inflection_max, linestyle='--', color = cc, alpha = 0.5)
            ax4[0,1].axhline(0, color='k')
            ax4[0,1].scatter( x, y, color = cc)
            
            corr_x =  np.hstack([option[0] for option in flatten(df.behavioural_data.values, 1)]) - inflection
            mod = sm.OLS(y[y != 0], corr_x[y != 0]).fit()
            print(' REGRESSION FIXED AT ZERO ======================================= ' )
            print('slope: ', mod.params[-1], '; p-val: ', mod.pvalues[-1], '; R^2: ' ,mod.rsquared)
#            print(mod.summary2())
            corr_x = sm.add_constant(corr_x, prepend=True)
            mod = sm.OLS(y[y != 0], corr_x[y != 0]).fit()
            corr_x =  np.hstack([option[0] for option in flatten(df.behavioural_data.values, 1)]) - inflection
            ax4[0,0].plot(corr_x[y != 0], y[y != 0], 'o', color = cc, alpha = 0.8, label='_nolegend_')
            ax5[0,0].plot(corr_x[y != 0], y[y != 0], 'o', color = cc, alpha = 0.55, label='_nolegend_', markersize = 9)
            [ax4[0,0].plot([xx-0.15, xx+0.15], [yy, yy], 
             color = cc, alpha = 0.05, linewidth=5,
             label='_nolegend_') for xx,yy in zip(corr_x[y != 0], y[y != 0])]
            [ax5[0,0].plot([xx-0.15, xx+0.15], [yy, yy], 
             color = cc, alpha = 0.05, linewidth=5,
             label='_nolegend_') for xx,yy in zip(corr_x[y != 0], y[y != 0])]
            ax4[0,0].plot(np.linspace(min(corr_x), max(corr_x)), (np.linspace(min(corr_x), max(corr_x)) * mod.params[-1]) + mod.params[0] , color = cc )

            all_data_x.extend( corr_x[y != 0] )
            all_data_y.extend( y[y != 0] )

            print('---------------------------------------')
            print(' REGRESSION WITH CONSTANT ------------------------------------- ' )
            print('range: ' , rr)
            print('slope: ', mod.params[-1], '; p-val: ', mod.pvalues[-1], '; R^2: ' ,mod.rsquared)
            print(mod.summary())
            legend.extend([rr])

    ax[0,0].legend(legend)
    align_yaxis(ax[0,0], 0, ax2, 0)
    align_yaxis(ax[0,1], 0, ax3, 0)
    ax[0,0].axhline(0, color='k')
    ax[0,1].axhline(0, color='k')
    plt.tight_layout()
    ax4[0,0].axvline(0, color = 'k'); ax4[0,0].axhline(0, color = 'k')
    ax4[0,0].legend(legend)
    squarePlot(ax4[0,0])
    squarePlot(ax4[0,1])
    plt.tight_layout()
    
#    corr_x = sm.add_constant(all_data_x, prepend=True)
    corr_x = sm.add_constant(all_data_x, prepend=True)
    mod = sm.OLS(all_data_y, corr_x).fit()
    print('\n ============================================= \n',
          'FIXED CORRELATION WITH ALL DATAPOINTS')
    print(mod.summary())
#    ax5[0,0].plot(np.linspace(min(all_data_x), max(all_data_x)), (np.linspace(min(all_data_x), max(all_data_x)) * mod.params[-1] + mod.params[0]), '--', color = 'k' )
    ax5[0,0].plot(np.linspace(min(all_data_x), max(all_data_x)), (np.linspace(min(all_data_x), max(all_data_x)) * mod.params[-1]) + mod.params[0] , '--', color = 'k' )
    ax5[0,0].text(-0.4, -0.3, 'R2 = ' + str(np.round(mod.rsquared, 3)))
    ax5[0,0].axhline(0, color = 'k')
    ax5[0,0].axvline(0, color = 'k')
    
#%%
def classify_parameters(fractile_MLE):
    sorting = np.argsort(np.mean(unique_listOfLists(fractile_MLE.mag_range.values), 1))
    all_params = [];     range_code = []
    for i,rr in tqdm(enumerate(np.array(unique_listOfLists(fractile_MLE.mag_range.values))[sorting])):
        data = fractile_MLE.loc[fractile_MLE.mag_range.apply(lambda x: all(x==rr))]
        dataPoints = flatten(data.behavioural_data.values, 1)
        dataPoints = np.hstack([np.vstack(points)[:,1:-1] for points in dataPoints])
        params =  np.vstack([[dd, pp] for dd, pp in zip(data.date.values, data.behavioural_fit.values) if np.size(pp) != 0])
        params = np.vstack(flatten(params[:,1:], 2)) 
        range_code.extend([i]*len(params))  
        all_params.extend(params)
        
    df = pd.DataFrame(np.vstack((np.vstack(all_params).T, np.array(range_code) )).T,
                 columns = ['parameter1', 'parameter2', 'range'])
        
    import statsmodels.formula.api as sm
    import statsmodels.api as sm
    y = df['range']
    X = df[['parameter1', 'parameter2']]
    Xc = sm.add_constant(X)
    model =sm.MNLogit(y, Xc)
    result = model.fit()
    print(result.get_margeff().summary())
    print(result.summary())
    

#%%
def extract_parameters(MLE_df, dataType = 'behaviour', minTrials = 0, revertLast = False):
    import itertools
    
    print('Using ' + dataType + ' data for the procedure')
    MLE_df = MLE_df.sort_values('date')
    MLE_df = MLE_df.loc[MLE_df.nTrials > minTrials]

    model = MLE_df.iloc[-1]
    parameters = []; ranges = []; dating = []; 
    index=[]; i=0
    if dataType.lower() == 'mle':
        for mle, dd, rr in tqdm(zip(MLE_df.params, MLE_df.date, MLE_df.mag_range), desc='Gathering Inflections'):
            ranges.extend([rr])
            dating.extend([dd])
            index.extend([i])
            parameters.extend([mle])
            i+=1
        parameters = np.vstack(parameters)
    elif dataType.lower() == 'behaviour':
        model = MLE_df.full_model.iloc[-1]
        model = model.model_parts['empty functions']['utility']
        for params, dd, rr in tqdm(zip(MLE_df.behavioural_fit, MLE_df.date, MLE_df.mag_range), desc='Gathering Inflections'):
            if np.size(params) > 2:
                for pp in params:
                    ranges.extend([rr])
                    dating.extend([dd])
                    index.extend([i])
                    parameters.extend([pp])
                i+=1
            elif params == []: 
                continue
            else:
                ranges.extend([rr])
                dating.extend([dd])
                index.extend([i])
                parameters.extend([params])
                i+=1
        parameters = np.vstack(parameters)

    index = np.array(index)
    dating = np.array(dating)
    minDate = []; dataList=[]; legend = []
    
    
    palette = itertools.cycle(sb.color_palette('colorblind'))
    color = np.array([next(palette) for i in unique_listOfLists(ranges)])
    
    if revertLast == True:
        ranking = np.array(unique_listOfLists(ranges))[np.argsort(np.sum(unique_listOfLists(ranges), 1))]
        ranking = ranking[np.argsort(np.diff(ranking, 1), 0)]
        ranking = np.array(flatten(ranking))
        color = color[[0,2,1]]
    else:
        ranking = np.sort(unique_listOfLists(ranges), 0) 
        
    for rr in ranking:
        where = [all(r1==rr) for r1 in ranges]
        dates = dating[where]
        y = parameters[where]
        x = index[where]
        dataList.append(parameters[where])
        legend.extend([str(rr)])
        
    if len(dataList) < 3:
        dataList = dataList[::-1]
        legend = legend[::-1]
        ranking = ranking[::-1]
        color = color[::-1]
        
    return dataList, legend, ranking, color

#%%
def compare_elicitationMethods(fractile_MLE, binary_MLE = None, revertLast = False):
    '''
    '''
    from scipy import stats
    import itertools
    import statsmodels.api as sm
    from macaque.f_Rfunctions import dv2_manova_2way
    
    f_data, legend, ranking1, c_specific = extract_parameters(fractile_MLE, dataType = 'behaviour', minTrials = 40, revertLast = revertLast)
    if np.size(binary_MLE) > 1:
        b_data, legend2, ranking2, c_specific = extract_parameters(binary_MLE, dataType = 'behaviour', minTrials = 40, revertLast = revertLast)
    else:
         b_data, legend2, ranking2, c_specific = extract_parameters(fractile_MLE, dataType = 'mle', minTrials = 40, revertLast = revertLast)
         b_data = [ff[:,2:4] for ff in b_data]
        
    palette = itertools.cycle(sb.color_palette('colorblind'))
    
    fig, ax = plt.subplots( 1, 3, squeeze = False, figsize=(12,6))
    gap = 0
    for ff, bb, color in zip(f_data, b_data, c_specific):
#        color = next(palette)
        ax[0,0].bar(np.array([0,2]) + gap, np.mean(ff, 0), 0.2, yerr = stats.sem(ff,0), color = color)
        ax[0,0].bar(np.array([0,2]) + gap + 0.2, np.mean(bb, 0), 0.2, yerr = stats.sem(bb,0), color = np.array(color)+0.1)
        gap += 0.5
     
    ax[0,0].legend(np.ravel(np.vstack((legend,legend2)).T))
    ax[0,0].set_xticks([0.6, 2.6])
    ax[0,0].set_xticklabels(['p1','p2'])
    
    #%%
    f_data = np.vstack((np.vstack(f_data).T, np.hstack([len(ff) * [i] for i, ff in enumerate(f_data)]))).T
    f_data = np.vstack((np.vstack(f_data).T, np.array(len(f_data) * [0]) )).T
    b_data = np.vstack((np.vstack(b_data).T, np.hstack([len(ff) * [i] for i, ff in enumerate(b_data)]))).T
    b_data = np.vstack((np.vstack(b_data).T, np.array(len(b_data) * [1]) )).T
    manovaData = np.vstack((f_data, b_data))   
    print('\n ==============================================')
    print('2-WAY mANOVA ON UTILITY PARAMETERS (METHODS AND RANGE)')
    print('-------------------------------------------- \n')
    dv2_manova_2way(DV1 = manovaData[:,0], DV2= manovaData[:,1], IV1 = manovaData[:,-1], IV2 = manovaData[:,-2])
    
    #%%
    model = fractile_MLE.full_model.iloc[-1]
    model = model.model_parts['empty functions']['utility']
    cc_ratio = []; inflection = []
    
    ranking = ranking1

    for dataset in tqdm(manovaData, desc='Gathering Inflections'):
            utility = model(np.linspace(0,1,10000), dataset[:2])
            where = np.argmax(np.gradient(utility))
            shape = np.gradient(np.gradient(utility))
            shape[shape<=0] = 0
            shape[shape>0] = 1
            cc_ratio.extend([np.mean(shape)])
            inflection.extend([np.linspace(ranking[int(dataset[2])][0],ranking[int(dataset[2])][1],10000)[where]])

    inflection = np.array(inflection)
    cc_ratio = np.array(cc_ratio)
    mData = [inflection[manovaData[:,-1] == i] for i in np.unique(manovaData[:,-1])]
    index = [manovaData[manovaData[:,-1] == i, -2] for i in np.unique(manovaData[:,-1])]
      
    correlations=[]
    gap = 0;  coloration = 0
    use_range = [ranking] * 2
        
    for mm, ii, rang in zip(mData, index, use_range):
        past = []
#        palette = itertools.cycle(sb.color_palette('colorblind'))
        for iii, rr, color in zip(np.unique(ii), rang, c_specific):
#            color = next(palette)
            dataset = mm[ii==iii]
            if np.size(past) != 0 :
                past = (past * (rr[1] - rr[0])) + rr[0]
                ax[0,1].plot( [(iii*0.5) + gap - 0.1, (iii*0.5) + gap + 0.1], 
                  [np.mean(past), np.mean(past)], '--', color='k')
                t,p = stats.ttest_ind(dataset,past ) 
            else:
                t=np.nan; p=np.nan
            x = np.arange(len(dataset))
            x = sm.add_constant(x, prepend=True)
            mod = sm.OLS(dataset, x).fit()
            correlations.append([mod.params[-1], mod.pvalues[-1], t, p, len(dataset)])
            ax[0,1].arrow( x = (iii*0.5) + gap, y = np.mean(dataset) / 2, 
              dx = 0, dy = np.sign(mod.params[-1]) * np.mean(dataset) / 4, 
              fc="k", ec="k", head_width=0.05, head_length=0.05 )
            ax[0,1].bar( (iii*0.5) + gap , np.mean(dataset), 0.2, yerr=stats.sem(dataset), color=np.array(color)+coloration)
            past = (dataset   - rr[0]) / (rr[1] - rr[0])
        gap = 0.2; coloration += 0.1

    correlations = np.vstack((np.array(correlations).T, legend + legend)).T
    correlations = np.vstack((np.array(correlations).T, flatten([[ll] * len(legend) for ll in ['fractile','binary']]))).T
    print('\n =================================================')
    print('inflection slopes:')
    df = pd.DataFrame(correlations, columns = [['slope','pval', 'past_t', 'past_p', 'N', 'range','method']])
    print(df)
    print(df[['past_t','past_p']])
    
    ax[0,1].legend(['fractile','binary'])
    ax[0,1].set_xticks([0.1, 0.6, 1.1])
    ax[0,1].set_xticklabels(legend)
    
    # NEED TO CHECK IF THESE ARE DIFFERENT TO NO ADAPTATION OR FULL ADAPTATION
    correlations = []
    mData = [cc_ratio[manovaData[:,-1] == i] for i in np.unique(manovaData[:,-1])]
    gap = 0;  coloration = 0
    use_range = [ranking] * 2
    for mm, ii, rang in zip(mData, index, use_range):
#        palette = itertools.cycle(sb.color_palette('colorblind'))
        past = []
        for iii, rr, color in zip(np.unique(ii), rang, c_specific):
#            color = next(palette)
            dataset = mm[ii==iii]
            if np.size(past) != 0 :
                past = (past - rr[0]) / (rr[1] - rr[0])
                ax[0,2].plot( [(iii*0.5) + gap - 0.1, (iii*0.5) + gap + 0.1], 
                  [np.mean(past), np.mean(past)], '--', color='k')
                t,p = stats.ttest_ind(dataset,past ) 
            else:
                t=np.nan; p=np.nan
            x = np.arange(len(dataset))
            x = sm.add_constant(x, prepend=True)
            mod = sm.OLS(dataset, x).fit()
            correlations.append([mod.params[-1], mod.pvalues[-1], t, p, len(dataset)])
            ax[0,2].arrow( x = (iii*0.5) + gap, y = np.mean(dataset) / 2, 
              dx = 0, dy = np.sign(mod.params[-1]) * np.mean(dataset) / 4, 
              fc="k", ec="k", head_width=0.05, head_length=0.05 )
            ax[0,2].bar( (iii*0.5) + gap , np.mean(dataset), 0.2, yerr=stats.sem(dataset), color=np.array(color)+coloration)
            past = (dataset  * (rr[1] - rr[0])) + rr[0]
        gap = 0.2; coloration += 0.1
        
    correlations = np.vstack((np.array(correlations).T, legend + legend)).T
    correlations = np.vstack((np.array(correlations).T, flatten([[ll] * len(legend) for ll in ['fractile','binary']]))).T
    print('\n =================================================')
    print('convexity ratio slopes:')
    df = pd.DataFrame(correlations, columns = [['slope','pval', 'past_t', 'past_p', 'N', 'range','method']])
    print(df)
    print(df[['past_t','past_p']])

    ax[0,2].legend(['fractile','binary'])
    ax[0,2].set_xticks([0.1, 0.6, 1.1])
    ax[0,2].set_xticklabels(legend)
    
    ax[0,1].axhline(0.5, color='k', linestyle='--')
    ax[0,2].axhline(0.5, color='k', linestyle='--')
    
    ax[0,1].set_ylim(0,1); ax[0,2].set_ylim(0,1)
    ax[0,1].set_ylabel('inflection'); ax[0,2].set_ylabel('convexity ratio')
    squarePlot(ax[0,0]); squarePlot(ax[0,1]); squarePlot(ax[0,2])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    #%%
    
    df = pd.DataFrame({'method' : manovaData[:,-1],
                       'range' : manovaData[:,-2],
                       'inflection' : inflection,
                       'convexity' : cc_ratio})
    
    print('\n ==============================================')
    print('2-WAY ANOVA ON INFLECTION POINT')
    print('-------------------------------------------- \n')
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    formula = 'inflection ~ C(range) + C(method) +  C(range):C(method)'
    model = ols(formula, data=df).fit()
    aov_table = anova_lm(model, typ=2)

    print('-------------------------------------------------')
    eta_squared(aov_table)
    #    omega_squared(aov_table)
    print(aov_table)
    print('-------------------------------------------------')
    print('Nb,Nm = \n', str(df.range.value_counts()))  
    print('\n', model.summary2())
    print('Bonferonni P is: ', str(0.05/len(model.params)))
    
    import statsmodels.api as sm
    
    data = []
    for ff in np.unique(df.method.values):
        where = np.array(df.method.values) == ff
        data.append(np.array(df.inflection.values)[where])
    data = np.array(data)
    
    print('\n=============================')
    print(stats.mstats.kruskalwallis(*[data[x] for x in np.arange(data.shape[0])]))
    print('\n') 
    post = sm.stats.multicomp.MultiComparison(df.inflection, df.method)
    print(post.allpairtest(stats.ranksums, method = 'holm')[0])
    
    print('\n ==============================================')
    print('2-WAY ANOVA ON CONVEXITY RATIO')
    print('-------------------------------------------- \n')
    formula = 'convexity ~ C(range) + C(method) +  C(range):C(method)'
    model = ols(formula, data=df).fit()
    aov_table = anova_lm(model, typ=2)

    print('-------------------------------------------------')
    eta_squared(aov_table)
    #    omega_squared(aov_table)
    print(aov_table)
    print('-------------------------------------------------')
    print('Nb,Nm = \n', str(df.range.value_counts()))  
    print('\n', model.summary2())
    print('Bonferonni P is: ', str(0.05/len(model.params)))
    
    import statsmodels.api as sm
    
    data = []
    for ff in np.unique(df.method.values):
        where = np.array(df.method.values) == ff
        data.append(np.array(df.convexity.values)[where])
    data = np.array(data)
    
    print('\n=============================')
    print(stats.mstats.kruskalwallis(*[data[x] for x in np.arange(data.shape[0])]))
    print('\n') 
    post = sm.stats.multicomp.MultiComparison(df.convexity, df.method)
    print(post.allpairtest(stats.ranksums, method = 'holm')[0])

    
#%%
def plot_freeRange(MLE_df, minTrials = 40):
    '''
    '''
    from scipy import stats
    from macaque.f_models import define_model

    dataList, legend, ranges, c_specific = extract_parameters(MLE_df, dataType = 'mle', minTrials = minTrials)
        
    #%%
    from macaque.f_uncertainty import bootstrap_sample, bootstrap_function
    from scipy.stats import norm
    
    fig, ax = plt.subplots( 2, 3, squeeze = False, figsize=(10,6))
    ff = define_model(MLE_df.model_used.iloc[-1])
    p0 = ff[1]
    utility = ff[-1](p0)['empty functions']['utility']
    softmax = ff[-1](p0)['empty functions']['pChooseA']
    probability = ff[-1](p0)['empty functions']['probability']
    
    full_range = np.sort(unique_listOfLists(ranges), 0)
    full_range = [np.min(full_range), np.max(full_range)]

    for params, rr in zip(dataList, np.sort(unique_listOfLists(ranges), 0)):
        uu = lambda pp: utility(np.linspace(0,1,100), pp)
        mean, lower, upper = bootstrap_function(uu, params[:,2:4], 'mean')
        ax[0,0].plot(np.linspace(full_range[0],full_range[1],100), mean)
        ax[0,0].fill_between(np.linspace(full_range[0],full_range[1],100), y1=lower, y2=upper, alpha=0.25)
        mean, lower, upper = bootstrap_function(uu,  params[:,2:4], 'median')
        ax[1,0].plot(np.linspace(full_range[0],full_range[1],100), mean)
        ax[1,0].fill_between(np.linspace(full_range[0],full_range[1],100), y1=lower, y2=upper, alpha=0.25)
        
    for params, rr in zip(dataList, np.sort(unique_listOfLists(ranges), 0)):
        sm = lambda pp: 1 / (1 + np.exp( -pp[0] * ( (np.linspace(-0.5,0.5,100) - pp[1]) )  ))
        mean, lower, upper = bootstrap_function(sm,  params[:,0], 'mean')
        ax[0,1].plot(np.linspace(-0.5,0.5,100), mean)
        ax[0,1].fill_between(np.linspace(-0.5,0.5,100), y1=lower, y2=upper, alpha=0.25)
        mean, lower, upper = bootstrap_function(sm,  params[:,0], 'median')
        ax[1,1].plot(np.linspace(-0.5,0.5,100), mean)
        ax[1,1].fill_between(np.linspace(-0.5,0.5,100), y1=lower, y2=upper, alpha=0.25)
        
    for params, rr in zip(dataList, np.sort(unique_listOfLists(ranges), 0)):
        prob = lambda pp: probability(np.linspace(0,1,100), pp)
        mean, lower, upper = bootstrap_function(prob,  params[:,-1], 'mean')
        ax[0,2].plot(np.linspace(0,1,100), mean)
        ax[0,2].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25)
        mean, lower, upper = bootstrap_function(prob,  params[:,-1], 'median')
        ax[1,2].plot(np.linspace(0,1,100), mean)
        ax[1,2].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25)

    ax[0,0].set_xlabel('reward magnitude')
    ax[0,0].set_ylabel('mean utility')
    squarePlot(ax[0,0])
    
    ax[1,0].set_xlabel('reward magnitude')
    ax[1,0].set_ylabel('median utility')
    squarePlot(ax[0,1])
    
    ax[0,1].set_xlabel('Δ value')
    ax[0,1].set_ylabel('mean pChA')
    squarePlot(ax[0,0])
    
    ax[1,1].set_xlabel('Δ value')
    ax[1,1].set_ylabel('median pChA')
    squarePlot(ax[0,1])
    
    ax[0,2].set_xlabel('reward probability')
    ax[0,2].set_ylabel('mean probability distortion')
    ax[0,2].legend(legend)
    ax[0,2].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,0])
    
    ax[1,2].set_xlabel('reward probability')
    ax[1,2].set_ylabel('median probability distortion')
    ax[1,2].legend(legend)
    ax[1,2].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,1])
    
    plt.suptitle('fitted functions')
    
    #%%

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((1, 3), (0, 2))
    gap = -0.9
    ax1=[]
    ax1.extend([ax]); ax1.extend([ax.twinx()]); ax1.extend([ax.twinx()]); ax1.extend([ax.twinx()])
    order =  [1,2,0,3]
        
    for sequenceType, color in zip(dataList, ['blue', 'green', 'red']):
        i = 1
        for param in order:
            if param == 1:
                bp = ax1[param].boxplot(np.log(sequenceType[:,param]), positions= [i *4.0+gap], sym='', widths=0.4)
            else:
                bp = ax1[param].boxplot(np.log(sequenceType[:,param]), positions= [i *4.0+gap], sym='', widths=0.4)
            set_box_color(bp, 'black') 

            risklessX = [i] * len(sequenceType)
            risklessX = np.random.normal(risklessX, 0.015, size=len(risklessX))
            if param == 1:
                ax1[param].scatter(risklessX*4.0+gap, np.log(np.ravel(sequenceType[:,param])), c=color, label='riskless', alpha=0.2)
            else:
                ax1[param].scatter(risklessX*4.0+gap, np.log(np.ravel(sequenceType[:,param])), c=color, label='riskless', alpha=0.2)

            if gap == -0.9:
                ax1[param].yaxis.set_ticks_position('left')
                ax1[param].spines['left'].set_position(('data', i*4-2))
            i+=1
        gap += 0.6

    ax.axhline(0, alpha=0.3, color='k')
    ax.set_xticks( np.arange(1, np.size(np.vstack(dataList),1)+1) *4 )
    ax.set_xticklabels( np.hstack((MLE_df.pNames.iloc[0][2:4], MLE_df.pNames.iloc[0][0],  MLE_df.pNames.iloc[0][3] )) )
    ax.set_xbound(2.5,17.5)
    fig.suptitle('ranges parameter comparisons')
    
    #%%
    for sequenceType, color in zip(dataList, ['blue', 'green', 'red']):
        ax2.scatter(np.log(sequenceType[:,1]), np.log(sequenceType[:,2]), color=color)
            
    ax2.set_xlabel(MLE_df.pNames.iloc[0][0][1])
    ax2.set_ylabel(MLE_df.pNames.iloc[0][0][2])
    ax2.grid()
    ax2.set_xbound(-5, 5)
    x0, x1 = ax2.get_xlim()
    y0, y1 = ax2.get_ylim()
    ax2.set_aspect((x1 - x0) / (y1 - y0))
    ax2.legend(legend)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    #%%
    from macaque.f_Rfunctions import dv4_manova
    IV = np.hstack([[i] * len(data) for i,data in enumerate(dataList)]).T
    dataList = np.vstack(dataList)
    dataList = np.vstack((dataList.T, IV)).T
    dv4_manova( dataList[:,1],  dataList[:,2],  dataList[:,0], dataList[:,3],  IV=IV)
    
#%%
def compare_timeHalves(MLE_df, dataType = 'behaviour'):
    '''
    '''
    from macaque.f_uncertainty import bootstrap_sample, bootstrap_function
    from scipy import stats
    range_code = []; equivalent = []; utility = []
    halve = []; manovaData = []
    
    sorting = np.argsort(np.mean(unique_listOfLists(MLE_df.mag_range.values), 1))
    fig, ax = plt.subplots(3,len(unique_listOfLists(MLE_df.mag_range.values)), squeeze = False, figsize=(12,8))
    
    for i,rr in tqdm(enumerate(np.array(unique_listOfLists(MLE_df.mag_range.values))[sorting])):
        if i == 0:
            print('\n')
        print(str(i) + ' is range: ' + str(rr))
        data = MLE_df.loc[MLE_df.mag_range.apply(lambda x: all(x==rr))]
        model = data.full_model.iloc[-1]
        model = model.model_parts['empty functions']['utility']
              
        firstHalf = data.iloc[:int(len(data)/2)]
        secondHalf = data.iloc[int(len(data)/2):]
        
        dataPoints_1 = flatten(firstHalf.behavioural_data.values, 1)
        dataPoints_1 = np.hstack([np.vstack(points)[:,1:-1] for points in dataPoints_1])
        dataPoints_2 = flatten(secondHalf.behavioural_data.values, 1)
        dataPoints_2= np.hstack([np.vstack(points)[:,1:-1] for points in dataPoints_2])
        half_index = np.array(flatten([[1]*len(dataPoints_1[0]), [2]*len(dataPoints_2[0])]))
        
        utility.extend(flatten([dataPoints_1[1],dataPoints_2[1]]) )
        equivalent.extend(flatten([dataPoints_1[0],dataPoints_2[0]]))
        halve.extend(half_index)
        range_code.extend([i]*(len(dataPoints_1[1]) + len(dataPoints_2[1])))
        legend = ['1st half','2nd half']
        
        ax[0,i].set_title('range: ' + str(rr))
        ax[0,i].scatter(dataPoints_1[0],dataPoints_1[1], alpha=0.6)
        ax[0,i].scatter(dataPoints_2[0],dataPoints_2[1], alpha=0.6)
        
        ax[0,i].set_ylabel('utility')
        ax[0,i].set_xlabel('normalized certainty equivalent')
        ax[0,i].legend(legend)
        ax[0,i].plot(np.linspace(0,1), np.linspace(0,1), '--', color='k')
        squarePlot(ax[0,i])
        
        if dataType.lower() == 'behaviour':
            width = 0.35
            fh = np.vstack(firstHalf.behavioural_fit.values[firstHalf.behavioural_fit.values.nonzero()])
            sh = np.vstack(secondHalf.behavioural_fit.values[secondHalf.behavioural_fit.values.nonzero()])
            
            p1 = np.mean(fh, 0); s1 = stats.sem(fh,0)
            ax[1,i].bar(np.arange(len(fh.T))-width/2, p1, yerr=s1, width=width)
            p2 = np.mean(sh, 0); s2 = stats.sem(sh,0)
            ax[1,i].bar(np.arange(len(sh.T))+width/2, p2, yerr=s2, width=width)
            ax[1,i].set_xticks(np.arange(len(sh.T)))
            ax[1,i].set_xticklabels(['p1', 'p2'])
            
            uu = lambda pp: model(np.linspace(0,1,100), pp)
            mean, lower, upper = bootstrap_function(uu, fh, 'mean')
            ax[2,i].plot(np.linspace(rr[0],rr[1],100), mean)
            ax[2,i].fill_between(np.linspace(rr[0],rr[1],100), y1=lower, y2=upper, alpha=0.25)
            
            uu = lambda pp: model(np.linspace(0,1,100), pp)
            mean, lower, upper = bootstrap_function(uu, sh, 'mean')
            ax[2,i].plot(np.linspace(rr[0],rr[1],100), mean)
            ax[2,i].fill_between(np.linspace(rr[0],rr[1],100), y1=lower, y2=upper, alpha=0.25)
            
            ax[2,i].legend(legend)
            squarePlot(ax[2,i])
            ax[2,i].set_ylabel('utility')
            ax[2,i].set_xlabel('reward magnitude')
            ax[2,i].plot(np.linspace(rr[0],rr[1]), np.linspace(0,1), '--', color='k')
            
        elif dataType.lower() == 'mle':
            fh = np.vstack(firstHalf.behavioural_fit.values)
            sh = np.vstack(secondHalf.behavioural_fit.values)
            
        fit_data = np.vstack((fh, sh))
        fit_data = np.vstack((fit_data.T, np.array([1]*len(fh) + [2]*len(sh)))).T
        fit_data = np.vstack((fit_data.T, [i]*len(fit_data))).T
        manovaData.append(fit_data)
    
    df = pd.DataFrame(dict({'utility' : utility,
                       'CEs' : equivalent,
                       'range' : range_code,
                       'half' : halve}))
    
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    formula = 'CEs ~ utility + C(range) + C(half) + C(range):C(half) +  C(range):utility'
    model = ols(formula, data=df).fit()
    aov_table = anova_lm(model, typ=2)

    print('-------------------------------------------------')
    eta_squared(aov_table)
    #    omega_squared(aov_table)
    print(aov_table)
    print('-------------------------------------------------')
    print('Nb,Nm = \n', str(df.range.value_counts()))  
    print('\n', model.summary2())
    
    print('Bonferonni P is: ', str(0.05/len(model.params)))
    
    # -----------------------------------------
    
    print('\n')
    print('--------------------------- Manova Result ---------------------------------------')
    from macaque.f_Rfunctions import dv4_manova_2way, dv2_manova_2way
    manovaData = np.vstack(manovaData)
    if dataType.lower() == 'behaviour':
        dv2_manova_2way(DV1 = manovaData[:,0], DV2= manovaData[:,1], IV1 = manovaData[:,-2], IV2 = manovaData[:,-1])
    elif dataType.lower() == 'mle':
        dv4_manova_2way(DV1 = manovaData[:,0], DV2= manovaData[:,1], 
                        DV3 = manovaData[:,2], DV4= manovaData[:,3],
                        IV1 = manovaData[:,-2], IV2 = manovaData[:,-1])
        
#%%
def compare_utilityLevels(fractile_MLE):
    '''
    '''
    from scipy import stats
    range_code = []
    equivalent = []
    utility = []
    legend = []
    
    fig, ax = plt.subplots(1,3, squeeze = False, figsize=(12,4))
    fig2, ax2 = plt.subplots(1,1, squeeze = False, figsize=(12,4))
    sorting = np.argsort(np.mean(unique_listOfLists(fractile_MLE.mag_range.values), 1))
    gap = 0.1
    for i,rr in tqdm(enumerate(np.array(unique_listOfLists(fractile_MLE.mag_range.values))[sorting])):
        if i == 0:
            print('\n')
        print(str(i) + ' is range: ' + str(rr))
        data = fractile_MLE.loc[fractile_MLE.mag_range.apply(lambda x: all(x==rr))]
        
        dataPoints = flatten(data.behavioural_data.values, 1)
        dataPoints = np.hstack([np.vstack(points) for points in dataPoints])
        
        utility.extend(dataPoints[1])
        equivalent.extend(dataPoints[0])
        range_code.extend([i]*len(dataPoints[1]))
        legend.extend([str(rr)])
        
        ax[0,0].scatter(dataPoints[0],np.array(dataPoints[1])-i/75, alpha=0.6)
        
        y = dataPoints[0]
        x = np.array(dataPoints[1])-i/75
        mean, error, X = np.vstack([[np.mean(y[x==Xs]), stats.sem(y[x==Xs]), Xs] for Xs in np.unique(x)]).T
        ax2[0,0].errorbar(mean,X, alpha=0.6, xerr=error, fmt = 'o')
    
        params =  np.vstack([[dd, pp] for dd, pp in zip(data.date.values, data.behavioural_fit.values) if np.size(pp) != 0])
        time = params[:,0]
        params = np.vstack(flatten(params[:,1:], 2)) 
        
        ax[0,1].scatter(np.log(params[:,0]), np.log(params[:,1]))
        ax[0,2].bar(np.arange(np.size(params,1)) + gap, np.mean(params,0), width = 0.2, yerr = stats.sem(params,0))
        gap += 0.2
        
    ax2[0,0].plot(np.linspace(0,1), np.linspace(0,1), '--', color = 'k')
#    ax2[0,0].set_ylim(-0.05,1.05)
#    ax2[0,0].set_xlim(-0.05,1.05)
#    ax[0,0].set_xlim(-0.05,1.05)
#    ax[0,0].set_ylim(-0.05,1.05)
    squarePlot(ax2[0,0])
    df = pd.DataFrame(dict({'utility' : utility,
                       'CEs' : equivalent,
                       'range' : range_code}))
    
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    formula = 'CEs ~ C(utility) + C(range) + C(range):C(utility)'
    model = ols(formula, data=df).fit()
    print(f"Overall model F({model.df_model: .0f}, {model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4}")

    aov_table = anova_lm(model, typ=2)

    print('-------------------------------------------------')
    eta_squared(aov_table)
    #    omega_squared(aov_table)
    print(aov_table)
    print('-------------------------------------------------')
    print('Nb,Nm = \n', str(df.range.value_counts()))  
    print('\n', model.summary2())
    print('Bonferonni P is: ', str(0.05/len(model.params)))
    
    import statsmodels.api as sm
    post = sm.stats.multicomp.MultiComparison(df.CEs, df.range)
    print(post.allpairtest(stats.ttest_ind, method = 'holm')[0])


    ax[0,0].set_ylabel('utility')
    ax[0,0].set_xlabel('normalized certainty equivalent')
    ax[0,0].legend(legend)
    ax[0,0].plot(np.linspace(0,1), np.linspace(0,1), '--', color='k')
    ax[0,1].axvline(0,color='k')
    ax[0,1].axhline(0,color='k')
    ax[0,1].set_ylabel('log p2'); ax[0,1].set_xlabel('log p1')
    
    ax[0,2].set_xticks([0.3, 1.3]); ax[0,2].set_xticklabels(['p1','p2'])
    
    squarePlot(ax[0,0])
    squarePlot(ax[0,2])
    squarePlot(ax[0,1])
    
#%%
def plot_schema(params):
    '''
    '''
    def sCDF(mm ):
        Xs = np.atleast_1d(mm) #range position
        inflection, temp = params[1], params[0]
        return np.where((inflection > 1 or inflection < 0),
                        [0] * len(Xs),
                        np.ravel([np.where(X<inflection, inflection*((X/inflection)**temp), 1-((1-inflection)*(((1-X)/(1-inflection))**temp))) for X in Xs])
                        )
        
    fig, ax2 = plt.subplots(1,2, squeeze = False, figsize=(8,4))
    ax2[0,0].set_title('full adaptation')
    ax2[0,0].plot(np.linspace(0,1,100), sCDF(np.linspace(0,1,100)), color=sb.color_palette()[0], linewidth=7)
    ax2[0,0].plot(np.linspace(0,0.5,100), sCDF(np.linspace(0,1,100)) * sCDF(0.5), color = sb.color_palette()[1], linewidth=3)
    
    ax2[0,0].plot(np.linspace(0,1,100), np.ones(100), '--', color=sb.color_palette()[0], linewidth=1)
    ax2[0,0].plot( np.ones(100), np.linspace(0,1,100), '--', color=sb.color_palette()[0], linewidth=1)
    ax2[0,0].plot(np.zeros(50), np.linspace(0, 1, 50), '--', color=sb.color_palette()[0], linewidth=1)
    ax2[0,0].plot(np.linspace(0, 1, 50), np.zeros(50), '--', color=sb.color_palette()[0], linewidth=1)
    
    ax2[0,0].plot(np.ones(50)/2, np.linspace(0, sCDF(0.5), 50), '--', color=sb.color_palette()[1], linewidth=1)
    ax2[0,0].plot( np.linspace(0, 0.5, 50),  [sCDF(0.5)]*50, '--', color=sb.color_palette()[1], linewidth=1)
    ax2[0,0].plot(np.zeros(50), np.linspace(0, sCDF(0.5), 50), '--', color=sb.color_palette()[1], linewidth=1)
    ax2[0,0].plot(np.linspace(0, 0.5, 50), np.zeros(50), '--', color=sb.color_palette()[1], linewidth=1)
    
    ax2[0,1].set_title('full adaptation')
    ax2[0,1].plot(np.linspace(0,1,100), sCDF(np.linspace(0,1,100)), color=sb.color_palette()[0], linewidth=7)
    ax2[0,1].plot(np.linspace(0,0.5,100), (sCDF(np.linspace(0,0.5,100)) / 0.5) * sCDF(0.5), color = sb.color_palette()[1], linewidth=3)
    
    ax2[0,1].plot(np.linspace(0,1,100), np.ones(100), '--', color=sb.color_palette()[0], linewidth=1)
    ax2[0,1].plot( np.ones(100), np.linspace(0,1,100), '--', color=sb.color_palette()[0], linewidth=1)
    ax2[0,1].plot(np.zeros(50), np.linspace(0, 1, 50), '--', color=sb.color_palette()[0], linewidth=1)
    ax2[0,1].plot(np.linspace(0, 1, 50), np.zeros(50), '--', color=sb.color_palette()[0], linewidth=1)
    
    ax2[0,1].plot(np.ones(50)/2, np.linspace(0, sCDF(0.5), 50), '--', color=sb.color_palette()[1], linewidth=1)
    ax2[0,1].plot( np.linspace(0, 0.5, 50),  [sCDF(0.5)]*50, '--', color=sb.color_palette()[1], linewidth=1)
    ax2[0,1].plot(np.zeros(50), np.linspace(0, sCDF(0.5), 50), '--', color=sb.color_palette()[1], linewidth=1)
    ax2[0,1].plot(np.linspace(0, 0.5, 50), np.zeros(50), '--', color=sb.color_palette()[1], linewidth=1)

    ax2[0,0].set_ylabel('utility')
    ax2[0,0].set_xlabel('reward magnitude')
    ax2[0,1].set_ylabel('utility')
    ax2[0,1].set_xlabel('reward magnitude')

        
    fig, ax1 = plt.subplots(1,2, squeeze = False, figsize=(8,4))
    ax1[0,0].set_title('full adaptation')
    ax1[0,0].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k' , linewidth=2)
    ax1[0,0].plot(np.linspace(0,1,100), sCDF(np.linspace(0,1,100)), color=sb.color_palette()[0], linewidth=7)
    ax15 = ax1[0,0].twiny()
    ax15.plot(np.linspace(0,0.5,100), sCDF(np.linspace(0,1,100)), color = sb.color_palette()[1], linewidth=3)
    ax1[0,1].set_title('full adaptation')
    ax1[0,1].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k' , linewidth=2)
    ax1[0,1].plot(np.linspace(0,1,100), sCDF(np.linspace(0,1,100)), color=sb.color_palette()[0], linewidth=7)
    ax15 = ax1[0,1].twiny()
    ax15.plot(np.linspace(0,0.5,100), sCDF(np.linspace(0,0.5,100)) / 0.5, color = sb.color_palette()[1], linewidth=3)
    
    ax1[0,0].set_ylabel('utility')
    ax1[0,0].set_xlabel('reward magnitude')
    ax1[0,1].set_ylabel('utility')
    ax1[0,1].set_xlabel('reward magnitude')


#%%
def compare_behaviouralFits(fractile_MLE_List, comparison = 'bic'):
    '''
    get ANOVA results on the residuals of each fit.  
    '''
    import statsmodels.api as sm
    
    AICrss = lambda RSS, n, k : (n*np.log(RSS/n)) + (2*k) 
    BICrss = lambda RSS, n, k : (n*np.log(RSS/n)) + (k*np.log(n)) 
    past_mean = 0
    
    from macaque.f_Rfunctions import oneWay_rmAnova
    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((1, 3), (0, 2))
   
    Xs = []; BIC = []; AIC = []; RSS = []
    realXs = []; dates=[]; names=[]; functions=[]
    for i, mle in enumerate(fractile_MLE_List):
        where = mle.fit_lsq.apply(lambda x: x!= [])
        rss = np.hstack(mle.fit_lsq.values)
        if np.size(rss) == 0:
            continue
        n = mle.behavioural_data.apply(lambda x: len(x[0][0])).values[where]
        k = len(mle.params.iloc[-1][2:])
        aic = AICrss(rss, n, k)
        bic = AICrss(rss, n, k)
        
        BIC.extend(bic)
        AIC.extend(aic)
        RSS.extend(rss)
        Xs.extend(np.random.normal(i, 0.08, size=len(rss)))
        realXs.extend([i]*len(rss))
        dates.extend(mle.date.values[where])
        names.extend(mle.model_used.unique())
        functions.append(mle.model_used.iloc[-1])
        
        if comparison.lower() == 'bic':
            if np.mean(bic) < past_mean:
                model_name = mle.model_used.iloc[-1]
                past_mean = np.mean(bic)
        elif comparison.lower() == 'aic':
            if np.mean(aic) < past_mean:
                model_name = mle.model_used.iloc[-1]
                past_mean = np.mean(aic)

    if comparison.lower() == 'bic':
        results = np.array(BIC)[np.isnan(BIC) == False]
        realXs = np.array(realXs)[np.isnan(BIC) == False]
        Xs = np.array(Xs)[np.isnan(BIC) == False]
        dates = np.array(dates)[np.isnan(BIC) == False]
        RSS = np.array(RSS)[np.isnan(BIC) == False]
    elif comparison.lower() == 'aic':
        results = np.array(AIC)[np.isnan(AIC) == False]
        realXs = np.array(realXs)[np.isnan(AIC) == False]
        Xs = np.array(Xs)[np.isnan(AIC) == False]
        dates = np.array(dates)[np.isnan(AIC) == False]
        RSS = np.array(RSS)[np.isnan(AIC) == False]
    else:
        results == LL
    sb.boxplot(np.array(realXs), results, ax=ax2, color='white', saturation=1, width=0.5, showfliers=False)
    plt.setp(ax2.artists, edgecolor = 'k', facecolor='w')
    plt.setp(ax2.lines, color='k')
    ax2.scatter(Xs, results, color='k', alpha=0.2)
    ax2.set_xticklabels(names, rotation=45)
    ax2.set_ylabel(comparison + ' score')
    ax2.set_xlabel('model tested')



    ax1.set_ylabel('RSS')
    for x in np.unique(realXs):
        where = realXs == x
        ax1.plot(RSS[where])
    ax1.legend(names)
    ax1.set_xticks(range(0, sum(where)))
    ax1.set_xticklabels(
        dates[where], rotation=45, horizontalalignment='center')
#    ax1.grid(axis='y')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlabel('session dates')

    plt.tight_layout()
    
    print('\n----------------------------------------------------------')
    rmAnova = oneWay_rmAnova(results, [x.toordinal() for x in dates], [functions[int(x)] for x in realXs])
    print('============================================================================')
    print('The best model is: ', model_name)
    
    data = []
    for ff in np.unique([functions[int(x)] for x in realXs]):
        where = np.array([functions[int(x)] for x in realXs]) == ff
        data.append(np.array(results)[where])
    data = np.array(data)
    
    print('\n================================================================================')
    print(stats.friedmanchisquare(*[data[x] for x in np.arange(data.shape[0])]))
    print('\n')
    post = sm.stats.multicomp.MultiComparison(results, [functions[int(x)] for x in realXs])
    print(post.allpairtest(stats.wilcoxon, method = 'holm')[0])


#%%
def get_firstFractiles(fractile, Trials, mismatch = 0.025):

    dating = Trials.loc[(Trials.GA_ev > 0.5)].sessionDate.unique()
    dating = dating[dating < fractile.sessionDate.min()]
    tt = Trials.loc[np.isin(Trials.sessionDate, dating)]
    riskySM = tt.get_Risky(mergeBy = 'all')
    riskySM = filter_utilityType(riskySM, 'fractile').copy()
    
    for date in tqdm(riskySM.sessionDate.unique()):
        sm = riskySM.loc[riskySM['sessionDate'] == date]
        mRange = np.concatenate(np.vstack(sm.primary.values)[:, 0::2])
        riskySM.loc[riskySM.sessionDate == date, 'min_rew'] = min(mRange)
        riskySM.loc[riskySM.sessionDate == date, 'max_rew'] = max(mRange)
    riskySM['reward_range'] = riskySM[['min_rew', 'max_rew']].values.tolist()
    riskySM.drop(columns=['min_rew'], inplace=True)
    riskySM.drop(columns=['max_rew'], inplace=True)
    
    riskySM.drop(riskySM.loc[riskySM.primary.apply(lambda x:x[1] != 0.5)].index, inplace = True)
    
    dList = []
    for date in riskySM.sessionDate.unique():
        STEP_75 = []; STEP_25 = []
        sm = riskySM.loc[riskySM.sessionDate == date]
        if max(sm.reward_range.iloc[-1]) < 1:
            continue
        
        primaries = np.vstack(sm.primary.values)
        reward_range = sm.reward_range.iloc[-1]
        reward_range[0] = min(primaries[primaries[:,2] == reward_range[1], 0])
        
        step1 = np.intersect1d(primaries[:,0], primaries[:,2])
        if np.size(step1) > 1:
            step1 = primaries[:,0][np.isin(primaries[:,0], step1)][0]
            
        if np.size(step1) == 0:
            CEs = sm.loc[(primaries[:,0] == reward_range[0]) & (primaries[:,2] == reward_range[1])].equivalent.values
            step1_low = np.intersect1d(primaries[:,0], np.round(CEs, 2))
            step1_high = np.intersect1d(primaries[:,2], np.round(CEs, 2))
            if np.size(step1_high) != 0:
                sm.loc[primaries[:,2] == step1_high, 'utility'] = 0.25
                sm.loc[primaries[:,2] == step1_high, 'fractile'] = 1
                STEP_25 = sm.loc[primaries[:,2] == step1_high, 'equivalent'].round(2).values

            if np.size(step1_low) != 0:
                sm.loc[primaries[:,0] == step1_low, 'utility'] = 0.75
                sm.loc[primaries[:,0] == step1_low, 'fractile'] = 1
                STEP_75 = sm.loc[primaries[:,0] == step1_low, 'equivalent'].round(2).values

        else:
            sm.loc[primaries[:,0] == step1, 'utility'] = 0.75
            sm.loc[primaries[:,0] == step1, 'fractile'] = 1
            sm.loc[primaries[:,2] == step1, 'utility'] = 0.25
            sm.loc[primaries[:,2] == step1, 'fractile'] = 1
            STEP_75 = sm.loc[(primaries[:,0] == step1) & (primaries[:,2] == reward_range[1]), 'equivalent'].round(2).values
            STEP_25 = sm.loc[(primaries[:,2] == step1)  & (primaries[:,0] == reward_range[0]), 'equivalent'].round(2).values
        
        #You still need
        if np.size(STEP_75) != 0:
            for step_75 in STEP_75:
                sm.loc[ np.abs(primaries[:,0] - step_75) < mismatch, 'utility'] = 0.875
                sm.loc[ np.abs(primaries[:,0] - step_75) < mismatch, 'fractile'] = 2
                sm.loc[ np.abs(primaries[:,2] - step_75) < mismatch, 'utility'] = 0.625
                sm.loc[ np.abs(primaries[:,2] - step_75) < mismatch, 'fractile'] = 2
        if np.size(STEP_25) != 0:
            for step_25 in STEP_25:
                sm.loc[ np.abs(primaries[:,0] - step_25) < mismatch, 'utility'] = 0.375
                sm.loc[ np.abs(primaries[:,0] - step_25) < mismatch, 'fractile'] = 2
                sm.loc[ np.abs(primaries[:,2] - step_25) < mismatch, 'utility'] = 0.125
                sm.loc[ np.abs(primaries[:,2] - step_25) < mismatch, 'fractile'] = 2
        
        sm.loc[(primaries[:,0] == reward_range[0]) & (primaries[:,2] == reward_range[1]), 'utility'] = 0.5
        sm.loc[(primaries[:,0] == reward_range[0]) & (primaries[:,2] == reward_range[1]), 'fractile'] = 0.0
        sm['division'] = 0
        dList.append(sm.drop(sm.loc[np.isnan(sm.utility.values)].index))
    dList = pd.concat(dList)
    
    for date in tqdm(dList.sessionDate.unique()):
        sm = dList.loc[dList['sessionDate'] == date]
        mRange = np.concatenate(np.vstack(sm.primary.values)[:, 0::2])
        dList.loc[dList.sessionDate == date, 'min_rew'] = min(mRange)
        dList.loc[dList.sessionDate == date, 'max_rew'] = max(mRange)
    dList['reward_range'] = dList[['min_rew', 'max_rew']].values.tolist()
    dList.drop(columns=['min_rew'], inplace=True)
    dList.drop(columns=['max_rew'], inplace=True)
    dList.reset_index(drop=True, inplace=True)
    for index,row in dList.iterrows():
        if max(row['reward_range']) > 0.7:
            dList.loc[index, 'reward_range'][0] = 0.1
            dList.loc[index, 'reward_range'][1] = 1.3
            
    cols = [
        'sessionDate', 'reward_range', 'division', 'fractile', 'primary',
        'primaryEV', 'equivalent', 'utility', 'secondary', 'secondaryEV',
        'm_range', 'freq_sCh', 'pFit', 'pSTE', 'no_of_Trials', 'nTrials',
        'primarySide', 'choiceList', 'choiceTimes', 'moveTime', 'trial_index',
        'oClock', 'func', 'metricType', 'seqCode', 'gList', 'chosenEV' ]
    return dList[cols]
 
#%%
def plot_illustration(model = 'risky-scdf'):
    '''
    '''
    from macaque.f_models import define_model
    import scipy.optimize as opt
#    y = [0, 0.125, 0.25, 0.5, 0.75, 0.875, 1]
#    x = [0, .3, .5, .6, 0.75, 0.85, 1]

    x = [ 0.225, 0.25, 0.275, 0.325, 0.35, 0.375,
         0.425, 0.45, 0.475, 0.525, 0.55, 0.575,
         0.725, 0.75, 0.775, 0.825, 0.85, 0.875]
    y = [ 0.1, 0.12, 0.15, 0.21, 0.25, 0.30,
         0.45, 0.52, 0.58, 0.68, 0.72, 0.76,
         0.93, 0.95, 0.96, 0.97, 0.98, 0.985]


    fig, ax = plt.subplots(1,1, squeeze = False, figsize=(4,4))
    ax[0,0].scatter(x,y, s=100, color='k')
#    ax[0,0].grid()

    outputs = define_model(model)
    p0 = outputs[1]
    utility = outputs[-1](p0)['empty functions']['utility']
    p0 = outputs[1][1:-1]
    function = lambda x, p1, p2 : utility(x, [p1,p2])
    pFit, pCov = opt.curve_fit( function, xdata=x,
                               ydata= y,
                               p0=[1, 0.5],
                               method='lm',
                               maxfev = 1000)
    ax[0,0].plot(np.linspace(0, 1, 100),
            function(np.linspace(0,1,100), pFit[0], pFit[1]), '--',
            color = 'k',
            linewidth = 2, alpha = 0.25)

    ax[0,0].set_xlabel('Certainty Equivalent')
    ax[0,0].set_ylabel('utility')


#%%
def get_archives(model = 'risky-scdf'):
    '''
    '''
    from macaque.f_models import define_model
    from macaque.f_uncertainty import bootstrap_y
    import scipy.optimize as opt
    norm = lambda x: (x - min(x)) / (max(x) - min(x))

    oldData = pd.read_csv('oldTigger_avg.csv', header=None, names = ['equivalents', 'utility'])
    oldData['utility'] = norm(oldData.utility)
    oldData['equivalents'] = norm(oldData['equivalents']) * (1.3 - 0.1) + 0.1

    oldData.plot.scatter(x='equivalents', y='utility', color='k', s=75)
    ax = plt.gca()
    ax.grid()

    outputs = define_model(model)
    p0 = outputs[1]
    utility = outputs[-1](p0)['empty functions']['utility']
    p0 = outputs[1][1:-1]
    function = lambda x, p1, p2 : utility(x, [p1,p2])
    pFit, pCov = opt.curve_fit( function, xdata=norm(oldData.equivalents.values),
                               ydata= oldData.utility.values,
                               p0=[1, 0.99],
                               method='lm',
                               maxfev = 1000)
    ax.plot(np.linspace(oldData['equivalents'].min(), oldData['equivalents'].max(), 100),
            function(np.linspace(0,1,100), pFit[0], pFit[1]), '--',
            color = 'k',
            linewidth = 3)

    #need to bootstrap a fit on this
    mean, bound_upper, bound_lower, yHat = bootstrap_y( norm(oldData.equivalents.values),
                                                       oldData.utility.values,
                                                       function,
                                                       pZero = [1, 0.99],
                                                       method='resampling', n=10000)
    ax.plot( np.linspace(oldData['equivalents'].min(), oldData['equivalents'].max(), 100), bound_lower, '--', color='k', alpha=0.4)
    ax.plot( np.linspace(oldData['equivalents'].min(), oldData['equivalents'].max(), 100), bound_upper, '--', color='k', alpha=0.4)
    squarePlot(ax)
    fig = plt.gcf()
    fig.suptitle('utility of 3 years past')

#%%
def filter_plausibleCEs(sm):
    mask = [True if (ce<=rr[1]) and (ce>=rr[0]) else False for ce, rr in zip(sm.equivalent.values,
        np.vstack(sm.reward_range))]
    sm = sm.loc[mask]
    return sm

#%%
def filter_utilityType(softmaxDF, mode):
    '''
    return only softmax sequences from trials from fract_util sequences
    ---------
    - Gambles need to be 2-outcome
    - m1,m2 need to be 0 and 0.5 respectively
    - p needs to be 0.1:0.9
    - the CE needs to be between 0 and 0.5
    '''
    softmaxDF = softmaxDF.loc[softmaxDF.primary.apply( lambda x: len(x) <= 4)]

    if mode.lower() == 'fractile':
        for date in softmaxDF.sessionDate.unique():
            sm = softmaxDF.loc[softmaxDF['sessionDate'] == date]
            for block in sm.division.values:
                mRange = np.concatenate(np.vstack(sm.primary.values)[:, 0::2])
                softmaxDF.loc[(softmaxDF.sessionDate == date) & (softmaxDF.division == block),
                              'min_rew'] = min( mRange)
                softmaxDF.loc[(softmaxDF.sessionDate == date) & (softmaxDF.division == block),
                              'max_rew'] = max( mRange)
                softmaxDF['reward_range'] = softmaxDF[['min_rew', 'max_rew']].values.tolist()
        softmaxDF.drop(columns=['min_rew'], inplace=True)
        softmaxDF.drop(columns=['max_rew'], inplace=True)

        fullRange = np.vstack(softmaxDF.reward_range.values)
        index = np.unique(softmaxDF.loc[softmaxDF.equivalent.values < fullRange[:,0]].index.tolist()\
                          + softmaxDF.loc[softmaxDF.equivalent.values > fullRange[:,1]].index.tolist())
        return softmaxDF.drop(index)

    elif mode.lower() == 'derivative':
        #filter CE sequences that have plausible CEs - i.e. set between the possible reward ranges
        fullRange = np.vstack(softmaxDF.secondary.values)[:, 0]
        softmaxDF = softmaxDF.loc[softmaxDF.equivalent.apply(
            lambda x: min(fullRange) <= x <= max(fullRange))]
        #filter for the lists that were used for the binary utility
        f_index = np.unique(
            np.concatenate((softmaxDF.loc[softmaxDF.gList.str.contains(
                'deriv')].index.values, softmaxDF.loc[
                    softmaxDF.gList.str.contains('Deriv')].index.values,
                            softmaxDF.loc[softmaxDF.gList.str.contains(
                                'trans')].index.values,
                            softmaxDF.loc[softmaxDF.gList.str.contains(
                                'gList_util')].index.values)))
        softmaxDF = softmaxDF.loc[f_index]
        for date in softmaxDF.sessionDate.unique():
            sm = softmaxDF.loc[softmaxDF['sessionDate'] == date]
            mRange = np.concatenate(np.vstack(sm.primary.values)[:, 0::2])
            softmaxDF.loc[softmaxDF.sessionDate == date, 'min_rew'] = min(
                mRange)
            softmaxDF.loc[softmaxDF.sessionDate == date, 'max_rew'] = max(
                mRange)
        softmaxDF['reward_range'] = softmaxDF[['min_rew',
                                               'max_rew']].values.tolist()
        softmaxDF.drop(columns=['min_rew'], inplace=True)
        softmaxDF.drop(columns=['max_rew'], inplace=True)
        
        #make sure the CEs and spreads make sense
        softmaxDF = softmaxDF.loc[[False if x[0]<x[1][0] or x[0]>x[1][1] else True for x in softmaxDF[['equivalent','reward_range']].values]]
        softmaxDF.drop(softmaxDF.loc[softmaxDF.primaryEV.values - softmaxDF.equivalent.values > 0.5].index, inplace = True)
        softmaxDF['spread'] = softmaxDF.primary.apply(lambda x: np.round(np.diff(x[::2]), 2)[0])

        #added to make results more reliable for trident
        softmaxDF = softmaxDF.loc[softmaxDF.gList != '\'gList_Tri_fullRange_deriv.txt\'']
        softmaxDF = softmaxDF.loc[(softmaxDF.spread == 0.15) | (softmaxDF.spread == 0.30)]
        return softmaxDF
    
    elif mode.lower() == 'random':
        pass


#%%
def get_fractileUtility(softmaxDF):
    '''
    '''

    # ------------------------------------------------------
    def splitFrac(div, times):
        arr = []
        fractile = div
        if times == 0:
            return div
        for n in range(0, times):
            fractile = fractile / 2
            arr.extend([fractile])
        return sum(arr)

    # ------------------------------------------------------
    fullRange = []

    #filter CEs
    softmaxDF = filter_utilityType(softmaxDF, mode='fractile').copy()

    #    binary_SM.reward_range = binary_SM.reward_range.apply(lambda x : precise_range(x[:]))

    for date in tqdm(
            softmaxDF.sessionDate.unique(), desc='assigning utility values'):
        sm = softmaxDF.loc[softmaxDF['sessionDate'] == date]
        for block in sm.division.unique():
            sd = sm.loc[sm.division == block].trial_index.values
            stepStart = [
                min(tt)
                for tt in [np.concatenate(list(val.values())) for val in sd]
            ]
            order = np.argsort(stepStart, axis=0)
            fr = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4][0:len(order)])

            EVs = sm.loc[sm.division == block].primaryEV.values

            util = [0.5]
            past = 0
            for ev, frac in zip(EVs[order][1:], fr[1:]):
                if frac != past:
                    div = splitFrac(util[0], frac)
                    past = frac
                if ev > np.array(EVs)[order][0]:
                    util.extend([util[0] + div])
                elif ev < np.array(EVs)[order][0]:
                    util.extend([util[0] - div])

            utility = [None] * len(order)
            fraction = [None] * len(order)
            for uu, oo, ff in zip(util, order, fr):
                utility[oo] = uu
                fraction[oo] = ff

            softmaxDF.loc[sm.loc[sm.division == block].index,
                          'fractile'] = fraction
            softmaxDF.loc[sm.loc[sm.division == block].index,
                          'utility'] = utility

    cols = [
        'sessionDate', 'reward_range', 'division', 'fractile', 'primary',
        'primaryEV', 'equivalent', 'utility', 'secondary', 'secondaryEV',
        'm_range', 'freq_sCh', 'pFit', 'pSTE', 'no_of_Trials', 'nTrials',
        'primarySide', 'choiceList', 'choiceTimes', 'moveTime', 'trial_index',
        'oClock', 'func', 'metricType', 'seqCode', 'gList', 'chosenEV'
    ]

    #make sure only trials with 50-50 probability
    softmaxDF = softmaxDF.loc[softmaxDF.primary.apply(lambda x: x[1] == x[3])]
    softmaxDF = softmaxDF.loc[softmaxDF.fractile < 3]
    
    return softmaxDF[cols].sort_values(
        by=['sessionDate', 'division', 'utility'])

#%%
def fit_fractileUtilities(fractile, Trials,
                           model = 'logit-2scdf-ev',
                           minPoints=3,
                           minTrials= 60,
                           fixedRange = False):
    '''
    '''
    from macaque.f_probabilityDistortion import plot_MLEfitting
    import scipy.optimize as opt
    from macaque.f_models import trials_2fittable, LL_fit
    from scipy.odr import ODR, Model, Data, RealData
    
    if type(model) == list:
        Fractile_MLE=[]
        for mm in model:
            Fractile_MLE.append(fit_fractileUtilities(fractile, Trials, model = mm, minPoints=minPoints))
        return Fractile_MLE
    else:
        tt = fractile.getTrials(Trials) #here I get the trials that form the fractile utility
        dList=[];
    
        for date in tqdm(fractile.sessionDate.unique(), desc = 'fitting fractiles'):
            df = fractile.loc[fractile.sessionDate == date]
            if len(df.loc[df.fractile > 0]) < 2:
                continue
            
            rr = np.vstack(df.reward_range.values )
            mRange = [rr.min(), rr.max()]
            
            X, Y = trials_2fittable(tt.loc[tt['sessionDate'] == date])
            if len(X) < minTrials:
                continue
            else:
                MLE = LL_fit(Y, X, model = model, fixedRange=False).fit(disp=False)
    
            pFit_all = []
            dataPoints = []; lsq=[]; pFit_y=[]
            
            x = []; y = [];
            for block in df.division.unique():
                yy = df.loc[df.division == block].utility.values
                xx = df.loc[df.division == block].equivalent.values
                if len(np.unique(yy)) >= minPoints:
                    x.extend(xx)
                    y.extend(yy)

            x = (np.array(x) - mRange[0]) / (mRange[1]-mRange[0])
            y = np.array(y)
            dataPoints.append(np.vstack((x,y)))
            try:
                data = RealData(x, y)
                function = lambda params, x : MLE.model_parts['empty functions']['utility'](x, params)
                model2fit = Model(function)
#                
                odr = ODR(data, model2fit, MLE.model.start_params[2:])
#                odr.set_job(fit_type=2)
                output = odr.run()
                if not all(output.beta == MLE.model.start_params[2:]):                
                    pFit_all.append(output.beta)
                    lsq.append(output.sum_square)
            except:
                pass
            
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
                'mag_range' : mRange,
                'behavioural_fit' : pFit_all,
                'behavioural_data' : dataPoints,
                'fit_lsq' : lsq,
                'pFit_in_y' : [pFit_y]})
        Fractile_MLE = pd.DataFrame(dList)
        plot_MLEfitting(Fractile_MLE, plotFittings=False)
        return Fractile_MLE

#%%
def plot_fractileUtilities(fractile_MLE, minPoints = 3):
    '''
    '''
    fig, ax = plt.subplots( 2,
                       len(fractile_MLE.date.unique()),
                       squeeze = False,
                       figsize= (int(np.ceil(len(fractile_MLE.date.unique())/3))*6, 5))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Daily Fittings')
    fig2, ax2 = plt.subplots( 1, 2, squeeze = False,  figsize= (6, 3))
    fig3, ax3 = plt.subplots( 1, len(unique_listOfLists(fractile_MLE.mag_range.values)), squeeze = False,  figsize= (9, 4))

    spreads = unique_listOfLists(fractile_MLE.mag_range.values)
    position = lambda rew_r: int(np.where([rew_r == spread for spread in spreads])[0])

    r=0
    for date in tqdm(np.sort(fractile_MLE.date.unique()), desc = 'plotting binary utilities'):
        if r == 0:
            ax[0,r].set_ylabel('fitted fractiles')
            ax[1,r].set_ylabel('mle fitting')
        df = fractile_MLE.loc[fractile_MLE.date == date]
        behav = df['behavioural_data'].values[0]
        fitting = flatten(df['behavioural_fit'].values)
        fitting_y = flatten(df['pFit_in_y'].values)
        
        mRange = df.mag_range.values[0]
        if sum(mRange) < 0.6:
            col = 'blue'
        elif sum(mRange) < 1.1:
            col='green'
        elif sum(mRange) > 1.1:
            col = 'red'
        ax[0,r].grid()
        ax[0,r].plot(np.linspace(mRange[0], mRange[1], 100),np.linspace(0, 1, 100), '--', color='k')
        ax[1,r].grid()
        ax[1,r].plot(np.linspace(mRange[0], mRange[1], 100),np.linspace(0, 1, 100), '--', color='k')
        if len(behav[0][0]) < minPoints:
                 continue
                 ax[0,r].text(0.1, 0.5, 'little points')
                 ax[1,r].text(0.1, 0.5, 'little points')
        for block, pFit in zip(behav, fitting):
            x = (block[0] * (mRange[1] - mRange[0])) +  mRange[0]
            y = block[1]
            ax[0,r].set_title(str(df.date.unique()[0]))
            ax[0,r].scatter(x,y, color='k')
            ax[1,r].scatter(x,y, color='k')
            ax2[0,0].scatter(x,y, color=col, alpha = 0.1)
            ax2[0,1].scatter(x,y, color=col, alpha = 0.1)
            
            ax3[0,position(mRange)].scatter(x,y, color='k', alpha = 0.7)

            function_fitted = lambda x : df.full_model.values[0].model_parts['empty functions']['utility'](x, pFit)
#            function_in_y = lambda x : df.full_model.values[0].model_parts['empty functions']['utility'](x, fitting_y[0])
            mle_fitted = df.full_model.values[0].utility

            if len(df)>=minPoints and len(X) < 100:
                ax[2,r].text(0.1, 0.5, 'Trials count under 100')
            ax[0,r].plot(np.linspace(mRange[0], mRange[1], 100), function_fitted(np.linspace(0,1, 100)), color=col, alpha=0.5)
#            ax[0,r].plot(np.linspace(mRange[0], mRange[1], 100), function_in_y(np.linspace(0,1, 100)), '--', color=col, alpha=0.5)
            ax[1,r].plot(np.linspace(mRange[0], mRange[1], 100), mle_fitted(np.linspace(0,1, 100)), color=col, alpha=0.5)
            ax2[0,0].plot(np.linspace(mRange[0], mRange[1], 100), function_fitted(np.linspace(0,1, 100)), color='dark'+col, alpha=0.1)
            ax2[0,1].plot(np.linspace(mRange[0], mRange[1], 100), mle_fitted(np.linspace(0,1, 100)), color='dark'+col, alpha=0.1)
            
            ax3[0,position(mRange)].plot(np.linspace(mRange[0], mRange[1], 100), function_fitted(np.linspace(0,1, 100)), color=col, alpha=0.5)
            ax3[0,position(mRange)].plot(np.linspace(mRange[0], mRange[1], 100), np.linspace(0, 1, 100),'--', color='k', alpha=0.5)
        r += 1
    ax2[0,0].grid(); ax2[0,1].grid()
    ax2[0,0].set_title('fitted fractiles')
    ax2[0,1].set_title('mle fitting')
    ax2[0,0].set_xlabel('certainty equivalent')
    ax2[0,1].set_xlabel('certainty equivalent')
    ax2[0,0].set_ylabel('utility')
    ax2[0,1].set_ylabel('utility')
    squarePlot(ax2[0,0])
    squarePlot(ax2[0,1])
    plt.suptitle('Aggregate Fittings')
    
    for _,ax in enumerate(fig3.axes):
        ax.set_title('fitted fractiles')
        ax.set_xlabel('certainty equivalent')
        ax.set_ylabel('utility')
        squarePlot(ax)

#    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#%%
def compare_midpoints(MLE_df, dataType='behaviour'):
    '''
    '''
    from scipy import stats
    from macaque.f_models import define_model

        
    MLE_df = MLE_df.sort_values('date')
    inflection = []; ranges = []; dating = []; cc_ratio = []
    index=[]; i=0

    if dataType.lower() == 'mle':
        for mle, dd, rr in tqdm(zip(MLE_df.full_model, MLE_df.date, MLE_df.mag_range), desc='Gathering Inflections'):
            where = np.argmax(np.gradient(mle.utility(np.linspace(0,1,10000))))
            ranges.extend([rr])
            dating.extend([dd])
            index.extend([i])
            inflection.extend([np.linspace(rr[0],rr[1],10000)[where]])
            i+=1
    elif dataType.lower() == 'behaviour':
        model = MLE_df.full_model.iloc[-1]
        model = model.model_parts['empty functions']['utility']
        for params, dd, rr in tqdm(zip(MLE_df.behavioural_fit, MLE_df.date, MLE_df.mag_range), desc='Gathering Inflections'):
            if np.size(params) > 2:
                for pp in params:
                    utility = model(np.linspace(0,1,10000), pp)
                    where = np.argmax(np.gradient(utility))
                    ranges.extend([rr])
                    dating.extend([dd])
                    index.extend([i])
                    inflection.extend([np.linspace(rr[0],rr[1],10000)[where]])
                i+=1
            elif params == []: 
                continue
            else:
                utility = model(np.linspace(0,1,10000), flatten(params))
                where = np.argmax(np.gradient(utility))
                shape = np.gradient(np.gradient(utility))
                shape[shape<=0] = 0
                shape[shape>0] = 1
                cc_ratio.extend([np.mean(shape)])
                ranges.extend([rr])
                dating.extend([dd])
                index.extend([i])
                inflection.extend([np.linspace(rr[0],rr[1],10000)[where]])
                i+=1

    fig, ax = plt.subplots( 2, 1, squeeze = False,  figsize= (10, 3))
    inflection = np.array(inflection)
    cc_ratio = np.array(cc_ratio)
    index = np.array(index)
    dating = np.array(dating)
    minDate = []; dataList=[]; legend = []; ratios = []
    for rr in np.sort(unique_listOfLists(ranges), 0):
        where = [all(r1==rr) for r1 in ranges]
        dates = dating[where]
        y = inflection[where]
        proportion = cc_ratio[where]
        x = index[where]
        
        dataList.append(y)
        ratios.append(proportion)
        
        ax[0,0].plot(x,y,'bo', color='k')
        minDate.extend([min(x)])
        ax[0,0].axvline(minDate[-1], color = 'magenta')
        ax[0,0].text(minDate[-1], 0.9, '   ' + str(rr))
        ax[0,0].text(minDate[-1], 0.05, '   ' + str(dates.min()))
        legend.extend([str(rr)])
        
        ax[1,0].plot(x, proportion, 'bo', color = 'k')
        ax[1,0].axvline(minDate[-1], color = 'magenta')
        ax[1,0].text(minDate[-1], 0.9, '   ' + str(rr))
        ax[1,0].text(minDate[-1], 0.05, '   ' + str(dates.min()))
        
        try:
            m, b = np.polyfit(x, y, 1)
            ff = (m*x) + b
            ax[0,0].plot(x, ff)
            ax[0,0].text(minDate[-1], 0.125, '  slope: ' + str(np.round(m, 3)))
        except:
            pass
        
        try:
            m, b = np.polyfit(x, proportion, 1)
            ff = (m*x) + b
            ax[1,0].plot(x, ff)
            ax[1,0].text(minDate[-1], 0.125, '  slope: ' + str(np.round(m, 3)))
        except:
            pass
    ax[0,0].set_ylim(0,1)
    ax[1,0].set_ylim(0,1)

    #%%
    fig, ax = plt.subplots( 1, 2, squeeze = False,  figsize= (6, 4)) 
    gap = 0.2
    for rr, ii in zip(dataList, ratios):
        ax[0,0].bar(0 + gap, np.mean(rr), width = 0.2, yerr = stats.sem(rr) )
        ax[0,1].bar(0 + gap, np.mean(ii), width = 0.2, yerr = stats.sem(ii) )
        gap+=0.2
    ax[0,0].legend(np.sort(unique_listOfLists(ranges), 0))
    ax[0,0].set_ylim([0,1]); ax[0,0].axhline(0.5, linestyle='--', color='k')
    ax[0,1].set_ylim([0,1]);  ax[0,1].axhline(0.5, linestyle='--', color='k')
    ax[0,0].set_ylabel('inflection point'); ax[0,1].set_ylabel('convexity ratio')
    squarePlot(ax[0,0]); squarePlot(ax[0,1])
    
    #%%

    fig, ax = plt.subplots( 1, 1, squeeze = False,  figsize= (3, 3)) 
    gap = -0.9
    i = 1
    for sequenceType, color in zip(dataList, ['blue', 'green', 'red']):
        bp = ax[0,0].boxplot(sequenceType, positions= [i *4.0+gap], sym='', widths=0.4)
        set_box_color(bp, 'black')
        ax[0,0].scatter(jitter([i *4.0+gap]*len(sequenceType), 0.04), sequenceType, c=color, label='riskless', alpha=0.2)
        gap += 0.6

    ax[0,0].axhline(0, alpha=0.3, color='k')
#    ax.set_xticks( np.arange(1, np.size(np.vstack(dataList),1)+1) *4 )
#    ax.set_xticklabels( np.hstack((df.pNames.iloc[0][1:3], df.pNames.iloc[0][0],  df.pNames.iloc[0][3] )) )
    ax[0,0].set_xbound(2.5,5)
#    ax1[1].set_ylabel('decision parameters')
    fig.suptitle('inflection point comparison')

    # -----------------------------------------------------------------------

    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    independent = flatten([len(dd) * [rr] for dd, rr, in zip(dataList, [1,2,3])])
    dependent = flatten(dataList)
    print('--------------------------------------------------------------')
    print('COMPARING MIDPOINTS')
    print('--------------------------------------------------------------')
    print('1 is [0,0.5] \n2 is [0, 1.0] \n3 is [0.5, 1.0]')
    
    data = pd.DataFrame(np.vstack((independent, dependent))).T
    data.columns = ['range', 'inflection']
    range_lm = ols('inflection ~ C(range)',  data=data).fit()
    table = sm.stats.anova_lm(range_lm, typ=3)
    print(table)
    
    data = []
    for ff in np.unique(independent):
        where = np.array(independent) == ff
        data.append(np.array(dependent)[where])
    data = np.array(data)
    
    print('\n=============================')
    print(stats.mstats.kruskalwallis(*[data[x] for x in np.arange(data.shape[0])]))
    print('\n')
    post = sm.stats.multicomp.MultiComparison(dependent, independent)
    print(post.allpairtest(stats.ranksums, method = 'holm')[0])
    print(range_lm.summary2())
    
    independent = flatten([len(dd) * [rr] for dd, rr, in zip(ratios, [1,2,3])])
    dependent = flatten(ratios)
    print('\n \n --------------------------------------------------------------')
    print('COMPARING CONVEXITY RATIO')
    print('--------------------------------------------------------------')
    print('1 is [0,0.5] \n2 is [0, 1.0] \n3 is [0.5, 1.0]')
    
    data = pd.DataFrame(np.vstack((independent, dependent))).T
    data.columns = ['range', 'convexity_ratio']
    range_lm = ols('convexity_ratio ~ C(range)',  data=data).fit()
    table = sm.stats.anova_lm(range_lm, typ=3)
    print(table)
    
    data = []
    for ff in np.unique(independent):
        where = np.array(independent) == ff
        data.append(np.array(dependent)[where])
    data = np.array(data)
    
    print('\n=============================')
    print(stats.mstats.kruskalwallis(*[data[x] for x in np.arange(data.shape[0])]))
    print('\n')
    post = sm.stats.multicomp.MultiComparison(dependent, independent)
    print(post.allpairtest(stats.ranksums, method = 'holm')[0])

    print(range_lm.summary2())
    
#%%
def parameterProgress(MLE_df, dataType = 'mle'):
    '''
    '''
    from macaque.f_models import define_model
    MLE_df = MLE_df.sort_values('date')

    fig, ax = plt.subplots( 1, 1, squeeze = False,  figsize= (10, 3))

    parameters = []; ranges = []; dating = []; 
    index=[]; i=0
    if dataType.lower() == 'mle':
        for mle, dd, rr in tqdm(zip(MLE_df.params, MLE_df.date, MLE_df.mag_range), desc='Gathering Inflections'):
            ranges.extend([rr])
            dating.extend([dd])
            index.extend([i])
            parameters.extend([mle])
            i+=1
        parameters = np.vstack(parameters)
    elif dataType.lower() == 'behaviour':
        model = MLE_df.full_model.iloc[-1]
        model = model.model_parts['empty functions']['utility']
        for params, dd, rr in tqdm(zip(MLE_df.behavioural_fit, MLE_df.date, MLE_df.mag_range), desc='Gathering Inflections'):
            if np.size(params) > 2:
                for pp in params:
                    ranges.extend([rr])
                    dating.extend([dd])
                    index.extend([i])
                    parameters.extend([pp])
                i+=1
            elif params == []: 
                continue
            else:
                ranges.extend([rr])
                dating.extend([dd])
                index.extend([i])
                parameters.extend([params])
                i+=1
        parameters = np.vstack(parameters)

    index = np.array(index)
    dating = np.array(dating)
    minDate = []; dataList=[]; legend = []
    for rr in np.sort(unique_listOfLists(ranges), 0):
        where = [all(r1==rr) for r1 in ranges]
        dates = dating[where]
        y = parameters[where]
        x = index[where]
        
        dataList.append(parameters[where])
        
        ax[0,0].plot(x,y[:,0],'bo', color='cyan')
        ax[0,0].plot(x,y[:,1],'bo', color='magenta')
        
        minDate.extend([min(x)])
        ax[0,0].axvline(minDate[-1], color = 'purple')
        ax[0,0].text(minDate[-1], 0.9, '   ' + str(rr))
        ax[0,0].text(minDate[-1], 0.05, '   ' + str(dates.min()))
        legend.extend([str(rr)])

    #%%
    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((1, 3), (0, 2))

    gap = -0.9
    ax1=[]
    ax1.extend([ax]); ax1.extend([ax.twinx()]); ax1.extend([ax.twinx()]); ax1.extend([ax.twinx()])
    if dataType.lower() == 'mle':
        order =  [1,2,0,3]
    else:
        order = [0,1]
        
    for sequenceType, color in zip(dataList, ['blue', 'green', 'red']):
        i = 1
        for param in order:
            if param == 1:
                bp = ax1[param].boxplot(np.log(sequenceType[:,param]), positions= [i *4.0+gap], sym='', widths=0.4)
            else:
                bp = ax1[param].boxplot(np.log(sequenceType[:,param]), positions= [i *4.0+gap], sym='', widths=0.4)
            set_box_color(bp, 'black') 

            risklessX = [i] * len(sequenceType)
            risklessX = np.random.normal(risklessX, 0.015, size=len(risklessX))
            if param == 1:
                ax1[param].scatter(risklessX*4.0+gap, np.log(np.ravel(sequenceType[:,param])), c=color, label='riskless', alpha=0.2)
            else:
                ax1[param].scatter(risklessX*4.0+gap, np.log(np.ravel(sequenceType[:,param])), c=color, label='riskless', alpha=0.2)

            if gap == -0.9:
                ax1[param].yaxis.set_ticks_position('left')
                ax1[param].spines['left'].set_position(('data', i*4-2))
            i+=1
        gap += 0.6

    ax.axhline(0, alpha=0.3, color='k')
    ax.set_xticks( np.arange(1, np.size(np.vstack(dataList),1)+1) *4 )
    if dataType.lower() == 'mle':
        ax.set_xticklabels( np.hstack((MLE_df.pNames.iloc[0][2:4], MLE_df.pNames.iloc[0][0],  MLE_df.pNames.iloc[0][3] )) )
    else:
        ax.set_xticklabels( np.hstack((MLE_df.pNames.iloc[0][2:4] )) )
    ax.set_xbound(2.5,17.5)
#    ax1[1].set_ylabel('decision parameters')
    fig.suptitle('ranges parameter comparisons')

    # -----------------------------------------------------------------------
    from macaque.f_probabilityDistortion import covarianceEllipse

    for sequenceType, color in zip(dataList, ['blue', 'green', 'red']):
        if dataType.lower() == 'mle':
            ax2.scatter(np.log(sequenceType[:,1]), np.log(sequenceType[:,2]), color=color)
#            covarianceEllipse(np.log(sequenceType[:,1]), sequenceType[:,2], ax2, color=color, draw='CI')
        else:
            ax2.scatter(np.log(sequenceType[:,0]), np.log(sequenceType[:,1]), color=color)
#            covarianceEllipse(np.log(sequenceType[:,0]), sequenceType[:,1], ax2, color=color, draw='CI')

    ax2.set_xlabel(MLE_df.pNames.iloc[0][0][1])
    ax2.set_ylabel(MLE_df.pNames.iloc[0][0][2])
    ax2.grid()
    ax2.set_xbound(-5, 5)
    x0, x1 = ax2.get_xlim()
    y0, y1 = ax2.get_ylim()
    ax2.set_aspect((x1 - x0) / (y1 - y0))
    ax2.legend(legend)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#    manova
    from macaque.f_Rfunctions import dv4_manova, dv2_manova
    IV = np.hstack([[i] * len(data) for i,data in enumerate(dataList)]).T
    dataList = np.vstack(dataList)
    dataList = np.vstack((dataList.T, IV)).T
    if dataType.lower() == 'mle':
        dv4_manova( dataList[:,1],  dataList[:,2],  dataList[:,0], dataList[:,3],  IV=IV)
    else:
        dv2_manova( dataList[:,0],  dataList[:,1],  IV=IV)

#%%
def fit_binaryUtilities(binary, Trials,
                           model = 'risky-scdf',
                           minPoints=3,
                           minTrials = 60):
    '''
    '''
    import scipy.optimize as opt
    from macaque.f_models import trials_2fittable, LL_fit
    d2 = lambda f_x : np.gradient(np.gradient(f_x))

    if type(model) == list:
        binary_MLE=[]
        for mm in model:
            binary_MLE.append(fit_fractileUtilities(fractile, Trials, model = mm, minPoints=3))
        return binary_MLE
    else:
#        binary = filter_plausibleCEs(binary)
        tt = binary.getTrials(Trials) #here I get the trials that form the fractile utility
        dList=[];
    
        for date in tqdm(np.sort(binary.sessionDate.unique()), desc = 'fitting fractiles'):
            df = binary.loc[binary.sessionDate == date]
            rewards = np.hstack(df.primary.apply(lambda x: x[::2]).values)
            if any(rewards > 0.5) and any(rewards < 0.5):
                mRange = [0.0, 1.0]
            elif any(rewards > 0.5) and not any(rewards < 0.45):
                mRange = [0.5, 1.0]
            elif any(rewards < 0.5) and not any(rewards > 0.55):
                mRange = [0.0, 0.5]
    
            df.sort_values(by='primaryEV', inplace=True)
            X, Y = trials_2fittable(tt.loc[tt['sessionDate'] == date])
            if len(X) < minTrials:
                continue
            else:
                MLE = LL_fit(Y, X, model = model).fit(disp=False)
    
            # fractile first
            function = lambda x, p1, p2: d2(MLE.model_parts['empty functions']['utility'](x, [p1,p2]))
            function2 = lambda x, p1, p2: MLE.model_parts['empty functions']['utility'](x, [p1,p2])
#            function = lambda x, p1, p2: d2(function(x, p1, p2))
    #        function = lambda x, p1, p2: d2(MLE.model_parts['empty functions']['utility'](x, [p1,p2]))
    #        from scipy import interpolate
    #        curve = lambda a,b : interpolate.LSQUnivariateSpline(a,b, t=[np.mean([min(Xs), max(Xs)])])
    #        mean, bound_upper, bound_lower, yHat = f_uncertainty.bootstrap(Xs,Ys, curve, \
    #                                                                        method = 'resampling', \
    #                                                                        n=10000)
            pFit_all = []
            dataPoints = []
            x = []; y = [];
            for block in df.division.unique():
                xx = df.loc[df.division == block].primaryEV.values
                yy = (df.loc[df.division == block].equivalent.values - xx)
                if len(yy) >= minPoints:
                    x.extend(xx)
                    y.extend(yy)
                    
            x = np.insert(x, [0, len(x)], [min(mRange), max(mRange)])
            y = np.insert(y, [0, len(y)], [0, 0])
            x = (x  - mRange[0]) / (mRange[1]-mRange[0])
            dataPoints.append(np.vstack((x,y)))

            try:
                pFit, pCov = opt.curve_fit( function, x, y,
                                           p0=MLE.model.start_params[2:4],
                                           method='lm',
                                           maxfev = 1000)
                pFit_all.append(pFit)
            except:
                continue
            
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
                'mag_range' : mRange,
                'behavioural_fit' : pFit_all,
                'behavioural_data' : dataPoints })
        binary_MLE = pd.DataFrame(dList)
        return binary_MLE

#%%
def plot_binaryUtilities(binary_MLE, minPoints = 3):
    '''
    '''
    d2 = lambda f_x : np.gradient(np.gradient(f_x))
    fig, ax = plt.subplots( 3,
                       len(binary_MLE.date.unique()),
                       squeeze = False,
                       figsize= (int(np.ceil(len(binary_MLE.date.unique())/3))*6, 5))

    fig2, ax2 = plt.subplots( 1, 2, squeeze = False,  figsize= (6, 3))
    fig3, ax3 = plt.subplots( 1, 3, squeeze = False,  figsize= (6, 3))
    
    binary_MLE = binary_MLE.loc[binary_MLE.NM_success == True]

    r=0
    for date in tqdm(np.sort(binary_MLE.date.unique()), desc = 'fitting fractiles'):
        if r == 0:
            ax[0,r].set_ylabel('fitted fractiles')
            ax[1,r].set_ylabel('mle fitting')
            ax[2,r].set_ylabel('derivative')
        df = binary_MLE.loc[binary_MLE.date == date]
        behav = df['behavioural_data'].values[0]
        fitting = flatten(df['behavioural_fit'].values)
        mRange = np.ravel(df.mag_range.values[0])
        if sum(mRange) < 0.6:
            col = 'blue'
        elif sum(mRange) < 1.1:
            col='green'
        elif sum(mRange) > 1.1:
            col = 'red'
        ax[0,r].set_title(str(df.date.unique()[0]))
        ax[0,r].grid()
        ax[0,r].plot(np.linspace(mRange[0], mRange[1], 100),
          np.linspace(0, 1, 100), '--', color='k')
        ax[1,r].grid()
        ax[1,r].plot(np.linspace(mRange[0], mRange[1], 100),
          np.linspace(0, 1, 100), '--', color='k')

        y = [0,0.5]
        for block, pFit in zip(behav, fitting):
            if len(df) < minPoints:
                 ax[0,r].text(0.1, 0.5, 'little points')
                 ax[1,r].text(0.1, 0.5, 'little points')
            x = (block[0]  * (mRange[1] - mRange[0])) + mRange[0]
            y = block[1]
#            ax[0,r].scatter(x,y, color='k')
            ax[2,r].scatter(x,y, color='k')
            ax3[0,0].scatter(x,y, color=col, alpha = 0.25)
#            ax3[0,1].scatter(x,y, color=col, alpha = 0.25)
            ax[2,r].axhline(0, color='k')

            function_fitted = lambda x : df.full_model.values[0].model_parts['empty functions']['utility'](x, pFit)
            ax[0,r].plot(np.linspace(mRange[0], mRange[1], 100), function_fitted(np.linspace(0,1, 100)), color='dark'+col, alpha=0.5)
            
            yy = convertRange(d2(function_fitted(np.linspace(0,1, 100))), 
                         [min(d2(function_fitted(np.linspace(0,1, 100)))), max(d2(function_fitted(np.linspace(0,1, 100))))],
                         [min(y), max(y)])
            
            ax[2,r].plot( np.linspace(mRange[0], mRange[1], 100),
               yy, color = 'dark' +col)
#             (d2(function_fitted(np.linspace(0,1, 100))) - min(y)) / (max(y) - min(y)), color = 'dark' +col)
            ax2[0,0].plot(np.linspace(mRange[0], mRange[1], 100),
               function_fitted(np.linspace(0,1, 100)), color='dark'+col, alpha=0.1)
            ax3[0,1].plot(np.linspace(mRange[0], mRange[1], 100),
               d2(function_fitted(np.linspace(0,1, 100))), color='dark'+col, alpha=0.1)
            
#            ax[2,r].scatter(x,-y, color='k')
            ax[2,r].axhline(0, color='k')
            


        mle_fitted = df.full_model.values[0].utility
        if len(df)>=minPoints and len(df.nobs.values[0]) < 100:
            ax[2,r].text(0.1, 0.5, 'Trials count under 100')
#        ax[0,r].plot(np.linspace(mRange[0], mRange[1], 100), function_fitted(np.linspace(0,1, 100)), color='dark'+col, alpha=0.5)
        ax[1,r].plot(np.linspace(mRange[0], mRange[1], 100), mle_fitted(np.linspace(0,1, 100)), color='dark'+col, alpha=0.5)
        
        yy = convertRange(d2(mle_fitted(np.linspace(0,1, 100))), 
                         [min(d2(mle_fitted(np.linspace(0,1, 100)))), max(d2(mle_fitted(np.linspace(0,1, 100))))],
                         [min(y), max(y)])
        
        ax[2,r].plot( np.linspace(mRange[0], mRange[1], 100),
          yy, color='dark'+col)
        ax3[0,2].plot(np.linspace(mRange[0], mRange[1], 100),
              d2(mle_fitted(np.linspace(0,1, 100))), color='dark'+col, alpha=0.1)

        ax2[0,1].plot(np.linspace(mRange[0], mRange[1], 100), mle_fitted(np.linspace(0,1, 100)), color='dark'+col, alpha=0.1)
        r += 1

    ax2[0,0].grid(); ax2[0,1].grid()
    ax2[0,0].set_title('fitted fractiles')
    ax2[0,1].set_title('mle fitting')
    ax2[0,0].set_xlabel('certainty equivalent')
    ax2[0,1].set_xlabel('certainty equivalent')
    ax2[0,0].set_ylabel('utility')
    ax2[0,1].set_ylabel('utility')
    squarePlot(ax2[0,0])
    squarePlot(ax2[0,1])
    plt.suptitle('Aggregate Fittings')
    
    ax3[0,0].grid(); ax2[0,1].grid()
    ax3[0,0].set_title('fitted fractiles')
    ax3[0,1].set_title('mle fitting')
    ax3[0,0].set_xlabel('certainty equivalent')
    ax3[0,1].set_xlabel('certainty equivalent')
    ax3[0,0].set_ylabel('utility d2')
    ax3[0,1].set_ylabel('utility d2')
    squarePlot(ax3[0,0])
    squarePlot(ax3[0,1])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#%%
def plot_avg_functions(MLE_df, minTrials = 40):
    '''
    '''
    from scipy import stats
    from macaque.f_models import define_model

    dataList, legend, ranges, c_specific = extract_parameters(MLE_df, dataType = 'mle', minTrials = minTrials)
        
    #%%
    from macaque.f_uncertainty import bootstrap_sample, bootstrap_function
    from scipy.stats import norm
    
    fig, ax = plt.subplots( 2, 3, squeeze = False, figsize=(10,6))
    ff = define_model(MLE_df.model_used.iloc[-1])
    p0 = ff[1]
    utility = ff[-1](p0)['empty functions']['utility']
    softmax = ff[-1](p0)['empty functions']['pChooseA']
    probability = ff[-1](p0)['empty functions']['probability']

    for params, rr in zip(dataList, np.sort(unique_listOfLists(ranges), 0)):
        uu = lambda pp: utility(np.linspace(0,1,100), pp)
        mean, lower, upper = bootstrap_function(uu, params[:,2:4], 'mean')
        ax[0,0].plot(np.linspace(rr[0],rr[1],100), mean)
        ax[0,0].fill_between(np.linspace(rr[0],rr[1],100), y1=lower, y2=upper, alpha=0.25)
        mean, lower, upper = bootstrap_function(uu,  params[:,2:4], 'median')
        ax[1,0].plot(np.linspace(rr[0],rr[1],100), mean)
        ax[1,0].fill_between(np.linspace(rr[0],rr[1],100), y1=lower, y2=upper, alpha=0.25)
        
    for params, rr in zip(dataList, np.sort(unique_listOfLists(ranges), 0)):
        sm = lambda pp: 1 / (1 + np.exp( -pp[0] * ( (np.linspace(-0.5,0.5,100) - pp[1]) )  ))
        mean, lower, upper = bootstrap_function(sm,  params[:,0:2], 'mean')
        ax[0,1].plot(np.linspace(-0.5,0.5,100), mean)
        ax[0,1].fill_between(np.linspace(-0.5,0.5,100), y1=lower, y2=upper, alpha=0.25)
        mean, lower, upper = bootstrap_function(sm,  params[:,0:2], 'median')
        ax[1,1].plot(np.linspace(-0.5,0.5,100), mean)
        ax[1,1].fill_between(np.linspace(-0.5,0.5,100), y1=lower, y2=upper, alpha=0.25)
        
    if np.size(dataList[0],1) > 4:    
        for params, rr in zip(dataList, np.sort(unique_listOfLists(ranges), 0)):
            prob = lambda pp: probability(np.linspace(0,1,100), pp)
            mean, lower, upper = bootstrap_function(prob,  params[:,-1], 'mean')
            ax[0,2].plot(np.linspace(0,1,100), mean)
            ax[0,2].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25)
            mean, lower, upper = bootstrap_function(prob,  params[:,-1], 'median')
            ax[1,2].plot(np.linspace(0,1,100), mean)
            ax[1,2].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25)

    ax[0,0].set_xlabel('reward magnitude')
    ax[0,0].set_ylabel('mean utility')
    squarePlot(ax[0,0])
    
    ax[1,0].set_xlabel('reward magnitude')
    ax[1,0].set_ylabel('median utility')
    squarePlot(ax[0,1])
    
    ax[0,1].set_xlabel('Δ value')
    ax[0,1].set_ylabel('mean pChA')
    squarePlot(ax[0,0])
    
    ax[1,1].set_xlabel('Δ value')
    ax[1,1].set_ylabel('median pChA')
    squarePlot(ax[0,1])
    
    ax[0,2].set_xlabel('reward probability')
    ax[0,2].set_ylabel('mean probability distortion')
    ax[0,2].legend(legend)
    ax[0,2].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,0])
    
    ax[1,2].set_xlabel('reward probability')
    ax[1,2].set_ylabel('median probability distortion')
    ax[1,2].legend(legend)
    ax[1,2].plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,1])
    
    plt.suptitle('fitted functions')
    
#%%
def plot_avg_utilities(MLE_df, dataType='mle', metric = 'mean', minTrials = 40):
    '''
    '''
    from scipy import stats
    from macaque.f_models import define_model
    dataList, legend, ranges, c_specific = extract_parameters(MLE_df, dataType = dataType, minTrials = minTrials)
    if dataType == 'mle':
        dataList = [ff[:,2:4] for ff in dataList]
        
    #%%
    from macaque.f_uncertainty import bootstrap_sample, bootstrap_function
    
    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(10,6))
    ff = define_model(MLE_df.model_used.iloc[-1])
    p0 = ff[1]
    utility = ff[-1](p0)['empty functions']['utility']
    i = 0
    for params, rr in zip(dataList, np.sort(unique_listOfLists(ranges), 0)):
        uu = lambda pp: utility(np.linspace(0,1,100), pp)
        mean, lower, upper = bootstrap_function(uu, params, 'mean')
        ax[0,0].plot(np.linspace(0,1,100), mean)
        ax[0,0].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25)
        mean, lower, upper = bootstrap_function(uu, params, 'median')
        ax[0,1].plot(np.linspace(0,1,100), mean)
        ax[0,1].fill_between(np.linspace(0,1,100), y1=lower, y2=upper, alpha=0.25)

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
    reference= np.sort(unique_listOfLists(ranges), 0)
    
    fig, ax = plt.subplots( 1, 2, squeeze = False, figsize=(10,6))
    i = np.sort(unique_listOfLists(ranges), 0)
    mean, _, _ = bootstrap_function(uu, dataList[1], 'mean')
#    mean = mean/1.3
    top = mean[int(((min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:])) / (max(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:]))*100))]
    bottom = mean[0]
    
    mean, _, _ = bootstrap_function(uu, dataList[1], 'median')
#    mean = mean/1.3
    top2 = mean[int(((min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:])) / (max(np.vstack(np.sort(unique_listOfLists(ranges), 0))[:,1]) - min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:]))*100))]
    bottom2 = mean[0]
    
    spreads = np.diff(np.sort(unique_listOfLists(ranges), 0))
    means = np.mean(np.sort(unique_listOfLists(ranges), 0), 1)
    
    for params, rr in zip(dataList, np.sort(unique_listOfLists(ranges), 0)):
        if np.diff(rr)==max(spreads):
            uu = lambda pp: utility(np.linspace(0,1,100), pp)
            mean, lower, upper = bootstrap_function(uu, params, 'mean')
            ax[0,0].plot(np.linspace(rr[0],rr[1],100), mean)
            ax[0,0].fill_between(np.linspace(rr[0],rr[1],100), y1=lower, y2=upper, alpha=0.25)
            mean, lower, upper = bootstrap_function(uu, params, 'median')
            ax[0,1].plot(np.linspace(rr[0],rr[1],100), mean)
            ax[0,1].fill_between(np.linspace(rr[0],rr[1],100), y1=lower, y2=upper, alpha=0.25)
        elif np.diff(rr)<max(spreads) and np.mean(rr) == min(means):
            if rr[0] != reference[1].min():
                uu = lambda pp: utility(np.linspace(0,1,100), pp)
                mean, lower, upper = bootstrap_function(uu, params, 'mean')
                bottom = -mean[int(((min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:]) - min(rr)) / (max(rr) - min(rr))*100))]
#                bottom = bottom / 1.3 
                bottom = (bottom*(top-0))+0

                ax[0,0].plot(np.linspace(rr[0],rr[1],100), (mean*(top-bottom))+bottom)
                ax[0,0].fill_between(np.linspace(rr[0],rr[1],100), y1=(lower*(top-bottom))+bottom, y2=(upper*(top-bottom))+bottom, alpha=0.25)
                
                mean, lower, upper = bootstrap_function(uu, params, 'median')
                bottom2 = -mean[int(((min(np.vstack(np.sort(unique_listOfLists(ranges), 0))[1,:]) - min(rr)) / (max(rr) - min(rr))*100))]
#                bottom2 = bottom2/1.3
                bottom2 = (bottom2*(top-0))+0
                ax[0,1].plot(np.linspace(rr[0],rr[1],100), (mean*(top2-bottom2))+bottom2)
                ax[0,1].fill_between(np.linspace(rr[0],rr[1],100), y1=(lower*(top2-bottom2))+bottom2, y2=(upper*(top2-bottom2))+bottom2, alpha=0.25)
            else:
                uu = lambda pp: utility(np.linspace(0,1,100), pp)
                mean, lower, upper = bootstrap_function(uu, params, 'mean')
                ax[0,0].plot(np.linspace(rr[0],rr[1],100), (mean*(top-bottom))+bottom)
                ax[0,0].fill_between(np.linspace(rr[0],rr[1],100), y1=(lower*(top-bottom))+bottom, y2=(upper*(top-bottom))+bottom, alpha=0.25)
                mean, lower, upper = bootstrap_function(uu, params, 'median')
                ax[0,1].plot(np.linspace(rr[0],rr[1],100), (mean*(top2-bottom2))+bottom2)
                ax[0,1].fill_between(np.linspace(rr[0],rr[1],100), y1=(lower*(top2-bottom2))+bottom2, y2=(upper*(top2-bottom2))+bottom2, alpha=0.25)
        elif np.diff(rr)<max(spreads) and np.mean(rr) == max(means):
            uu = lambda pp: utility(np.linspace(0,1,100), pp)
            mean, lower, upper = bootstrap_function(uu, params, 'mean')
            ax[0,0].plot(np.linspace(rr[0],rr[1],100), (mean*(1.0-top))+top)
            ax[0,0].fill_between(np.linspace(rr[0],rr[1],100), y1=(lower*(1.0-top))+top, y2=(upper*(1.0-top))+top, alpha=0.25)
            mean, lower, upper = bootstrap_function(uu, params, 'median')
            ax[0,1].plot(np.linspace(rr[0],rr[1],100), (mean*(1.0-top2))+top2)
            ax[0,1].fill_between(np.linspace(rr[0],rr[1],100), y1=(lower*(1.0-top2))+top2, y2=(upper*(1.0-top2))+top2, alpha=0.25)

    ax[0,0].set_xlabel('reward magnitude')
    ax[0,0].set_ylabel('mean utility')
    ax[0,0].legend(legend)
    ax[0,0].plot(np.linspace(reference[1].min(),reference[1].max(),100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,0])
    
    ax[0,1].set_xlabel('reward magnitude')
    ax[0,1].set_ylabel('median utility')
    ax[0,1].legend(legend)
    ax[0,1].plot(np.linspace(reference[1].min(),reference[1].max(),100), np.linspace(0,1,100), '--', color='k')        
    squarePlot(ax[0,1])
    plt.suptitle('overlapping utilities')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#%%
def filter_byMidpoints(softmaxDF):
    '''
    '''
    midpoints = []
    gaps = []
    secondaries = softmaxDF.secondary.values
    primaries = softmaxDF.primary.values
    for primary, secondary in zip(primaries, secondaries):
        secondary = np.vstack(secondary)[:, 0]
        primary = np.array([primary[0]] * len(secondary))
        midpoints.append( np.round(np.mean([primary, secondary], axis=0), decimals=3))
        gaps.append(np.round(np.abs(primary - secondary), decimals=3))
    softmaxDF['midpoint'] = midpoints
    softmaxDF['gap'] = gaps
    softmaxDF['iTrials'] = [ str(np.sort(np.concatenate(list(val.values()))))for val in softmaxDF.get('trial_index').values]

    if ('division' in softmaxDF.columns):
        softmaxDF.drop_duplicates(
            subset=['iTrials', 'division', 'sessionDate'],
            keep='first',
            inplace=True)
        softmaxDF.sort_values(
            by=['sessionDate', 'division', 'primaryEV'], inplace=True)
    else:
        softmaxDF.drop_duplicates( subset=['iTrials', 'sessionDate'], keep='first', inplace=True)
        softmaxDF.sort_values(by=['sessionDate', 'primaryEV'], inplace=True)

    try:
        f_index = np.unique(
            np.concatenate(
                (softmaxDF.loc[softmaxDF.gList.str.contains(
                    'random')].index.values, softmaxDF.loc[
                        softmaxDF.gList.str.contains('rand')].index.values,
                 softmaxDF.loc[softmaxDF.gList.str.contains('rU')].index.values
                )))
        softmaxDF = softmaxDF.iloc[f_index].copy()
    except:
        pass

    return softmaxDF


#%%
def gap_random(df, repeated_midpoints, axis=None, subplace=None):
    '''
    get the collection of points calculated from softmax-fitted gap sequences
    '''
    import matplotlib.cm as cm
    import scipy.optimize as opt

    norm = lambda x: (x - min(x)) / (max(x) - min(x))
    sigmoid = lambda x, p2: np.array(1 / (1 + np.exp(-(x - 0) / p2)))
    xx = np.linspace(-0.03, max(df.gap) + 0.01)

    colors = cm.rainbow(np.linspace(
        0, 1, len(repeated_midpoints)))  #UNIQUE LIST OF LIST!!!

    gaps_all = []
    pChoice_all = []
    popt_all = []
    midpoint_all = []
    for col, point in zip(colors, repeated_midpoints):
        subdf = df.loc[df.midpoint == point].sort_values(
            'gap')  #find index for specfic option1 gamble
        gaps = subdf.gap.values
        pChoice = []  #probability of choosing the higher value option
        for _, row in subdf.iterrows():
            if row.primaryEV > row.secondaryEV:
                pChoice.extend([1 - row.freq_sCh])
            else:
                pChoice.extend([row.freq_sCh])
        param_bounds = ([0.01], [1])
        popt, pcov = opt.curve_fit(
            sigmoid, gaps, pChoice, p0=[1], method='trf', bounds=param_bounds)

        if axis is not None:
            axis[0, subplace].plot(
                xx, sigmoid(xx, popt), color=col, linestyle='--')
            axis[0, subplace].scatter(gaps, pChoice, color=col)
            axis[0, subplace].grid(True)
            axis[0, subplace].vlines(0, 0, 1, color='red')

        gaps_all.append(gaps)
        pChoice_all.append(pChoice)
        popt_all.append(popt)
        midpoint_all.append(point)
    if popt_all == [] or len(popt_all) < 2:
        pass

    return gaps_all, pChoice_all, popt_all, midpoint_all

#%%


def precise_range(x):
    '''
    '''
    if x[0] < 0.4:
        x[0] = 0.0
    elif x[0] > 0.4:
        x[0] = 0.5

    if x[1] > 0.6:
        x[1] = 1.0
    elif x[1] < 0.6:
        x[1] = 0.5
    return x


#%%


def fit_parametricModel(filteredDF, Trials, Models = [], plotTQDM=True, plotFitting=False):
    '''
    '''
    #some of the pvalues dont work for the rpelec fit
    from macaque.f_models import get_modelLL, trials_2fittable, LL_fit

    sTrials = filteredDF.getTrials(Trials)
    if Models == []:
        Models = ['utility-specific', 'utility-scdf', 'utility-prelec']

    np.warnings.filterwarnings('ignore')
    mle_list=[]
    for Model in Models:
        dList = [];
        for date in tqdm(
                sTrials.sessionDate.unique(), desc=Model, disable=not plotTQDM):
            tt = sTrials.loc[sTrials['sessionDate'] == date]

            X, Y = trials_2fittable(tt)
            MLE = LL_fit(Y, X, model = Model).fit(disp=False)

            safes = [
                A if side[0] == 1 else B
                for A, B, side in zip(tt.gambleA, tt.gambleB, tt.outcomesCount)
            ]
            safes = np.array(safes)[:, 0]
            mLow = safes.min(
            )  #min(np.concatenate(Trials[['GA_ev','GB_ev']].values))
            mMax = safes.max(
            )  #max(np.concatenate(Trials[['GA_ev','GB_ev']].values))

            context = tt.trialSequenceMode.unique()
            dList.append({
                'date': MLE.model.sessionDate,
                'nTrials': MLE.nobs,
                'params': MLE.params,
                'pvalues': MLE.pvalues,
                'NM_success': MLE.mle_retvals['converged'],
                'model_used': MLE.model.model_name,
                'LL': MLE.llf,
                'context': np.unique(context)[-1],
                'm_range': [mLow, mMax],
                'pNames': MLE.model.exog_names,
                'Nfeval': MLE.Nfeval,
                'all_fits': MLE.res_x,
                'full_model': MLE,
                'AIC': MLE.aic,
                'BIC': MLE.bic
            })

        MLE_fits = MLE_object(pd.DataFrame(dList))
        from macaque.f_probabilityDistortion import plot_MLEfitting
        plot_MLEfitting(MLE_fits, plotFittings=False)

        mle_list.append(MLE_fits)
    return mle_list

class MLE_object(pd.DataFrame):
    '''
    '''
    @property
    def _constructor(self):
        return MLE_object

    def plotSoftmax(self):
        ax = plt.gca()
        plotData = np.vstack(( np.linspace(0,1.2,100), np.ones(100), np.zeros(100),
            np.zeros(100), np.linspace(1.2,0,100), np.ones(100),
            np.zeros(100), np.zeros(100))).T
        for xx in self.full_model.apply(lambda x: x.softmax).values:
            ax.plot(plotData[:,0] - plotData[:,-4], xx, color = 'k', alpha = 0.2)
        params = self.params.mean()
        functions = self.full_model.apply(lambda x: x.model.model_parts)[0](params)
        ax.plot(plotData[:,0] - plotData[:,-4], functions['prob_chA'](plotData), color = 'r', linewidth = 3)
        ax.grid(which='both',axis='x')

    def plotUtility(self):
        ax = plt.gca()
        plotData = np.linspace(0,1.2,100)
        for xx in self.full_model.apply(lambda x: x.utility).values:
            ax.plot(plotData, xx, color = 'k', alpha = 0.2)
        params = self.params.mean()
        functions = self.full_model.apply(lambda x: x.model.model_parts)[0](params)
        ax.plot(plotData, functions['utility'](plotData), color = 'r', linewidth = 3)
        ax.plot(plotData, plotData, '--', color = 'k', linewidth = 3)

    def plotProbability(self):
        ax = plt.gca()
        plotData = np.linspace(0,1,100)
        for xx in self.full_model.apply(lambda x: x.probability).values:
            ax.plot(plotData, xx, color = 'k', alpha = 0.2)
        params = self.params.mean()
        functions = self.full_model.apply(lambda x: x.model.model_parts)[0](params)
        ax.plot( plotData,  functions['probability_distortion'](plotData), color = 'r', linewidth = 3)
        ax.plot(plotData, plotData, '--', color = 'k', linewidth = 3)


#%%
def fractile_RT():
    '''
    '''
    # ---------------------------------------------------------------------
    responseDF = filteredDF.get_RTs()
    fractiles = flatten([[seq] * nTrials for nTrials, seq in zip(filteredDF.nTrials, filteredDF.fractile)])
    responseDF['fractile'] = fractiles

    utilities = flatten([[seq] * nTrials for nTrials, seq in zip(filteredDF.nTrials, filteredDF.utility)])
    responseDF['utility'] = utilities
    # --------------------------------------------------------------------

    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(10, 4))
    responseDF.groupby(by=['deltaEV', 'fractile']).mean()['RTs'].unstack().plot(
        ax=ax[0, 1],
        grid=True)
    ax[0, 1].legend(['first', 'second', 'third'])
    responseDF.groupby(by=['chosenEV', 'fractile']).mean()['RTs'].unstack().plot(
        ax=ax[0, 2],
        grid=True)
    ax[0, 2].legend(['first', 'second', 'third'])
    responseDF.groupby(by=['fractile']).mean()['RTs'].plot(
        kind='bar',
        ax=ax[0, 0],
        grid=False)
    ax[0, 0].set_xticklabels(['first', 'second', 'third'])
    ax[0, 0].set_ylabel('response time')
    # --------------------------------------------------------------------

    #get the trials where gambles were chosen
    gChosen = responseDF.loc[responseDF.chosenSide == responseDF.primarySide]
    #get the trials where gambles were chosen
    sChosen = responseDF.loc[responseDF.chosenSide != responseDF.primarySide]

    gChosen.groupby(by=['chosenEV', 'fractile']).mean()['RTs'].unstack().plot()
    gChosen.groupby(by=['chosenEV', 'fractile']).mean()['RTs'].unstack().boxplot()
    gChosen.groupby(by=['chosenEV', 'fractile']).mean()['RTs'].unstack().hist(bins=20)

    scatter_matrix(gChosen, alpha=0.2, figsize=(6, 6), diagonal='kde')
    sns.pairplot(gChosen, hue='fractile', size=2.5)
    sns.pairplot(gChosen.groupby(by=['chosenEV', 'fractile']).mean()['RTs'].unstack(), hue='fractile', size=2.5)

#%%
from types import MethodType
class utilityDF(pd.DataFrame):
    @property
    def _constructor(self):
        return utilityDF

    def get_fractiles(self,  minSecondaries = 3, minChoices = 4):
        from macaque.f_choices import get_options, get_psychData
        from macaque.f_psychometrics import get_softmaxData
        fractChoices = get_options(self.loc[self.trialSequenceMode == 9050], mergeBy = 'block',
                                   byDates = True,
                                   mergeSequentials=False)
        filteredDF = get_softmaxData(fractChoices, metricType = 'CE', minSecondaries = 4, minChoices = 4)
        filteredDF = get_fractileUtility(filteredDF) #this adds the fractile number, and the utility measure to softmaxDF

        filteredDF.plotUtility = MethodType(plot_fractileUtilities, filteredDF )
        return filteredDF

    def get_equivariates(self, minSecondaries = 3, minChoices = 4):
        from macaque.f_choices import get_options, get_psychData
        from macaque.f_psychometrics import get_softmaxData
        listChoices = get_options(self.loc[self.trialSequenceMode == 9020], mergeBy = 'block',
                                   byDates = True,
                                   mergeSequentials=True)
        filteredDF = get_softmaxData(listChoices, metricType = 'CE', minSecondaries = 3, minChoices = 4)
        filteredDF.plotUtility = MethodType( f_utility.get_binaryUtility, filteredDF )
        return filteredDF

    def get_random(self, minPoints = 3, minSecondaries = 3, minChoices = 6):
        from macaque.f_choices import get_options, get_psychData
        from macaque.f_psychometrics import get_softmaxData
        listChoices = get_options(self.loc[self.trialSequenceMode == 9020], mergeBy = 'block',
                                   byDates = True,
                                   mergeSequentials=True)
        rChoices = get_psychData(listChoices, metricType = 'trans', transitType = 'safes')
        filteredDF = get_softmaxData(rChoices, metricType = 'trans', minSecondaries = 0, minChoices = 4)
        filteredDF.plotUtility = MethodType( get_randomUtility, filteredDF )
        return filteredDF
#        multiGAP, uniGAP = f_utility.get_randomUtility(rChoices, minPoints = 3, minSecondaries = 3, minChoices = 6, plotit = True)

    def get_Merged(self, minPoints = 3, minSecondaries = 3, minChoices = 6):
        from macaque.f_choices import get_options, get_psychData
        from macaque.f_psychometrics import get_softmaxData
        listChoices = get_options(self)
#        rChoices = get_psychData(listChoices, metricType = 'trans', transitType = 'safes')
        filteredDF = get_softmaxData(rChoices, metricType = 'trans', minSecondaries = 0, minChoices = 4)
        filteredDF.plotUtility = MethodType( get_randomUtility, filteredDF )
        return filteredDF