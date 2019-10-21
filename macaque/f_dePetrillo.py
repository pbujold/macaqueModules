import pandas as pd

data = pd.read_csv(r"C:\Users\phbuj\Google Drive\Github\PhD_Analysis\DePetrillo.csv")

#%%
data['Trial_type'].loc[data.Trial_type.apply(lambda x: x=='neutral')] = 0.5
data['Trial_type'].loc[data.Trial_type.apply(lambda x: x=='dis')] = 0.333
data['Trial_type'].loc[data.Trial_type.apply(lambda x: x=='adv')] = 0.666

#%% Remake into new table of X and Y based on food and presentation

def split_conditions(data, selector = 'Subject')
    '''
    selector can also be 'Subject', 'Cigarette_N', 'BMI', 'Age', 'Sex'
    '''
    selector = 'Subject'
    # 
    
    splitters = unique_listOfLists(data[['Condition','Reward_type']].values)
    
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
        
    return subTable

#%%
def fit_randomUtility(subTable, model= 'logit-power-1prelec-cumm'):
    """
    """  
    from macaque.f_models import trials_2fittable, LL_fit
    
    if type(model) == list:
        dList=[]
        for mm in model:
            dList.append(fit_randomUtility(subTable, model= mm))
        return dList
    else:
        dList=[];
    
    for to_convert in tqdm(subTable):
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
        
        MLE = LL_fit(Y, X.astype(float), model = model, fixedRange=False).fit(disp=False)
        
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
    
    return results
    
#%%
    
def compare_fittedModels(MLE_list, comparison='BIC', selection = 'mean'):
    '''
    '''
    from macaque.f_Rfunctions import oneWay_rmAnova
    import statsmodels.api as sm
    from scipy import stats
    
    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((1, 3), (0, 2))

    forPlot = []
    Xs = []; BIC = []; AIC = []; LL = []
    realXs = []; dates=[]; names=[]; functions=[]
    for i,mle in enumerate(MLE_list):
        BIC.extend(mle.BIC.values)
        AIC.extend(mle.AIC.values)
        LL.append(mle.LL.values)
        Xs.extend(np.random.normal(i, 0.08, size=len(mle)))
        realXs.extend([i]*len(mle))
        dates.append(mle.condition.values)
        names.extend(mle.model_used.unique())
        functions.extend(mle.model_used.values)

    if comparison.lower() == 'bic':
        results = BIC
    elif comparison.lower() == 'aic':
        results = AIC
    else:
        results == LL
    sb.boxplot(realXs, results, ax=ax2, color='white', saturation=1, width=0.5)

    [ax2.plot(Xs[n::len(MLE_list)+1], results[n::len(MLE_list)+1], alpha = 0.75, color = 'grey') for n in range(len(MLE_list[-1]))]

    plt.setp(ax2.artists, edgecolor = 'k', facecolor='w')
    # plt.setp(ax2.lines, color='k')
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

    print('\n================================================================================')
    print('PARAMETRIC')
    print('\n----------------------------------------------------------')
    rmAnova = oneWay_rmAnova(results, np.concatenate(dates), functions)

    data = []
    for ff in np.unique(functions):
        where = np.array(functions) == ff
        data.append(np.array(results)[where])
    data = np.array(data)
    
    print('\n================================================================================')
    print('NON PARAMETRIC')
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
    
    
if plotit: 
    for n in range(len(results)):
        results.iloc[n].full_model.plot_fullModel()
        
    