# -*- coding: utf-8 -*-
"""
load_monkey : load_monkey(mCode)
    returns: *trials, dates*
get_short : get_short(trials)
    **
del_errorType : del_errorTrials(trials, errorType = 'all')
    .
: by_sequenceType(trials, sequenceType)
    .
: by_sequenceMode(trials, sequenceMode)
    .
: by_date(df_full, date, sequence_mode = 'all', sequence_type = 'all', filter_errors = True)
    .
by_range :
    .
: by_gamble(df_full)
    .
"""
import pandas as pd
import numpy as np
import scipy.optimize as opt
from macaque.f_toolbox import *
from macaque.f_choices import get_psychData, get_options
from scipy.stats.distributions import t
import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
plt.rcParams['svg.fonttype'] = 'none'
tqdm = ipynb_tqdm()

#%%


def get_softmaxData(choiceData,
                    metricType='ce',
                    minSecondaries=4,
                    minChoices=4,
                    plotTQDM=True):
    '''
    From a 'choiceData' dataFrame, retrieve psychometric data used in certainty/probability equivalents, choice ratios, reaction times.  \n
    **IMPORTANT**:\n
    If 'choiceData' is *divided into blocked or sequence-specific choices*, get_softmaxData returns block or sequence specific results (per day of testing).

    Parameters
    ----------
    choiceData : DataFrame
        DataFrame of psychophysics data i.e. CE or PE sequences
    metricType : string
        'CE' / 'certainty equivalent' or 'PE'/'probability equivalent' psychometric fits on the choice Data, \
        'Trans' orders without computing psychometrics
    minSecondaries : int
        Number of secondary options against which the primary is tested (e.g. safes for a single gamble)
    minChoices : int
        Number or choice made between the primary and secondary options (e.g. safes repeated n times per gamble)
    trials : None or DataFrame
        Dataframe from which original trials can be used to merge similar blocks that come one after another *(only useful for blocked choice data)*

    Returns
    ----------
    softmaxDF : DataFrame
        Returns psychometric data used to plot softmax curves, reaction times, and choice rations between gamble/safe pairs and sequences

    ----------
    future: needs to print proper Confidence Intervals
    '''

    # This is in case thedata has been divided in blocks/sequence types
    # (useful for further analysis)
    if ('division' in choiceData.columns) and (len(
            choiceData.sessionDate.unique()) > 1):
        dfs = []
        for day in tqdm(
                choiceData.sessionDate.unique(),
                desc='Computing block-based Psychophysics',
                disable=not plotTQDM):
            for div in choiceData.loc[choiceData.sessionDate ==
                                      day].division.unique():
                tailEnd = get_softmaxData(
                    (choiceData.loc[choiceData.sessionDate == day]
                     .loc[choiceData.division == div]), metricType,
                    minSecondaries, minChoices)
                if tailEnd is None:
                    continue
                else:
                    dfs.append(
                        tailEnd.assign(division=div).assign(sessionDate=day))

        softmaxDF = pd.concat(dfs, ignore_index=True)

        if metricType.lower() == 'ce' or metricType.lower() == 'certainty equivalent':
            cols = [
                'sessionDate', 'primary', 'primaryEV', 'equivalent',
                'secondary', 'secondaryEV', 'm_range', 'freq_sCh', 'pFit',
                'pSTE', 'no_of_Trials', 'nTrials', 'primarySide', 'choiceList',
                'filteredRT', 'choiceTimes', 'moveTime', 'trial_index',
                'oClock', 'func', 'metricType', 'division', 'seqCode', 'gList',
                'chosenEV'
            ]

        elif metricType.lower() == 'pe' or metricType.lower() == 'probability equivalent':
            cols = [
                'sessionDate', 'primary', 'primaryEV', 'equivalent', 'freq_sCh',
                'secondary', 'secondaryEV', 'm_range', 'pFit', 'pSTE',
                'no_of_Trials', 'nTrials', 'primarySide', 'choiceList',
                'filteredRT', 'choiceTimes', 'moveTime', 'trial_index',
                'oClock', 'func', 'metricType', 'division', 'seqCode', 'gList',
                'chosenEV'
            ]
        elif metricType.lower() == 'none':
            cols = [
                'sessionDate', 'primary', 'primaryEV', 'secondary',
                'secondaryEV', 'm_range', 'freq_sCh', 'no_of_Trials', 'nTrials',
                'primarySide', 'choiceList', 'filteredRT', 'choiceTimes',
                'moveTime', 'trial_index', 'oClock', 'metricType', 'division',
                'seqCode', 'gList', 'chosenEV'
            ]
        else:
            cols = [
                'sessionDate', 'primary', 'primaryEV', 'secondary',
                'secondaryEV', 'm_range', 'freq_sCh', 'no_of_Trials', 'nTrials',
                'primarySide', 'choiceList', 'filteredRT', 'choiceTimes',
                'moveTime', 'trial_index', 'oClock', 'metricType', 'division',
                'seqCode', 'gList', 'chosenEV'
            ]

        return psychometricDF(softmaxDF[cols])

    #-------------------------------------------------------------------------

    else:
        cols = [
            'primary', 'primaryEV', 'secondary', 'secondaryEV', 'm_range',
            'freq_sCh', 'primarySide', 'no_of_Trials', 'nTrials', 'choiceList',
            'filteredRT', 'choiceTimes', 'moveTime', 'trial_index', 'oClock',
            'metricType', 'chosenEV'
        ]
        #        softmaxDF = pd.DataFrame(columns=cols)
        dfs = []

        psychData = get_psychData(choiceData, metricType, transitType='None')
        unique_options = unique_listOfLists(psychData.option1)
        for option in unique_options:
            # find index for specfic option1 gamble
            index = psychData['option1'].apply(lambda x: x == option)
            mags = []
            igg = {}
            trialType = []

            # here we define different secondary gambles from their magnitudes
            #   LOOK HERE FOR THE ISSUE OF != 2
            if psychData.loc[index].loc[psychData.loc[index].option2.apply(
                    lambda x: len(x)) != 2].option2.values.tolist() != []:
                gg = psychData.loc[index].loc[
                    psychData.loc[index].option2.apply(lambda x: len(x)) !=
                    2].option2.apply(lambda x: [x[0], x[2]])
                mags, igg = unique_listOfLists(gg, returnIndex=True)
                for nn in mags:
                    igg[tuple(nn)] = gg.iloc[igg[tuple(nn)]].index
                trialType = mags[:]

            # here we define safe secondary options as unique
            if psychData.loc[index].loc[psychData.loc[index].option2.apply(
                    lambda x: len(x)) == 2].index.tolist() != []:
                listy = psychData.loc[index].loc[psychData.loc[
                    index].option2.apply(lambda x: len(x)) == 2].option2.apply(
                        lambda x: x[0])
                mags.append([min(listy), max(listy)])  # add the safes to this
                igg[tuple([
                    min(listy), max(listy)
                ])] = psychData.loc[index].loc[psychData.loc[
                    index].option2.apply(lambda x: len(x)) == 2].index.tolist()
                trialType.append(['safe'])

            for m_range, tt in zip(mags, trialType):
                # make series of trial numbers for minChoices filter
                choiceRepeats = psychData.loc[igg[tuple(
                    m_range)]].no_of_Trials.values.tolist()

                if len([lens for lens in choiceRepeats if lens >= minChoices
                       ]) >= minSecondaries:
                    # condition to evaluate the options
                    # import pdb ; pdb.set_trace() #AWESOME WAY TO DEBUG
                    subDf = psychData.loc[igg[tuple(m_range)]].loc[
                        psychData.loc[igg[tuple(m_range)]].no_of_Trials >=
                        minChoices].sort_values('option2')
                    if np.size(subDf) == 0:
                        continue

                    if tt != ['safe']:
                        # look at the magnitude in option 2 fields
                        marker = [m[-1] for m in subDf.option2]
                    else:
                        marker = [m[0] for m in subDf.option2]

                    try:
                        seq = int(subDf.seqType.unique()[0])
                        gList = subDf.gList.unique()[0]
                    except BaseException:
                        seq = []
                        gList = []

                    dfs.append(
                        pd.DataFrame({
                            'primary': [
                                flatten(
                                    unique_listOfLists(
                                        subDf.option1.values.tolist()))
                            ],
                            'primaryEV':
                            np.unique(subDf.G1_ev.values.tolist()).tolist(),
                            'secondary': [subDf.option2.values.tolist()],
                            'secondaryEV':
                            [np.unique(subDf.G2_ev.values.tolist()).tolist()],
                            'm_range': [m_range],
                            'freq_sCh': [(subDf.chose2 /
                                          subDf.no_of_Trials).values.tolist()],
                            'no_of_Trials':
                            [subDf.no_of_Trials.values.tolist()],
                            'nTrials':
                            [sum(subDf.no_of_Trials.values.tolist())],
                            'choiceList': [{
                                key: value for key, value in zip(
                                    marker, subDf.choiceList.values.tolist())
                            }],
                            'choiceTimes': [{
                                key: value for key, value in zip(
                                    marker, subDf.choiceTimes.values.tolist())
                            }],
                            'filteredRT': [{
                                key: value for key, value in zip(
                                    marker, subDf.filteredRT.values.tolist())
                            }],
                            'moveTime': [{
                                key: value for key, value in zip(
                                    marker, subDf.moveTime.values.tolist())
                            }],
                            'trial_index': [{
                                key: value for key, value in zip(
                                    marker, subDf.trial_index.values.tolist())
                            }],
                            'oClock': [{
                                key: value for key, value in zip(
                                    marker, subDf.oClock.values.tolist())
                            }],
                            'primarySide': [{
                                key: value for key, value in zip(
                                    marker, subDf.side_of_1.values.tolist())
                            }],
                            'metricType': [metricType.upper()],
                            'seqCode': [seq],
                            'gList': [gList],
                            'chosenEV': [{
                                key: value for key, value in zip(
                                    marker, subDf.chosenEV.values.tolist())
                            }]
                        }))

        if dfs == []:
            softmaxDF = pd.DataFrame(columns=cols)
        else:
            softmaxDF = pd.concat(dfs, ignore_index=True)
        if softmaxDF.empty:
            return None

        if metricType.lower() == 'ce' or metricType.lower(
        ) == 'certainty equivalent' or metricType.lower(
        ) == 'pe' or metricType.lower() == 'probability equivalent':
            cols = [
                'primary', 'primaryEV', 'equivalent', 'secondary',
                'secondaryEV', 'm_range', 'freq_sCh', 'pFit', 'pSTE',
                'primarySide', 'no_of_Trials', 'nTrials', 'choiceList',
                'filteredRT', 'choiceTimes', 'moveTime', 'trial_index',
                'oClock', 'func', 'metricType', 'seqCode', 'gList', 'chosenEV'
            ]
            softmaxDF = fit_softmax(softmaxDF, metricType)
        elif metricType.lower() == 'trans' or metricType.lower(
        ) == 'transitivity':
            cols = [
                'primary', 'primaryEV', 'secondary', 'secondaryEV', 'm_range',
                'freq_sCh', 'primarySide', 'no_of_Trials', 'nTrials',
                'choiceList', 'filteredRT', 'choiceTimes', 'moveTime',
                'trial_index', 'oClock', 'metricType', 'seqCode', 'gList',
                'chosenEV'
            ]
#        import pdb ; pdb.set_trace() #AWESOME WAY TO DEBUG
        return psychometricDF(softmaxDF[cols])


#%%
def fit_softmax(softmaxDF, metricType='CE'):
    '''
    '''
    np.warnings.filterwarnings('ignore')
    dList = []

    # sigmoid = lambda x, p1, p2: 1/(1+ np.exp( -(1/p2) * (x - p1) ) )
    # #logistic sigmoid function

    # logistic sigmoid function (SAME AS ABOVE)
    def sigmoid(x, p1, p2):
        return np.array(1 / (1 + np.exp(-(x - p1) / p2)))

    #softmax = lambda x, p1, p2: np.exp( -(x - p1)/p2 ) / np.exp( -(x - p1)/p2 ) + np.exp( -(x - p1)/p2 )

    for i, situation in (softmaxDF.iterrows()):
        if metricType.lower() == 'ce':
            x_mag = [m[0] for m in situation.secondary]
            #            primary = situation.primary[::2]
            chX_freq = situation.freq_sCh  # need the frequency of picking safe, not gamble
        elif metricType.lower() == 'pe':
            x_mag = [g[3] for g in situation.secondary]
            #            primary = flatten([g[::2] for g in situation.secondary])
            chX_freq = situation.freq_sCh  # need the frequency of picking safe, not gamble

        # define p0 as dynamic points of entry
        p0 = [max(x_mag) / 2, 0.015]

        try:  # get the modelling done, but have a catch if it doesn't work
            #            param_bounds=([min(primary), -np.inf],[max(primary), np.inf])
            pFit, pCov = opt.curve_fit(
                sigmoid, x_mag, chX_freq, p0=p0,
                method='trf')  # , bounds=param_bounds)
            # standard error on parameters
            pSTE = np.sqrt(np.diag(pCov).tolist())
            dList.append({
                'pFit': pFit.tolist(),
                'pSTE': pSTE.tolist(),
                'func': sigmoid,
                'equivalent': pFit[0]
            })
        except RuntimeError:
            dList.append({
                'pFit': np.nan,
                'pSTE': np.nan,
                'func': sigmoid,
                'equivalent': np.nan
            })
    return pd.concat(
        [softmaxDF, pd.DataFrame(dList)], axis=1, join_axes=[softmaxDF.index])


#%%


def plot_softmax(softmaxDF,
                 sortBy='primaryEV',
                 printRatios=True,
                 plot_ci='fit',
                 color=None):
    '''
    From a softmax dataFrame, plot the softmax curves either individually or all at once.

    Parameters
    ----------
    softmaxDF : DataFrame
        DataFrame of psychophysics data i.e. CE or PE sequences
    info : DataFrame
        Contains the information thta we can plot about the day's session (ml rank, choice percentages, etc...)

    Returns
    ----------
    Plots softmax-normalized sigmoid curves that fit to the choice bahviour of the animals

    ----------
    future: needs to print proper Confidence Intervals
    '''
    from operator import add
    from operator import sub
    import matplotlib.cm as cm

    def get_range(sm):
        softmaxType = np.unique(sm['metricType'])  # plotting the model
        if softmaxType.item().lower() == 'ce':
            primary = np.concatenate([x[::2] for x in sm['primary']])
            return min(primary), max(primary)
        elif softmaxType.item().lower() == 'pe':
            return 0, 1
        else:
            raise NameError('No psychometrics possible')

    def get_XYpoints(row):
        if row.metricType.lower() == 'ce':
            x_mag = [m[0] for m in row.secondary]
            chX_freq = row.freq_sCh  # need the frequency of picking safe, not gamble
            EV = row.primaryEV
        elif row.metricType.lower() == 'pe':
            x_mag = [g[3] for g in row.secondary]
            chX_freq = row.freq_sCh  # need the frequency of picking safe, not gamble
            EV = row.secondary
        return x_mag, chX_freq, EV

    def smSubplots(sm, printRatios=printRatios, plot_ci=plot_ci, color=None):
        rows = np.ceil(len(sm) / 9)
        if len(sm) < 5:
            fig, ax = plt.subplots(
                int(rows), len(sm), squeeze=False, figsize=(10, int(rows * 4)))
            maxC = len(sm) - 1
        else:
            fig, ax = plt.subplots(
                int(rows), 9, squeeze=False, figsize=(15, int(rows * 2)))
            maxC = 8
        minX, maxX = get_range(sm)
        xx = np.linspace(minX, maxX, 100)
        c = 0
        r = 0
        # --------------------------------------
        for ind, row in sm.iterrows():
            x_mag, chX_freq, EV = get_XYpoints(row)
            func = row.func
            if color:
                col = color[c]
            else:
                col = cm.rainbow(row.primaryEV * 2)
            ax[r, c].set_title(str(row['primary']))
            # plot points of selection
            ax[r, c].plot(x_mag, chX_freq, 'bo ', color=col)
            # plots a line at the expected value
            ax[r, c].axvline(x=EV, linestyle='--', color='k', alpha=0.7)
            if printRatios:
                for m, p, nn in zip(x_mag, chX_freq, row['no_of_Trials']):
                    ax[r, c].text(
                        m + 0.015,
                        p + 0.015,
                        str(int(p * nn)) + '/' + str(int(nn)),
                        style='italic',
                        color='k',
                        alpha=0.65)
            ax[r, c].grid(b=True, which='major', axis='y')
            if not np.isnan(row['equivalent']):
                ax[r, c].plot(xx, func(xx, *row['pFit']), color=col)
                if row['primaryEV'] > np.mean((minX, maxX)):
                    ax[r, c].text(minX + 0.02, 1.0,
                                  'CE=' + str(round(row['pFit'][0], 2)))
                else:
                    ax[r, c].text(row['pFit'][0] + 0.02, 0.05,
                                  'CE=' + str(round(row['pFit'][0], 2)))
                if row['pFit'][0] > EV:
                    ax[r, c].axvline(x=row['pFit'][0], linestyle='-', color='g')
                elif row['pFit'][0] < EV:
                    ax[r, c].axvline(x=row['pFit'][0], linestyle='-', color='r')
                elif row['pFit'][0] == EV:
                    ax[r, c].axvline(x=row['pFit'][0], linestyle='-', color='k')
            ax[r, c].set_xlim(minX - 0.05, maxX + 0.05)  # x axis length
            ax[r, c].set_ylim(-0.1, 1.2)  # y axis length
            if plot_ci.lower() == 'residuals':
                bound_upper, bound_lower = softmax_CI(row, xx, method=plot_ci)
            elif plot_ci.lower() == 'resampling':
                bound_upper, bound_lower = softmax_CI(row, xx, method=plot_ci)
            elif plot_ci.lower() == 'fit':
                bound_upper = func(xx, *map(add, row['pFit'], row['pSTE']))
                bound_lower = func(xx, *map(sub, row['pFit'], row['pSTE']))
            ax[r, c].fill_between(
                xx, bound_lower, bound_upper, color=col, alpha=0.2)
            c += 1
            if c > maxC:
                c = 0
                r += 1
        #--------------------------------------------------------------
        if c < maxC and r == rows - 1:
            while c <= maxC:
                fig.delaxes(ax[r, c])
                c += 1
#        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        for axis in ax.reshape(-1):
            x0, x1 = axis.get_xlim()
            y0, y1 = axis.get_ylim()
            axis.set_aspect((x1 - x0) / (y1 - y0))
        if 'sessionDate' in sm.columns:
            plt.suptitle(str(sm['sessionDate'].unique()[0]))
        plt.show()

    # -----------------------------------------------

    if isinstance(softmaxDF, pd.core.series.Series):
        softmaxDF = softmaxDF.to_frame().transpose()

    if 'sessionDate' in softmaxDF.columns:
        for date in softmaxDF['sessionDate'].unique():
            sm = softmaxDF.loc[softmaxDF['sessionDate'] == date]
            sm.sort_values([sortBy], inplace=True)
            smSubplots(
                sm, printRatios=printRatios, plot_ci=plot_ci, color=color)
    else:
        softmaxDF.sort_values([sortBy], inplace=True)
        smSubplots(
            softmaxDF, printRatios=printRatios, plot_ci=plot_ci, color=color)
    return


#%%


def softmax_CI(softmaxDF, xx, method='resampling', n=1000):
    import numpy as np
    import scipy.optimize as opt

    #------------------------------

    def get_choiceRatio(data):
        ch = []
        chX_freq = []
        x_mag = []
        for dd in data:
            if dd[0] not in x_mag and ch == []:
                x_mag.extend(dd[0])
            elif dd[0] not in x_mag and ch != []:
                x_mag.extend(dd[0])
                chX_freq.extend([sum(ch) / len(ch)])
                ch = []
            ch.extend(dd[1])
        chX_freq.extend([sum(ch) / len(ch)])
        return x_mag, chX_freq

    #-------------------------------

    safes = []
    chS = []
    safes.extend(
        np.concatenate([
            np.repeat(item, len(values))
            for item, values in softmaxDF.choiceList.items()
        ]))
    chS.extend(
        np.concatenate(
            [values for item, values in softmaxDF.choiceList.items()]) - 1)

    data = np.array(np.split(np.array([safes, chS]), len(chS), axis=1))
    x_mag, chX_freq = get_choiceRatio(data)

    p0 = [max(x_mag) / 2, 0.015]

    # logistic sigmoid function (SAME AS ABOVE)
    def sigmoid(x, p1, p2):
        return np.array(1 / (1 + np.exp(-(x - p1) / p2)))

    # this give same result as matlab softmax
    pFit_1, pCov = opt.curve_fit(sigmoid, x_mag, chX_freq, p0=p0, method='trf')
    resid = sigmoid(x_mag, pFit_1[0], pFit_1[1]) - chX_freq
    yHat = sigmoid(x_mag, pFit_1[0], pFit_1[1])

    #    xx=xx = np.linspace(0,1,100)
    b1 = []

    #b1.append(sigmoid(xx, pFit_1[0], pFit_1[1]))
    for i in np.arange(1, n):
        if method.lower() == 'residuals':
            residBoot = np.random.permutation(resid)
            yBoot = yHat + residBoot
            # this give same result as matlab softmax
            pFit, pCov = opt.curve_fit(
                sigmoid, x_mag, yBoot, p0=p0, method='trf')
        elif method.lower() == 'resampling':
            xb = np.random.choice(range(len(data)), len(data), replace=True)
            bootSample = np.hstack(data[xb])
            bootSample = bootSample[:, np.argsort(bootSample[0])]
            bootSample = np.array(np.split(bootSample, len(data), axis=1))
            bootx, booty = get_choiceRatio(bootSample)
            try:
                # this give same result as matlab softmax
                pFit, pCov = opt.curve_fit(
                    sigmoid, bootx, booty, p0=p0, method='trf')
            except BaseException:
                continue
            if pFit[1] < 0.002:

                def sigmoid(x, p1):
                    return np.array(1 / (1 + np.exp(-(x - p1) / 0.002)))

                # this give same result as matlab softmax
                pFit, pCov = opt.curve_fit(
                    sigmoid, bootx, booty, p0=p0[0], method='trf')
                pFit = [pFit, 0.002]

                # logistic sigmoid function (SAME AS ABOVE)
                def sigmoid(x, p1, p2):
                    return np.array(1 / (1 + np.exp(-(x - p1) / p2)))

        b1.append(sigmoid(xx, pFit[0], pFit[1]))

    b1 = np.vstack(b1)
    upper, lower = np.percentile(b1, [5, 95], axis=0)

    return upper, lower


#%%


def plot_transitivity(softmaxDF):
    '''
    From a softmax dataFrame, plot the softmax curves either individually or all at once.

    Parameters
    ----------
    softmaxDF : DataFrame
        DataFrame of psychophysics data i.e. CE or PE sequences
    info : DataFrame
        Contains the information thta we can plot about the day's session (ml rank, choice percentages, etc...)

    Returns
    ----------
    Plots softmax-normalized sigmoid curves that fit to the choice bahviour of the animals

    ----------
    future: needs to print proper Confidence Intervals
    '''
    if softmaxDF.empty:
        return
    import numpy as np
    from macaque.f_toolbox import flatten

    # -------------------------------------------- where primary function starts

    if ('sessionDate' in softmaxDF.columns) and (len(
            softmaxDF.sessionDate.unique()) > 1):
        for day in softmaxDF.sessionDate.unique():
            for div in softmaxDF.seqCode.unique():
                # .sort_values(['primaryEV']))
                plot_transitivity(softmaxDF.loc[softmaxDF.sessionDate == day]
                                  .loc[softmaxDF.seqCode == div])
    else:
        # if there is a date to the softmax row, add the date to the subplot
        i = 0
        ratios = []
        indice = []
        leftAxis = []
        rightAxis = []
        lookup = []
        for index, row in softmaxDF.iterrows():
            #            np.sort(row.secondary)
            leftAxis.extend(
                np.repeat(str(row.primary), len(row.freq_sCh), axis=0).tolist())
            rightAxis.extend(row.secondary)
            for choice_ratio in row.freq_sCh:
                ratios.extend([choice_ratio - 0.5])
                indice.extend([i])
                i += 1
            lookup.extend([i])

        colors = []
        for ii, ration in enumerate(ratios):
            if ration > 0:
                colors.extend('g')
            elif ration < 0:
                colors.extend('r')
            else:
                colors.extend('k')

        fig, axarr = plt.subplots(
            figsize=(8, len(flatten(softmaxDF.freq_sCh.tolist())) / 4))
        if 'sessionDate' in softmaxDF.columns:
            axarr.set_title(
                softmaxDF.sessionDate.apply(lambda x: x.strftime("%Y-%m-%d"))
                .unique().tolist()[0] + ': division ' + str(
                    softmaxDF.seqCode.unique().tolist()[0])
            )  # this sets the subplot's title
        axarr.barh(indice, ratios, color=colors)

        axarr.axvline(x=0, linestyle='-', color='k', alpha=1)
        axarr.axvline(x=0.25, linestyle='--', color='k', alpha=0.6)
        axarr.axvline(x=-0.25, linestyle='--', color='k', alpha=0.6)
        plt.yticks(indice, leftAxis)
        axarr.set_ylim(min(indice) - 1, max(indice) + 2)  # y axis length
        plt.tight_layout()

        axarr2 = axarr.twinx()
        axarr2.barh(indice, ratios, alpha=0)
        for ii, chR, nT in zip(indice, flatten(softmaxDF.freq_sCh.tolist()),
                               flatten(softmaxDF.no_of_Trials.tolist())):
            if chR > 0.5:
                axarr2.text(
                    chR - 0.5 + 0.015,
                    ii - 0.25,
                    str(int(chR * nT)) + '/' + str(int(nT)),
                    style='italic',
                    color='k',
                    alpha=0.65,
                    fontsize='smaller')
            else:
                axarr2.text(
                    chR - 0.5 - 0.08,
                    ii - 0.25,
                    str(int(chR * nT)) + '/' + str(int(nT)),
                    style='italic',
                    color='k',
                    alpha=0.65,
                    fontsize='smaller')
        for lines in lookup:
            axarr2.axhline(y=lines - 0.5, linestyle='-', color='b', alpha=1)
        plt.yticks(indice, rightAxis)
        axarr2.set_ylim(min(indice) - 1, max(indice) + 2)  # y axis length
        axarr2.set_xlim(-0.6, 0.6)  # y axis length
        plt.tight_layout()

        plt.show()


#%%


def plot_reactionTime(softmaxDF):
    '''
    CHANGE THIS BECAUSE IT DOES NOT PRINT PROPER CONFIDENCE INTERVAL
    '''
    import scipy.stats as stats
    import numpy as np

    if ('sessionDate' in softmaxDF.columns) and (len(
            softmaxDF.sessionDate.unique()) > 1):
        for day in softmaxDF.sessionDate.unique():
            for div in softmaxDF.seqCode.unique():
                # .sort_values(['primaryEV']))
                plot_reactionTime(softmaxDF.loc[softmaxDF.sessionDate == day]
                                  .loc[softmaxDF.seqCode == div])
    else:
        if isinstance(softmaxDF, pd.core.series.Series):
            plot_Times(softmaxDF)
        else:
            softmaxDF.sort_values(['primaryEV'], inplace=True)
            fig, axarr = plt.subplots(
                len(softmaxDF),
                1,
                squeeze=False,
                figsize=(8,
                         len(softmaxDF) * 2.3))  # this is the subplot command
            ii = 0
            if ('sessionDate' in softmaxDF.columns):
                plt.suptitle(
                    softmaxDF.sessionDate.apply(
                        lambda x: x.strftime("%Y-%m-%d")).unique().tolist()[0] +
                    ': division ' + str(softmaxDF.seqCode.unique().tolist()[0]),
                    fontsize=16)
            for index, row in softmaxDF.iterrows():
                plot_Times(row, subPlace=axarr, subPlot=ii)
                ii += 1
#                plt.tight_layout()
            plt.show()


#%%


def plot_Times(situation, subPlace=None, subPlot=0):
    import scipy.stats as stats
    import numpy as np
    plt.rc('axes', axisbelow=True)

    choiceDict = situation.choiceTimes  # plotting the model
    moveDict = situation.moveTime  # plotting the model

    if np.unique([len(x) for x in situation.secondary]) == 2:
        x_mag = [m[0] for m in situation.secondary]
        EV = situation.primaryEV
    elif np.unique([len(x) for x in situation.secondary]) == 4:
        x_mag = [g[3] for g in situation.secondary]
        EV = situation.primaryEV

    steCH = []
    avgCH = []
    steMV = []
    avgMV = []
    for x in x_mag:
        chTimes = choiceDict[x]
        steCH.append(stats.sem(np.array(chTimes)))
        avgCH.append(np.mean(chTimes))
        mvTimes = moveDict[x]
        steMV.append(stats.sem(np.array(mvTimes)))
        avgMV.append(np.mean(mvTimes))

#        import pdb ; pdb.set_trace() #AWESOME WAY TO DEBUG
#fig, axarr = plt.subplots(figsize=(8, len(flatten(softmaxDF.freq_sCh.tolist()))/4))
    subPlace[subPlot, 0].grid(b=True, which='major', axis='y')
    subPlace[subPlot, 0].bar(
        x_mag, avgCH, width=0.035, yerr=steCH, color='k', alpha=1, capsize=2)
    subPlace[subPlot, 0].bar(
        x_mag, avgMV, width=0.035, yerr=steMV, color='b', alpha=1, capsize=2)
    subPlace[subPlot, 0].axvline(x=EV, linestyle='--', color='k', alpha=0.7)
    plt.xlabel('safe magnitude')
    plt.ylabel('reactionTime')
    for x in x_mag:
        subPlace[subPlot, 0].plot(
            np.linspace(x, x, len(choiceDict[x])),
            choiceDict[x],
            '.',
            color='k',
            alpha=0.3)
        subPlace[subPlot, 0].plot(
            np.linspace(x, x, len(moveDict[x])),
            moveDict[x],
            '.',
            color='k',
            alpha=0.7)
    subPlace[subPlot, 0].text(
        max(x_mag) + (x_mag[-1] - x_mag[-2]), 0.25, str(situation.primary))


#%%


def plot_equivalents(softmaxDF, withFit=False):
    from collections import Counter
    import numpy as np
    import pandas as pd

    if ('sessionDate' in softmaxDF.columns) and (len(
            softmaxDF.sessionDate.unique()) > 1):
        for day in softmaxDF.sessionDate.unique():
            # .sort_values(['primaryEV']))
            plot_equivalents(softmaxDF.loc[softmaxDF.sessionDate == day])
    else:
        plt.rc('axes', axisbelow=True)
        fig, axarr = plt.subplots(
            1, len(softmaxDF.seqCode.unique()), squeeze=False, figsize=(8, 4))

        for n, div in enumerate(softmaxDF.seqCode.unique()):
            softmaxDF.loc[softmaxDF.seqCode == div].sort_values(
                ['primaryEV'], inplace=True)

            mGroup = Counter(softmaxDF.loc[softmaxDF.seqCode == div]
                             .primary.apply(lambda x: x[2]))
            # find the most common magnitude for the plotting
            mGroup = mGroup.most_common(1)[0][0]
            selectedDF = softmaxDF.loc[softmaxDF.seqCode == div].loc[
                softmaxDF.primary.apply(lambda x: x[2]) == mGroup]
            p = selectedDF.primary.apply(lambda x: x[3])
            m = selectedDF.primary.apply(lambda x: x[2])
            CE = selectedDF.equivalent

            if ('sessionDate' in softmaxDF.loc[softmaxDF.seqCode == div]
                    .columns):
                plt.suptitle(
                    softmaxDF.loc[softmaxDF.seqCode == div].sessionDate.apply(
                        lambda x: x.strftime("%Y-%m-%d")).unique().tolist()[0],
                    fontsize=16)
            axarr[0, n].scatter(p, CE, color='b', alpha=0.7)
            axarr[0, n].plot(
                np.linspace(0, 1, 1000),
                np.linspace(0, 0.5, 1000),
                color='k',
                linestyle='--')
            # this sets the subplot's title
            axarr[0, n].set_title(str(np.unique(selectedDF.division.tolist())))
            #        axarr.set_ylim(0, mGroup) #y axis length
            #        axarr.set_xlim(0, 1) #y axis length
            #        axarr.axis('scaled')
            axarr[0, n].set_adjustable('datalim')
            axarr[0, n].grid(b=True, which='major')
            x0, x1 = axarr[0, n].get_xlim()
            y0, y1 = axarr[0, n].get_ylim()
            axarr[0, n].set_aspect((x1 - x0) / (y1 - y0))


#%%
def expand_softmax(softmaxDF):
    '''
    Expands a softmax dataframe so that each secondary option has its own row - as opposed to aggregating them
    '''
    if len(softmaxDF.loc[softmaxDF.secondary.apply(lambda x: len(x) > 1)]) == 0:
        return softmaxDF  # in case the softmaxDF is already in a expanded-like form

    if 'sessionDate' not in softmaxDF:
        softmaxDF['sessionDate'] = 0
        softmaxDF['division'] = 0

    dfs = []
    count = 0
    for date in tqdm(softmaxDF['sessionDate'].unique(), desc='Expanding DF'):
        dateSM = softmaxDF.loc[softmaxDF['sessionDate'] == date]
        miniDF = []
        for _, row in dateSM.loc[dateSM.secondary.apply(
                lambda x: len(x) > 1)].iterrows():
            for i, secondary in enumerate(row.secondary):
                count += 1
                new_row = row.copy()
                new_row['secondary'] = new_row.secondary[i]
                new_row['secondaryEV'] = new_row.secondaryEV[i]
                new_row['freq_sCh'] = new_row.freq_sCh[i]
                new_row['no_of_Trials'] = new_row.no_of_Trials[i]
                new_row['nTrials'] = new_row.nTrials
                new_row['primarySide'] = {secondary[0]: new_row.primarySide[secondary[0]]}
                new_row['choiceList'] = {secondary[0]: new_row.choiceList[secondary[0]]}
                new_row['moveTime'] = {secondary[0]: new_row.moveTime[secondary[0]]}
                new_row['choiceTimes'] = {secondary[0]: new_row.choiceTimes[secondary[0]]}
                new_row['filteredRT'] = {secondary[0]: new_row.filteredRT[secondary[0]]}
                new_row['trial_index'] = {secondary[0]: new_row.trial_index[secondary[0]]}
                new_row['oClock'] = {secondary[0]: new_row.oClock[secondary[0]]}
                miniDF.append(new_row)
        dfs.append(pd.DataFrame(miniDF))
    dfs.append(softmaxDF.loc[softmaxDF.secondary.apply(lambda x: len(x) == 1)])

    softmaxDF = pd.concat(dfs, ignore_index=True)
    softmaxDF['secondaryEV'] = softmaxDF.secondaryEV.apply(
        lambda x: x[0] if isinstance(x, list) else x)
    softmaxDF['freq_sCh'] = softmaxDF.freq_sCh.apply(
        lambda x: x[0] if isinstance(x, list) else x)

    softmaxDF['gap'] = np.abs(np.round(softmaxDF.primaryEV - softmaxDF.secondaryEV, decimals = 3))
    softmaxDF['midpoint'] =np.round(np.mean((softmaxDF.primaryEV.values, softmaxDF.secondaryEV.values), axis=0), decimals=3)
    softmaxDF['iTrials'] = softmaxDF.trial_index.apply(lambda x : str(np.sort(np.squeeze(list(x.values())))))

    Ndecimals = 2
    decade = 10**Ndecimals
    softmaxDF['midpoint'] = np.trunc(softmaxDF['midpoint'].values*decade)/decade
    softmaxDF['secondary'] = softmaxDF['secondary'].apply(lambda x: np.squeeze(x))
    softmaxDF['no_of_Trials'] = softmaxDF['no_of_Trials'].apply(lambda x: x[0] if type(x) == list else x)

    if 'division' in softmaxDF.columns:
        softmaxDF.sort_values( by=['sessionDate', 'division', 'primaryEV'], inplace = True)
    else:
        softmaxDF.sort_values(by=['sessionDate', 'primaryEV'], inplace = True)
    softmaxDF.drop_duplicates( subset=['iTrials', 'division', 'sessionDate'], keep='first', inplace = True)
    softmaxDF['iTrials'] = softmaxDF.trial_index.apply(lambda x : np.sort(np.squeeze(list(x.values()))))
    cols = ['primary', 'primaryEV', 'secondary', 'secondaryEV', 'gap', 'midpoint', 'm_range',
       'freq_sCh', 'primarySide', 'no_of_Trials', 'nTrials', 'choiceList',
       'filteredRT', 'choiceTimes', 'moveTime', 'trial_index', 'oClock',
       'metricType', 'seqCode', 'gList', 'chosenEV', 'sessionDate', 'division',
       'iTrials']
    return  psychometricDF(softmaxDF[cols])

#%%
def sort_byMidpoint(softmaxDF, minGapN=3, maxGap=0.1, specific_width = None):
    '''
    '''
    import scipy.optimize as opt
    from macaque.f_psychometrics import psychometricDF
    np.warnings.filterwarnings('ignore')
    sigmoid = lambda x, p2: np.array(1 / (1 + np.exp(-(x - 0) / p2)))
    newDF = []
    
    softmaxDF = softmaxDF.loc[softmaxDF.gap != 0]
            
    param_bounds = ([0.01], [1])
    for date in tqdm(softmaxDF.sessionDate.unique()):
        df = softmaxDF.loc[softmaxDF.sessionDate == date]
        unique_midpoints = np.unique(df.midpoint)
        for midpoint in unique_midpoints:
            subdf = df.loc[df.midpoint == midpoint].sort_values(by = 'gap')
            if specific_width != None:
                where = np.isin(subdf['gap'].values, np.array(specific_width))
                subdf = subdf.loc[where]

            subdf = subdf.loc[subdf['gap'] < maxGap]
#            subdf = subdf.loc[subdf.freq_sCh >= 0.5]
            if len(subdf) < minGapN:
                continue

            gaps = subdf.gap.values
            pChooseSecondary = subdf.freq_sCh.values
            popt, pcov = opt.curve_fit(sigmoid, gaps, pChooseSecondary, p0=[1], method='lm')
#                                       bounds = param_bounds)
            if len(gaps) < minGapN:
                popt = [np.nan]; pcov = [np.nan]

            newDF.append(pd.DataFrame({'sessionDate' : date,
                     'primary' : [subdf.primary.tolist()],
                     'primaryEV' : [subdf.primaryEV.tolist()],
                     'secondary' : [subdf.secondary.tolist()],
                     'secondaryEV' : [subdf.secondaryEV.tolist()],
                     'temperature' : popt[0],
                     'temp_error' : pcov[0],
                     'm_range' : [subdf.m_range.min()],
                     'freq_sCh' : [subdf.freq_sCh.tolist()],
                     'no_of_Trials' : [subdf.no_of_Trials.tolist()],
                     'nTrials' : subdf.no_of_Trials.sum(),
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
                     'gap' : [subdf.gap.values],
                     'midpoint' : np.unique(subdf.midpoint.values)[0],
                     'iTrials' : [subdf.iTrials.tolist()]}))

    if len(newDF) == 0:
        return []
    else:
        newDF = pd.concat(newDF, ignore_index=True)
        newDF = psychometricDF(newDF.sort_values(by=['sessionDate', 'midpoint']))
        cols = ['sessionDate', 'primary', 'primaryEV', 'secondary', 'secondaryEV',
           'midpoint', 'temperature', 'temp_error', 'm_range',  'gap', 'freq_sCh', 'no_of_Trials',
           'nTrials', 'primarySide', 'choiceList', 'filteredRT', 'choiceTimes',
           'moveTime', 'trial_index', 'oClock', 'metricType', 'division',
           'seqCode', 'gList', 'chosenEV', 'iTrials']
        return newDF[cols]

#%%

class psychometricDF(pd.DataFrame):
    '''
    '''
    @property
    def _constructor(self):
        return psychometricDF

    def plotSoftmax(self,
                    sortBy='primaryEV',
                    printRatios=True,
                    plot_ci='fit',
                    color=None):
        plot_softmax(
            self,
            sortBy=sortBy,
            printRatios=printRatios,
            plot_ci=plot_ci,
            color=color)

    def getTrials(self, trials):
        regTrials = np.sort(
            np.unique(
                np.concatenate([
                    np.concatenate(list(val.values()))
                    for val in self.get('trial_index').values
                ]))
        )  # gets the index for all the trials I should use in regression
        # get only trials that were used in the rest of the analysis
        return trials.loc[regTrials].copy()

    def collapse(self, trials):
        '''
        Need the original trials dataframe.
        '''
        from macaque.f_choices import get_options
        return get_softmaxData( get_options(self.getTrials(trials)),
                                metricType = 'ce',
                                mergeBy='all' )

    def expand(self):
        '''
        '''
        return expand_softmax(self)

    def byMidpoints(self, minGapN=3, maxGap=0.1, specific_width = None):
        '''
        '''
        softmaxDF = expand_softmax(self)
        return sort_byMidpoint(softmaxDF,
                               minGapN=minGapN,
                               maxGap=maxGap,
                               specific_width=specific_width)


    def get_RTs(self):
        '''
        '''
        RTs = flatten([list(val.values()) for val in self.choiceTimes ])
        sEV = flatten([[np.array(y) for y in val.keys()] for val in self.choiceTimes ])
        pEV = flatten([[np.array(ev) for y in val.keys()]  for val, ev in zip( self.choiceTimes, self.primaryEV) ])
        seqType = flatten([[seq for y in val.keys()] for val, seq in zip(self.choiceTimes.values, self.seqCode) ])
        chosenEV = flatten( flatten([list(val.values()) for val in self.chosenEV]))
        date = flatten([[seq.toordinal() for y in val.keys()] for val, seq in zip(self.choiceTimes.values, self.sessionDate) ])
        date = np.array(date) - date[0]
        chosenSide = flatten( flatten([list(val.values()) for val in self.choiceList]))
        primarySide = flatten( flatten([list(val.values()) for val in self.primarySide]))

        chosenEV = flatten( flatten([list(val.values()) for val in self.chosenEV]))

        sEVs = []; pEVs = []
        seqTypes = []; day = []
        for rt, sev, pev, cc, dd in zip(RTs, sEV, pEV, seqType, date):
            sEVs.extend([sev] * len(rt))
            pEVs.extend([pev] * len(rt))
            seqTypes.extend([cc] * len(rt))
            day.extend([dd] * len(rt))
        del sEV, pEV, seqType, date
        deltaEV = np.round(np.array(pEVs) - np.array(sEVs), decimals=2)

        RTs = np.array(flatten(RTs))
        variables = np.array((RTs, deltaEV, np.array(seqTypes), np.array(chosenEV), day,  np.array(chosenSide), np.array(primarySide))).T
        aovDF = pd.DataFrame( variables, columns=['RTs', 'deltaEV', 'context', 'chosenEV', 'day', 'chosenSide', 'primarySide'])
        aovDF['RTs'] = aovDF['RTs'].astype(float)
        return aovDF

    def plot_descriptive(self, Trials):
        
        trials = self.getTrials(Trials)
        #Make a 9-grid figure
        fig, ax = plt.subplots(3, 3, squeeze = False, figsize=(8,8))
        plt.tight_layout()
        #plot distribution of all EVs
        sb.distplot(np.concatenate(trials[['GA_ev','GB_ev']].values), bins=20, norm_hist=True, ax=ax[1,0])
        ax[1,0].set( xlabel ='EVs', ylabel='frequency')
        #plot distribution of all RTs
        sb.distplot(trials['choiceTime'].values, bins=100, norm_hist=True, ax=ax[0,0])
        ax[0,0].set( xlabel ='RTs', ylabel='frequency')
        #plot MLs received
        sb.distplot(trials['ml_received'].values, bins=20, norm_hist=True, ax=ax[2,0])
        ax[2,0].set( xlabel ='rewarded ml', ylabel='frequency')

        #plot distribution of probabilities
        pA = trials.gambleA.apply(lambda x: x[-1] if x!= [] else np.nan)
        pB = trials.gambleB.apply(lambda x: x[-1] if x!= [] else np.nan)
        pA = pA[pA != 1]; pA=pA[~np.isnan(pA)]
        pB = pB[pB != 1]; pB=pB[~np.isnan(pB)]
        sb.distplot(np.concatenate((pA.values,pB.values)), bins=9, norm_hist=True, ax=ax[1,2])
        ax[1,2].set( xlabel ='probabilities', ylabel='frequency')

        mA = trials.gambleA.apply(lambda x: x[::2] if x!= [] else np.nan).values
        mB = trials.gambleB.apply(lambda x: x[::2] if x!= [] else np.nan).values
        sb.distplot(mA, bins=9, norm_hist=True, ax=ax[1,1])
        ax[1,1].set( xlabel ='probabilities', ylabel='frequency')

        #
        sb.distplot(trials.loc[[any(np.isin(x, 2)) for x in trials.outcomesCount.values]].choiceTime.values,
                    bins=20, norm_hist=True, ax=ax[0,2])
        ax[0,2].set( xlabel ='probabilities', ylabel='frequency')

        sb.distplot(trials.loc[[any(np.isin(x, 2)) for x in trials.outcomesCount.values]].ml_received.values,
                    bins=20, norm_hist=True, ax=ax[2,2])
        ax[2,2].set( xlabel ='rewarded ml', ylabel='frequency')

        #plot

        trials.loc[trials.outcomesCount.apply(lambda x: x==[1,1])]
