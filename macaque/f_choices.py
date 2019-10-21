"""
Module of functions that apply to/get choice data for a trials dataframe (monkeys).

"""
from macaque.f_toolbox import *
from collections import Counter
import pandas as pd
import numpy as np
tqdm = ipynb_tqdm()
from macaque.f_trials import add_chosenEV


#%%
#from numba import jit
#from numpy import arange
#import numba
#@jit(debug=True)
def get_options(trials,
                mergeBy='all',
                byDates=False,
                mergeSequentials=True,
                sideSpecific=None,
                plotTQDM=True):
    '''
    From a trials dataFrame, make a new dataframe which has every outcome pair and the choices between them.  If sideSpecific is True,
    return two dataframes for options presented on the right, and options presented on the left.

    Parameters
    ----------
    trials : DataFrame
        DataFrame of trials.
    sideSpecific : Boolean
        if True, returns two Dataframes - one where primaries are all on the left, and one where they are all on the right
    mergeBy: String
        'all', 'sequenceType', or 'block'.  Chooses if you want to separate psychometric data based on where they came from.
    byDates: list
        list of dates to select trials from + looks at choices per-day rather than all together.  If byDates is None, merge all across days.
    mergeSequentials: Boolean
        if True merge trials from sequential blocks that share glist, or all the same gamble/safe pairs
    Returns
    ----------
    choiceDate : DataFrame
        DataFrame of choices for outcome pairs, choiceData captures multiple choices, reaction times, errors, and indices
    *--if sideSpecific is True--*
        choiceDataA, choiceDataB are returned rather than ChoiceData.
    '''

    if 'chosenEV' not in trials.columns:
        trials = add_chosenEV(trials)  #add the chosenEV column

    if 'filteredRT' not in trials.columns:
        trials.loc[:,'filteredRT'] = np.nan  #add the chosenEV column

    #this runs first, and recurse the function over itself per day.
    if byDates:
        dates = trials.sessionDate.unique()
        dfs = []
        for session in tqdm(
                dates, desc='Gathering Daily Choice Data',
                disable=not plotTQDM):
            dfs.append(
                get_options(
                    trials.loc[trials.sessionDate == session],
                    mergeBy=mergeBy,
                    mergeSequentials=mergeSequentials,
                    sideSpecific=sideSpecific).assign(sessionDate=session)
            )
        choiceData = pd.concat(dfs, ignore_index=True)

        print('\nMerge choices from', str(len(dfs)), 'different days.')
        print('There were', str(len(choiceData)),
              'unique choice situations in these sessions')

        return choiceDF(choiceData)

    #-------------------------------------------------------------

    if mergeBy.lower() == 'block':
        dfs = []
        for i, block in enumerate(trials.blockNo.unique()):
            if len(np.unique(trials.loc[trials.blockNo == block].trialSequenceMode.unique())) > 1:
                print('\n', trials.sessionDate.tolist()[0].strftime('%Y-%m-%d'),': deleted block', str(block), 'due to inconsistency')
                continue
            sequence = int(
                np.unique(trials.loc[trials.blockNo == block]
                          .trialSequenceMode.unique()))
            glist = np.unique(
                trials.loc[trials.blockNo == block].sequenceFilename.unique())
            dfs.append(
                get_options(
                    trials.loc[trials.blockNo == block],
                    sideSpecific=sideSpecific).assign(
                        division=i,
                        seqType=sequence,
                        gList=str(glist).strip('[]')))
        choiceData = pd.concat(dfs, ignore_index=True)
        if mergeSequentials is True:
            choiceData = merge_sequential(choiceData, trials)
        return choiceDF(choiceData)

    elif mergeBy.lower() == 'sequencetype':
        sequences = trials.trialSequenceMode.unique()
        glist = np.unique(trials.sequenceFilename.unique()).tolist()
        dfs = []
        for i, sequence in enumerate(sequences):
            dfs.append(
                get_options(
                    trials.loc[trials.trialSequenceMode == sequence],
                    sideSpecific=sideSpecific).assign(
                        division=i,
                        seqType=sequence,
                        gList=str(glist).strip('[]')))
        choiceData = pd.concat(dfs, ignore_index=True)
        return choiceDF(choiceData)

    elif mergeBy.lower() == 'all':
        unique_Outcomes = unique_listOfLists(trials['gambleA'].tolist() +
                                             trials['gambleB'].tolist())
        cols2 = [
            'side_of_1', 'option1', 'chose1', 'option2', 'chose2', 'noChoice',
            'no_of_Trials', 'outcomesCount', 'choiceList', 'filteredRT',
            'choiceTimes', 'moveTime', 'errors', 'trial_index', 'oClock',
            'G1_ev', 'G2_ev', 'chosenEV'
        ]

        dfs = []
        for situation in unique_Outcomes:
            if sideSpecific == None:
                aTr = trials.loc[[
                    True if x == situation else False for x in trials.gambleA
                ]]
                bTr = trials.loc[[
                    True if x == situation else False for x in trials.gambleB
                ]]
            elif sideSpecific.lower() == 'left':
                aTr = trials.loc[[
                    True if x == situation else False for x in trials.gambleA
                ]]
                bTr = trials.loc[[]]
            elif sideSpecific.lower() == 'right':
                aTr = trials.loc[[]]
                bTr = trials.loc[[
                    True if x == situation else False for x in trials.gambleB
                ]]
            secondaries = unique_listOfLists(aTr.gambleB.tolist() + bTr.gambleA.tolist())  #get list of unique secondaries to combine

            for secondary in secondaries:  #might be faster in a nested loop than two separate loops
                cA = aTr.loc[[
                    True if x == secondary else False for x in aTr.gambleB
                ]]
                cB = bTr.loc[[
                    True if x == secondary else False for x in bTr.gambleA
                ]]

                sideChosen = cA.gambleChosen.tolist() + swapAB(
                    cB.gambleChosen.tolist())
                counted = Counter(sideChosen)

                sideChosen = [ord(char.lower()) - 96 for char in sideChosen]
                dfs.append(
                    pd.DataFrame({
                        'side_of_1': [['1'] * len(cA) + ['2'] * len(cB)],
                        'option1': [situation],
                        'chose1':
                        counted['A'],
                        'option2': [secondary],
                        'chose2':
                        counted['B'],
                        'noChoice':
                        len(sideChosen) - (counted['A'] + counted['B']),
                        'no_of_Trials':
                        len(sideChosen),
                        'outcomesCount':
                        [[int(len(situation) / 2),
                          int(len(secondary) / 2)]],
                        'choiceList': [sideChosen],
                        'choiceTimes':
                        [cA.choiceTime.tolist() + cB.choiceTime.tolist()],
                        'filteredRT':
                        [cA.filteredRT.tolist() + cB.filteredRT.tolist()],
                        'moveTime': [
                            cA.j_firstStimulus.tolist() +
                            cB.j_firstStimulus.tolist()
                        ],
                        'errors': [
                            cA.errorType.tolist() + cB.errorType.tolist()
                        ],
                        'trial_index': [cA.index.tolist() + cB.index.tolist()],
                        'oClock': [cA.time.tolist() + cB.time.tolist()],
                        'G1_ev': [
                            np.unique(cA.GA_ev.tolist() + cB.GB_ev.tolist())
                        ],
                        'G2_ev': [
                            np.unique(cA.GB_ev.tolist() + cB.GA_ev.tolist())
                        ],
                        'chosenEV': [
                            cA.chosenEV.tolist() + cB.chosenEV.tolist()
                        ]
                    }))
        if dfs == []:
            choiceData = choiceData = pd.DataFrame(columns=cols2)
        else:
            choiceData = pd.concat(dfs, ignore_index=True)

        choiceData = choiceData[cols2]
        return choiceDF(choiceData.assign(division=1, sessionDate=np.nan))


#%%
def get_gambles(choiceData, option):
    '''
    From a choiceData dataFrame, get all choices where the primary or secondary options are gambles.
    This is regardless of the side on which they are presented.

    Parameters
    ----------
    choiceData : DataFrame
        DataFrame of trials.
    option : Str
        'primary or 'secondary' to decide from which options to filter for gambles.

    Returns
    ----------
    choiceData.loc[index] : DataFrame
        DataFrame of choices that involve a gamble as their primary or secondary option, regardless of side.
    '''
    from macaque.f_toolbox import unique_listOfLists

    if option.lower() == 'primary':
        unique_Outcomes = unique_listOfLists(choiceData.option1)
        index = []
        for i, option in enumerate(unique_Outcomes):
            if len(option) > 2:  #lenght of gambles outcomes
                index.extend(choiceData.loc[choiceData.option1.apply(
                    lambda x: x == option)].index.tolist())
        return choiceData.loc[index]
    elif option.lower() == 'secondary':
        unique_Secondaries = unique_listOfLists(choiceData.option2)
        index = []
        for i, option in enumerate(unique_Secondaries):
            if len(option) > 2:  #lenght of gambles outcomes
                index.extend(choiceData.loc[choiceData.option2.apply(
                    lambda x: x == option)].index.tolist())
        return choiceData.loc[index]


#%%
def get_safes(choiceData, option):
    '''
    From a choiceData dataFrame, get all choices where the primary or secondary options are safe, certain outcomes.
    This is regardless of the side on which they are presented.

    Parameters
    ----------
    choiceData : DataFrame
        DataFrame of trials.
    option : Str
        'primary or 'secondary' to decide from which options to filter for safe, unique outcomes.

    Returns
    ----------
    choiceData.loc[index] : DataFrame
        DataFrame of choices that involve a safe outcome as their primary or secondary option, regardless of side.
    '''
    from macaque.f_toolbox import unique_listOfLists

    if option.lower() == 'primary':
        unique_Outcomes = unique_listOfLists(choiceData.option1)
        index = []
        for i, option in enumerate(unique_Outcomes):
            if (len(option) == 2) and (
                    option[-1] == 1.00):  #lenght of safe outcomes
                index.extend(choiceData.loc[choiceData.option1.apply(
                    lambda x: x == option)].index.tolist())
        return choiceData.loc[index]

    elif option.lower() == 'secondary':
        unique_Secondaries = unique_listOfLists(choiceData.option2)
        index = []
        for i, option in enumerate(unique_Secondaries):
            if (len(option) == 2) and (
                    option[-1] == 1.00):  #lenght of safe outcomes
                index.extend(choiceData.loc[choiceData.option2.apply(
                    lambda x: x == option)].index.tolist())
        return choiceData.loc[index]


#%%
def get_imp(choiceData, option):
    '''
    From a choiceData dataFrame, get all choices where there is only the primary or secondary option (imperative trials)
    This is regardless of the side on which they are presented.

    Parameters
    ----------
    choiceData : DataFrame
        DataFrame of trials.
    option : Str
        'primary or 'secondary' to decide from which options to filter for imperatives.

    Returns
    ----------
    choiceData.loc[index] : DataFrame
        DataFrame of choices that present only a primary or secondary option, i.e. imperative trials.
    '''
    from macaque.f_toolbox import unique_listOfLists

    if option.lower() == 'primary':
        unique_Secondaries = unique_listOfLists(choiceData.option1)
        index = []
        for i, option in enumerate(unique_Secondaries):
            if len(option) == 0:  #lenght for no outcomes
                index.extend(choiceData.loc[choiceData.option1.apply(
                    lambda x: x == option)].index.tolist())
        return choiceData.loc[index]

    elif option.lower() == 'secondary':
        unique_Secondaries = unique_listOfLists(choiceData.option2)
        index = []
        for i, option in enumerate(unique_Secondaries):
            if len(option) == 0:  #lenght for no outcomes
                index.extend(choiceData.loc[choiceData.option2.apply(
                    lambda x: x == option)].index.tolist())
        return choiceData.loc[index]


#%%
def get_psychData(choiceData, metricType='transitivity', transitType='gambles'):
    '''
    filters data so that it is ready for psychometrics
    '''
    if (metricType.lower() == 'ce') or (
            metricType.lower() == 'certainty equivalent'):
        psychData = get_gambles(choiceData, 'primary')
        psychData = get_safes(psychData, 'secondary')
        return choiceDF(psychData)
    elif (metricType.lower() == 'pe') or (
            metricType.lower() == 'probability equivalent'):
        psychData = get_safes(choiceData, 'primary')
        psychData = get_gambles(psychData, 'secondary')
        return choiceDF(psychData)
    elif metricType.lower() == 'trans' or metricType.lower() == 'transitivity':
        if transitType.lower() == 'gambles':
            psychData = get_gambles(choiceData, 'primary')
            psychData = get_gambles(psychData, 'secondary')
            return choiceDF(psychData)
        elif transitType.lower() == 'safes':
            psychData = get_safes(choiceData, 'primary')
            psychData = get_safes(psychData, 'secondary')
            return choiceDF(psychData)
        else:
            #simply remove imperatives
            choiceData.drop(
                choiceData.loc[choiceData.option2.apply(lambda x: x == [])]
                .index.tolist() + choiceData.loc[choiceData.option1.apply(
                    lambda x: x == [])].index.tolist(),
                inplace=True)
            return choiceDF(choiceData)
    if metricType.lower() == 'none':
        return choiceData


#%%
def get_indexedTrials(choiceData, trials):
    '''
    '''
    from macaque.f_toolbox import flatten
    index = flatten(choiceData.trial_index.values.tolist())
    iTrials = trials.loc[index]
    return iTrials


#%%
def merge_sequential(choiceData, trials):
    '''
    Merge sequential blocks into single blocks when they all share the same trials and/or gList name
    '''
    #     if ('division' in choiceData.columns) and (len(choiceData.sessionDate.unique())>1):
    #THIS NEEDS TO MERGE SOFTMAXES TOGETHER WHY THEY ARE FROM SIMILAR CONTEXTS
    from macaque.f_toolbox import flatten, unique_listOfLists
    from macaque.f_psychometrics import get_softmaxData

    #----------------------------------------------------------------------

    def merge_seqChoices(sDivs, sTrials, oldCD, tt):
        sDivs = np.unique(sDivs)
        sTrials = np.unique(sTrials)
        for div in sDivs:
            oldCD.drop(oldCD.loc[oldCD.division == div].index, inplace=True)
        newRowsChoice = get_options(tt.loc[sTrials], mergeBy='sequencetype')
        newRowsChoice = newRowsChoice.assign(division=sDivs[0])
        return oldCD.append(newRowsChoice).sort_index()

    #----------------------------------------------------------------------

#    choiceData.index = range(len(choiceData.index))
    sTrials = []
    sDivs = []
    oDivs = choiceData.division.unique()  #find the different blocks
    for ii in range(len(choiceData.division.unique()) - 1):
        div1 = oDivs[ii]
        div2 = oDivs[ii + 1]  #assign block numbers for the current and next
        df1 = choiceData.loc[choiceData.division ==
                             div1]  #find row attributable to primary block
        df2 = choiceData.loc[choiceData.division ==
                             div2]  #find row attributable to secondary block
        if df1.seqType.unique() == df2.seqType.unique() and df1.gList.unique(
        ) == df2.gList.unique():  #check if their gList matches
            if df1.seqType.unique(
            ) < 9020.0:  #now check to see if these are both custom sequences (9001)
                gambleList_1 = np.sort(
                    unique_listOfLists(
                        flatten([df1.option1.tolist(),
                                 df1.option2.tolist()])), 0)
                gambleList_2 = np.sort(
                    unique_listOfLists(
                        flatten([df2.option1.tolist(),
                                 df2.option2.tolist()])), 0)

                #find all gambles and all safes taken from their unique list
                gg1 = [
                    option
                    for x, option in zip([len(x)
                                          for x in gambleList_1], gambleList_1)
                    if x == max([len(x) for x in gambleList_1])
                ]
                gg2 = [
                    option
                    for x, option in zip([len(x)
                                          for x in gambleList_2], gambleList_2)
                    if x == max([len(x) for x in gambleList_2])
                ]
                ss1 = [
                    option
                    for x, option in zip([len(x)
                                          for x in gambleList_1], gambleList_1)
                    if x == min([len(x) for x in gambleList_1])
                ]
                ss2 = [
                    option
                    for x, option in zip([len(x)
                                          for x in gambleList_2], gambleList_2)
                    if x == min([len(x) for x in gambleList_2])
                ]

                if type(gg1[0]) != list:
                    gg1[0] = gg1[0].tolist()
                if type(gg2[0]) != list:
                    gg2[0] = gg2[0].tolist()
                if type(ss1[0]) != list:
                    ss1[0] = ss1[0].tolist()
                if type(ss2[0]) != list:
                    ss2[0] = ss2[0].tolist()

                if gg1[0] == gg2[0] and (len(gg1) == 1) and (
                        len(gg2) == 1
                ):  #means there is no difference between the two sets
                    pass
                elif ss1[0] == ss2[0] and (len(ss1) == 1) and (len(ss2) == 1):
                    pass
                else:
                    if len(sTrials) != 0:
                        sf = []
                        for dd in np.unique(sDivs):
                            sf.append(
                                get_softmaxData(
                                    choiceData.loc[choiceData.division == dd],
                                    metricType='CE',
                                    minSecondaries=4,
                                    minChoices=4))
                        try:
                            sf = pd.concat(sf, ignore_index=True)
                            if len(sf) >= 2 and not any(
                                    np.abs(np.diff(sf.equivalent.values)) > 0.1):
                                pass  #this is done so that I could look at past blocks
                            else:
                                choiceData = merge_seqChoices(
                                    sDivs, sTrials, choiceData, trials)
                        except:
                            choiceData = merge_seqChoices(
                                sDivs, sTrials, choiceData, trials)
                            pass
                    sTrials = []
                    sDivs = []
                    continue

            sTrials.extend(
                np.unique(
                    flatten(
                        [df1.trial_index.tolist(),
                         df2.trial_index.tolist()], 2)))
            sDivs.extend([div1, div2])
        elif len(sTrials) != 0:
            choiceData = merge_seqChoices(sDivs, sTrials, choiceData, trials)
            sTrials = []
            sDivs = []
        else:
            sTrials = []
            sDivs = []

    if len(sTrials) != 0:
        choiceData = merge_seqChoices(sDivs, sTrials, choiceData, trials)

    choiceData.sort_values(by=['sessionDate', 'division'], inplace=True)
    choiceData.index = range(len(choiceData.index))
    return choiceData  #removes trials that were recorded by error


#%%
#def fix_wrongBlock(trials):


def merge_sequential_test(choiceData, trials):
    '''
    '''
    from macaque.f_toolbox import unique_listOfLists, flatten

    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    dfs = []
    for seqCode, gList in unique_listOfLists(choiceData[['seqType',
                                                         'gList']].values):
        df = choiceData.loc[(choiceData.seqType == seqCode) &
                            (choiceData.gList == gList)]
        for toMerge in consecutive(df.division.unique()):
            if seqCode < 9020:
                options = [
                    flatten(
                        np.unique([
                            df.loc[df.division == block].option1.tolist(),
                            df.loc[df.division == block].option2.tolist()
                        ])) for block in toMerge
                ]
                indices = np.array([
                    np.unique(df.loc[df.division == block].trial_index.tolist())
                    for block in toMerge
                ])

                consecutives = [[np.nan]]
                for i in range(len(options) - 1):
                    if options[i] == options[i + 1]:
                        if consecutives[-1][-1] == i:
                            consecutives[-1].extend([i + 1])
                        else:
                            consecutives.append([i, i + 1])
                    else:
                        if consecutives[-1][-1] != i:
                            consecutives.append([i])
                if len(consecutives) == 1:
                    consecutives.append([0])
                elif len(consecutives[-1]) == 1:
                    consecutives.append([len(options)])
                consecutives = consecutives[1:]

                for cc in consecutives:
                    iMerge = np.unique(flatten(indices[cc]))
                    dfs.append(
                        get_options(
                            trials.loc[iMerge],
                            mergeBy='sequencetype').assign(division=toMerge[0]))

            else:
                iMerge = np.hstack(df.loc[[
                    True if x in toMerge else False for x in df.division
                ]].trial_index.values)
                dfs.append(
                    get_options(
                        trials.loc[iMerge],
                        mergeBy='sequencetype').assign(division=toMerge[0]))

    mCD = pd.concat(dfs, ignore_index=True)
    mCD.sort_values(by=['sessionDate', 'division'], inplace=True)
    mCD.index = range(len(mCD.index))
    return mCD

#%%
class choiceDF(pd.DataFrame):
    '''
    '''
    @property
    def _constructor(self):
        return choiceDF

    def getPsychometrics(self,  metricType = 'CE',  minSecondaries = 4,  minChoices = 4):
        '''
        '''
        from macaque.f_psychometrics import get_softmaxData
        return get_softmaxData(self,
                               metricType = metricType,
                               minSecondaries = minSecondaries,
                               minChoices = minChoices)

    def selectTrialType(self,  metricType = 'ce', transitType='None'):
        return get_psychData(self,
                             metricType = metricType,
                             transitType=transitType)

    def getTrials(self, trials):
        regTrials = np.sort( np.unique( np.concatenate( [x for x in self['trial_index']])) )
        return trials.loc[regTrials].copy()
