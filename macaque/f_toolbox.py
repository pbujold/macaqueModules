"""
Created on Mon Jan 15 18:07:16 2018

@author: PB-Modig
"""

#%%

def calculate_pvalues(df):
    import pandas as pd
    from scipy.stats import pearsonr
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

#%%
def is_outlier_doubleMAD(points):
    """
    FOR ASSYMMETRIC DISTRIBUTION
    Returns : filtered array excluding the outliers

    Parameters : the actual data Points array

    Calculates median to divide data into 2 halves.(skew conditions handled)
    Then those two halves are treated as separate data with calculation same as for symmetric distribution.(first answer) 
    Only difference being , the thresholds are now the median distance of the right and left median with the actual data median
    """
    import numpy as np

    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    medianIndex = np.int(points.size/2)

    leftData = np.copy(points[0:medianIndex])
    rightData = np.copy(points[medianIndex:points.size])

    median1 = np.median(leftData, axis=0)
    diff1 = np.sum((leftData - median1)**2, axis=-1)
    diff1 = np.sqrt(diff1)

    median2 = np.median(rightData, axis=0)
    diff2 = np.sum((rightData - median2)**2, axis=-1)
    diff2 = np.sqrt(diff2)

    med_abs_deviation1 = max(np.median(diff1),0.000001)
    med_abs_deviation2 = max(np.median(diff2),0.000001)

    threshold1 = ((median-median1)/med_abs_deviation1)*3
    threshold2 = ((median2-median)/med_abs_deviation2)*3

    #if any threshold is 0 -> no outliers
    if threshold1==0:
        threshold1 = sys.maxint
    if threshold2==0:
        threshold2 = sys.maxint
    #multiplied by a factor so that only the outermost points are removed
    modified_z_score1 = 0.6745 * diff1 / med_abs_deviation1
    modified_z_score2 = 0.6745 * diff2 / med_abs_deviation2

    filtered1 = []
    i = 0
    for data in modified_z_score1:
        if data < threshold1:
            filtered1.append(leftData[i])
        i += 1
    i = 0
    filtered2 = []
    for data in modified_z_score2:
        if data < threshold2:
            filtered2.append(rightData[i])
        i += 1

    filtered = filtered1 + filtered2
    return filtered

#%%
def mad_based_outlier(points, thresh=3.5):
    import numpy as np
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

#%%
def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

#%%
def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2,(y1-y2)/2,v2)
    adjust_yaxis(ax1,(y2-y1)/2,v1)

#%%
def adjust_yaxis(ax,ydif,v):
    """shift axis ax by ydiff, maintaining point v at the same location"""
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny>maxy or (-miny==maxy and dy > 0):
        nminy = miny
        nmaxy = miny*(maxy+dy)/(miny+dy)
    else:
        nmaxy = maxy
        nminy = maxy*(miny+dy)/(maxy+dy)
    ax.set_ylim(nminy+v, nmaxy+v)
    
#%%
def set_box_color(bp, color):
    import matplotlib.pyplot as plt
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
#%%
def convertRange( value, r1, r2 ):
    return ( value - r1[ 0 ] ) * ( r2[ 1 ] - r2[ 0 ] ) / ( r1[ 1 ] - r1[ 0 ] ) + r2[ 0 ]

#%%
def jitter(x, spread=0.05):
    import numpy as np
    return np.random.normal(x,spread)
#%%
def squarePlot(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))

#%%
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

#%%
def ipynb_tqdm():
    '''
    Returns true in in a notebook
    '''
    import sys
    try:
        if sys.argv != ['']:
            from tqdm import tqdm_notebook as tqdm
#            from tqdm import tqdm as tqdm
            return tqdm
        elif sys.argv == ['']:
            from tqdm import tqdm
            return tqdm
    except NameError:
        return None


#%%
def unique_listOfLists(listOfList, returnIndex=False):
    '''
    get a list of unique lists within a list of list.

    also returns a dictionary for the index if necessary
    '''
    uniques = [list(x) for x in set(tuple(x) for x in listOfList)]
    if returnIndex:
        index = {}
        for unique in uniques:
            index[tuple(unique)] = [
                i for i, x in enumerate(listOfList) if x == unique
            ]
        return uniques, index
    else:
        return uniques

    #THIS NEED TO ALSO POTENTIALLY RETURN THE INDEX OF WHERE THE UNIQUES WERE


#%%
def flatten(listOfList, levels=1):
    '''
    collapse lists of lists across dimensions
    '''
    for _ in range(levels):
        listOfList = [item for sublist in listOfList for item in sublist]
    return listOfList


#%%
def swapAB(toSwap):
    toSwap = [1 if x == 'B' else x for x in toSwap]
    toSwap = [2 if x == 'A' else x for x in toSwap]
    swapped = ['A' if x == 1 else x for x in toSwap]
    swapped = ['B' if x == 2 else x for x in swapped]
    return swapped


#%%
def sideStr_toNum(sides):
    newSides = []
    for side in sides:
        if side.lower() == 'left':
            newSides.extend([1])
        elif side.lower() == 'right':
            newSides.extend([2])
    return newSides


#%%
def ab2int(toInt):
    toInt = [1 if x == 'B' else x for x in toInt]
    toInt = [0 if x == 'A' else x for x in toInt]
    return toInt


#%%
def table_ends(df, x=3):
    import numpy as np
    """Returns both head and tail of the dataframe or series.

    Args:
        x (int): Optional number of rows to return for each head and tail
    """
    print('{} rows x {} columns'.format(np.shape(df)[0], np.shape(df)[1]))
    return df.head(x).append(df.tail(x))


#%%
def save_data(directory, name='temp'):
    '''
    Store only dataFrames (in case you are having issues with saving an entire session)
    '''
    import shelve
    import pandas as pd
    from pathlib import Path
    home = str(Path.home())
    filename = home + r'\University Of Cambridge\OneDrive - University Of Cambridge\Lab Computer\UtilityRangeAdaptation\tmp' + "\\" + name

    my_shelf = shelve.open(filename, 'n')  # 'n' for new
    print(directory)
    alldfs = [
        var for var in directory
        if isinstance(eval(var), pd.core.frame.DataFrame)
    ]

    print('Storing DataFrames: \n')
    for key in alldfs:
        print(key)
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()


#%%
def load_data(name='temp'):
    '''
    Load only dataFrames (in case you are having issues with saving an entire session)
    '''
    import shelve
    from pathlib import Path
    home = str(Path.home())
    filename = home + r'\University Of Cambridge\OneDrive - University Of Cambridge\Lab Computer\UtilityRangeAdaptation\tmp' + "\\" + name

    my_shelf = shelve.open(filename)
    print('Restoring DataFrames: \n')
    for key in my_shelf:
        print(key)
        globals()[key] = my_shelf[key]
    my_shelf.close()


#%%
def save_Session(name='temp'):
    import shelve
    from pathlib import Path
    import dill
    home = str(Path.home())

    import os
    cwd = os.getcwd()
    filename = cwd + "\\tmp\\" + name + '.pk1'

    #    filename = home + r'\University Of Cambridge\OneDrive - University Of Cambridge\Lab Computer\UtilityRangeAdaptation\tmp' + "\\" + name + '.pk1'
    #    dill.settings['recurse'] = True
    dill.dump_session(filename)


#%%
def load_Session(name='temp'):
    import dill
    from pathlib import Path
    home = str(Path.home())

    import os
    cwd = os.getcwd()
    filename = cwd + "\\tmp\\" + name + '.pk1'
    dill.load_session(filename)


#    try:
#        filename = home + r'\University Of Cambridge\OneDrive - University Of Cambridge\Lab Computer\UtilityRangeAdaptation\tmp' + "\\" + name + '.pk1'
#        dill.load_session(filename)
#    except:
#        filename = home + r'\OneDrive - University Of Cambridge\Lab Computer\UtilityRangeAdaptation\tmp' + "\\" + name + '.pk1'
#        dill.load_session(filename)
#%%
def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
    return aov


#%%
def omega_squared(aov):
    mse = aov['sum_sq'][-1] / aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq'] -
                       (aov[:-1]['df'] * mse)) / (sum(aov['sum_sq']) + mse)
    return aov


#%%
def ICweights(ICs, models):
    import numpy as np

    try:
        mIC = np.mean(ICs, axis=1)
    except:
        mIC = []
        for ics in ICs:
            mIC.extend([np.mean(ics)])
    ICweight = []
    for ic in mIC:
        numerator = ic - min(mIC)
        denominator = mIC - min(mIC)
        ICweight.extend(
            [np.exp(-0.5 * numerator) / sum(np.exp(-0.5 * denominator))])
    print('\n IC weights (lkelihood that a model is the best model): ')
    print('-----------------------------------------------------------------')
    print(np.array((ICweight, models)).T)


#%%


def label_diff(i, j, text, X, Y, ax):
    x = (X[i] + X[j]) / 2
    y = 1.1 * max(Y[i], Y[j])
    dx = abs(X[i] - X[j])

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':20,'shrinkB':20,'linewidth':2}
    ax.annotate(text, xy=(X[i], y + (y / 2.7)), zorder=10)
    ax.annotate('', xy=(X[i], y), xytext=(X[j], y), arrowprops=props)

    return y + (y / 2.7)


#%%
def gitCommit(token='66764e318aed52254d7549b8329a26aa05739fe4',
              RepoName='macaque',
              WhatFiles='all'):
    '''
    Updates the python module on my github repository

    Parameters
    ----------
    token : str
        Token string for accessing the repository
    RepoName : string
        String name of the folder you want to update
    WhatFiles : str
        'all' to update a full folder, or the specific string for the module you want to update within the folder

    Returns
    ----------
    ----------
    future: needs to print proper Confidence Intervals
    '''
    import datetime
    now = datetime.datetime.now()
    from github import Github
    import base64
    from pathlib import Path
    home = str(
        Path.home()
    ) + '\\University Of Cambridge\\OneDrive - University Of Cambridge\\Lab Computer\\UtilityRangeAdaptation'
    gh = Github(token)
    #    for repo in g.get_user().get_repos():
    #        print(repo.name)
    #        repo.edit(has_wiki=False)

    user = gh.get_user()
    repo = user.get_repo("mpy_analysis")

    if WhatFiles.lower() == 'all':
        #        filing = repo.get_file_contents('/')
        filing = repo.get_file_contents('/' + RepoName)
        #        for folder in [f for f in filing if f.path == RepoName]:
        #            f_path = folder.path
        #            repo.update_file(f_path, 'updated on'+str(now) , f_path, folder.sha)
        for file in filing:
            git_path = '/' + file.path
            content_path = home + git_path.replace('/', '\\')
            with open(content_path, 'rb') as input_file:
                content = input_file.read()
                repo.update_file(git_path, 'updated on ' + str(now), content,
                                 file.sha)


#%%
def getAsterisk(p):
    '''
    '''
    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    else:
        stars = 'x'
    return stars

#%%
def merge_dicts(listOfdicts):
    results = {}
    for d in listOfdicts:
        results.update(d)
    return results

#%%
def sample_power_probtest(p1, p2, power=0.8, sig=0.05):
    z = norm.isf([sig/2]) #two-sided t test
    zp = -1 * norm.isf([power])
    d = (p1-p2)
    s =2*((p1+p2) /2)*(1-((p1+p2) /2))
    n = s * ((zp + z)**2) / (d**2)
    return int(round(n[0]))
