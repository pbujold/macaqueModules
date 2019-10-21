'''
Applyng R code to python variables.

This is an attempt to fill in missing statistic functions using R from rpy2
'''

#from pathlib import Path
#home = str(Path.home())
import os
cwd = os.getcwd()
os.environ['R_HOME'] = cwd + '\\R'

import pandas as pd
import numpy as np

from rpy2 import robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
R = ro.r


#%%
def to_rData(variable):
    import pandas as pd
    import numpy as np
    from rpy2.robjects import FloatVector, int2ri, StrVector
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    if type(variable) == pd.core.frame.DataFrame:
        output = pandas2ri.py2ri(variable)
    elif type(variable) == np.ndarray:
        output = numpy2ri(variable)
    elif type(variable) == list:
        output = FloatVector(variable)
    elif type(variable) == int:
        output = int2ri(variable)
    elif type(variable) == str:
        output = StrVector(variable)

    pandas2ri.deactivate()


#%%
def dv3_manova(DV1, DV2, DV3, IV):
    '''
    '''
    stats = importr('stats')

    formula = R.formula("cbind(factor0, factor1, factor2) ~ IV")
    env = formula.environment
    env["factor0"] = numpy2ri(DV1)
    env["factor1"] = numpy2ri(DV2)
    env["factor2"] = numpy2ri(DV3)
    env["IV"] = numpy2ri(IV)
    ols_str = stats.lm(formula)
    results = stats.manova(ols_str)

    #report manova test
    print(R.summary(results, test='Wilks').rx('stats'))
    print(R.summary(R.aov(ols_str)))


#%%
def oneWay_rmAnova(DV, ID, IV):
    '''
    Parameters
     ----------
    DV : list/array
        Dependent variable as a singular list/array (will make a df-longside)
    ID : list/array
        Repeated measure: list/array of assigned identities
    IV : list/array
        The independent variable (condition) you are testing across as a list/array
    '''
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    df = pd.DataFrame({'DV': DV, 'ID': ID, 'IV': IV})
    r_df = pandas2ri.py2ri(df)

    afex = importr('afex')
    model = afex.aov_ez('ID', 'DV', r_df, within='IV')
    print(R.summary(model))

    #    esm = importr("emmeans", on_conflict="warn")
    esm = importr("lsmeans")

    pairwise = esm.lsmeans(model, "IV", contr="pairwise", adjust="holm")
    print(R.summary(pairwise))

    pandas2ri.deactivate()
    return R.summary(pairwise)

#    robjects.globalenv["c_param"] = numpy2ri(np.log(c))
#    robjects.globalenv["u_param"] = numpy2ri(np.log(u))
#    robjects.globalenv["w_param"] = numpy2ri(np.log(w))
#    robjects.globalenv["group"] = numpy2ri(g)
#    ols_str = stats.lm("cbind(c_param, u_param, w_param) ~ group")
#report variable-specific differences


def dv2_manova(DV1, DV2, IV):
    '''
    '''
    stats = importr('stats')

    formula = R.formula("cbind(factor0, factor1) ~ IV")
    env = formula.environment
    env["factor0"] = numpy2ri(DV1)
    env["factor1"] = numpy2ri(DV2)
    env["IV"] = numpy2ri(IV)
    ols_str = stats.lm(formula)
    results = stats.manova(ols_str)

    #report manova test
    print(R.summary(results, test='Wilks').rx('stats'))
    print(R.summary(R.aov(ols_str)))


def dv4_manova(DV1, DV2, DV3, DV4, IV):
    '''
    '''
    stats = importr('stats')

    formula = R.formula("cbind(factor0, factor1, factor2, factor3) ~ IV")
    env = formula.environment
    env["factor0"] = numpy2ri(DV1)
    env["factor1"] = numpy2ri(DV2)
    env["factor2"] = numpy2ri(DV3)
    env["factor3"] = numpy2ri(DV4)
    env["IV"] = numpy2ri(IV)
    ols_str = stats.lm(formula)
    results = stats.manova(ols_str)

    #report manova test
    print(R.summary(results, test='Wilks').rx('stats'))
    print(R.summary(R.aov(ols_str)))
    
#%%
def dv5_manova(DV1, DV2, DV3, DV4, DV5, IV):
    '''
    '''
    stats = importr('stats')

    formula = R.formula("cbind(factor0, factor1, factor2, factor3, factor4) ~ IV")
    env = formula.environment
    env["factor0"] = numpy2ri(DV1)
    env["factor1"] = numpy2ri(DV2)
    env["factor2"] = numpy2ri(DV3)
    env["factor3"] = numpy2ri(DV4)
    env["factor4"] = numpy2ri(DV5)
    env["IV"] = numpy2ri(IV)
    ols_str = stats.lm(formula)
    results = stats.manova(ols_str)

    #report manova test
    print(R.summary(results, test='Wilks').rx('stats'))
    print(R.summary(R.aov(ols_str)))
    
    
#%%
def dv4_manova_2way(DV1, DV2, DV3, DV4, IV1, IV2):
    '''
    '''
    stats = importr('stats')

    formula = R.formula("cbind(factor0, factor1, factor2, factor3) ~ factor(IV1) + factor(IV2) + factor(IV1):factor(IV2)")
    env = formula.environment
    env["factor0"] = numpy2ri(DV1)
    env["factor1"] = numpy2ri(DV2)
    env["factor2"] = numpy2ri(DV3)
    env["factor3"] = numpy2ri(DV4)
    env["IV1"] = numpy2ri(IV1)
    env["IV2"] = numpy2ri(IV2)
    ols_str = stats.lm(formula)
    results = stats.manova(ols_str)

    #report manova test
    print(R.summary(results, test='Wilks').rx('stats'))
    print(R.summary(R.aov(ols_str)))

#%%
def dv2_manova_2way(DV1, DV2, IV1, IV2):
    '''
    '''
    stats = importr('stats')

    formula = R.formula("cbind(factor0, factor1) ~ factor(IV1) + factor(IV2) + factor(IV1):factor(IV2)")
    env = formula.environment
    env["factor0"] = numpy2ri(DV1)
    env["factor1"] = numpy2ri(DV2)
    env["IV1"] = numpy2ri(IV1)
    env["IV2"] = numpy2ri(IV2)
    ols_str = stats.lm(formula)
    results = stats.manova(ols_str)

    #report manova test
    print(R.summary(results, test='Wilks').rx('stats'))
    print(R.summary(R.aov(ols_str)))


