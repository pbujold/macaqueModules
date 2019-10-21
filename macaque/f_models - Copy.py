# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:06:53 2018

@author: phbuj
"""
import numpy as np
import pandas as pd
from macaque.f_toolbox import *
tqdm = ipynb_tqdm()

from statsmodels.base.model import GenericLikelihoodModel, GenericLikelihoodModelResults
import matplotlib.pyplot as plt
from types import MethodType
from scipy.stats import norm
from inspect import signature

    
    
#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:18:48 2019

@author: phbuj
"""

def define_model(decisionFunctions, fixedRange = False, dynamic = False):
    '''
    enter nomenclature for the model: 'utility-probability-cumulative-sigmoid'
    '''
    #fix this to input cummulative in the name
    decisionFunctions = decisionFunctions.split('-')
    if len(decisionFunctions) == 3:
        if decisionFunctions[0] == 'dynamic':
            dynamic = True
            sfm_function, util_function = decisionFunctions[1:]
            prob_function, cumm = ['none',False]
        else:
            sfm_function, util_function, prob_function = decisionFunctions
            cumm = False
    elif len(decisionFunctions) == 4:
        sfm_function, util_function, prob_function, cumm = decisionFunctions
    elif len(decisionFunctions) == 2:
        if decisionFunctions[0].lower() == 'control':
            sfm_function, util_function, prob_function, cumm = ['random','none','none',False]
    # define the functions to use in the model
    
    def rangePosition(Xs): #need to normalize the utilities
            X = Xs.copy()
            if fixedRange == True:
                fullMags = [0, 0.5]
            else:
                fullMags = np.hstack((X[:,0], X[:,2][(X[:,2] != 0)], X[:,4],  X[:,6][(X[:,6] != 0)]))     
                
            if  min(fullMags) < 0.15:
                newX = (X[:,0::2] - 0) / (max(fullMags) -0)
            else:
                newX = (X[:,0::2] - min(fullMags)) / (max(fullMags) - min(fullMags))
            newX[:,1] = [0 if np.sign(x) == -1.0  else x for x in newX[:,1]]
            newX[:,-1] = [0 if np.sign(x) == -1.0  else x for x in newX[:,-1]]
            X[:,0::2] = newX
            return X
    
    U, u_parameters, u0 = get_functions(util_function)
    u_parameters = ['u_' + pp for pp in u_parameters]
    W, w_parameters, w0 = get_functions(prob_function)
    w_parameters = ['w_' + pp for pp in w_parameters]
    parameterPositions = np.hstack([[0] * len(u0), [1] * len(w0)])
    VALUE = get_value(U, W, parameterPositions, cumulative = cumm, dynamic = dynamic)
    SFM, sfm_parameters, sfm0 = get_sigmoid(sfm_function, VALUE, dynamic = dynamic)
    if dynamic == True:
        LL = lambda y, X, params : sum(y * np.log(SFM(X, params))) + sum((1-y) * np.log(1-SFM(X, params)))
    else:
        LL = lambda y, X, params : sum(y * np.log(SFM(rangePosition(X), params))) + sum((1-y) * np.log(1-SFM(rangePosition(X), params)))

    parameterPositions = np.hstack([[0] * len(sfm0), [1] * len(u0), [2] * len(w0)])
    pNames = np.hstack((sfm_parameters, u_parameters, w_parameters)).tolist()
    x0 = np.hstack((sfm0, u0, w0))
    
    def model_parts(params):
            param_sfm = params[parameterPositions == 0]
            param_u = params[parameterPositions == 1]
            param_w = params[parameterPositions == 2]
            param_past = []

            def utility(mm):
                if len(signature(U).parameters) > 1:
                    return U(mm, param_u)
                else: 
                     return U(mm)

            def probability(pp):
                if len(signature(W).parameters) > 1:
                    return W(pp, param_w)
                else: 
                     return W(pp)
                 
            if dynamic:
                def value(option, X):
                    return VALUE(option, X, np.hstack((param_u, param_w)))
                def utility(mm, X):
                    Utility = U(X)
                    return Utility(mm, param_u)
                param_past = int(param_u[-1] / np.finfo('double').eps)
                
            else:
                def value(option):
                    if len(signature(VALUE).parameters) > 1:
                        return VALUE(option, np.hstack((param_u, param_w)))
                    else: 
                         return VALUE(option)

            def PchA(X):
                return SFM(X, params)

            return {
                'prob_chA': PchA,
                'value': value,
                'utility': utility,
                'probability_distortion': probability,
                'parameterNames': pNames,
                'description': 'PchA( sigmoid( VALUE_A(U*W) - VALUE_ B(U*W) ) )',
                'empty functions' : {'utility' : U,
                                     'probability' : W,
                                     'value' : VALUE,
                                     'pChooseA' : SFM}, 
                'past_effect': param_past
            }
                
    return LL, x0, pNames, model_parts
    
#%%
def get_sigmoid(sfm_function, VALUE, dynamic=False):
    '''
    '''
    sig = signature(VALUE)
    
    if dynamic == False:
        if len(sig.parameters) > 1:
            if sfm_function.lower() == 'probit':
                SFM = lambda X, params : norm.cdf(x=(VALUE( X[:,:4] , params[2:]) - VALUE(X[:,4:], params[2:])), loc=params[1], scale=1/params[0])
                sfm_parameters = ['noise', 'side_bias']
                sfm0 = [1, 0]
            elif sfm_function.lower() == 'logit':
                SFM = lambda X, params : 1 / (1 + np.exp( -params[0] * ( (VALUE( X[:,:4] , params[2:]) - VALUE(X[:,4:], params[2:]) - params[1]) )  ))
                sfm_parameters = ['noise', 'side_bias']
                sfm0 = [1, 0]
            elif sfm_function.lower() == 'random':
                SFM = lambda y, params : np.random.rand(len(y))
                sfm_parameters = ['noise']
                sfm0 = [1]
            elif sfm_function.lower() == 'webb':
                VARn21 = np.exp(s+(g*time))
                SFM = lambda X, params : 1 / (1 + np.exp( ( (VALUE( X[:,:4] , params[2:]) - VALUE(X[:,4:], params[2:]) - params[1]) ) / np.sqrt(VARn21)  ))
                sfm_parameters = ['noise', 'side_bias']
                sfm0 = [1, 0]
            elif sfm_function.lower() == 'noside':
                SFM = lambda X, params : norm.cdf(x=(VALUE( X[:,:4] , params[2:]) - VALUE(X[:,4:], params[2:])), loc=0, scale=1/params[0])
                sfm_parameters = ['noise']
                sfm0 = [1, 0]
        else:
            if sfm_function.lower() == 'probit':
                SFM = lambda X, params : norm.cdf(x=(VALUE( X[:,:4]) - VALUE(X[:,4:])), loc= params[1], scale=1/params[0])
                sfm_parameters = ['noise', 'side_bias']
                sfm0 = [1, 0]
            elif sfm_function.lower() == 'logit':
                SFM = lambda X, params : 1 / (1 + np.exp( -params[0] *  (VALUE( X[:,:4] ) - VALUE(X[:,4:]) - params[1])   ))
                sfm_parameters = ['noise', 'side_bias']
                sfm0 = [1, 0]
            elif sfm_function.lower() == 'random':
                SFM = lambda y, params : np.random.rand(len(y))
                sfm_parameters = ['noise', 'side_bias']
                sfm0 = [1]
            elif sfm_function.lower() == 'webb':
                VARn21 = np.exp(s+(g*time))
                SFM = lambda X, params : 1 / (1 + np.exp( ( (VALUE( X[:,:4] , ) - VALUE(X[:,4:]) - params[1]) ) / np.sqrt(VARn21)  ))
                sfm0 = [1,0]
            elif sfm_function.lower() == 'noside':
                SFM = lambda X, params : norm.cdf(x=(VALUE( X[:,:4] , params[2:]) - VALUE(X[:,4:], params[2:])), loc=0, scale=1/params[0])
                sfm_parameters = ['noise']
                sfm0 = [1]
    #%%
    elif dynamic == True:
        if sfm_function.lower() == 'probit':
            def SFM(X, params):
                return norm.cdf(x=(VALUE( X[:,:4], X, params[1:]) - VALUE(X[:,4:], X, params[1:])), loc=0, scale=1/params[0])
        elif sfm_function.lower() == 'logit':
            def SFM(X, params):
                return 1 / (1 + np.exp( -params[0] * ( VALUE( X[:,:4], X, params[1:]) - VALUE(X[:,4:], X, params[1:]) )))
        sfm_parameters = ['noise']
        sfm0 = [1]
    return SFM, sfm_parameters, sfm0

#%%
def get_functions(functionName):
    '''
    U(X, params, t)(x[i])
    '''
    if functionName.lower() == 'dual':
        def define_utilityFromPast(options):
#            rho, gamma = params
#            ref_function = lambda past, p1, t : np.where( (t==0), 0, np.sum([np.mean((p1 * ((1-p1)**(t-1-tau))) * xPast) for tau, xPast in enumerate(past)]) )
                
            aEVs = [np.array([gg[0]]) if gg[1] == 0 else gg for gg in options[:, :4:2]]
            bEVs = [np.array([gg[0]]) if gg[1] == 0 else gg for gg in options[:, 4::2]]
            allEVs = np.vstack((aEVs, bEVs)).T
            allEVs = [np.concatenate(evs) for evs in allEVs]
            allEVs =np.vstack(allEVs)
            
            #this is where p1 matters
            def ref(gamma):
                pastImpact = int(gamma / np.finfo('double').eps)
                if pastImpact > len(allEVs) or pastImpact <= 0:
                    return np.nan
                past = [ allEVs[i-pastImpact:i] if i-pastImpact > 0 else allEVs[0:i] for i in range(len(allEVs)) ]
                past[0] = np.mean(allEVs[0])
                ref_function = lambda past: np.array([np.mean(pp) for pp in past])
                return ref_function(past)
            
            def Utility_dynamic(x, params, reference = None):
                if reference == None:
                    if x.ndim == 1:
                        return (x ** params[0]) / ((x ** params[0]) + ( ref(params[1]) ** params[0]) )       
                    else:
                        return [(xx ** params[0]) / ((xx ** params[0]) + ( ref(params[1]) ** params[0])) for xx in np.array(x).T]
                else:
                    if x.ndim == 1:
                        return (x ** params[0]) / ((x ** params[0]) + ( reference ** params[0]) )       
                    else:
                        return [(xx ** params[0]) / ((xx ** params[0]) + ( reference ** params[0])) for xx in np.array(x).T]

            def Utility(x, params):
                if params[0]<0 or params[0]>1:
                    return [np.nan]*len(x)
                return ((1-params[0]) * Utility_dynamic(x, params[1:])) + (params[0] * Utility_dynamic(x, params[1:], reference = 0.5))
            
            return Utility
        
        pNames, p0 =  [['weighting', 'predisposition', 'n_pastTrials'], [0.5, 1.0, 0.000000000000005]]
        return define_utilityFromPast, pNames, p0
    
    #%%
    if functionName.lower() == 'esvt':
        def define_utilityFromPast(options):
#            rho, gamma = params
#            ref_function = lambda past, p1, t : np.where( (t==0), 0, np.sum([np.mean((p1 * ((1-p1)**(t-1-tau))) * xPast) for tau, xPast in enumerate(past)]) )
                
            aEVs = [np.array([gg[0]]) if gg[1] == 0 else gg for gg in options[:, :4:2]]
            bEVs = [np.array([gg[0]]) if gg[1] == 0 else gg for gg in options[:, 4::2]]
            allEVs = np.vstack((aEVs, bEVs)).T
            allEVs = [np.concatenate(evs) for evs in allEVs]
            allEVs =np.vstack(allEVs)
            
            #this is where p1 matters
            def ref(gamma):
                if gamma < 0 or gamma > 0.3:
                    return [np.nan]*len(allEVs)
                past = [ allEVs[:i] for i in range(len(allEVs)) ]
                past[0] = [np.mean(allEVs[0])]
                ref_function = lambda past: np.array([np.sum([ (gamma * ((1-gamma)**(t-tau))) * pp for tau, pp in enumerate(xPast)]) for t, xPast in enumerate(past)])
                return ref_function(past)
            
            def Utility(x, params):
                print(params[1])
                if x.ndim == 1:
                    return (x ** params[0]) / ((x ** params[0]) + ( ref(params[1]) ** params[0]) )       
                else:
                    return [(xx ** params[0]) / ((xx ** params[0]) + ( ref(params[1]) ** params[0])) for xx in np.array(x).T]
        
            return Utility
        
        pNames, p0 = [['predisposition', 'gamma'], [1, 0]]
        return define_utilityFromPast, pNames, p0
    
    #%%  
    if functionName.lower() == 'esvt_mean':
        def define_utilityFromPast(options):
#            rho, gamma = params
#            ref_function = lambda past, p1, t : np.where( (t==0), 0, np.sum([np.mean((p1 * ((1-p1)**(t-1-tau))) * xPast) for tau, xPast in enumerate(past)]) )
                
            aEVs = [np.array([gg[0]]) if gg[1] == 0 else gg for gg in options[:, :4:2]]
            bEVs = [np.array([gg[0]]) if gg[1] == 0 else gg for gg in options[:, 4::2]]
            allEVs = np.vstack((aEVs, bEVs)).T
            allEVs = [np.concatenate(evs) for evs in allEVs]
            allEVs =np.vstack(allEVs)
            
            #this is where p1 matters
            def ref(gamma):
                pastImpact = int(gamma / np.finfo('double').eps)
                if pastImpact > len(allEVs) or pastImpact <= 0:
                    return [np.nan]*len(allEVs)
                past = [ allEVs[i-pastImpact:i] if i-pastImpact > 0 else allEVs[0:i] for i in range(len(allEVs)) ]
                past[0] = np.mean(allEVs[0])
                ref_function = lambda past: np.array([np.mean(pp) for pp in past])
                return ref_function(past)
            
            def Utility(x, params):
                if x.ndim == 1:
                    return (x ** params[0]) / ((x ** params[0]) + ( ref(params[1]) ** params[0]) )       
                else:
                    return [(xx ** params[0]) / ((xx ** params[0]) + ( ref(params[1]) ** params[0])) for xx in np.array(x).T]
                    
            return Utility
        
        pNames, p0 = [['predisposition', 'n_pastTrials'], [1, 0.000000000000005]]
        return define_utilityFromPast, pNames, p0
    
    #%%
    if functionName.lower() == 'sigmoid':
        def define_utilityFromPast(options):
#            rho, gamma = params
#            ref_function = lambda past, p1, t : np.where( (t==0), 0, np.sum([np.mean((p1 * ((1-p1)**(t-1-tau))) * xPast) for tau, xPast in enumerate(past)]) )
                
            aEVs = [np.array([gg[0]]) if gg[1] == 0 else gg for gg in options[:, :4:2]]
            bEVs = [np.array([gg[0]]) if gg[1] == 0 else gg for gg in options[:, 4::2]]
            allEVs = np.vstack((aEVs, bEVs)).T
            allEVs = [np.concatenate(evs) for evs in allEVs]
            allEVs =np.vstack(allEVs)
            
            #this is where p1 matters
            def ref(gamma):
                pastImpact = int(gamma / np.finfo('double').eps)
#                if pastImpact > len(allEVs) or pastImpact <= 0:
#                    return [np.nan]*len(allEVs)
                past = [ allEVs[i-pastImpact:i] if i-pastImpact > 0 else allEVs[0:i] for i in range(len(allEVs)) ]
                past[0] = np.mean(allEVs[0])
                ref_function = lambda past: np.vstack([[np.mean(pp), np.mean(pp)] if i==0 else [np.mean(pp), np.std(pp)] for i, pp in enumerate(past)])
                return ref_function(past)
            
            def Utility(x, params):
                if x.ndim == 1:
                    return norm.cdf(x,  loc=ref(params)[:,0], scale=ref(params)[:,1])   
                else:
                    return [norm.cdf(xx,  loc=ref(params)[:,0], scale=ref(params)[:,1]) for xx in np.array(x).T]
            return Utility
        
        pNames, p0 =  [['n_pastTrials'], [ 0.000000000000005]]
        return define_utilityFromPast, pNames, p0
    
    #%%
    if functionName.lower() == '3power':
        def simple_rho(mm, params):
            p1, p2, p3 = params
            if p3 < 0 or p3 > 1 or p2 < 1 or p1 > 1:
                return [np.nan] * len(mm)
            mm = np.array(mm)
            result = np.ravel(np.where((mm >= p3),
                                 (mm-p3)**p1,
                                -p2*(p3-mm)**p1))
#            return (result - min(result)) / (max(result) - min(result))
            return result
        function = lambda mm, param_u: simple_rho(mm, param_u) 
        pNames = ['temperature', 'lossAversion', 'inflection']
        p0 = [1, 1, 0.25]
        return function, pNames, p0
    #%%
    if functionName.lower() == '2power':
        def simple_rho(mm, params):
            p1, p3 = params
            if p3 < 0 or p3 > 1 or p1 > 1:
                return [np.nan] * len(mm)
            mm = np.array(mm)
            result = np.ravel(np.where((mm >= p3),
                                 (mm-p3)**p1,
                                -(p3-mm)**p1))
#            return (result - min(result)) / (max(result) - min(result))
            return result
        function = lambda mm, param_u: simple_rho(mm, param_u) 
        pNames = ['temperature', 'inflection']
        p0 = [1, 0.25]
        return function, pNames, p0
    #%%    
    if functionName.lower() == '3scdf':
        def sCDF(mm, params):
            Xs = np.atleast_1d(mm) #range position
            inflection, temp, lossAversion = params[1], params[0], params[2]
            result = np.ravel([np.where(X<inflection, 
                                        (lossAversion*(inflection*((X/inflection)**temp))) - (1-((1-inflection)*(((1-inflection)/(1-inflection))**temp))), 
                                        1-((1-inflection)*(((1-X)/(1-inflection))**temp))) for X in Xs])
            return np.where((inflection > 1 or inflection < 0),
                            [0] * len(Xs),
                            (result - min(result)) / (max(result) - min(result)))
        function = lambda mm, param_u: sCDF(mm, param_u) 
        pNames = ['temperature', 'lossAversion', 'inflection']
        p0 = [1, 1, 0.5]
        return function, pNames, p0
    #%%
    if functionName.lower() == '2glimcher' or functionName.lower() == 'glimcher' :
        function = lambda mm, param_u: np.where( (param_u[1] > 1.2 or param_u[1] < 0),
                                                np.nan,                
                                                (mm ** param_u[0]) / ((mm ** param_u[0]) + ( param_u[1] ** param_u[0]) ) )      
        pNames = ['predisposition', 'reference']
        p0 = [1, 0.5]
        return function, pNames, p0
    #%%
    if functionName.lower() == '2prelec':
        def prelec(mm, params):
            p1, p2 = params
            mm = np.array(mm)
            return np.where((p2 > 10 or p1 > 10 or p1<0.1),
                                np.nan,
                                np.exp(-p2 * (-np.log(mm))**p1))
        function = lambda mm, param_u: prelec(mm, param_u) 
        pNames = ['temperature', 'height']
        p0 = [1, 0.5]
        return function, pNames, p0
    #%%        
    if functionName.lower() == 'prelec' or functionName.lower() == '1prelec':
        function = lambda mm, param_u: np.exp(-((-np.log(mm))**param_u))
        pNames = ['temperature']
        p0 = [1]
        return function, pNames, p0
    #%%    
    if functionName.lower() == '2scdf' or functionName.lower() == 'scdf':
        def sCDF(mm, params):
            Xs = np.atleast_1d(mm) #range position
            inflection, temp = params[1], params[0]
            return np.where((inflection > 1 or inflection < 0 or temp > 20 or temp < 0),
                            [0] * len(Xs),
                            np.ravel([np.where(X<inflection, inflection*((X/inflection)**temp), 1-((1-inflection)*(((1-X)/(1-inflection))**temp))) for X in Xs])
                            )
        function = lambda mm, param_u: sCDF(mm, param_u) 
        pNames = ['temperature', 'inflection']
        p0 = [1, 0.5]
        return function, pNames, p0
    #%%    
    if functionName.lower() == 'power' or functionName.lower() == '1power':
        function = lambda mm, param_u: mm**param_u
        pNames = ['temperature']
        p0 = [1]
        return function, pNames, p0
    #%%
    if functionName.lower() == '2beta':
        from scipy.stats import beta
        function = lambda mm, param_u: beta.cdf(mm, a=param_u[0], b=param_u[1]) 
        #sCDF (squared two-sided cummmulative distribution function)
        pNames = ['mean', 'variance']
        p0 = [1, 1]
        return function, pNames, p0
    #%%    
    if functionName.lower() == '3triangle':
        function = lambda mm, param_u: triangle.cdf(mm, c=param_u[2],loc= param_u[0], scale=param_u[1]) #sCDF (squared two-sided cummmulative distribution function)
        #sCDF (squared two-sided cummmulative distribution function)
        pNames = ['p1', 'p2', 'p3']
        p0 = [1, 1, 1]
        return function, pNames, p0
    #%%    
    if functionName.lower() == '2gonzalez':
        gonzalez = lambda p, param_w: (param_w[1] * p**param_w[0]) / (param_w[1] * p**param_w[0] + (1 - p)**param_w[0])
        function = lambda mm, param_u: gonzalez(mm, param_u)  #sCDF (squared two-sided cummmulative distribution function)
        pNames = ['temperature', 'height']
        p0 = [1, 1]
        return function, pNames, p0
    #%%    
    if functionName.lower() == 'tversky' or functionName.lower() == '1tversky':
        function = lambda p, param_w: (p ** param_w) / (((p**param_w) + ((1 - p)**param_w))**(1 / param_w))
        pNames = ['temperature']
        p0 = [1]
        return function, pNames, p0
    #%%    
    if functionName.lower() == 'ev':
        function = lambda mm: mm  #sCDF (squared two-sided cummmulative distribution function)
        pNames = []
        p0 = []
        return function, pNames, p0
    #%%
    if functionName.lower() == 'none':
        function = lambda mm: mm
        pNames = []
        p0 = []
        return function, pNames, p0

#%%
def get_value(U, W, parameterPositions, cumulative = False, dynamic = False):
    '''
    '''
    uParams = (parameterPositions == 0)
    wParams = (parameterPositions == 1)
    
    # Value = U(mm) + mu(U(mm) − U(reference)) #value given rabin's model
    # reference can be the past mean
    # or 1/2 * (u(x) + u(y)) + 1/2 * (1 − λ)|u(x) − u(y)|
    
    if cumulative != False:
        def VALUE(option, params):
            return np.sum([ U(option[:,i], params[uParams]) *  (1-W(option[:,3], params[wParams])) if i == 0 else U(option[:,i], params[uParams]) *  W(option[:,i+1], params[wParams]) for i in range(0,np.size(option, 1),2)], 0)  

#        np.sum([ (U(option[:,i], params[uParams]) * (1-W(option[:,i+3], params[wParams]))) if i==0 else (U(option[:,i], params[uParams]) * W(option[:,i+1], params[wParams])) for i in np.arange(0, np.size(option,1), 2) ], 0)
#        def VALUE(option, params):
#            val = []
#            for x in option:
#                val.extend( sum([((1-W(x[2+1], params[wParams])) * U(x[i],params[uParams])) if i==0 else (W(x[i+1], params[wParams]) * U(x[i],params[uParams])) for i in range(0, len(x), 2)]))
#            return np.array(val)
            
        return VALUE
#     ----------------------------------------------------
    else:
        if sum(uParams == True) == 0 and sum(wParams == True) == 0:
            VALUE = lambda option :  np.sum([ U(option[:,i]) *  W(option[:,i+1]) for i in range(0,np.size(option, 1),2)], 0)  
            return VALUE
        elif sum(uParams == True) == 0 and sum(wParams == True) > 0:
            VALUE = lambda option, params :  np.sum([ U(option[:,i]) *  W(option[:,i+1], params[wParams]) for i in range(0,np.size(option, 1),2)], 0)  
            return VALUE
        elif sum(wParams == True) == 0 and  sum(uParams == True) > 0:
            if dynamic == True: 
                def VALUE(option, X, params):
                    Utility = U(X)
                    return np.sum([ Utility(option[:,i], params[uParams]) *  W(option[:,i+1]) for i in range(0,np.size(option, 1),2)], 0)                  
                return VALUE
            else:
                VALUE = lambda option, params :  np.sum([ U(option[:,i], params[uParams]) *  W(option[:,i+1]) for i in range(0,np.size(option, 1),2)], 0)  
                return VALUE
        elif sum(uParams == True) > 0 and sum(wParams == True) > 0:
            VALUE = lambda option, params : np.sum([ U(option[:,i], params[uParams]) *  W(option[:,i+1], params[wParams]) for i in range(0,np.size(option, 1),2)], 0)  
            return VALUE

#%%
def trials_2fittable(trials, use_Outcomes = False):
    LL_df = pd.DataFrame([])
    LL_df['sessionDate'] = trials.sessionDate.values
    LL_df['A'] = trials.gambleA.values
    LL_df['A'] = LL_df['A'].apply(lambda x: x + [0] * (4 - len(x)))
    LL_df[['A_m1', 'A_p1', 'A_m2', 'A_p2']] = pd.DataFrame(
        LL_df.A.values.tolist(), index=LL_df.index)
    LL_df['B'] = trials.gambleB.values
    LL_df['B'] = LL_df['B'].apply(lambda x: x + [0] * (4 - len(x)))
    LL_df[['B_m1', 'B_p1', 'B_m2', 'B_p2']] = pd.DataFrame(
        LL_df.B.values.tolist(), index=LL_df.index)
    LL_df['chosenSide'] = [
        ord(char.lower()) - 96 for char in trials.gambleChosen
    ]
    #        LL_df['chG'] = np.array([1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
    #        LL_df['chS'] = 1 - LL_df['chG']
    LL_df['outcomes'] = trials['ml_received'].values
    #        LL_df['past'] = [[LL_df['outcomes'][i], LL_df['outcomes'][i-1]] for i in range(1, len(LL_df['outcomes']) )]
    LL_df['evA'] = np.array([
        sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in LL_df['A']
    ])
    LL_df['evB'] = np.array([
        sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in LL_df['B']
    ])
    LL_df.index = trials.index
    LL_df.drop(columns='A', inplace=True)
    LL_df.drop(columns='B', inplace=True)

    exog = LL_df[['A_m1', 'A_p1', 'A_m2', 'A_p2', 'B_m1', 'B_p1', 'B_m2', 'B_p2' ]]
    endog = np.abs(LL_df.chosenSide.values - 2)

    if use_Outcomes == True:
#        exog['chosenSide'] = trials.gambleChosen.apply(lambda x : ord(x) - 65)
        exog['outcomes'] = trials.ml_received
        exog['RT'] = trials.choiceTime
    return exog, endog


#%%
def plot_MLEfitting(MLE_df):
    '''
    '''
    fig, ax = plt.subplots(
        int(np.ceil(len(MLE_df) / 6)),
        6,
        squeeze=False,
        figsize=(2.5 * int(np.ceil(len(MLE_df) / 10)), 15))
    fig.suptitle(
        'Function Value Estimates (and parameters) during minimization')
    row = 0
    col = 0
    colors = ['orange', 'teal', 'm', 'limegreen']
    for n in range(10 * int(np.ceil(len(MLE_df) / 10))):
        if col == 0:
            ax[row, col].set_ylabel('- LL')
        if row == int(np.ceil(len(MLE_df) / 6)) - 1:
            ax[row, col].set_xlabel('runs')
        if n >= len(MLE_df):
            ax[row, col].set_visible(False)
            col += 1
            continue
        ax[row, col].plot(MLE_df.iloc[n]['Nfeval'], linewidth=3, color='k')
        ax[row, col].set_title(str(MLE_df.loc[n]['date']))

        for dd in range(len(MLE_df.loc[n]['params'])):
            cc = colors[dd]
            ax2 = ax[row, col].twinx()
            ax2.spines["right"].set_position(("axes", 1 + 0.1 * dd))
            ax2.spines["right"].set_visible(True)

            ax2.plot(np.vstack(MLE_df.loc[n].all_fits)[:, dd], cc, alpha=0.4)
            if col == 5:
                ax2.set_ylabel('param %d' % dd, color=cc)
            ax2.tick_params('y', colors=cc)

        col += 1
        if col > 5:
            row += 1
            col = 0
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    #plot the evolution of the LL and of the parameters
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(15, 4))
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

    colors = ['orange', 'teal', 'm', 'limegreen']
    for dd in range(len(NM_fit.params.values[0])):
        cc = colors[dd]
        ax2 = ax[0, 0].twinx()
        ax2.spines["right"].set_position(("axes", 1 + 0.055 * dd))
        ax2.spines["right"].set_visible(True)

        ax2.plot(np.vstack(NM_fit.params.values)[:, dd], cc, alpha=0.4)
        ax2.set_ylabel('param %d' % dd, color=cc)
        ax2.tick_params('y', colors=cc)
    plt.show()

#%%

def get_modelLL(trials, Model):
    '''
    '''
    #    import pdb ; pdb.set_trace() #AWESOME WAY TO DEBUG
    np.warnings.filterwarnings('ignore')
    mLow = 0  #min(np.concatenate(Trials[['GA_ev','GB_ev']].values))
    mMax = 0.5  #max(np.concatenate(Trials[['GA_ev','GB_ev']].values))

    #%%
    if Model.lower() == 'ev_attention':
        pNames = ['softmax', 'attention']
        bounds = None

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]

        gambleSide = [1 if side[0] == 2 else 2 for side in trials.outcomesCount]
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        sEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in safes
        ])
        gEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in gambles
        ])

        Pg = lambda params: 1 / (1 + np.exp(-params[0] * (params[1]*gEV - (1-params[1])*sEV)))
        LL = lambda params: sum(chG * np.log((Pg(params) ))) + sum(chS * np.log(((1 - Pg(params)) )))
        x0 = [10, 0.5]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'prospect_3param_attention':
        U = lambda mm, param_u: ((mm - mLow) / (mMax - mLow))**param_u
        W = lambda p, param_w: np.exp(-((-np.log(p))**param_w))
        #prelec 1 param
        pNames = ['softmax', 'utility', 'probability', 'attention']
        bounds = None

        #need to add the attention parameter at the correct place

        def func_Pg(U, W, gamble, safes):
            m1 = gamble[:, 0::2][:, 0]
            m2 = gamble[:, 0::2][:, 1]
            p1 = gamble[:, 1::2][:, 0]
            p2 = gamble[:, 1::2][:, 1]
            Pg = lambda params : 1 / (1 + np.exp( -params[0] * ( params[3]*(W(p1,params[2])*U(m1,params[1]) + W(p2,params[2])*U(m2,params[1])) - (1-params[3])*U(safes,params[1]) )  )  )
            return Pg

        safes = [ A if side[0] == 1 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount) ]
        gambles = [ A if side[0] == 2 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount) ]

        gambleSide = [1 if side[0] == 2 else 2 for side in trials.outcomesCount]
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(  [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)] )
        chS = 1 - chG

        safes = np.array(safes)[:, 0]
        gambles = np.vstack(gambles)

        Pg = func_Pg(U, W, gambles, safes)
        LL = lambda params: sum(chG * np.log((Pg(params)))) + \
             sum(chS * np.log((1 - Pg(params))))
        x0 = [10, 1, 1, 0.5]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'ev_withwsls_2param':
        pNames = ['softmax', 'wsls-chance']
        bounds = None

        def WL_outcome(past):
            xx = np.vstack([
                np.array([1, 0]) if past1 >= past2 else np.array([0, 1])
                for past1, past2 in past
            ])
            if past[0][1] > 0:
                xx = np.insert(xx, 0, np.array([1, 0]), 0)
            else:
                xx = np.insert(xx, 0, np.array([0, 1]), 0)
            return xx

        def dynamicProbabilities(past, chG):
            pwstay = 1
            plshift = 1
            pchG = [0.5]
            pchS = [0.5]
            win = WL_outcome(past)[:, 0]

            for w, chg in zip(win, chG):
                if chg == 1:
                    if w == 1:
                        pchG.extend([pwstay])
                        pchS.extend([1 - pwstay])
                    else:
                        pchG.extend([1 - plshift])
                        pchS.extend([plshift])
                elif chg == 0:
                    if w == 1:
                        pchG.extend([1 - pwstay])
                        pchS.extend([pwstay])
                    else:
                        pchG.extend([plshift])
                        pchS.extend([1 - plshift])
            pchG = np.array(pchG[:-1])
            pchS = np.array(pchS[:-1])
            return pchG, pchS

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]

        gambleSide = [1 if side[0] == 2 else 2 for side in trials.outcomesCount]
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        outcomes = trials['ml_received'].values
        past = [[outcomes[i], outcomes[i - 1]] for i in range(1, len(outcomes))]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        sEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in safes
        ])
        gEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in gambles
        ])

        Pg = lambda params: 1 / (1 + np.exp(-params * (gEV - sEV)))

        WSLS = dynamicProbabilities(past, chG)
        LL = lambda params: sum(chG * np.log((Pg(params[0]) * (1 - params[1])) + params[1] * WSLS[0])) + sum(chS * np.log(((1 - Pg(params[0])) * (1 - params[1])) + params[1] * WSLS[1]))

        x0 = [10, 0.5]
        return LL, x0, pNames

    #%%

    elif Model.lower() == 'ev_withwsls_4param':
        pNames = ['softmax', 'wsls-chance', 'winStay', 'loseShift']
        bounds = None

        def WL_outcome(past):
            xx = np.vstack([
                np.array([1, 0]) if past1 >= past2 else np.array([0, 1])
                for past1, past2 in past
            ])
            if past[0][1] > 0:
                xx = np.insert(xx, 0, np.array([1, 0]), 0)
            else:
                xx = np.insert(xx, 0, np.array([0, 1]), 0)
            return xx

        def dynamicProbabilities(past, chG, params):
            pwstay = params[0]
            plshift = params[1]
            pchG = [0.5]
            pchS = [0.5]
            win = WL_outcome(past)[:, 0]

            for w, chg in zip(win, chG):
                if chg == 1:
                    if w == 1:
                        pchG.extend([pwstay])
                        pchS.extend([1 - pwstay])
                    else:
                        pchG.extend([1 - plshift])
                        pchS.extend([plshift])
                elif chg == 0:
                    if w == 1:
                        pchG.extend([1 - pwstay])
                        pchS.extend([pwstay])
                    else:
                        pchG.extend([plshift])
                        pchS.extend([1 - plshift])
            pchG = np.array(pchG[:-1])
            pchS = np.array(pchS[:-1])
            return pchG, pchS

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]

        gambleSide = [1 if side[0] == 2 else 2 for side in trials.outcomesCount]
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        outcomes = trials['ml_received'].values
        past = [[outcomes[i], outcomes[i - 1]] for i in range(1, len(outcomes))]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        sEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in safes
        ])
        gEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in gambles
        ])

        Pg = lambda pSM: 1 / (1 + np.exp(-pSM * (gEV - sEV)))

        def LL(params):
            pchG, pchS = dynamicProbabilities(past, chG, params[2::])
            return sum(chG*np.log( (Pg(params[0])*(1-params[1])) + params[1]*pchG )) +\
                   sum(chS*np.log( ((1-Pg(params[0]))*(1-params[1])) + params[1]*pchS ))

        x0 = [10, 0.5, 0.5, 0.5]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'prospect_3param':
        U = lambda mm, param_u: ((mm - mLow) / (mMax - mLow))**param_u
        W = lambda p, param_w: np.exp(-((-np.log(p))**param_w))  #prelec 1 param
        pNames = ['softmax', 'utility', 'probability']
        bounds = None

        def func_Pg(U, W, gamble, safes):
            m1 = gamble[:, 0::2][:, 0]
            m2 = gamble[:, 0::2][:, 1]
            p1 = gamble[:, 1::2][:, 0]
            p2 = gamble[:, 1::2][:, 1]
            Pg = lambda params : 1 / (1 + np.exp( -params[0] * ( (W(p1,params[2:])*U(m1,params[1]) + W(p2,params[2:])*U(m2,params[1])) - U(safes,params[1]))  )  )
            return Pg

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]

        gambleSide = [1 if side[0] == 2 else 2 for side in trials.outcomesCount]
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        safes = np.array(safes)[:, 0]
        gambles = np.vstack(gambles)

        Pg = func_Pg(U, W, gambles, safes)
        gambles
        safes
        LL = lambda params: sum(chG * np.log(Pg(params))) + sum(chS * np.log(1 - Pg(params)))
        x0 = [1, 1, 1]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'prospect_3param_side':
        U = lambda mm, param_u: ((mm - mLow) / (mMax - mLow))**param_u
        W = lambda p, param_w: np.exp(-((-np.log(p))**param_w))
        #prelec 1 param
        dS = lambda gSide, param_s: param_s * gSide  #deltaSide #p1=choice, p2=sideBias, p3=utility, p4=probability
        pNames = ['softmax', 'sideBias', 'utility', 'probability']
        bounds = None

        def func_Pg(U, W, dS, gamble, safes, gSide):
            m1 = gamble[:, 0::2][:, 0]
            m2 = gamble[:, 0::2][:, 1]
            p1 = gamble[:, 1::2][:, 0]
            p2 = gamble[:, 1::2][:, 1]
            Pg = lambda params : 1 / (1 + np.exp( -params[0] *\
                                                 ( ((W(p1,params[3:])*U(m1,params[2]) + W(p2,params[3:])*U(m2,params[2]))\
                                                    + dS(gSide, params[1])) - U(safes,params[2]) )  )  )
            return Pg

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]

        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        safes = np.array(safes)[:, 0]
        gambles = np.vstack(gambles)

        Pg = func_Pg(U, W, dS, gambles, safes, gambleSide)
        LL = lambda params: sum(chG * np.log(Pg(params))) + sum(chS * np.log(1 - Pg(params)))
        x0 = [1, 0, 1, 1]
        return LL, x0, pNames
        #calculates the least square error on the predicted and elicited CEs

    #%%
    elif Model.lower() == 'wsls_simple':
        pNames = ['pWinStay', 'pLoseShift']
        bounds = None

        def WL_outcome(past):
            xx = np.vstack([
                np.array([1, 0]) if past1 >= past2 else np.array([0, 1])
                for past1, past2 in past
            ])
            if past[0][1] > 0:
                xx = np.insert(xx, 0, np.array([1, 0]), 0)
            else:
                xx = np.insert(xx, 0, np.array([0, 1]), 0)
            return xx

        def dynamicProbabilities(past, chG, params):
            pwstay = params[0]
            plshift = params[1]
            pchG = [0.5]
            pchS = [0.5]
            win = WL_outcome(past)[:, 0]

            for w, chg in zip(win, chG):
                if chg == 1:
                    if w == 1:
                        pchG.extend([pwstay])
                        pchS.extend([1 - pwstay])
                    else:
                        pchG.extend([1 - plshift])
                        pchS.extend([plshift])
                elif chg == 0:
                    if w == 1:
                        pchG.extend([1 - pwstay])
                        pchS.extend([pwstay])
                    else:
                        pchG.extend([plshift])
                        pchS.extend([1 - plshift])
            pchG = np.array(pchG[:-1])
            pchS = np.array(pchS[:-1])
            return pchG, pchS

        outcomes = trials['ml_received'].values
        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = np.array(
            [ord(char.lower()) - 96 for char in trials.gambleChosen])

        #this needs to be the y of it all!
        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG
        past = [[outcomes[i], outcomes[i - 1]] for i in range(1, len(outcomes))]

        def LL(params):
            pchG, pchS = dynamicProbabilities(past, chG, np.array(params))
            return sum(chG * np.log(pchG)) + sum(chS * np.log(pchS))

        x0 = [0.5, 0.5]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'wsls_dynamic':
        pNames = ['pWStay', 'pLShift', 'dWStay', 'dLShift']
        bounds = [[0, 1], [0, 1], [-np.inf, np.inf], [-np.inf, np.inf]]

        def WL_outcome(past):
            xx = np.vstack([
                np.array([1, 0]) if past1 >= past2 else np.array([0, 1])
                for past1, past2 in past
            ])
            if past[0][1] > 0:
                xx = np.insert(xx, 0, np.array([1, 0]), 0)
            else:
                xx = np.insert(xx, 0, np.array([0, 1]), 0)
            return xx

        def dynamicProbabilities(past, chG, params):
            pwstay = params[0]
            plshift = params[1]
            deltaPws = params[2]
            deltaPls = params[3]
            pchG = [0.5]
            pchS = [0.5]
            win = WL_outcome(past)[:, 0]

            for w, chg in zip(win, chG):
                if chg == 1:
                    if w == 1:
                        pwstay = pwstay + deltaPws * (1 - pwstay)
                        plshift = (1 - deltaPls) * plshift
                        pchG.extend([pwstay])
                        pchS.extend([1 - pwstay])
                    else:
                        pwstay = (1 - deltaPws) * pwstay
                        plshift = plshift + deltaPls * (1 - plshift)
                        pchG.extend([1 - plshift])
                        pchS.extend([plshift])
                elif chg == 0:
                    if w == 1:
                        pwstay = pwstay + deltaPws * (1 - pwstay)
                        plshift = (1 - deltaPls) * plshift
                        pchG.extend([1 - pwstay])
                        pchS.extend([pwstay])
                    else:
                        pwstay = (1 - deltaPws) * pwstay
                        plshift = plshift + deltaPls * (1 - plshift)
                        pchG.extend([plshift])
                        pchS.extend([1 - plshift])

            pchG = np.array(pchG[:-1])
            pchS = np.array(pchS[:-1])
            return pchG, pchS

        outcomes = trials['ml_received'].values
        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = np.array(
            [ord(char.lower()) - 96 for char in trials.gambleChosen])

        #this needs to be the y of it all!
        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG
        past = [[outcomes[i], outcomes[i - 1]] for i in range(1, len(outcomes))]

        def LL(params):
            pchG, pchS = dynamicProbabilities(past, chG, np.array(params))
            return sum(chG * np.log(pchG)) + sum(chS * np.log(pchS))

        x0 = [0.5, 0.5, 0, 0]
        return LL, x0, pNames

    #%%
    # remodel using daeyol lee's psychometric function
    elif Model.lower() == 'rl':

        pNames = ['gamma', 'learning rate']
        bounds = [[0, 1]]

        def rlEVs(gIndex, outcomes, chS, safes, param):
            nEV = lambda pEV, pRew, alpha: pEV + (alpha * (pRew - pEV))

            startEV = np.array(
                [[.05, .1, .15, .2, .25, .3, .35, .4, .45, 0]] * len(chS))
            startEV[:, -1] = safes
            gIndex = gIndex.astype(int)

            for i in range(0, len(gIndex) - 1):
                if chS[i] == 1:
                    startEV[[startEV[:, 9] == safes[i]]][:, -1] = nEV(
                        safes[i], outcomes[i], param)
                elif chS[i] == 0:
                    startEV[i + 1:, :][:, gIndex[i]] = nEV(
                        startEV[i, :][gIndex[i]], outcomes[i], param)
            gEVs = np.array([gEV[gLoc] for gEV, gLoc in zip(startEV, gIndex)])
            sEVs = startEV[:, -1]
            return gEVs, sEVs

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)
        ]

        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        safes = np.array(safes)
        gambles = np.array(gambles)
        gIndex = (gambles * 2 * 10) - 1
        outcomes = trials['ml_received'].values

        def LL(params):
            gEVs, sEVs = rlEVs(gIndex, outcomes, chS, safes, params[1])
            Pg = lambda gamma: 1 / (1 + np.exp(-gamma * (gEVs - sEVs)))
            return sum(chG * np.log(Pg(params[0]))) + sum(
                chS * np.log(1 - Pg(params[0])))

        x0 = [1, 0]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'rl_pastgamble':
        '''
        '''
        pNames = ['gamma', 'learning rate', 'ws_gamble']
        bounds = [[0, 1], [-.5, .5]]

        def rlEVs(gIndex, outcomes, chS, safes, param):
            nEV = lambda pEV, pRew, alpha: pEV + (alpha * (pRew - pEV))

            startEV = np.array(
                [[.05, .1, .15, .2, .25, .3, .35, .4, .45, 0]] * len(chS))
            startEV[:, -1] = safes
            gIndex = gIndex.astype(int)

            chG = 1 - chS
            gOutcome = [
                rew if chg else np.nan for chg, rew in zip(chG, outcomes)
            ]
            gWin = [1 if gg == 0.5 else gg for gg in gOutcome]
            WLstate = [0]

            for i in range(0, len(gIndex) - 1):
                if not np.isnan(gWin[i]):
                    WLstate.extend([gWin[i]])
                else:
                    WLstate.extend([WLstate[i]])

                if chS[i] == 1:
                    startEV[[startEV[:, 9] == safes[i]]][:, -1] = nEV(
                        safes[i], outcomes[i], param)
                elif chS[i] == 0:
                    startEV[i + 1:, :][:, gIndex[i]] = nEV(
                        startEV[i, :][gIndex[i]], outcomes[i], param)
            gEVs = np.array([gEV[gLoc] for gEV, gLoc in zip(startEV, gIndex)])
            sEVs = startEV[:, -1]
            return gEVs, sEVs, np.array(WLstate)

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)
        ]

        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        safes = np.array(safes)
        gambles = np.array(gambles)
        gIndex = (gambles * 2 * 10) - 1
        outcomes = trials['ml_received'].values

        def LL(params):
            gEVs, sEVs, WLstate = rlEVs(gIndex, outcomes, chS, safes, params[1])
            Pg = lambda gamma, gWSLS: 1 / (1 + np.exp(-gamma * (gEVs - sEVs + (gWSLS * WLstate))))
            return sum(chG * np.log(Pg(params[0], params[2]))) + sum(
                chS * np.log(1 - Pg(params[0], params[2])))
            #redu this formula with daeyole lee's function rather than this one

        x0 = [1, 0, 0]
        return LL, x0, pNames
    #%%

    elif Model.lower() == 'rl_reinforce_winlose':
        '''
        '''
        pNames = ['gamma', 'learning rate', 'winlose', 'ceta']
        bounds = [[0, 1], [-.5, .5]]

        def rlEVs(gIndex, outcomes, chS, safes, param):
            alpha = param[0]
            beta = param[1]
            ceta = param[2]

            nEV = lambda pEV, pRew, alpha: pEV + (alpha * (pRew - pEV))
            def pEV( pEV, gOut, beta, ceta):
                if gOut == 0.5:
                     return pEV + beta * (gOut-pEV)
                elif gOut == 0.0:
                     return pEV + ceta * (gOut-pEV)
                else:
                    return pEV

            startEV = np.array(
                [[.05, .1, .15, .2, .25, .3, .35, .4, .45, 0]] * len(chS))
            startEV[:, -1] = safes
            gIndex = gIndex.astype(int)

            chG = 1 - chS
            gOutcome = [  rew if chg else np.nan for chg, rew in zip(chG, outcomes)  ]
            gWin = [1 if gg == 0.5 else gg for gg in gOutcome]
            gWin = [-1 if gg == 0 else gg for gg in gWin]
            state = 0
            for i in range(0, len(gIndex) - 1):
                if chS[i] == 1:
                    startEV[[startEV[:, 9] == safes[i]]][:, -1] = nEV(safes[i], outcomes[i], alpha)
                elif chS[i] == 0:
                    startEV[i + 1:, :][:, gIndex[i]] = nEV( startEV[i, :][gIndex[i]], outcomes[i], alpha)

                if not np.isnan(gWin[i]):
                    startEV[i + 1, :9] = pEV(startEV[i+1, :9], gOutcome[i], beta, ceta)
                    state = gOutcome[i]
                else:
                    startEV[i+1,:9] = pEV(startEV[i+1, :9], state, beta, ceta)

            # ----------------------------------------------------------

            gEVs = np.array([gEV[gLoc] for gEV, gLoc in zip(startEV, gIndex)])
            sEVs = startEV[:, -1]
            return gEVs, sEVs

        safes = [  A if side[0] == 1 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount) ]
        gambles = [ A if side[0] == 2 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)  ]

        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        safes = np.array(safes)
        gambles = np.array(gambles)
        gIndex = (gambles * 2 * 10) - 1
        outcomes = trials['ml_received'].values

        def LL(params):
            gEVs, sEVs = rlEVs(gIndex, outcomes, chS, safes, params[1:])
            Pg = lambda gamma : 1 / (1 + np.exp(-gamma * (gEVs - sEVs)))
            return sum(chG * np.log(Pg(params[0]))) + sum( chS * np.log(1 - Pg(params[0])))

        x0 = [1, 0, 0, 0]
        return LL, x0, pNames
    #%%
    elif Model.lower() == 'rl_simple_winlose':
        '''
        '''
        pNames = ['gamma', 'learning rate', 'win', 'ceta']
        bounds = [[0, 1], [-.5, .5]]

        def rlEVs(gIndex, outcomes, chS, safes, param):
            beta = param[0]
            ceta = param[1]
#            ceta = param[2]

            def pEV( pEV, gOut, beta, ceta):
                if gOut == 0.5:
                     return pEV + beta * (gOut-pEV)
                elif gOut == 0.0:
                     return pEV + ceta * (gOut-pEV)
                else:
                    return pEV

            startEV = np.array(
                [[.05, .1, .15, .2, .25, .3, .35, .4, .45, 0]] * len(chS))
            startEV[:, -1] = safes
            gIndex = gIndex.astype(int)

            chG = 1 - chS
            gOutcome = [  rew if chg else np.nan for chg, rew in zip(chG, outcomes)  ]
            gWin = [1 if gg == 0.5 else gg for gg in gOutcome]
            gWin = [-1 if gg == 0 else gg for gg in gWin]
            state = 0
            for i in range(0, len(gIndex) - 1):
                if not np.isnan(gWin[i]):
                    startEV[i + 1:, :9] = pEV(startEV[i, :9], gOutcome[i], beta, ceta)

            # ----------------------------------------------------------

            gEVs = np.array([gEV[gLoc] for gEV, gLoc in zip(startEV, gIndex)])
            sEVs = startEV[:, -1]
            return gEVs, sEVs

        safes = [  A if side[0] == 1 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount) ]
        gambles = [ A if side[0] == 2 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)  ]

        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        safes = np.array(safes)
        gambles = np.array(gambles)
        gIndex = (gambles * 2 * 10) - 1
        outcomes = trials['ml_received'].values

        def LL(params):
            gEVs, sEVs = rlEVs(gIndex, outcomes, chS, safes, params[1:])
            Pg = lambda gamma : 1 / (1 + np.exp(-gamma * (gEVs - sEVs)))
            return sum(chG * np.log(Pg(params[0]))) + sum( chS * np.log(1 - Pg(params[0])))

        x0 = [1, 0, 0]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'rl_reinforce_winlose_fixed':
        '''
        '''
        pNames = ['gamma', 'learning rate', 'ws_gamble', 'ceta']
#        bounds = [[0, 1], [-.5, .5]]

        def rlEVs(gIndex, outcomes, chS, safes, param):
            alpha = param[0]
            beta = param[1]
            ceta = param[2]
            nEV = lambda pEV, pRew, alpha: pEV + (alpha * (pRew - pEV))

            startEV = np.array(
                [[.05, .1, .15, .2, .25, .3, .35, .4, .45, 0]] * len(chS))
            startEV[:, -1] = safes
            gIndex = gIndex.astype(int)

            chG = 1 - chS
            gOutcome = [  rew if chg else np.nan for chg, rew in zip(chG, outcomes)  ]
            gWin = [1 if gg == 0.5 else gg for gg in gOutcome]
            gWin = [-1 if gg == 0 else gg for gg in gWin]
            state = 0
            for i in range(0, len(gIndex) - 1):
                if chS[i] == 1:
                    startEV[[startEV[:, 9] == safes[i]]][:, -1] = nEV(safes[i], outcomes[i], alpha)
                elif chS[i] == 0:
                    startEV[i + 1:, :][:, gIndex[i]] = nEV( startEV[i, :][gIndex[i]], outcomes[i], alpha)
#                 ------------------------------
                if not np.isnan(gWin[i]):
                    if gWin[i] == 1:
                        startEV[i + 1, :9] = startEV[i + 1, :9] + beta * gWin[i]
                    else:
                        startEV[i + 1, :9] = startEV[i + 1, :9] + ceta * gWin[i]
                    state = gWin[i]
                else:
                    if gWin[i] == 1:
                        startEV[i + 1, :9] = startEV[i + 1, :9] + beta * state
                    else:
                        startEV[i + 1, :9] = startEV[i + 1, :9] + ceta * state
#                    startEV[i+1,:9] = startEV[i + 1, :9] + beta * state

            gEVs = np.array([gEV[gLoc] for gEV, gLoc in zip(startEV, gIndex)])
            sEVs = startEV[:, -1]
            return gEVs, sEVs

        safes = [  A if side[0] == 1 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount) ]
        gambles = [ A if side[0] == 2 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)  ]

        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        safes = np.array(safes)
        gambles = np.array(gambles)
        gIndex = (gambles * 2 * 10) - 1
        outcomes = trials['ml_received'].values

        def LL(params):
            gEVs, sEVs = rlEVs(gIndex, outcomes, chS, safes, params[1:])
            Pg = lambda gamma : 1 / (1 + np.exp(-gamma * (gEVs - sEVs)))
            return sum(chG * np.log(Pg(params[0]))) + sum( chS * np.log(1 - Pg(params[0])))

        x0 = [1, 0, 0, 0]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'rl_reinforce_winlose_adhoc':
        '''
        '''
        pNames = ['gamma', 'learning rate', 'ws_gamble', 'ceta']
#        bounds = [[0, 1], [-.5, .5]]

        def rlEVs(gIndex, outcomes, chS, safes, param):
            alpha = param[0]
            beta = param[1]
            nEV = lambda pEV, pRew, alpha: pEV + (alpha * (pRew - pEV))

            startEV = np.array(
                [[.05, .1, .15, .2, .25, .3, .35, .4, .45, 0]] * len(chS))
            startEV[:, -1] = safes
            gIndex = gIndex.astype(int)

            chG = 1 - chS
            gOutcome = [  rew if chg else np.nan for chg, rew in zip(chG, outcomes)  ]
            gWin = [1 if gg == 0.5 else gg for gg in gOutcome]
            gWin = [-1 if gg == 0 else gg for gg in gWin]
            WLstate=[0]
            for i in range(0, len(gIndex) - 1):
                if not np.isnan(gWin[i]):
                    WLstate.extend([gWin[i]])
                else:
                    WLstate.extend([gWin[i]])

                if chS[i] == 1:
                    startEV[[startEV[:, 9] == safes[i]]][:, -1] = nEV(safes[i], outcomes[i], alpha)
                elif chS[i] == 0:
                    startEV[i + 1:, :][:, gIndex[i]] = nEV( startEV[i, :][gIndex[i]], outcomes[i], alpha)
#                 ------------------------------

            gEVs = np.array([gEV[gLoc] for gEV, gLoc in zip(startEV, gIndex)])
            gEVs = [gg+(beta*wl) for gg,wl in zip(gEVs, np.nancumsum(np.array(WLstate)))]
            sEVs = startEV[:, -1]
            return gEVs, sEVs

        safes = [  A if side[0] == 1 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount) ]
        gambles = [ A if side[0] == 2 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)  ]

        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        safes = np.array(safes)
        gambles = np.array(gambles)
        gIndex = (gambles * 2 * 10) - 1
        outcomes = trials['ml_received'].values

        def LL(params):
            gEVs, sEVs = rlEVs(gIndex, outcomes, chS, safes, params[1:])
            Pg = lambda gamma : 1 / (1 + np.exp(-gamma * (gEVs - sEVs)))
            return sum(chG * np.log(Pg(params[0]))) + sum( chS * np.log(1 - Pg(params[0])))

        x0 = [1, 0, 0, 0]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'rl_simone':
        '''
        '''
        pNames = ['gamma', 'learning rate']
#        bounds = [[0, 1], [-.5, .5]]

        def rlEVs(gIndex, outcomes, chS, safes, param):
            alpha = param[0]
            nEV = lambda pEV, alpha, winlose: pEV + (alpha * (winlose))

            startEV = np.array(
                [[.05, .1, .15, .2, .25, .3, .35, .4, .45, 0]] * len(chS))
            startEV[:, -1] = safes
            gIndex = gIndex.astype(int)

            chG = 1 - chS
            gOutcome = [  rew if chg else np.nan for chg, rew in zip(chG, outcomes)  ]
            gWin = [1 if gg == 0.5 else gg for gg in gOutcome]
            gWin = [-1 if gg == 0 else gg for gg in gWin]
            gWin = np.nan_to_num(gWin)
            WLstate=[0]
            for i in range(0, len(gIndex) - 1):
#                if not np.isnan(gWin[i]):
#                    WLstate.extend([gWin[i]])
#                else:
#                    WLstate.extend([gWin[i]])

                if chS[i] == 1:
                    startEV[[startEV[:, 9] == safes[i]]][:, -1] = nEV(safes[i], alpha, gWin[i])
                elif chS[i] == 0:
                    startEV[i + 1:, :][:, gIndex[i]] = nEV( startEV[i, :][gIndex[i]],  alpha, gWin[i])
#                 ------------------------------

            gEVs = np.array([gEV[gLoc] for gEV, gLoc in zip(startEV, gIndex)])
#            gEVs = [gg+(beta*wl) for gg,wl in zip(gEVs, np.nancumsum(np.array(WLstate)))]
            sEVs = startEV[:, -1]
            return gEVs, sEVs

        safes = [  A if side[0] == 1 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount) ]
        gambles = [ A if side[0] == 2 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)  ]

        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        safes = np.array(safes)
        gambles = np.array(gambles)
        gIndex = (gambles * 2 * 10) - 1
        outcomes = trials['ml_received'].values

        def LL(params):
            gEVs, sEVs = rlEVs(gIndex, outcomes, chS, safes, params[1:])
            Pg = lambda gamma : 1 / (1 + np.exp(-gamma * (gEVs - sEVs)))
            return sum(chG * np.log(Pg(params[0]))) + sum( chS * np.log(1 - Pg(params[0])))

        x0 = [1, 0]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'rl_simone_utility':
        '''
        '''
        pNames = ['gamma', 'learning rate', 'utility']

        def rlEVs(gIndex, outcomes, chS, safes, param):
            alpha = param[0]
            rho = param[1]
            nEV = lambda pEV, alpha, winlose: pEV + (alpha * (winlose))

            startEV = np.array(
                [[.05, .1, .15, .2, .25, .3, .35, .4, .45, 0]] * len(chS))
            startEV[:, -1] = safes
            gIndex = gIndex.astype(int)

            chG = 1 - chS
            gOutcome = [  rew if chg else np.nan for chg, rew in zip(chG, outcomes)  ]
            gWin = [1 if gg == 0.5 else gg for gg in gOutcome]
            gWin = [-1 if gg == 0 else gg for gg in gWin]
            gWin = np.nan_to_num(gWin)
            WLstate=[0]
            for i in range(0, len(gIndex) - 1):
                if chS[i] == 1:
                    startEV[[startEV[:, 9] == safes[i]]][:, -1] = nEV(safes[i], alpha, gWin[i])
                elif chS[i] == 0:
                    startEV[i + 1:, :][:, gIndex[i]] = nEV( startEV[i, :][gIndex[i]],  alpha, gWin[i])
#                 ------------------------------

            gEVs = np.array([gEV[gLoc] for gEV, gLoc in zip(startEV, gIndex)]) * 2
#            gEVs = [gg+(beta*wl) for gg,wl in zip(gEVs, np.nancumsum(np.array(WLstate)))]
            sEVs = (startEV[:, -1]/0.5) ** rho
            return gEVs, sEVs

        safes = [  A if side[0] == 1 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount) ]
        gambles = [ A if side[0] == 2 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)  ]

        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        safes = np.array(safes)
        gambles = np.array(gambles)
        gIndex = (gambles * 2 * 10) - 1
        outcomes = trials['ml_received'].values

        def LL(params):
            gEVs, sEVs = rlEVs(gIndex, outcomes, chS, safes, params[1:])
            Pg = lambda gamma : 1 / (1 + np.exp(-gamma * (gEVs - sEVs)))
            return sum(chG * np.log(Pg(params[0]))) + sum( chS * np.log(1 - Pg(params[0])))

        x0 = [1, 0, 1]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'rl_wsls_5param':
        pNames = ['pWinStay', 'pLoseShift', 'useWSLS', 'gamma', 'alpha']
        bounds = None

        def rlEVs(gIndex, outcomes, chS, safes, param):
            nEV = lambda pEV, pRew, alpha: pEV + (alpha * (pRew - pEV))

            startEV = np.array(
                [[.05, .1, .15, .2, .25, .3, .35, .4, .45, 0]] * len(chS))
            startEV[:, -1] = safes
            gIndex = gIndex.astype(int)

            for i in range(0, len(gIndex) - 1):
                if chS[i] == 1:
                    startEV[[startEV[:, 9] == safes[i]]][:, -1] = nEV(
                        safes[i], outcomes[i], param)
                elif chS[i] == 0:
                    startEV[i + 1:, :][:, gIndex[i]] = nEV(
                        startEV[i, :][gIndex[i]], outcomes[i], param)
            gEVs = np.array([gEV[gLoc] for gEV, gLoc in zip(startEV, gIndex)])
            sEVs = startEV[:, -1]
            return gEVs, sEVs

        def WL_outcome(past):
            xx = np.vstack([
                np.array([1, 0]) if past1 >= past2 else np.array([0, 1])
                for past1, past2 in past
            ])
            if past[0][1] > 0:
                xx = np.insert(xx, 0, np.array([1, 0]), 0)
            else:
                xx = np.insert(xx, 0, np.array([0, 1]), 0)
            return xx

        def dynamicProbabilities(past, chG, params):
            pwstay = params[0]
            plshift = params[1]
            pchG = [0.5]
            pchS = [0.5]
            win = WL_outcome(past)[:, 0]

            for w, chg in zip(win, chG):
                if chg == 1:
                    if w == 1:
                        pchG.extend([pwstay])
                        pchS.extend([1 - pwstay])
                    else:
                        pchG.extend([1 - plshift])
                        pchS.extend([plshift])
                elif chg == 0:
                    if w == 1:
                        pchG.extend([1 - pwstay])
                        pchS.extend([pwstay])
                    else:
                        pchG.extend([plshift])
                        pchS.extend([1 - plshift])
            pchG = np.array(pchG[:-1])
            pchS = np.array(pchS[:-1])
            return pchG, pchS

        outcomes = trials['ml_received'].values
        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = np.array(
            [ord(char.lower()) - 96 for char in trials.gambleChosen])

        #this needs to be the y of it all!
        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG
        past = [[outcomes[i], outcomes[i - 1]] for i in range(1, len(outcomes))]

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)
        ]

        safes = np.array(safes)
        gambles = np.array(gambles)
        gIndex = (gambles * 2 * 10) - 1

        def LL(params):
            pchG, pchS = dynamicProbabilities(past, chG, np.array(params[0:2]))
            gEVs, sEVs = rlEVs(gIndex, outcomes, chS, safes, params[4])
            Pg = lambda gamma: 1 / (1 + np.exp(-gamma * (gEVs - sEVs)))
            return sum(
                chG * np.log(pchG * params[2] +
                             (Pg(params[3]) * (1 - params[2])))) + sum(
                                 chS * np.log(pchS * params[2] + (
                                     (1 - Pg(params[3])) * (1 - params[2]))))

        x0 = [0.5, 0.5, 0.5, 1, 0]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'rl_wsls_3param':
        pNames = ['alpha', 'beta', 'gamma']
        bounds = None

        def rlEVs(gIndex, outcomes, chS, safes, param):
            nEV = lambda pEV, pRew, alpha: pEV + (alpha * (pRew - pEV))
            wsls = lambda outcome: 0.5 if outcome > 0 else 0
            wslsEV = lambda pEV, outcome, beta: pEV + (beta * (wsls(outcome) - pEV))

            startEV = np.array(
                [[.05, .1, .15, .2, .25, .3, .35, .4, .45, 0]] * len(chS))
            startEV[:, -1] = safes
            gIndex = gIndex.astype(int)

            for i in range(0, len(gIndex) - 1):
                if chS[i] == 1:
                    startEV[[startEV[:, 9] == safes[i]]][:, -1] = nEV(
                        safes[i], outcomes[i], param[0])
                elif chS[i] == 0:
                    startEV[i + 1:, :][:, gIndex[i]] = nEV(
                        startEV[i, :][gIndex[i]], outcomes[i],
                        param[0])  #specific prediction error
                    startEV[i + 1:, :][:, gIndex[i]] = wslsEV(
                        startEV[i + 1:, :][:, gIndex[i]], outcomes[i],
                        param[1])  #wsls prediction error

            gEVs = np.array([gEV[gLoc] for gEV, gLoc in zip(startEV, gIndex)])
            sEVs = startEV[:, -1]
            return gEVs, sEVs

        outcomes = trials['ml_received'].values
        gambleSide = np.array(
            [1 if side[0] == 2 else 2 for side in trials.outcomesCount])
        chosenSide = np.array(
            [ord(char.lower()) - 96 for char in trials.gambleChosen])

        #this needs to be the y of it all!
        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG
        past = [[outcomes[i], outcomes[i - 1]] for i in range(1, len(outcomes))]

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.GA_ev, trials.GB_ev, trials.outcomesCount)
        ]

        safes = np.array(safes)
        gambles = np.array(gambles)
        gIndex = (gambles * 2 * 10) - 1

        def LL(params):
            gEVs, sEVs = rlEVs(gIndex, outcomes, chS, safes, params[0:2])
            Pg = lambda gamma: 1 / (1 + np.exp(-gamma * (gEVs - sEVs)))
            return sum(chG * np.log(Pg(params[2]))) + sum(chS * np.log(
                (1 - Pg(params[2]))))

        x0 = [0, 0, 10]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'ev':
        pNames = ['softmax']
        bounds = None

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]

        gambleSide = [1 if side[0] == 2 else 2 for side in trials.outcomesCount]
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        sEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in safes
        ])
        gEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in gambles
        ])

        Pg = lambda params: 1 / (1 + np.exp(-params * (gEV - sEV)))
        LL = lambda params: sum(chG * np.log(Pg(params[0]))) + sum(chS * np.log(1 - Pg(params[0])))

        x0 = [10]
        return LL, x0, pNames

    #%%
    elif Model.lower() == 'ev+wsls':

        pNames = ['softmax', 'winBonus']
        bounds = None

        def gstate(chG, chS, outcomes):
            gOutcome = [
                rew if chg else np.nan for chg, rew in zip(chG, outcomes)
            ]
            gWin = [1 if gg == 0.5 else gg for gg in gOutcome]
            gWin = [-1 if gg == 0 else gg for gg in gWin]
            WLstate = [0]

            for i in range(0, len(gOutcome) - 1):
                if not np.isnan(gWin[i]):
                    WLstate.extend([gWin[i]])
                else:
                    WLstate.extend([WLstate[i]])
            return np.array(WLstate)

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]

        gambleSide = [1 if side[0] == 2 else 2 for side in trials.outcomesCount]
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        sEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in safes
        ])
        gEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in gambles
        ])
        outcomes = trials['ml_received'].values

        Pg = lambda params : 1 / (1 + np.exp( -params[0] * ( gEV + gstate(chG,chS,outcomes)*params[1] - sEV ) ) )
        LL = lambda params: sum(chG * np.log(Pg(params))) + sum(chS * np.log(1 - Pg(params)))

        x0 = [10, 0]
        return LL, x0, pNames

    #%%

    elif Model.lower() == 'prospect+wsls':

        def gstate(chG, chS, outcomes):
            gOutcome = [
                rew if chg else np.nan for chg, rew in zip(chG, outcomes)
            ]
            gWin = [1 if gg == 0.5 else gg for gg in gOutcome]
            #            gWin = [ -1 if gg==0 else gg for gg in gWin]
            WLstate = [0]

            for i in range(0, len(gOutcome) - 1):
                if not np.isnan(gWin[i]):
                    WLstate.extend([gWin[i]])
                else:
                    WLstate.extend([WLstate[i]])
            return np.array(WLstate)

        def func_Pg(U, W, gamble, safes, wsls):
            m1 = gamble[:, 0::2][:, 0]
            m2 = gamble[:, 0::2][:, 1]
            p1 = gamble[:, 1::2][:, 0]
            p2 = gamble[:, 1::2][:, 1]
            Pg = lambda params : 1 / (1 + np.exp( -params[0] * ( (W(p1,params[2])*U(m1,params[1]) + W(p2,params[2])*U(m2,params[1])) +\
                                                 wsls*params[3] - U(safes,params[1]))  )  )
            return Pg

        U = lambda mm, param_u: ((mm - mLow) / (mMax - mLow))**param_u
        W = lambda p, param_w: np.exp(-((-np.log(p))**param_w))  #prelec 1 param
        pNames = ['softmax', 'utility', 'probability', 'winBonus']
        bounds = None

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]

        gambleSide = [1 if side[0] == 2 else 2 for side in trials.outcomesCount]
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        sEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in safes
        ])
        gEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in gambles
        ])
        outcomes = trials['ml_received'].values

        safes = np.array(safes)[:, 0]
        gambles = np.vstack(gambles)

        wsls = gstate(chG, chS, outcomes)

        Pg = func_Pg(U, W, gambles, safes, wsls)
        LL = lambda params: sum(chG * np.log(Pg(params))) + sum(chS * np.log(1 - Pg(params)))

        x0 = [10, 1, 1, 0]
        return LL, x0, pNames

    #%%

    elif Model.lower() == 'ev_wsls_prob':

        pNames = ['softmax', 'winBonus']
        bounds = None

        def gstate(chG, chS, outcomes):
            gOutcome = [
                rew if chg else np.nan for chg, rew in zip(chG, outcomes)
            ]
            gWin = [1 if gg == 0.5 else gg for gg in gOutcome]
            gWin = [-1 if gg == 0 else gg for gg in gWin]
            WLstate = [0]

            for i in range(0, len(gOutcome) - 1):
                if not np.isnan(gWin[i]):
                    WLstate.extend([gWin[i]])
                else:
                    WLstate.extend([WLstate[i]])
            return np.array(WLstate)

        safes = [
            A if side[0] == 1 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]
        gambles = [
            A if side[0] == 2 else B for A, B, side in zip(
                trials.gambleA, trials.gambleB, trials.outcomesCount)
        ]

        gambleSide = [1 if side[0] == 2 else 2 for side in trials.outcomesCount]
        chosenSide = [ord(char.lower()) - 96 for char in trials.gambleChosen]

        chG = np.array(
            [1 if gS == cS else 0 for gS, cS in zip(gambleSide, chosenSide)])
        chS = 1 - chG

        sEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in safes
        ])
        gEV = np.array([
            sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]) for x in gambles
        ])
        outcomes = trials['ml_received'].values

        Pg = lambda params: 1 / (1 + np.exp(-params[0] * (gEV - sEV)))
        LL = lambda params : sum(chG*np.log( Pg(params) + gstate(chG,chS,outcomes)*params[1] )) +  sum(chS*np.log( 1-(Pg(params)+gstate(chG,chS,outcomes)*params[1]) ))

        x0 = [10, 0]
        return LL, x0, pNames

#%%

#%%
def plot_softmax(self, params = [], color='k'):
    plotData = np.vstack(( np.linspace(0,1,100), np.ones(100), np.zeros(100),
                np.zeros(100), np.linspace(1,0,100), np.ones(100),
                np.zeros(100), np.zeros(100))).T

    if params != []:
        functions = self.model.model_parts['prob_chA'](params)
        yy = self.model.model_parts['prob_chA'](params)
    else:
        yy = self.softmax(plotData)
    xx = np.linspace(0,1,100) - np.linspace(1,0,100)

    ax = plt.gca()
    ax.plot(xx,yy, linewidth = 2, color=color)

    if params != []:
        value = self.model.model_parts['value']
        vA = value(self.model.exog[:,0:4], params[2:])
        vB = value(self.model.exog[:,4:], params[2:])
    else:
        value = self.model_parts['value']
        vA = value(self.model.exog[:,0:4])
        vB = value(self.model.exog[:,4:])
    deltaVal = vA - vB
    chA = self.endog

    dvu = np.unique(deltaVal)
    chu = []
    for dd in dvu:
        chu.extend([sum(chA[deltaVal == dd]) / len(chA[deltaVal == dd])])

    ax.plot(dvu,chu, 'o', color = color)
    ax.axvline(0, color = 'k', linestyle = '--')
    ax.grid(which='both',axis='x')
    ax.set_title('choice noise')
    ax.set_ylabel('probability of left choice')
    ax.set_xlabel('value difference')

def plot_utility(self, CI = False, color='k'):
    util = self.utility(np.linspace(0,1,100))
    ax = plt.gca()
    ax.plot(np.linspace(0,1,100), util, color = color, linewidth=3)
    ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), color = 'k',
            linestyle = '--', alpha = 0.7)
    ax.grid()
    ax.set_title('utility')
    ax.set_ylabel('utility')
    ax.set_xlabel('reward magnitude')

    if CI == True:
        conf_int = MLE.conf_int().T[:,1:-1].T
        if len(conf_int,1) == 2:
            yy =[]
            product = ((i, j) for i in conf_int[0] for j in conf_int[1])
            for i, j in product:
                yy.append(self.model_parts['empty functions']['utility'](np.linspace(0,1,100), [i,j]))
            upper, lower = np.percentile(yy, [2.5, 97.5], axis=0)
        elif len(conf_int,1) == 1:
            for i in conf_int[0]:
                yy.append(self.model_parts['empty functions']['utility'](np.linspace(0,1,100), i))
            upper, lower = np.percentile(yy, [2.5, 97.5], axis=0)
        ax.plot( np.linspace(0,1,100), lower, color='b', alpha=0.25)
        ax.plot(np.linspace(0,1,100), upper, color='b', alpha=0.25)

def plot_probability(self, color='k'):
    prob = self.probability(np.linspace(0,1,100))
    ax = plt.gca()
    ax.plot(np.linspace(0,1,100) ,prob, color = color)
    ax.plot(np.linspace(0,1,100),np.linspace(0,1,100), color = 'b',
            linestyle = '--', alpha = 0.7)
    ax.grid()
    ax.set_title('probability distortion')
    ax.set_ylabel('subjective probability')
    ax.set_xlabel('objective probability')

def plot_fullModel(self, title=None, fig = None, ax = None, color = None, return_fig=False):
    if fig == None and ax == None:
        fig, ax = plt.subplots(1, 3,figsize=(8, 3))
    if color == None:
        color = 'lightgrey'
        
    if title == None:
        fig.suptitle(self.model.model_name)
    else:
        fig.suptitle(title)
    plt.sca(ax[2])
    plot_probability(self, color=color)
    plt.sca(ax[0])
    plot_softmax(self, color=color)
    plt.sca(ax[1])
    plot_utility(self, color=color)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if return_fig == True:
        return fig, ax

def animateBehaviour(self, threeAxis = True, saveName=None):
    '''
    '''
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from scipy.stats import norm
    import matplotlib.pylab as pl

    defData = np.linspace(np.min(self.exog),np.max(self.exog),100)
    Utilities = np.vstack(self.utility(np.tile(defData, (len(self.exog),1)), self.exog)).T
    normalize = lambda x :  x / max( x )

    noise = (1/self.params[0]) / np.sqrt(2)
    param_past = int(self.params[1] / np.finfo('double').eps)
    def multipleDistributions(ax, data, utility):
        ax.collections.clear()
        locations = np.percentile(utility, np.array([10,30,50,70,90]) * np.max(self.exog[:,0]) )
        ax.fill_between(data, np.zeros(len(data)), norm.pdf(data, loc=locations[0], scale=noise), color = 'blue', alpha = 0.3)
        ax.fill_between(data, np.zeros(len(data)), norm.pdf(data, loc=locations[1], scale=noise), color = 'blue', alpha = 0.15)
        ax.fill_between(data, np.zeros(len(data)), norm.pdf(data, loc=locations[2], scale=noise), color = 'purple', alpha = 0.10)
        ax.fill_between(data, np.zeros(len(data)), norm.pdf(data, loc=locations[3], scale=noise), color = 'red', alpha = 0.15)
        ax.fill_between(data, np.zeros(len(data)), norm.pdf(data, loc=locations[4], scale=noise), color = 'red', alpha = 0.3)

#    plt.ion()
    if threeAxis == True:
        fig, ax = plt.subplots( 1, 3, figsize=( 9,4 ))
    else:
        fig, ax = plt.subplots( 1, 2, figsize=( 6,3 ))
    plt.tight_layout(rect=[0.003, 0.03, 0.95, 0.95])
    plt.suptitle("Change in 'Utility'")

    def prep_axes(g, xlim, ylim):
        g.set_xlim(xlim)
        g.set_ylim(ylim)
    def animate(utility):
        global text, line, line2, i
        text.set_visible(False)
        line.set_alpha(0.05); line2.set_alpha(0.05)
        line.set_color('k'); line2.set_color('k')
        prep_axes(ax[1], [-0.05,1.05], [-0.05,1.05])
        prep_axes(ax[0], [-0.05,1.05], [-0.05,1.05])
        line, = ax[1].plot( defData, utility, color='b', alpha = 1, linewidth = 2)
        line2, = ax[0].plot( defData, normalize(np.gradient(utility)),
                   color='b', alpha = 1, linewidth = 2)
        text = ax[1].text(0.05,.95,'Current Trial: ' + str(i),
                 bbox=dict(facecolor='none', edgecolor='black', pad=2))
        if threeAxis == True:
            multipleDistributions(ax[2], defData, utility)
        i+=1
    def init_animation():
        global line, line2, text, i
        i=0
        ax[0].clear(); ax[1].clear()
        ax[0].set_title('last ' + str(param_past) + ' values distribution')
        ax[1].set_title('normalized value')
        ax[1].plot(defData, np.mean(Utilities.T, axis=1), color='r', linewidth=3)
        ax[0].plot(defData, normalize(np.gradient(np.mean(Utilities.T, axis=1))), color='r', linewidth=3)
        ax[1].plot(defData, defData, '--', color='k')
        ax[0].axhline(0, linestyle='--', color = 'k')
        line, = ax[1].plot( defData, defData, color='b', alpha = 1, linewidth = 2)
        line2, = ax[0].plot( defData, normalize(np.gradient(defData)),
                   color='b', alpha = 1, linewidth = 2)
        ax[1].set_ylabel('relative value'); ax[1].set_xlabel('reward magnitude')
        ax[0].set_ylabel('past distribution'); ax[0].set_xlabel('reward magnitude')
        text = ax[0].text(0.05,.95,'Current Trial: ' + str(i),
                 bbox=dict(facecolor='none', edgecolor='black', pad=2))
        if threeAxis == True:
            multipleDistributions(ax[2], defData, np.mean(Utilities, axis=0))
            ax[2].set_ylabel('density of value signals'); ax[2].set_xlabel('value of magnitudes')
            ax[2].set_title('value distributions')
            ax[2].axhline(0, color = 'k')
    ani = animation.FuncAnimation(fig, animate, init_func=init_animation,
                                             frames=Utilities, repeat=True,
                                             interval=0.1, repeat_delay=3000)
    fig.show()
    if saveName !=None:
        import matplotlib.pyplot as plt
        plt.rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.8-Q16/magick.exe'
        writer = animation.ImageMagickFileWriter()
        ani.save(saveName + '.gif', writer = 'imagemagick', bitrate=1800)

    #need to make a 3rd subplot where I can show points change in real time.
    return ani

def animateValue(self, saveName=None):
    '''
    '''
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from scipy.stats import norm
    import matplotlib.pylab as pl

    defData = np.linspace(0,1,11)
    Gambles = self.model_parts['changingGambleValues'](self.exog)

    noise = ((1/self.params[0]) / np.sqrt(2) ) /2 #half the noise (for each side)
    param_past = self.params[1]

    fig, ax = plt.subplots( 1, 2, figsize=( 6,3 ))
    plt.tight_layout(rect=[0.003, 0.03, 0.95, 0.95])
    plt.suptitle("Change in gamble 'value'")

    def multipleDistributions(ax, data, utility):
        ax.collections.clear()
        ax.fill_between(data, utility-noise/4, utility+noise/4, color = 'blue', alpha = 0.2)
        ax.fill_between(data, utility-noise/2, utility+noise/2, color = 'blue', alpha = 0.1)
        ax.fill_between(data, utility-noise, utility+noise, color = 'blue', alpha = 0.05)

    def prep_axes(g, xlim, ylim):
        g.set_xlim(xlim)
        g.set_ylim(ylim)
    def animate(gambles):
        global text, line, line2, i
        ax[1].clear()
        ax[1].grid(axis='x')
        prep_axes(ax[1], [-0.05,1.05], [-0.15,0.15])
        ax[1].plot(defData, defData-defData, '--', color='k')
        text.set_visible(False)
        line.set_alpha(0.01)
        line.set_color('k')
        prep_axes(ax[0], [-0.05,1.05], [-0.05,1.05])
        line, = ax[0].plot( defData, gambles, color='b', alpha = 1, linewidth = 2)
        line2, = ax[1].plot( defData, gambles-defData, color='b', alpha = 1, linewidth = 2)
        multipleDistributions(ax[1], defData, gambles-defData)
        #put the noise on it
        text = ax[0].text(0.05,.95,'Current Trial: ' + str(i),
                 bbox=dict(facecolor='none', edgecolor='black', pad=2))
        i+=1
    def init_animation():
        global line, line2, text, i
        i=0
        prep_axes(ax[1], [-0.05,1.05], [-0.15,0.15])
        ax[0].clear(); ax[1].clear()
        ax[0].set_title('gamble value')
        ax[1].set_title('value differences')
        ax[0].plot(defData, np.mean(Gambles.T, axis=1), '--', color='r', linewidth=3)
        ax[0].plot(defData, defData, '--', color='k')
        ax[1].plot(defData, np.mean(Gambles.T, axis=1) - defData, '--', color='r', linewidth=3)
        ax[1].plot(defData, defData-defData, '--', color='k')
        line, = ax[0].plot( defData, defData, color='b', alpha = 1, linewidth = 2)
        line2, = ax[1].plot( defData, defData-defData, color='b', alpha = 1, linewidth = 2)
        ax[0].set_ylabel('relative value'); ax[1].set_xlabel('value of gambles')
        ax[1].set_ylabel('relative value'); ax[1].set_xlabel('valGmb - valSafe')
        text = ax[0].text(0.05,.95,'Current Trial: ' + str(i),
                 bbox=dict(facecolor='none', edgecolor='black', pad=2))
        ax[0].grid(axis='both')
        ax[1].grid(axis='x')

    ani = animation.FuncAnimation(fig, animate, init_func=init_animation,
                                             frames=Gambles, repeat=True,
                                             interval=0.1, repeat_delay=3000)
    fig.show()
    if saveName !=None:
        import matplotlib.pyplot as plt
        plt.rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.8-Q16/magick.exe'
        writer = animation.ImageMagickFileWriter()
        ani.save(saveName + '.gif', writer = 'imagemagick', bitrate=1800)
    #need to make a 3rd subplot where I can show points change in real time.
    return ani

def simulate_Choices(self, trials, n=1, params = None, plotTQDM = True):
    '''
    '''
    from macaque.f_choices import get_options
    from macaque.f_psychometrics import get_softmaxData
    simulatedTrials = pd.DataFrame(columns = ['monkey', 'sessionDate',
       'time', 'trialNo', 'blockNo', 'gambleA',
       'gambleB', 'outcomesCount', 'GA_ev', 'GB_ev', 'gambleChosen',
       'trialSequenceMode', 'sequenceFilename', 'choiceTime',
       'j_firstStimulus', 'error', 'errorType', 'ml_received', 'ml_drank',
       'j_leftGain', 'j_rightGain', 'j_xOffset', 'j_yOffset', 'stimulusOn',
       'j_onStimulus'])

    simulatedTrials = trials.copy()
    if params != None:
        pchA = self.model.model_parts(params)['prob_chA'](self.exog)
    else:
        pchA = self.p_chooseA(self.exog)

    simulatedChoices = []
    df = []
    for _ in tqdm(range(n),  desc='simulating dataset', disable=not plotTQDM):
        choice = [1 if pp>=rr else 0 for pp, rr in zip(pchA, np.random.rand(len(pchA)))]
#        simulatedChoices.append( ['A' if cc==1 else 'B' for cc in choice] )
        simulatedTrials['gambleChosen'] = ['A' if cc==1 else 'B' for cc in choice]
        df.append(simulatedTrials.copy())
    return df

def simulate_CE(self, trials, n=1, mergeBy = 'block',
                                metricType = 'CE',
                                mergeSequentials=True,
                                minSecondaries = 4,
                                minChoices = 4,
                                plotTQDM = True):
    '''
    '''
    from macaque.f_choices import get_options
    from macaque.f_psychometrics import get_softmaxData
    simulatedTrials = pd.DataFrame(columns = ['monkey', 'sessionDate',
       'time', 'trialNo', 'blockNo', 'gambleA',
       'gambleB', 'outcomesCount', 'GA_ev', 'GB_ev', 'gambleChosen',
       'trialSequenceMode', 'sequenceFilename', 'choiceTime',
       'j_firstStimulus', 'error', 'errorType', 'ml_received', 'ml_drank',
       'j_leftGain', 'j_rightGain', 'j_xOffset', 'j_yOffset', 'stimulusOn',
       'j_onStimulus'])

    #get real CEs
    choiceData = get_options(trials, mergeBy = mergeBy,
                                      byDates = False,
                                      mergeSequentials = mergeSequentials)
    softmax = choiceData.getPsychometrics(metricType = metricType,
                                            minSecondaries = minSecondaries,
                                            minChoices = minChoices)
    softmax.sort_values(by='primaryEV', inplace=True)
    realCEs = softmax.equivalent.values
    realEVs = softmax.primaryEV.values

    #Simulate Datapoints
    simulatedTrials = trials.copy()
    gA = [[gg[0],gg[1]] if gg[1] == 1.0 else gg for gg in self.model.exog[:,0:4].tolist()]
    gB = [[gg[0],gg[1]] if gg[1] == 1.0 else gg for gg in self.model.exog[:,4:].tolist()]
    simulatedTrials['gambleA'] = gA
    simulatedTrials['gambleB'] = gB

    pchA = self.p_chooseA(self.exog)
    simulatedCEs = []
    simulatedOptions = []
    for _ in tqdm(range(n),  desc='simulating dataset', disable=not plotTQDM):
        choice = [1 if pp>=rr else 0 for pp, rr in zip(pchA, np.random.rand(len(pchA)))]
        simulatedTrials['gambleChosen'] = ['A' if cc==1 else 'B' for cc in choice]
        simulatedChoice = get_options(simulatedTrials, mergeBy = mergeBy,
                                      byDates = False,
                                      mergeSequentials = mergeSequentials)
        simulatedPsychometric = simulatedChoice.getPsychometrics(metricType = metricType,
                                                minSecondaries = minSecondaries,
                                                minChoices = minChoices)
        simulatedOptions.append(simulatedPsychometric.primaryEV.values)
        simulatedCEs.append(simulatedPsychometric.equivalent.values)

    simulatedOptions = np.vstack(simulatedOptions)
    jitter = np.vstack([np.random.normal(gg, 0.005, size=len(gg)) for gg in simulatedOptions])

    ax = plt.gca()
    ax.plot(realEVs*2, realCEs*2, color = 'k', marker='o')
    ax.scatter(jitter*2, simulatedCEs, alpha = 0.25)
    ax.scatter(realEVs*2, np.median(np.vstack(simulatedCEs), axis=0),  marker='_', s=1000, color='r')
    ax.plot(np.linspace(0,2,100), np.linspace(0,2,100), '--', color = 'k', alpha = 0.5)
    ax.set_xlim(left = 0, right = 1)
    ax.set_ylim(bottom = -0, top = 1.2)
    ax.grid()

#%%
class LL_fit(GenericLikelihoodModel):
    '''
    Class on which the maximum log likelihood is defined - simply put trials in, and a model.
    '''
    def __init__(self, endog, exog, **kwds):
        from macaque.f_models import define_model

        if 'model' in kwds:
            model = kwds['model']
        else:
            print('NEED TO SPECIFY MODEL')
            print('Using utility-specific as a default')
            model = 'utility-specific'
            
        if 'fixedRange' in kwds:
            fixedRange = kwds['fixedRange']
        else:
            fixedRange = False
            
#        if 'dynamic' in model:
#            ll, start_params, exog_names, model_parts = define_util_LL(model, fixedRange=fixedRange)
#        else:
        ll, start_params, exog_names, model_parts = define_model(model, fixedRange=fixedRange)
        #defining import class variables#
        self.ll = ll
        self.start_params = start_params
        self.xnames = exog_names
        self.model_name = model
        self.model_parts = model_parts
        super(LL_fit, self).__init__(endog, exog, **kwds)

    #%%
    def nloglikeobs(self, params):
        return -self.ll(self.endog, self.exog, params)
    #%%
    def cross_validation(self):
        #Simulate Datapoints
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        error = []; params=[]
        Y = self.endog
        X = self.exog

        for train_index, test_index in tqdm(loo.split(X), desc='leave-one-out CV'):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            fitting = LL_fit( Y_train, X_train, model = self.model_name).fit(disp=0)
            function = fitting.model.model_parts(fitting.params)['prob_chA']
            params.append(fitting.params)

            pchA_test = function(X_test)
            chA_test = sum(pchA_test >= np.random.rand(1))
            error.extend(chA_test - Y_test)

        CVresults = {'MSE' : np.mean(np.array(error)**2),
                     'score' : 1-np.mean(np.array(error)**2),
                     'parameters' : np.vstack(params),
                     'falsePositives' : np.mean([1 if cc == 1 else 0 for cc in error]),
                     'falseNegatives' : np.mean([1 if cc == -1 else 0 for cc in error])}
        return CVresults
    #%%
    def fit(self, maxiter=10000, maxfun=5000, disp = 0, **kwds):
        #define parameter names for the summary
        self.data.xnames = self.xnames
        #define parameter for callback
        res_x = []

        fitted = super(LL_fit, self).fit(
            start_params=self.start_params,
            disp=disp,
            maxiter=maxiter,
            maxfun=maxfun,
            trend='nc',
            callback=res_x.append,
            retall=True,
            **kwds)

        # now the unique characteristics of the model that I want to keep
        fitted.res_x = res_x
        fitted.Nfeval = [-(self.ll(self.endog, self.exog, xs)) for xs in fitted.res_x]
        fitted.model_parts = self.model_parts(fitted.params)

        fitted.p_chooseA = fitted.model_parts['prob_chA']
        fitted.utility = fitted.model_parts['utility']
        fitted.probability = fitted.model_parts['probability_distortion']
        plotData = np.vstack(( np.linspace(0,1,100), np.ones(100), np.zeros(100),
                np.zeros(100), np.linspace(1,0,100), np.ones(100),
                np.zeros(100), np.zeros(100))).T
        fitted.softmax = fitted.model_parts['prob_chA']

#        if self.model_name == 'utility-random':
#            print(0)
        fitted.plot_softmax = MethodType( plot_softmax, fitted )
        fitted.plot_utility = MethodType( plot_utility, fitted )
        fitted.plot_probability = MethodType( plot_probability, fitted )
        fitted.plot_fullModel = MethodType( plot_fullModel, fitted )
        fitted.simulate_Choices = MethodType ( simulate_Choices, fitted )
        fitted.simulate_CE = MethodType( simulate_CE, fitted )

        if 'dynamic' in self.model_name.lower():
           fitted.values = np.vstack((fitted.model_parts['value'](fitted.exog[:,:4], fitted.exog),
                           fitted.model_parts['value'](fitted.exog[:,4:], fitted.exog))).T
           fitted.animateValue = MethodType( animateBehaviour, fitted)
#        elif self.model_name.lower() == 'dynamic-rl':
#           fitted.animateValue = MethodType( animateValue, fitted)
#           fitted.values = np.vstack((fitted.model_parts['value'](fitted.exog[:,:4]),
#                                       fitted.model_parts['value'](fitted.exog[:,4:8]))).T
        else:
            fitted.values = np.vstack((fitted.model_parts['value'](fitted.exog[:,:4]),
                                       fitted.model_parts['value'](fitted.exog[:,4:8]))).T
        return fitted
