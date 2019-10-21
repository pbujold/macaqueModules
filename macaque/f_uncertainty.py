# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 17:24:52 2018

@author: phbuj
"""
import numpy as np
import pandas as pd
from macaque.f_toolbox import *
tqdm = ipynb_tqdm()

#%%


def bootstrap_y(xData, yData, function, pZero, method='residuals', n=10000):
    '''
    '''
    import scipy.optimize as opt
    from scipy import interpolate
    np.warnings.filterwarnings('ignore')

    def ff(a, b):
        popt, pcov = opt.curve_fit(function, a, b, p0=pZero, method='lm')
        return lambda x: function(x, popt[0], popt[1])
    #-----------------------------------
    #apply this to the spline CE... might be wrong.
    def get_meanXY(data):
        x = []
        y = []
        data = np.squeeze(data)
        for xx in np.unique(data[:, 0]):
            index = data[:, 0] == xx
            x.extend([xx])
            y.extend([np.mean(data[:, 1][index])])
        return np.array(x), np.array(y)
    #----------------------------------------------------------
    data = np.array(np.split(np.array([xData, yData]), len(xData), axis=1))
    xMean, yMean = get_meanXY(data)
    yMean = yMean[np.argsort(xMean)]
    xMean = np.sort(xMean)

    xFull = np.linspace(min(xMean), max(xMean), 100)

    fun = ff(xMean, yMean)
    yHat = fun(xMean)
    resid = yMean - yHat

    b1 = []
    for i in tqdm(range(0, n), desc='bootstrapping confidence interval: '):
        if method.lower() == 'residuals':
            residBoot = np.random.permutation(resid)
            booty = yHat + residBoot
            bootFun = ff(xMean, booty)
        elif method.lower() == 'resampling':
            xb = np.random.choice(range(len(data)), len(data), replace=True)
            bootSample = np.hstack(data[xb])
            bootSample = bootSample[:, np.argsort(bootSample[0])]
            bootSample = np.array(np.split(bootSample, len(data), axis=1))
            bootx, booty = get_meanXY(bootSample)
            try:
                bootFun = ff(bootx, booty)
            except:
                continue
        b1.append(bootFun(xFull))

    b1 = np.vstack(b1)
    mean = fun(xFull)
    upper, lower = np.percentile(b1, [2.5, 97.5], axis=0)
    return mean, upper, lower, yHat


#%%
def bootstrap_fractile(xData,
                       yData,
                       function='sCDF',
                       method='residuals',
                       n=10000):
    '''
    '''
    import scipy.optimize as opt
    from scipy import interpolate
    sigmoid = lambda x, p1, p2: np.array(1 / (1 + np.exp(-(x - p1) / p2)))  #logistic sigmoid function (SAME AS ABOVE)
    prelec = lambda x, p1, p2: np.exp(-p2 * (-np.log(x))**p1)

    #prelec 2 params

    #-----------------------------------
    #apply this to the spline CE... might be wrong.
    def get_CEmean(data):
        x = []
        y = []
        data = np.squeeze(data)
        for util in np.unique(data[:, 0]):
            indexer = data[:, 0] == util
            y.extend([util])
            x.extend([np.mean(data[:, 1][indexer])])
        return np.array(x), np.array(y)

    #----------------------------------------------------------
    data = np.array(np.split(np.array([yData, xData]), len(xData), axis=1))
    xMean, yMean = get_CEmean(data)
    xFull = np.linspace(min(xMean), max(xMean), 100)

    if function.lower() == 'scdf':
        param_bounds = ([min(xFull), 1], [max(xFull), np.inf])
        popt, pcov = opt.curve_fit(
            sCDF,
            xMean,
            yMean,
            p0=[np.mean([min(xFull), max(xFull)]), 1],
            method='trf',
            bounds=param_bounds)
        #        popt, pcov = opt.curve_fit(sCDF, xMean, yMean, p0 = [0.99,2],  method='trf')
        fun = lambda x: sCDF(x, popt[0], popt[1])
    elif function.lower() == 'spline':
        fun = interpolate.interp1d(xMean, yMean, kind='quadratic')


#        fun = interpolate.LSQUnivariateSpline(xMean,yMean, t=np.mean([min(yMean),max(yMean)]))
    elif function.lower() == 'softmax':
        popt, pcov = opt.curve_fit(
            sigmoid, xMean, yMean, p0=[1, 1], method='trf')
        fun = lambda x: sigmoid(x, popt[0], popt[1])
    elif function.lower() == 'prelec':
        popt, pcov = opt.curve_fit(
            prelec, xMean, yMean, p0=[1, 1], method='trf')
        fun = lambda x: prelec(x, popt[0], popt[1])
    yHat = fun(xMean)
    resid = yHat - yMean

    b1 = []
    for i in range(0, n):
        if method.lower() == 'residuals':
            residBoot = np.random.permutation(resid)
            booty = yHat + residBoot
            if function.lower() == 'scdf':
                popt, pcov = opt.curve_fit(
                    sCDF,
                    xMean,
                    booty,
                    p0=[np.mean([min(xMean), max(xMean)]), 1],
                    method='trf',
                    bounds=param_bounds)
                #                popt, pcov = opt.curve_fit(sCDF, xMean, booty, p0 = [0.99,2],  method='trf')
                bootFun = lambda x: sCDF(x, popt[0], popt[1])
            elif function.lower() == 'spline':
                #                bootFun = interpolate.LSQUnivariateSpline(xMean,booty, t=np.mean([min(yMean),max(yMean)]))
                bootFun = interpolate.interp1d(xMean, booty, kind='quadratic')
            elif function.lower() == 'softmax':
                popt, pcov = opt.curve_fit(
                    sigmoid, xMean, booty, p0=[1, 1], method='trf')
                bootFun = lambda x: sigmoid(x, popt[0], popt[1])
            elif function.lower() == 'prelec':
                popt, pcov = opt.curve_fit(
                    prelec, xMean, booty, p0=[1, 1], method='trf')
                bootFun = lambda x: prelec(x, popt[0], popt[1])
        elif method.lower() == 'resampling':
            xb = np.random.choice(range(len(data)), len(data), replace=True)
            bootSample = np.hstack(data[xb])
            bootSample = bootSample[:, np.argsort(bootSample[0])]
            bootSample = np.array(np.split(bootSample, len(data), axis=1))
            bootx, booty = get_CEmean(bootSample)
            try:
                if function.lower() == 'scdf':
                    #                    popt, pcov = opt.curve_fit(sCDF, bootx, booty, p0 = [0.99,2],  method='trf')
                    popt, pcov = opt.curve_fit(
                        sCDF,
                        bootx,
                        booty,
                        p0=[np.mean([min(xFull), max(xFull)]), 1],
                        method='trf',
                        bounds=param_bounds)
                    bootFun = lambda x: sCDF(x, popt[0], popt[1])
                elif function.lower() == 'spline':
                    #                    bootFun = interpolate.LSQUnivariateSpline(bootx,booty, t=np.mean([min(yMean),max(yMean)]))
                    bootFun = interpolate.interp1d(
                        bootx, booty, kind='quadratic')
                elif function.lower() == 'softmax':
                    popt, pcov = opt.curve_fit(
                        sigmoid, bootx, booty, p0=[1, 1], method='trf')
                    bootFun = lambda x: sigmoid(x, popt[0], popt[1])
                elif function.lower() == 'prelec':
                    popt, pcov = opt.curve_fit(
                        prelec, bootx, booty, p0=[1, 1], method='trf')
                    bootFun = lambda x: prelec(x, popt[0], popt[1])
            except:
                continue
        b1.append(bootFun(xFull))

    b1 = np.vstack(b1)
    mean = fun(xFull)
    upper, lower = np.percentile(b1, [5, 95], axis=0)
    return mean, upper, lower, yHat

#%%
    
def bootstrap_sample(sample, measure, n=10000):
    xx = []
    for _ in range(n):
        xb = np.random.choice(range(len(sample)), len(sample), replace=True)
        bootSample = sample[xb,:]
        if measure.lower() == 'mean':
            xx.append(np.mean(bootSample, 0))
        elif measure.lower() == 'median':
            xx.append(np.median(bootSample, 0))
    lower, upper = np.percentile(xx, [2.5, 97.5], axis=0)
    return lower, upper

#%%
def bootstrap_function(function, sample, measure, n=10000):
    if measure.lower() == 'mean':
        real = function(np.mean(sample, 0))
    elif measure.lower() == 'median':
        real = function(np.median(sample, 0))

    xx = []; curve = []
    for _ in range(n):
        xb = np.random.choice(range(len(sample)), len(sample), replace=True)
        if sample.ndim > 1:
            bootSample = sample[xb,:]
        else:
            bootSample = sample[xb]
        if measure.lower() == 'mean':
            xx.append(np.mean(bootSample, 0))
        elif measure.lower() == 'median':
            xx.append(np.median(bootSample, 0))
        curve.append(function(xx[-1]))
    lower, upper = np.percentile(curve, [2.5, 97.5], axis=0)
    return real, lower, upper