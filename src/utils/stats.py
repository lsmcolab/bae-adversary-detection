import numpy as np
import scipy.stats
from scipy.stats import ttest_ind
from src.utils import *

def by_iteration(results, target_data,
 complete_with='zero', cumsum=False, fixed_max_len=None):
    # 1. Formating data
    # finding the maximum length
    if not fixed_max_len:
        max_len = max([len(exp[target_data]) for exp in results])
    else:
        max_len = fixed_max_len

    # building the y-axis data
    y = []
    for exp in results:
        if len(exp[target_data]) > max_len:
            y.append(exp[target_data][:max_len])
        else:
            y.append(exp[target_data])
            while len(y[-1]) < max_len:
                if complete_with == 'zero':
                    y[-1].append(0)
                elif complete_with == 'last':
                    y[-1].append(y[-1][-1])
                else:
                    raise NotImplemented
        if cumsum:
            y[-1] = np.cumsum(y[-1])

    # retrieving the mean and the confidence interval
    m, l, u = mean_confidence_interval(y, by_='iteration')
    return m, l, u

def by_experiment(results,target_data,
 complete_with='zero', cumsum=False, fixed_max_len=200):
    # 1. Formating data
    # finding the maximum length
    if not fixed_max_len:
        max_len = max([len(exp[target_data]) for exp in results])
    else:
        max_len = fixed_max_len
        
    
    # building the y-axis data
    y = []
    for exp in results:
        if len(exp[target_data]) > max_len:
            y.append(exp[target_data][:max_len])
        else:
            y.append(exp[target_data])
            while len(y[-1]) < max_len:
                if complete_with == 'zero':
                    y[-1].append(0)
                elif complete_with == 'last':
                    y[-1].append(y[-1][-1])
                else:
                    raise NotImplemented
        if cumsum:
            y[-1] = np.cumsum(y[-1])
            
    y_m = []
    y_l, y_u = [], []
    for i in range(len(y)):
        # retrieving the mean and the confidence interval
        m, l, u = mean_confidence_interval(y[i], by_='experiment')
        y_m.append(m)
        y_l.append(l)
        y_u.append(u)
    return y_m, np.array(y_l), np.array(y_u)

def mean_confidence_interval(data, by_='iteration', confidence=0.95):
    a = 1.0 * np.array(data)
    if by_=='iteration':
        n = len(a)-1
        m, se = np.mean(a,axis=0), scipy.stats.sem(a)
        h = scipy.stats.t.interval(alpha=confidence, df=n,loc=m,scale=se)
        l, u = h[0], h[1]
    elif by_=='experiment':
        n = len(a)-1
        m, se = np.mean(a), scipy.stats.sem(a)
        h = scipy.stats.t.interval(alpha=confidence, df=n,loc=m,scale=se)
        l, u = h[0], h[1]
    else:
        raise NotImplemented
    return [m, l, u]

def summary(results, target_data, LaTeX=False):
    print('|||',target_data,'SUMMARY |||')
    for method in results:
        m, l, u = by_experiment(results=results[method], target_data=target_data)
        m, l, u = mean_confidence_interval(m, by_='iteration')
        
        if LaTeX:
            print(method,':\n$ %.3f \\pm %.3f $' % (m,(u-l)/2))
        else:
            print(method,':',m,(u-l)/2)

def pvalues(results,target_data,by_='iteration',complete_with='zero',cumsum=False):
    pvalues = {}

    if by_=='iteration':
        # finding the maximum length
        max_len = 0
        for method in results:
            if max_len < max([len(exp[target_data]) for exp in results[method]]):
                max_len = max([len(exp[target_data]) for exp in results[method]]) 

        # building the y-axis data
        y = {}
        for method in results:
            y[method] = []
            for exp in results[method]:
                y[method].append(exp[target_data])
                while len(y[method][-1]) < max_len:
                    if complete_with == 'zero':
                        y[method][-1].append(0)
                    elif complete_with == 'last':
                        y[method][-1].append(y[-1][-1])
                    else:
                        raise NotImplemented
                if cumsum:
                    y[method][-1] = np.cumsum(y[-1])
                
        for method1 in y:
            for method2 in y:
                pvalues[(method1,method2)] = ttest_ind(y[method1],y[method2],equal_var=False)[1]

        for it in range(max_len):
            print('\nIteration',it)
            print([method for method in results])
            for method1 in y:
                for method2 in y:
                    print('%.2f' % (pvalues[(method1,method2)][it]) + '\t',end='')
                print()
                
    elif by_=='experiment':
        y = {}
        for method in results:
            y[method] = []
            for exp in results[method]:
                y[method].append(np.mean(exp[target_data]))
                
        for method1 in y:
            for method2 in y:
                pvalues[(method1,method2)] = ttest_ind(y[method1],y[method2],equal_var=False)[1]

        print('\n',[method for method in results])
        for method1 in y:
            for method2 in y:
                print('%.2f' % (pvalues[(method1,method2)]) + '\t',end='')
            print()

    else:
        raise NotImplemented