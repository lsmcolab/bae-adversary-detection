import matplotlib.pyplot as plt
import numpy as np
import os
import src.utils.stats as sts

NEXP = 50

FIG_COUNTER = 0
FIGSIZE = (6.4,5.4)

FONTSIZE = 28
LEGEND_FONTSIZE = 12
FONT_DICT = {
        'weight': 'bold',
        'size': 26,
        }
TICK_FONTSIZE = 20

COLOR   = ['tab:blue','tab:green','tab:red','tab:orange','tab:purple','tab:brown','tab:pink','tab:olive']
            
MARKER_SIZE = 18
MARK_EVERY = 20
MARKER = ['o','^','p','s','X','o']

LINEWIDTH = 5
LINESTYLE = ['--','-',':','-.','-.']

def lines(results,target_data,ylabel='y-axis',xlabel='x-axis',save=False,savepath='./plots/',env_name=''):
    global FIG_COUNTER, FIGSIZE
    plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)

    y = {}
    y_lower = {}
    y_upper = {}
    counter = 0
    for method in results:
        y[method], y_lower[method], y_upper[method] =\
            sts.by_iteration(results[method],target_data=target_data,complete_with='zero')
        plt.fill_between(range(len(y_lower[method])),y_lower[method],y_upper[method],color=COLOR[counter%len(COLOR)],alpha=0.4)
        plt.plot(y[method],label=method,
            color=COLOR[counter%len(COLOR)],marker=MARKER[counter%len(MARKER)], markersize=MARKER_SIZE,markevery=MARK_EVERY,
            linewidth=LINEWIDTH,linestyle=LINESTYLE[counter%len(LINESTYLE)], markeredgecolor='black')
        counter += 1
    plt.legend(loc='best',ncol=1,fontsize=LEGEND_FONTSIZE,edgecolor='black')
    plt.xlabel(xlabel,fontdict=FONT_DICT)
    plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.tight_layout()
    plt.savefig(savepath+target_data+'_lines.pdf')
    plt.show()
    FIG_COUNTER += 1

def cumlines(results,target_data,ylabel='y-axis',xlabel='x-axis',save=False,savepath='./plots/',env_name=''):
    global FIG_COUNTER, FIGSIZE
    plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)

    y = {}
    y_lower = {}
    y_upper = {}
    counter = 0
    for method in results:
        y[method], y_lower[method], y_upper[method] =\
            sts.by_iteration(results[method],target_data=target_data,complete_with='zero',cumsum=True,fixed_max_len=200)
        plt.fill_between(range(len(y_lower[method])),y_lower[method],y_upper[method],color=COLOR[counter%len(COLOR)],alpha=0.4)
        plt.plot(y[method],label=method,
            color=COLOR[counter%len(COLOR)],marker=MARKER[counter%len(MARKER)], markersize=MARKER_SIZE,markevery=MARK_EVERY,
            linewidth=LINEWIDTH,linestyle=LINESTYLE[counter%len(LINESTYLE)], markeredgecolor='black')
        counter += 1
    plt.legend(loc='best',ncol=1,fontsize=18,edgecolor='black')
    plt.xlabel(xlabel,fontdict=FONT_DICT)
    plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.tight_layout()

    if save:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig(savepath+env_name+'_'+target_data+'_cumlines.pdf')
    else:
        plt.show()
    FIG_COUNTER += 1

def bars(results,target_data,savepath='.plots/',ylabel='y-axis'):
    global FIG_COUNTER, FIGSIZE
    plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)

    xlabels = []
    heights = []
    errors = []
    for method in results:
        m, l, u =\
            sts.by_experiment(results[method],target_data=target_data)
        xlabels.append(method)
        heights.append(np.mean(m))
        errors.append(np.mean(u-l)/2)

    plt.bar(range(len(heights)),heights,width=0.8,align='center',color=COLOR[:len(heights)],edgecolor='black',
                linewidth=1, tick_label=xlabels, yerr=errors,capsize=5)
    plt.xticks(fontsize=TICK_FONTSIZE,fontweight='bold',rotation=45)
    plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.tight_layout()
    plt.savefig(savepath+target_data+'_cumlines.pdf')
    plt.show()
    FIG_COUNTER += 1