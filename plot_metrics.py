import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from ltp import LTP
from cycler import cycler

from typing import Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import matplotlib.font_manager as mfm
from matplotlib.font_manager import FontManager, _rebuild
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams




def plot_distribution():
    '''数据集中事件类型的分布情况
    '''
    plt.rc('font', size=LARGE_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=LARGE_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=LARGE_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LARGE_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    chi_to_eng = {
        '竞赛行为-胜负': 'Competition.Result',
        '产品行为-发布': 'Product.Release',
        '人生-死亡': 'Life.Death',
        '司法行为-拘捕': 'Justice.Arrest',
        '组织关系-辞/离职': 'Organization.Resign',
        '竞赛行为-夺冠': 'Competition.Champion'
    }
    event_counter = Counter()
    with open('data/train.json', mode='r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            example = json.loads(line.strip())
            events = example['event_list']
            for event in events:
                event_type = event['event_type']
                event_counter[event_type] += 1
    total = sum(event_counter.values())
    most_common_events = event_counter.most_common(6)
    event_counts = {chi_to_eng[k]: v for k, v in most_common_events}
    
    other = total - sum(event_counts.values())
    
    event_counts['Other'] = other
    #rcParams['font.family'] = 'sans-serif'
    #rcParams['font.sans-serif'] = ['Tahoma']
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))


    event_types = list(event_counts.keys())
    event_counts = list(event_counts.values())
    
    
    theme = plt.get_cmap('binary')
    ax.set_prop_cycle("color", [theme(2. * (i+1) / 20)
                             for i in range(len(event_types))])
    def func(pct):
        #absolute = int(round(pct/100.*np.sum(allvals)))
        return f"{pct:.0f}%"
    wedges, texts, autotexts = ax.pie(event_counts, labels=tuple(event_types), 
                                       autopct=lambda pct: func(pct))
    plt.tight_layout()
    #ax.set_title("Matplotlib bakery: A pie")
    plt.savefig('outputs/en_distribution_no_color.pdf', format='pdf')
    plt.show()

def combine_plot():
    
    ############### setting #########################
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 15

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    x_labels = ['RoBERTa', 'RoBERTa-CRF', 'M-RoBERTa', 'RoBERTa-CRF*', 'M-RoBERT-CRF*']
    
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(18,12))
    gs = gridspec.GridSpec(1, 1, figure=fig)
    gs0 = gs[0].subgridspec(2, 2)
    #fig, axs = plt.subplots(2, 1, figsize=(8,8))
    ax0 = fig.add_subplot(gs0[:, 0])
    ax1 = fig.add_subplot(gs0[0, 1])
    ax3 = fig.add_subplot(gs0[1, 1])
    #ax1, ax3 = axs[0], axs[1]

    ##################### for distribution ###############################
    event_counter = Counter()
    with open('data/train.json', mode='r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            example = json.loads(line.strip())
            events = example['event_list']
            for event in events:
                event_type = event['event_type']
                event_counter[event_type] += 1
    total = sum(event_counter.values())
    most_common_events = event_counter.most_common(6)
    event_counts = {k:v for k, v in most_common_events}
    
    other = total - sum(event_counts.values())
    
    event_counts['其他'] = other
    event_types = list(event_counts.keys())
    event_counts = list(event_counts.values())
    
    def func(pct):
        #absolute = int(round(pct/100.*np.sum(allvals)))
        return f"{pct:.0f}%"
    wedges, texts, autotexts = ax0.pie(event_counts, labels=tuple(event_types), 
                                       autopct=lambda pct: func(pct))
    # ax0.legend(wedges, event_types,
    #         title="事件类型"
    #         # loc="center left",
    #         # bbox_to_anchor=(1, 0, 0.5, 1)#,prop=font_properties
    # )

    ##################### for classification precision ####################
    x = np.arange(len(x_labels))

    total = np.array([362, 316, 301, 243, 229])
    class_wrong = np.array([285, 260, 240, 200, 164])
    wrong_ratio = class_wrong / total

    width = 0.35 # the width of the bars
    rects1 = ax1.bar(x - width/2, total, width, label='触发词错误总量')
    rects2 = ax1.bar(x + width/2, class_wrong, width, label='分类错误量')

    color = '#000000'#'tab:red'

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel('模型')
    ax1.set_ylabel('错误数量', color=color)

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(ymax=460)
    ax1.legend(loc=2)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'#'tab:orange'
    ax2.set_ylabel('分类错误占比', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, wrong_ratio, color=color, label='分类错误占比')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(ymin=0.2, ymax=0.93)
    ax2.legend(loc=1)

    ##################### for event num ################################
    stat_dict = {
        'M-RoBERT-CRF*': {1: 0.8542635658914729, 2: 0.9135802469135802, 3: 0.7931034482758621, 4: 0.875, 5: 0.78},
        'RoBERTa-CRF*': {1: 0.8496124031007752, 2: 0.9074074074074074, 3: 0.6896551724137931, 4: 0.875, 5: 0.8888888888888888},
        'RoBERTa': {1: 0.8155038759689922, 2: 0.7777777777777778, 3: 0.41379310344827586, 4: 0.625, 5: 0.3333333333333333},
        'RoBERTa-CRF': {1: 0.8232558139534883, 2: 0.845679012345679, 3: 0.6551724137931034, 4: 0.875, 5: 0.5555555555555556},
        'M-RoBERTa': {1: 0.8310077519379845, 2: 0.845679012345679, 3: 0.5517241379310345, 4: 0.5, 5: 0.770} 
    }
    stat_frame = pd.DataFrame(stat_dict).round(4)

    ax3.set_xlabel('事件数量')
    ax3.set_xticks(list(range(5)))
    ax3.set_xticklabels(['1', '2', '3', '4', '>= 5'])
    #ax3.yaxis.tick_right()
    #ax3.yaxis.set_label_position("right")
    ax3.set_ylabel('正确率')
    for model_name in x_labels:
        y_data = stat_frame[model_name].tolist()
        ax3.plot(y_data, label=model_name)

    ax3.legend()


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def plot_event_num():
    '''各模型性能随事件数量变化的趋势
    '''
    # for event num
    stat_dict = {
        'M-RoBERT-CRF*': {1: 0.8542635658914729, 2: 0.9135802469135802, 3: 0.7931034482758621, 4: 0.875, 5: 0.78},
        'RoBERTa-CRF*': {1: 0.8496124031007752, 2: 0.9074074074074074, 3: 0.6896551724137931, 4: 0.875, 5: 0.8888888888888888},
        'RoBERTa': {1: 0.8155038759689922, 2: 0.7777777777777778, 3: 0.41379310344827586, 4: 0.625, 5: 0.3333333333333333},
        'RoBERTa-CRF': {1: 0.8232558139534883, 2: 0.845679012345679, 3: 0.6551724137931034, 4: 0.875, 5: 0.5555555555555556},
        'M-RoBERTa': {1: 0.8310077519379845, 2: 0.845679012345679, 3: 0.5517241379310345, 4: 0.5, 5: 0.770} 
    }
    stat_frame = pd.DataFrame(stat_dict).round(4)

    # for classification precision
    total = np.array([362, 316, 301, 243, 229])
    class_wrong = np.array([285, 260, 240, 200, 164])
    wrong_ratio = class_wrong / total
    x_labels = ['RoBERTa', 'RoBERTa-CRF', 'M-RoBERTa', 'RoBERTa-CRF*', 'M-RoBERT-CRF*']
    x = np.arange(len(x_labels))

    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax3 = plt.subplots(figsize=(9, 6))

    #ax3.set_title('模型在多事件句子中的表现')
    ax3.set_xlabel('The number of events')
    ax3.set_xticks(list(range(5)))
    ax3.set_xticklabels(['1', '2', '3', '4', '>= 5'])
    #ax3.yaxis.tick_right()
    #ax3.yaxis.set_label_position("right")
    ax3.set_ylabel('SL accuracy')

    markers = ['o', '*', 'v', 'X', 's']
    #linestyles = ['dashdot', 'dashed', 'dotted', 'solid']
    # default_cycler = (cycler(marker=markers) + 
    #                   cycler(linestyle=linestyles) +
    #                   cycler(color=list('rgby')))
    # ax3.set_prop_cycle(default_cycler)
    fmts = ['k*-', 'k+-', 'ko--', 'ks--', 'k^-.']
    for i, model_name in enumerate(x_labels):
        y_data = stat_frame[model_name].tolist()
        #ax3.plot(y_data, label=model_name, linestyle=linestyles[i], lw='3')
        ax3.plot(y_data, fmts[i], label=model_name, lw='3', markersize=12)

    ax3.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('outputs/en_metric_by_event_num_no_color.pdf', format='pdf')
    plt.show()

def plot_class_wrong():
    '''事件类型分类错误占总体错误的情况
    '''
    total = np.array([362, 316, 301, 243, 229])
    class_wrong = np.array([285, 260, 240, 200, 164])
    wrong_ratio = class_wrong / total
    x_labels = ['RoBERTa', 'RoBERTa-CRF', 'M-RoBERTa', 'RoBERTa-CRF*', 'M-RoBERT-CRF*']
    x = np.arange(len(x_labels))

    plt.rcParams['axes.unicode_minus'] = False
    width = 0.35 # the width of the bars

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ##################### for classification precision ####################

    rects1 = ax1.bar(x - width/2, total, width, color='k', linestyle='solid', label='Total errors')
    rects2 = ax1.bar(x + width/2, class_wrong, width, color='grey', edgecolor='black', linestyle='dashed',label='TE errors')

    color = 'black'#'tab:red'
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=5)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Number of errors', color=color)

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(ymax=480)
    ax1.legend(loc=2)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'grey'#'tab:orange'
    ax2.set_ylabel('Percentage', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, wrong_ratio, color=color, label='Percentage', marker='o')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(ymin=0.2, ymax=0.96)
    ax2.legend(loc=1)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('outputs/en_classification_precision_no_color.pdf', format='pdf')
    plt.show()

def plot_metric():

    # for event num
    stat_dict = {
        'M-RoBERT-CRF*': {1: 0.8542635658914729, 2: 0.9135802469135802, 3: 0.7931034482758621, 4: 0.875, 5: 0.78},
        'RoBERTa-CRF*': {1: 0.8496124031007752, 2: 0.9074074074074074, 3: 0.6896551724137931, 4: 0.875, 5: 0.8888888888888888},
        'RoBERTa': {1: 0.8155038759689922, 2: 0.7777777777777778, 3: 0.41379310344827586, 4: 0.625, 5: 0.3333333333333333},
        'RoBERTa-CRF': {1: 0.8232558139534883, 2: 0.845679012345679, 3: 0.6551724137931034, 4: 0.875, 5: 0.5555555555555556},
        'M-RoBERTa': {1: 0.8310077519379845, 2: 0.845679012345679, 3: 0.5517241379310345, 4: 0.5, 5: 0.770} 
    }
    stat_frame = pd.DataFrame(stat_dict).round(4)

    # for classification precision
    total = np.array([362, 316, 301, 243, 229])
    class_wrong = np.array([285, 260, 240, 200, 164])
    wrong_ratio = class_wrong / total
    x_labels = ['RoBERTa', 'RoBERTa-CRF', 'M-RoBERTa', 'RoBERTa-CRF*', 'M-RoBERT-CRF*']
    x = np.arange(len(x_labels))

    plt.rcParams['axes.unicode_minus'] = False
    width = 0.35 # the width of the bars

    fig, axs = plt.subplots(2, 1, figsize=(8,8))
    ax1, ax3 = axs[0], axs[1]

    ##################### for classification precision ####################

    rects1 = ax1.bar(x - width/2, total, width, label='触发词错误总量')
    rects2 = ax1.bar(x + width/2, class_wrong, width, label='分类错误量')

    color = '#000000'#'tab:red'
    ax1.set_title('触发词分类错误的情况')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel('模型')
    ax1.set_ylabel('错误数量', color=color)

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(ymax=460)
    ax1.legend(loc=2)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'#'tab:orange'
    ax2.set_ylabel('分类错误占比', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, wrong_ratio, color=color, label='分类错误占比')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(ymin=0.2, ymax=0.93)
    ax2.legend(loc=1)

    ##################### for event num ################################
    ax3.set_title('模型在多事件句子中的表现')
    ax3.set_xlabel('事件数量')
    ax3.set_xticks(list(range(5)))
    ax3.set_xticklabels(['1', '2', '3', '4', '>= 5'])
    #ax3.yaxis.tick_right()
    #ax3.yaxis.set_label_position("right")
    ax3.set_ylabel('正确率')
    for model_name in x_labels:
        y_data = stat_frame[model_name].tolist()
        ax3.plot(y_data, label=model_name)

    ax3.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('outputs/metric.pdf', format='pdf')
    plt.show()

def temp():
    
    stat_dict = {
        'M-RoBERT-CRF*': {1: 0.8542635658914729, 2: 0.9135802469135802, 3: 0.7931034482758621, 4: 0.875, 5: 0.7777777777777778},
        'RoBERTa-CRF*': {1: 0.8496124031007752, 2: 0.9074074074074074, 3: 0.6896551724137931, 4: 0.875, 5: 0.8888888888888888},
        'RoBERTa': {1: 0.8155038759689922, 2: 0.7777777777777778, 3: 0.41379310344827586, 4: 0.625, 5: 0.3333333333333333},
        'RoBERTa-CRF': {1: 0.8232558139534883, 2: 0.845679012345679, 3: 0.6551724137931034, 4: 0.875, 5: 0.5555555555555556},
        'M-RoBERTa': {1: 0.8310077519379845, 2: 0.845679012345679, 3: 0.5517241379310345, 4: 0.5, 5: 0.7777777777777778} 
    }

    stat_frame = pd.DataFrame(stat_dict).round(4)
    print(stat_frame)

if __name__ == '__main__':

    MEDIUM_SIZE = 13
    BIGGER_SIZE = 15
    LARGE_SIZE = 18

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plot_event_num()
    plot_class_wrong()
    plot_distribution()
    