"""
Additional data processing and post-processing
functions for neural graphical model analytics.
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pyvis import network as net
from PIL import Image
import io
import pandas as pd
from sklearn import covariance
from sklearn import preprocessing
from time import time
import torch



def function_plots_for_target(plot_dict):
    """
    plot_dict ={
        target: {
            source1: [x, fx, title], 
            source2: [x, fx, title], 
            ...,
            }
    }
    """
    # Get the target
    target = list(plot_dict.keys())
    if len(target)==1:
        target = target[0]
    num_sources = len(plot_dict[target])
    # fig = plt.figure(figsize=(int(3*num_sources), 25))
    # fig = plt.figure(figsize=(5, int(5*num_sources)))
    fig = plt.figure(figsize=(15, 15))
    p=min(num_sources, 3)
    for i, source in enumerate(plot_dict[target].keys()):
        ax = plt.subplot(p+1, int(num_sources/p), i+1) # (grid_x, grid_y, plot_num)
        # plt.subplot(num_sources, 1, i+1)
        x, fx, title = plot_dict[target][source]
        # plot the function
        plt.plot(x, fx, 'b')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        plt.title(title, fontsize=20)
    # show the plot
    plt.savefig(f'plot_{target}.jpg', dpi=300)
    plt.show()


def plot_function(x, fx, title=f'plot of (x, fx)'):
    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot the function
    plt.plot(x, fx, 'b')
    plt.title(title)
    # show the plot
    # plt.savefig('plot.jpg', dpi=300)
    # plt.show()
    return 


def retrieve_graph(graph_edges):
    """ Read the graph edgelist and 
    convert it to a networkx graph.
    """
    graph_edges = graph_edges.replac