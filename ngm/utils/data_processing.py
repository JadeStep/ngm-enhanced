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
    graph_edges = graph_edges.replace('(', '').replace(')', '')
    graph_edges = graph_edges[2:-1].split("', '")
    edge_list = []
    for e in graph_edges:
        e = e.split(',')
        edge_list.append(
            (e[0], ''.join(e[1:-2]).lstrip(), 
            {"weight":float(e[-2]), 'color':e[-1][1:]})
        )
    G = nx.Graph()
    G.add_edges_from(edge_list)
    for n in G.nodes():
        G.nodes[n].update({'category':'unknown'})
    return G


def get_interactive_graph(G, title='', node_PREFIX='ObsVal'):
    Gv = net.Network(
        notebook=True, 
         height='750px', width='100%', 
    #     bgcolor='#222222', font_color='white',
        heading=title
    )
    Gv.from_nx(G.copy(), show_edge_weights=True, edge_weight_transf=(lambda x:x) )
    for e in Gv.edges:
        e['title'] = str(e['weight'])
        e['value'] = abs(e['weight'])
    if node_PREFIX is not None:
        for n in Gv.nodes:
            n['title'] = node_PREFIX+':'+n['category']
    Gv.show_buttons()
    return Gv


def set_feature_values(features_dict, features_known):
    """Updates the feature values with the known categories

    Args:
        features_dict (dict): {'name':'category'}
        node_attribute_konwn (dict): {'name':'category'}

    Returns:
        features_dict (dict): {'name':'category'}
    """
    for n, c in features_known.items():
        if n in features_dict.keys():
            features_dict[n] = c
        else:
            print(f'node {n} not found in features_dict')
    return features_dict


def series2df(series):
    "Convert a pd.Series to pd.Dataframe and set the index as header."
    # Convert the series to dictionary.
    series_dict = {n:v for n, v in zip(series.index, series.values)}
    # Create the dataframe from series and transpose.
    df = pd.DataFrame(series_dict.items()).transpose()
    # Set the index row as header and drop it from values.
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    return df


def t2np(x):
    "Convert torch to numpy"
    return x.detach().cpu().numpy()


def convertToTorch(data, req_grad=False, use_cuda=False):
    """Convert data from numpy to torch variable, if the req_grad
    flag is on then the gradient calculation is turned on.
    """
    if not torch.is_tensor(data):
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        data = torch.from_numpy(data.astype(np.float, copy=False)).type(dtype)
    data.requires_grad = req_grad
    return data


def normalize_table(X, method='min_max'):
    """Normalize the input data X.

    Args:
        X (pd.Dataframe): Samples(M) x Features(D).
        methods (str): min_max/mean 

    Returns:
        Xnorm (pd.Dataframe): Samples(M) x Features(D).
        scaler (object): The scaler to scale X
    """
    if method=='min_max':
        scaler = preprocessing.MinMaxScaler()
    elif method=='mean':
        scaler = preprocessing.StandardScaler()
    else:
        print(f'Scaler "{method}" not found')
    # Apply the scaler on the data X
    Xnorm = scaler.fit_transform(X)
    # Convert back to pandas dataframe
    Xnorm = pd.DataFrame(Xnorm, columns=X.columns)
    return Xnorm, scaler


def inverse_norm_table(Xnorm, Xscaler):
    """
    Apply the inverse transform on input normalized
    data to get back the original data.
    """
    return Xscaler.inverse_transform(Xnorm)

def analyse_condition_number(table, MESSAGE=''):
    S = covariance.empirical_covariance(table, assume_centered=False)
    eig, con = eig_val_condition_num(S)
    print(f'{MESSAGE} covariance matrix: The condition number {con} and min eig {min(