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
        target = t