
"""
Neural graphical models for the conditional 
independence graphs. The conditional independence
graphs show the partial correlations between the 
nodes (features). 

Functions for NGMs:
1. Learning
2. Inference
3. Sampling

Note that this implementation is for 
1. Undirected graphs.
2. Input data should be real valued.

TODO: Implementation for the directed graphs. 
TODO: Extend to images and categorical variables.
"""
import copy
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sys
import torch
import torch.nn as nn

# local imports
import ngm.utils.neural_view as neural_view
import ngm.utils.data_processing as dp


######################################################################
# Functions for NGM learning
######################################################################

def product_weights_MLP(model):
    """
    Reads the input model (MLP) and returns the normalized
    product of the neural network weight matrices. 
    """
    for i, (n, p) in enumerate(model.MLP.named_parameters()):
        if i==0:
            if 'weight' in n:
                W = torch.abs(p).t() # DxH
                # Normalizing the weight using L2-norm
                W = torch.nn.functional.normalize(W)
        else: # i > 0
            if 'weight' in n:
                curr_W = torch.abs(p).t()
                # Normalizing the current weight using L2-norm
                curr_W = torch.nn.functional.normalize(curr_W)
                W = torch.matmul(W, curr_W)
                # Normalizing the running weight product using L2-norm
                W = torch.nn.functional.normalize(W)
    return W


def forward_NGM(X, model, S, structure_penalty='hadamard', lambd=0.1):
    """Pass the input X through the NGM model
    to obtain the X_pred. 

    LOSS = reg_loss + lambd * structure_loss

    The 'hadamard' ||prodW * Sc|| is more theoretically sound as it just 
    focuses on the terms needed to zero out and completely drop the 
    non-zero terms. 
    The 'diff' ||prodW-S|| also tries to make the non-zero terms go to 1.

    Args:
        X (torch.Tensor BxD): Input data
        model (torch.nn.object): The MLP model for NGM's `neural' view
        S (pd.DataFrame): Adjacency matrix from graph G
        structure_penalty (str): 'hadamard':||prodW * Sc||, 'diff':||prodW-S||
        lambd (float): reg_loss + lambd * structure_loss
            Recommended lambd=1 as the losses are scaled to the same range.
    
    Returns:
        (list): [
            Xp (torch.Tensor BxD): The predicted X