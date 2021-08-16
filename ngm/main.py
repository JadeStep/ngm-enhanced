
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
            loss (torch.scalar): The NGM loss 
            reg_loss (torch.scalar): The regression term loss
            structure_loss (torch.scalar): The structure penalty loss
        ]
    """
    # 1. Running the NGM model 
    Xp = model.MLP(X)
    # 2. Calculate the regression loss
    mse = nn.MSELoss() 
    reg_loss = mse(Xp, X)
    # 3. Calculate the structure loss
    # 3.1 Get the frame of the graph structure
    if structure_penalty=='hadamard':
        # Get the complement of S (binarized)
        Sg = (S==0).astype(int)
        Sg = dp.convertToTorch(np.array(Sg), req_grad=False)
    elif structure_penalty=='diff':
        # Binarize the adjacency matrix S
        Sg = (S!=0).astype(int)
        Sg = dp.convertToTorch(np.array(Sg), req_grad=False)
    else:
        print(f'Structure penalty {structure_penalty} is not defined')
        sys.exit(0)
    # 3.2 Initialize the structure loss
    structure_loss = torch.zeros(1)[0]
    if lambd > 0:
        # 3.3 Get the product of weights (L2 normalized) of the MLP
        prod_W = product_weights_MLP(model)
        D = prod_W.shape[-1]
        # 3.4 Calculate the penalty
        if structure_penalty=='hadamard':
            # Using the L2 norm for high structure penalty
            structure_loss = torch.linalg.norm(prod_W*Sg, ord=2)
        elif structure_penalty=='diff':
            struct_mse = nn.MSELoss() 
            structure_loss = struct_mse(prod_W, Sg)
        # 3.5 Scale the structure loss
        structure_loss = structure_loss/(D**2)
        # Adding the log scaling
        structure_loss = torch.log(structure_loss)
    # 4. Calculate the total loss = reg_loss + lambd * struct_loss
    loss = reg_loss + lambd * structure_loss
 
    return Xp, loss, reg_loss, structure_loss


def learning(
    G, 
    X,
    lambd=1.0,
    hidden_dim=20,
    epochs=1200, 
    lr=0.001,
    norm_type='min_max',
    k_fold=1,
    structure_penalty='hadamard',
    VERBOSE=True
    ):
    """Learn the distribution over a conditional independence graph. 
    1. Fit a MLP (autoencoder) to learn the data representation from X->X. 
    2. The input-output path of dependence structure of the MLP 
       should match the conditional independence structure of the
       input graph. This is achieved using a regularization term.
    3. Return the learned model representing the NGM

    Normalize X and select the best model using K-fold CV. 

    Fit the MLP on the input data X to get the `neural' view of NGM 
    while maintaining the conditional independence structure defined 
    by the complement structure matrix Sc. Does cross-validation to 
    get better generalization.

    Args:
        G (nx.Graph): Conditional independence graph.
        X (pd.DataFrame): Samples(M) x Features(D).
        lambd (float): reg_loss + lambd * structure_loss
            Recommended lambd=1 as the losses are scaled to the same range.
        hidden_dim (int): The size of the hidden unit of the MLP. 
            Each layer will have the same value.
        epochs (int): The training epochs number.
        lr (float): Learning rate for the optimizer.
        norm_type (str): min_max/mean
        k_fold (int): #splits for the k-fold CV.
        structure_penalty (str): 'hadamard':||prodW * Sc||, 'diff':||prodW-S||
        VERBOSE (bool): if True, prints to output.
        
    Returns:
        model_NGM (list): [
            model (torch.nn.object): A MLP model for NGM's `neural' view,
            scaler (sklearn object): Learned normalizer for the input data,
            feature_means (pd.Series): [feature:mean val]
        ]
    """
    # Get the graph structure
    S = nx.to_pandas_adjacency(G)
    # Arrange the columns of X to match the adjacency matrix
    X = X[S.columns]
    feature_means = X.mean()
    print(f'Means of selected features {feature_means, len(feature_means)}')
    # Normalize the data
    print(f'Normalizing the data: {norm_type}')
    X, scaler = dp.process_data_for_CI_graph(X, norm_type)
    # Converting the data to torch 
    X = dp.convertToTorch(np.array(X), req_grad=False)
    M, D = X.shape
    # Splitting into k-fold for cross-validation 
    n_splits = k_fold if k_fold > 1 else 2
    kf = KFold(n_splits=n_splits, shuffle=True)
    # For each fold, collect the best model and the test-loss value
    results_Kfold = {}
    for _k, (train, test) in enumerate(kf.split(X)):
        if _k >= k_fold: # No CV if k_fold=1
            continue
        if VERBOSE: print(f'Fold num {_k}')
        X_train, X_test = X[train], X[test] # KxD, (M-K)xD

        # Initialize the MLP model
        if VERBOSE: print(f'Initializing the NGM model')
        model = neural_view.DNN(I=D, H=hidden_dim, O=D)
        optimizer = neural_view.get_optimizers(model, lr=lr)

        # TODO: Add base initialization only on the regression loss
        # model = base_initialization_NGM(model, X_train)

        # Defining optimization & model tracking parameters
        best_test_loss = np.inf
        PRINT = int(epochs/10) # will print only 10 times
        lambd_increase = int(epochs/10)
        # updating with the best model and loss for the current fold
        results_Kfold[_k] = {}

        # Training the NGM model
        for e in range(epochs):
            # TODO: Keep increasing the lambd penalty as epochs proceed
            # if not e % lambd_increase:
            #     lambd *= 10 # increase in lambd value
            #     print(f'epoch={e}, lambda={lambd}')
            # reset the grads to zero
            optimizer.zero_grad()
            # calculate the loss for train data
            _, loss_train, reg_loss_train, struct_loss_train = forward_NGM(
                X_train, 
                model, 
                S,
                structure_penalty,
                lambd=lambd
            )
            with torch.no_grad(): # prediction on test 
                _, loss_test, reg_loss_test, struct_loss_test = forward_NGM(
                    X_test, 
                    model, 
                    S,
                    structure_penalty, 
                    lambd=lambd 
                )
            # calculate the backward gradients
            loss_train.backward()
            # updating the optimizer params with the grads
            optimizer.step()
            # Printing output
            if not e%PRINT and VERBOSE: 
                print(f'\nFold {_k}: epoch:{e}/{epochs}')
                print(f'Train: loss={dp.t2np(loss_train)}, reg={dp.t2np(reg_loss_train)}, struct={dp.t2np(struct_loss_train)}')
                print(f'Test: loss={dp.t2np(loss_test)}, reg={dp.t2np(reg_loss_test)}, struct={dp.t2np(struct_loss_test)}')
            # Updating the best model for this fold
            _loss_test = dp.t2np(loss_test)
            if _loss_test < best_test_loss: # and e%10==9:
                results_Kfold[_k]['best_model_updates'] = f'Fold {_k}: epoch:{e}/{epochs}:\n\
                    Train: loss={dp.t2np(loss_train)}, reg={dp.t2np(reg_loss_train)}, struct={dp.t2np(struct_loss_train)}\n\
                    Test: loss={dp.t2np(loss_test)}, reg={dp.t2np(reg_loss_test)}, struct={dp.t2np(struct_loss_test)}'
                # if VERBOSE and not e%PRINT or e==epochs-1:
                    # print(f'Fold {_k}: epoch:{e}/{epochs}: Updating the best model with test loss={_loss_test}')
                best_model_kfold = copy.deepcopy(model)
                best_test_loss = _loss_test
            # else: # loss increasing, reset the model to the previous best
            #     # print('re-setting to the previous best model')
            #     model = best_model_kfold
            #     optimizer = neural_view.get_optimizers(model, lr=lr)
        results_Kfold[_k]['test_loss'] = best_test_loss
        results_Kfold[_k]['model'] = best_model_kfold
        if VERBOSE: print('\n')
    # Select the model from the results Kfold dictionary 
    # with the best score on the test fold.
    best_loss = np.inf
    for _k in results_Kfold.keys():
        curr_loss = results_Kfold[_k]['test_loss']
        if curr_loss < best_loss:
            model = results_Kfold[_k]['model']
            best_loss = curr_loss
            best_model_details = results_Kfold[_k]["best_model_updates"]

    print(f'Best model selected: {best_model_details}')
    # Checking the structure of the prodW and Sc
    prod_W = dp.t2np(product_weights_MLP(model))
    # print(f'Structure Check: prodW={prod_W}, S={(np.array(S)!=0).astype(int)}')
    return [model, scaler, feature_means]


######################################################################
# Functions to run inference over the learned NGM
######################################################################

def inference(
    model_NGM, 
    node_feature_dict, 
    unknown_val='u', 
    lr=0.001, 
    max_itr=1000,
    VERBOSE=True,
    reg_loss_th=1e-6
    ):
    """Algorithm to run the feature inference among the nodes of the
    NGM learned over the conditional independence graph.

    We only optimize for the regression of the known values as that 
    is the only ground truth information we have and the prediction
    should be able to recover the observed datapoints.
    Regression: Xp = f(Xi) 
    Input Xi = {Xi[k] (fixed), Xi[u] (learned)}
    Reg loss for inference = ||Xp[k] - Xi[k]||^2_2

    Run gradient descent over the input, which modifies the unobserved
    features to minimize the inference regression loss. 

    Args:
        model_NGM (list): [
            model (torch.nn.object): A MLP model for NGM's `neural' view,
            scaler (sklearn object): Learned normalizer for the input data,
            feature_means (pd.Series): [feature:mean val]
        ]
        node_feature_dict (dict): {'name':value}.
        unknown_val (str): The marker for the unknown value.
        lr (float): Learning rate for the optimizer.
        max_itr (int): For the convergence.
        VERBOSE (bool): enable/disable print statements.
        reg_loss_th (float): The threshold for reg loss convergence.

    Returns:
        Xpred (pd.DataFrame): Predictions for the unobserved features.
            {'feature name': pred-value} 
    """
    # Get the NGM params
    model, scaler, feature_means = model_NGM
    # Get the feature names and input dimension
    D = len(feature_means)
    feature_names = feature_means.index
    # Freeze the model weights
    for p in model.parameters():
        p.requires_grad = False
    # Initializin the input vector Xi
    _Xi = feature_means.copy()
    # TODO: Try min and max init as well
    # Assign the known (observed) values to the Xi
    for _n, v in node_feature_dict.items():
        if v!=unknown_val:
            _Xi[_n] = v
    # Normalize the values of Xi using the scaler
    _Xi = scaler.transform(dp.series2df(_Xi))[0]
    # Convert to dataseries to maintain the column name associations
    _Xi = pd.Series(
        {n:v for n, v in zip(feature_names, _Xi)}, 
        index=feature_names
    )
    # Creating the feature list with unobserved (unkonwn) tensors as learnable.
    # and observed (known) tensors as fixed
    feature_tensors = [] # List of feature tensors
    # Setting the optimization parameters
    optimizer_parameters = []
    for i, _n in enumerate(feature_names):
        _xi = torch.as_tensor(_Xi[_n])
        # set the value to learnable or not
        _xi.requires_grad = node_feature_dict[_n]==unknown_val
        feature_tensors.append(_xi)
        if node_feature_dict[_n]==unknown_val:
            optimizer_parameters.append(_xi)
    # Init a mask for the known & unknown values
    mask_known = torch.zeros(1, D)
    mask_unknown = torch.zeros(1, D)
    for i, _n in enumerate(feature_names):
        if node_feature_dict[_n]==unknown_val:
            mask_unknown[0][i] = 1
        else:
            mask_known[0][i] = 1
    # Define the optimizer
    optimizer = torch.optim.Adam(
        optimizer_parameters,
        lr=lr, 
        betas=(0.9, 0.999),
        eps=1e-08,
        # weight_decay=0
    )
    # Minimizing for the regression loss for the known values.
    itr = 0
    curr_reg_loss = np.inf
    PRINT = int(max_itr/10) + 1 # will print only 10 times
    mse = nn.MSELoss() # regression loss 
    best_reg_loss = np.inf
    while curr_reg_loss > reg_loss_th and itr<max_itr: # Until convergence
        # The tensor input to the MLP model
        Xi = torch.zeros(1, D) 
        for i, f in enumerate(feature_tensors):
            Xi[0][i] = f
        # reset the grads to zero
        optimizer.zero_grad()
        # Running the NGM model 
        Xp = model.MLP(Xi)
        # Output should be Xi*known_mask with no grad
        Xo = Xi.clone().detach()
        # Set the gradient to False
        Xo.requires_grad = False
        # Calculate the Inference loss using the known values
        reg_loss = mse(mask_known*Xp, mask_known*Xo)
        # reg_loss = mse(Xp, Xo)
        # calculate the backward gradients
        reg_loss.backward()
        # updating the optimizer params with the grads
        optimizer.step()
        # Selecting the output with the lowest inference loss
        curr_reg_loss = dp.t2np(reg_loss)
        if curr_reg_loss < best_reg_loss: