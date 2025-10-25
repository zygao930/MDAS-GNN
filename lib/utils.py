import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from time import time
from scipy.sparse.linalg import eigs


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    """
    Load adjacency matrix from file
    
    Parameters:
        distance_df_filename: path to CSV file containing edge information
        num_of_vertices: number of vertices
        
    Returns:
        A: adjacency matrix
    """
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

        # Remap node IDs if needed
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}

            with open(distance_df_filename, 'r') as f:
                f.readline()  # Skip header
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:  # Node IDs start from 0
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    """
    Load bidirectional adjacency matrix from file
    
    Parameters:
        distance_df_filename: path to CSV file containing edge information
        num_of_vertices: number of vertices
        
    Returns:
        A: bidirectional adjacency matrix
    """
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

        # Remap node IDs if needed
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}

            with open(distance_df_filename, 'r') as f:
                f.readline()  # Skip header
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA

        else:  # Node IDs start from 0
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA


def get_Laplacian(A):
    """
    Compute graph Laplacian matrix: L = D - A
    
    Parameters:
        A: adjacency matrix (N, N)
        
    Returns:
        Laplacian matrix (N, N)
    """
    assert (A-A.transpose()).sum() == 0  # Ensure A is symmetric
    
    D = np.diag(np.sum(A, axis=1))  # Degree matrix
    L = D - A  # Laplacian matrix
    
    return L


def scaled_Laplacian(W):
    """
    Compute scaled Laplacian matrix
    
    Parameters:
        W: weight matrix (N, N)
        
    Returns:
        scaled Laplacian matrix (N, N)
    """
    assert W.shape[0] == W.shape[1]
    
    D = np.diag(np.sum(W, axis=1))  # Degree matrix
    L = D - W  # Laplacian matrix
    
    lambda_max = eigs(L, k=1, which='LR')[0].real  # Maximum eigenvalue
    
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def sym_norm_Adj(W):
    """
    Compute symmetric normalized adjacency matrix
    
    Parameters:
        W: weight matrix (N, N)
        
    Returns:
        Symmetric normalized adjacency matrix (N, N)
    """
    assert W.shape[0] == W.shape[1]
    
    N = W.shape[0]
    W = W + np.identity(N)  # Add self-connections
    D = np.diag(np.sum(W, axis=1))
    sym_norm_Adj_matrix = np.dot(np.sqrt(D), W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix, np.sqrt(D))
    
    return sym_norm_Adj_matrix


def trans_norm_Adj(W):
    """
    Compute transition matrix (random walk normalized adjacency)
    
    Parameters:
        W: weight matrix (N, N)
        
    Returns:
        Transition matrix (N, N)
    """
    assert W.shape[0] == W.shape[1]
    
    N = W.shape[0]
    W = W + np.identity(N)  # Add self-connections
    D = np.diag(1.0/np.sum(W, axis=1))
    trans_norm_Adj_matrix = np.dot(D, W)
    
    return trans_norm_Adj_matrix


def compute_val_loss(net, val_loader, criterion, sw, epoch):
    """
    Compute validation loss for the model
    
    Parameters:
        net: neural network model
        val_loader: validation data loader
        criterion: loss function
        sw: summary writer for logging
        epoch: current epoch number
        
    Returns:
        validation_loss: average validation loss
    """
    net.train(False)  # Set to evaluation mode
    
    with torch.no_grad():
        val_loader_length = len(val_loader)
        tmp = []
        
        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, decoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
            labels = labels.unsqueeze(-1)  # (B, N, T, 1)
            
            predict_length = labels.shape[2]  # T
            
            # Encode
            encoder_output = net.encode(encoder_inputs)
            
            # Decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]
            
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]
            
            loss = criterion(predict_output, labels)
            tmp.append(loss.item())
        
        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    
    return validation_loss


def predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type):
    """
    Make predictions and save results
    
    Parameters:
        net: neural network model
        data_loader: data loader
        data_target_tensor: target tensor
        epoch: current epoch
        _max: maximum values for denormalization
        _min: minimum values for denormalization
        params_path: path for saving results
        type: data type (train/val/test)
    """
    net.train(False)  # Set to evaluation mode
    
    start_time = time()
    
    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        loader_length = len(data_loader)
        
        prediction = []
        input = []
        
        start_time = time()
        
        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, decoder_inputs, labels = batch_data
            
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
            labels = labels.unsqueeze(-1)  # (B, N, T, 1)
            
            predict_length = labels.shape[2]  # T
            
            # Encode
            encoder_output = net.encode(encoder_inputs)
            input.append(encoder_inputs[:, :, :, 0:1].cpu().numpy())
            
            # Decode step by step
            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]
            
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]
            
            prediction.append(predict_output.detach().cpu().numpy())
        
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
        
        prediction = np.concatenate(prediction, 0)
        prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
        data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
        
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)
        
        # Calculate metrics
        excel_list = []
        prediction_length = prediction.shape[2]
        
        for i in range(prediction_length):
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i, 0])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i, 0]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i, 0], 0)
            excel_list.extend([mae, rmse, mape])
        
        # Overall metrics
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        excel_list.extend([mae, rmse, mape])


def load_graphdata_normY_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, percent=1.0):
    """
    Load and preprocess graph data with normalization
    
    Parameters:
        graph_signal_matrix_filename: path to data file
        num_of_hours: number of hours to look back
        num_of_days: number of days to look back  
        num_of_weeks: number of weeks to look back
        DEVICE: computation device
        batch_size: batch size for data loaders
        shuffle: whether to shuffle training data
        percent: percentage of training data to use
        
    Returns:
        train_loader, train_target_tensor, val_loader, val_target_tensor, 
        test_loader, test_target_tensor, _max, _min
    """
    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
    dirpath = os.path.dirname(graph_signal_matrix_filename)
    
    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) + '.npz')
    
    file_data = np.load(filename)
    train_x = file_data['train_x']
    train_target = file_data['train_target']
    train_timestamp = file_data['train_timestamp']
    
    # Apply percentage scaling
    train_x_length = train_x.shape[0]
    scale = int(train_x_length*percent)
    train_x = train_x[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]
    
    val_x = file_data['val_x']
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']
    
    test_x = file_data['test_x']
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']
    
    _max = file_data['mean']
    _min = file_data['std']
    
    # Normalize targets to [-1,1] range
    train_target_norm = max_min_normalization(train_target, _max[:, :, 0, :], _min[:, :, 0, :])
    test_target_norm = max_min_normalization(test_target, _max[:, :, 0, :], _min[:, :, 0, :])
    val_target_norm = max_min_normalization(val_target, _max[:, :, 0, :], _min[:, :, 0, :])
    
    # Prepare training data
    train_decoder_input_start = train_x[:, :, 0:1, -1:]  # Last known value as decoder input
    train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)
    train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]), axis=2)
    
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)
    
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Prepare validation data
    val_decoder_input_start = val_x[:, :, 0:1, -1:]
    val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)
    
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)
    
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # Prepare test data
    test_decoder_input_start = test_x[:, :, 0:1, -1:]
    test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)
    
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)
    
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min