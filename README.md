# MDAS-GNN: Multi-Dimensional Adaptive Spatial-Graph Neural Network

A PyTorch implementation of MDAS-GNN for spatio-temporal prediction tasks, particularly focused on traffic accident risk prediction.

## Overview

MDAS-GNN is a neural network model that combines spatial graph convolution with multi-dimensional attention mechanisms to predict future values in spatio-temporal data. The model is especially designed for predicting traffic accident risks on road networks.

## Features

- **Multi-dimensional feature processing**: Handles accident severity, infrastructure risk, and environmental risk
- **Spatial-temporal modeling**: Combines GCN with transformer-style attention
- **Flexible temporal dependencies**: Supports weekly, daily, and hourly patterns
- **Adaptive attention**: Feature-specific spatial attention mechanisms

## Quick Start

### 1. Data Preparation

First, prepare your accident data:

```bash
python npz.py
```

Then generate the training dataset:

```bash
python prepareData.py --config configurations/accident.conf
```

### 2. Training

Train the model:

```bash
python train_MDASGNN.py --config configurations/accident.conf --cuda=0
```

For background training:

```bash
nohup python -u train_MDASGNN.py --config configurations/accident.conf --cuda=0 > accident.out &
```

## Project Structure

```
├── MDASGNN.py           # Main model implementation
├── train_MDASGNN.py     # Training script
├── prepareData.py       # Data preprocessing
├── npz.py              # Data preparation from raw sources
├── utils.py            # Utility functions
├── metrics.py          # Evaluation metrics
├── configurations/
│   └── accident.conf   # Configuration file
└── data/
    ├── accident/       # Accident data
    └── processed/      # Processed data
```

## Configuration

Key parameters in `accident.conf`:

- **Data Settings**:
  - `num_of_vertices`: Number of road segments (2144 for Central London)
  - `points_per_hour`: Time resolution (12 = 5-minute intervals)
  - `num_for_predict`: Prediction horizon

- **Model Settings**:
  - `num_layers`: Number of attention layers
  - `d_model`: Model dimension
  - `nb_head`: Number of attention heads
  - `encoder_input_size`: Input feature dimensions (3)

- **Training Settings**:
  - `batch_size`: Training batch size
  - `learning_rate`: Learning rate
  - `epochs`: Training epochs

## Data Format

The model expects data in the following format:

- **Input**: `(batch_size, num_nodes, time_steps, features)`
- **Features**: 
  - Accident Severity Intensity
  - Infrastructure Risk
  - Environmental Risk
- **Output**: Predicted values for next time steps

## Requirements

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- GeoPandas (for spatial data processing)
- TensorboardX (for logging)

## Model Architecture

MDASGNN combines:

1. **Spatial Graph Convolution**: Captures spatial dependencies between road segments
2. **Multi-Head Attention**: Models temporal dependencies and feature interactions
3. **Feature-Specific Processing**: Separate attention mechanisms for different risk factors
4. **Encoder-Decoder Structure**: Flexible prediction horizon

## Results

The model outputs predictions for accident risk at each road segment for the specified time horizon. Evaluation uses standard metrics including MAE, RMSE, and MAPE.

