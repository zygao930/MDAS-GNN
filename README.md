# MDAS-GNN: Multi-Dimensional Spatiotemporal Graph Neural Network

A PyTorch implementation of MDAS-GNN for predicting traffic accident risks on urban road networks.

## Overview

MDAS-GNN models traffic accident risk through three complementary perspectives:
- **Traffic Safety Risk**: Historical accident patterns and severity
- **Infrastructure Risk**: Road characteristics and built environment
- **Environmental Risk**: Weather, lighting, and temporal conditions

The model uses spatial diffusion to understand how risks propagate through road networks, combined with attention mechanisms to capture temporal patterns.

## Quick Start

### 1. Preprocess Data

```bash
python npz.py
```

### 2. Prepare Training Data

```bash
python prepareData.py --config configurations/accident.conf
```

### 3. Train Model

```bash
python train_MDASGNN.py --config configurations/accident.conf --cuda=0
```

For background training:
```bash
nohup python -u train_MDASGNN.py --config configurations/accident.conf --cuda=0 > accident.out &
```

## Requirements

```
torch>=1.9.0
numpy>=1.19.0
pandas>=1.2.0
scipy>=1.6.0
scikit-learn>=0.24.0
```

## Key Features

- Multi-dimensional risk modeling with feature-specific spatial diffusion
- Encoder-decoder architecture with attention mechanisms
- Weekly temporal aggregation for improved prediction stability
- Flexible configuration for different prediction horizons

