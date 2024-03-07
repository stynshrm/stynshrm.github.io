---
author: Satyan Sharma
title: Solubility Prediction using GNN
date: 2023-11-20
math: true
tags: ["Machine Learning", "Cheminformatics"]
thumbnail: /th/th_sol.png
---

 We will use ESOL dataset and train GNN model to predict solubility directly from chemical structures


```python
! pip install rdkit-pypi
```


```python
!python -c "import torch; print(torch.__version__)"
```

    2.1.0+cu121



```python
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
!pip install torch-geometric
```

# Dataset


```python
import rdkit
from torch_geometric.datasets import MoleculeNet

# Load the ESOL dataset
data = MoleculeNet(root=".", name="ESOL")
data
```

    Downloading https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
    Processing...
    Done!

    ESOL(1128)




```python
# Investigating the dataset
print("Dataset type: ", type(data))
print("Dataset features: ", data.num_features)
print("Dataset target: ", data.num_classes)
print("Dataset length: ", data.len())
print("Sample  nodes: ", data[0].num_nodes)
print("Sample  edges: ", data[0].num_edges)
```

    Dataset type:  <class 'torch_geometric.datasets.molecule_net.MoleculeNet'>
    Dataset features:  9
    Dataset target:  734
    Dataset length:  1128
    Sample  nodes:  32
    Sample  edges:  68



```python
print("Dataset sample: ", data[0])
```

    Dataset sample:  Data(x=[32, 9], edge_index=[2, 68], edge_attr=[68, 3], smiles='OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ', y=[1, 1])



```python
# Look at features
data[0].x
```




    tensor([[8, 0, 2, 5, 1, 0, 4, 0, 0],
            [6, 0, 4, 5, 2, 0, 4, 0, 0],
            [6, 0, 4, 5, 1, 0, 4, 0, 1],
            [8, 0, 2, 5, 0, 0, 4, 0, 1],
            [6, 0, 4, 5, 1, 0, 4, 0, 1],
            [8, 0, 2, 5, 0, 0, 4, 0, 0],
            [6, 0, 4, 5, 2, 0, 4, 0, 0],
            [6, 0, 4, 5, 1, 0, 4, 0, 1],
            [8, 0, 2, 5, 0, 0, 4, 0, 1],
            [6, 0, 4, 5, 1, 0, 4, 0, 1],
            [8, 0, 2, 5, 0, 0, 4, 0, 0],
            [6, 0, 4, 5, 1, 0, 4, 0, 0],
            [6, 0, 2, 5, 0, 0, 2, 0, 0],
            [7, 0, 1, 5, 0, 0, 2, 0, 0],
            [6, 0, 3, 5, 0, 0, 3, 1, 1],
            [6, 0, 3, 5, 1, 0, 3, 1, 1],
            [6, 0, 3, 5, 1, 0, 3, 1, 1],
            [6, 0, 3, 5, 1, 0, 3, 1, 1],
            [6, 0, 3, 5, 1, 0, 3, 1, 1],
            [6, 0, 3, 5, 1, 0, 3, 1, 1],
            [6, 0, 4, 5, 1, 0, 4, 0, 1],
            [8, 0, 2, 5, 1, 0, 4, 0, 0],
            [6, 0, 4, 5, 1, 0, 4, 0, 1],
            [8, 0, 2, 5, 1, 0, 4, 0, 0],
            [6, 0, 4, 5, 1, 0, 4, 0, 1],
            [8, 0, 2, 5, 1, 0, 4, 0, 0],
            [6, 0, 4, 5, 1, 0, 4, 0, 1],
            [8, 0, 2, 5, 1, 0, 4, 0, 0],
            [6, 0, 4, 5, 1, 0, 4, 0, 1],
            [8, 0, 2, 5, 1, 0, 4, 0, 0],
            [6, 0, 4, 5, 1, 0, 4, 0, 1],
            [8, 0, 2, 5, 1, 0, 4, 0, 0]])




```python
# Investigating the edges in sparse COO format
data[0].edge_index.t()
```




    tensor([[ 0,  1],
            [ 1,  0],
            [ 1,  2],
            [ 2,  1],
            [ 2,  3],
            [ 2, 30],
            [ 3,  2],
            [ 3,  4],
            [ 4,  3],
            [ 4,  5],
            [ 4, 26],
            [ 5,  4],
            [ 5,  6],
            [ 6,  5],
            [ 6,  7],
            [ 7,  6],
            [ 7,  8],
            [ 7, 24],
            [ 8,  7],
            [ 8,  9],
            [ 9,  8],
            [ 9, 10],
            [ 9, 20],
            [10,  9],
            [10, 11],
            [11, 10],
            [11, 12],
            [11, 14],
            [12, 11],
            [12, 13],
            [13, 12],
            [14, 11],
            [14, 15],
            [14, 19],
            [15, 14],
            [15, 16],
            [16, 15],
            [16, 17],
            [17, 16],
            [17, 18],
            [18, 17],
            [18, 19],
            [19, 14],
            [19, 18],
            [20,  9],
            [20, 21],
            [20, 22],
            [21, 20],
            [22, 20],
            [22, 23],
            [22, 24],
            [23, 22],
            [24,  7],
            [24, 22],
            [24, 25],
            [25, 24],
            [26,  4],
            [26, 27],
            [26, 28],
            [27, 26],
            [28, 26],
            [28, 29],
            [28, 30],
            [29, 28],
            [30,  2],
            [30, 28],
            [30, 31],
            [31, 30]])




```python
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
molecule = Chem.MolFromSmiles(data[0]["smiles"])
molecule
```




    
![png](/solGNN_9_0.png)
    



# Graph Neural Network




```python
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
```


```python
embedding_size = 64

class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(data.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size*2, 1)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden
```


```python
model = GCN()
print(model)
```

    GCN(
      (initial_conv): GCNConv(9, 64)
      (conv1): GCNConv(64, 64)
      (conv2): GCNConv(64, 64)
      (conv3): GCNConv(64, 64)
      (out): Linear(in_features=128, out_features=1, bias=True)
    )


# Train


```python
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Wrap data in a data loader
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 64
loader = DataLoader(data[:int(data_size * 0.8)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.8):],
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

def train(data):
    # Enumerate over the data
    for batch in loader:
      # Use GPU
      batch.to(device)
      # Reset gradients
      optimizer.zero_grad()
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
      # Calculating the loss and gradients
      loss = loss_fn(pred, batch.y)
      loss.backward()
      # Update using the gradients
      optimizer.step()
    return loss, embedding

print("Starting training...")
losses = []
for epoch in range(1000):
    loss, h = train(data)
    losses.append(loss)
    if epoch % 100 == 0:
      print(f"Epoch {epoch} | Train Loss {loss}")
```

    Starting training...
    Epoch 0 | Train Loss 0.7323977947235107
    Epoch 100 | Train Loss 0.5643615126609802
    Epoch 200 | Train Loss 0.8129488825798035
    Epoch 300 | Train Loss 0.5515668988227844
    Epoch 400 | Train Loss 0.26473188400268555
    Epoch 500 | Train Loss 0.3548230826854706
    Epoch 600 | Train Loss 0.10742906481027603
    Epoch 700 | Train Loss 0.29880979657173157
    Epoch 800 | Train Loss 0.08752292394638062
    Epoch 900 | Train Loss 0.0839475765824318


# Predictions


```python
import pandas as pd
```


```python
# Analyze the results for one batch
test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(device)
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
    df = pd.DataFrame()
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()
df["y_real"] = df["y_real"].apply(lambda row: row[0])
df["y_pred"] = df["y_pred"].apply(lambda row: row[0])
df
```





  <div id="df-3eafda81-0e14-4f21-857d-2ca53efc1bd7" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_real</th>
      <th>y_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.300</td>
      <td>-1.867198</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-3.953</td>
      <td>-4.592132</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.091</td>
      <td>-3.610150</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.210</td>
      <td>-2.107021</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-5.850</td>
      <td>-4.743058</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>59</th>
      <td>-4.522</td>
      <td>-4.377628</td>
    </tr>
    <tr>
      <th>60</th>
      <td>-4.286</td>
      <td>-1.482177</td>
    </tr>
    <tr>
      <th>61</th>
      <td>-3.900</td>
      <td>-3.672484</td>
    </tr>
    <tr>
      <th>62</th>
      <td>-5.060</td>
      <td>-4.966655</td>
    </tr>
    <tr>
      <th>63</th>
      <td>-7.200</td>
      <td>-7.057222</td>
    </tr>
  </tbody>
</table>
<p>64 rows × 2 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3eafda81-0e14-4f21-857d-2ca53efc1bd7')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>


  </div>


<div id="df-d809fc59-3ab1-4c5c-9490-ebb0de241859">
  <button class="colab-df-quickchart" onclick="quickchart('df-d809fc59-3ab1-4c5c-9490-ebb0de241859')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-d809fc59-3ab1-4c5c-9490-ebb0de241859 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
plt = sns.scatterplot(data=df, x="y_real", y="y_pred")
plt.set(xlim=(-7, 2))
plt.set(ylim=(-7, 2))
plt
```
    
![png](/solGNN_19_1.png)
    

