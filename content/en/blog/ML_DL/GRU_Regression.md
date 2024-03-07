---
author: Satyan Sharma
title: GRU Regression for Molcular Property Prediction
date: 2023-08-20
math: true
tags: ["Machine Learning", "Cheminformatics"]
thumbnail: /th/th_gru.png
---

# Gated Recurrent Units (GRU)

Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) that was introduced by Cho et al. in 2014. It uses gating mechanisms to selectively update the hidden state of the network at each time step, allowing them to effectively model sequential data such as time series, natural language, and speech.  

The network process sequential data by passing the hidden state from one time step to the next using gating mechanisms.


![png](/gru01a.png)

* ***Reset Gate*** -  identifies the unnecessary information and what information to delete at the specific timestamp. 

* ***Update Gate*** - identifies what current GRU cell will pass information to the next GRU cell thus, keeping track of the most important information.

* ***Current Memory Gate*** or ***Candidate  Hidden State*** 
Candidate  Hidden State is used to determine the information stored from the past. This is generally called the memory component in a GRU cell. 


![png](/gru01b.png)

* New Hidden State - the new hidden state and depends on the update gate and candidate hidden state. whenever 
$\boldsymbol{Z}_t$ is $0$, the information at the previously hidden layer gets forgotten. It is updated with the value of the new candidate hidden layer. If $\boldsymbol{Z}_t$ is $1$, then the information from the previously hidden layer is maintained. This is how the most relevant information is passed from one state to the next.




## Using GRU for Regression on qm9 dataset.

```python
import os
import re
from collections import Counter
from pathlib import Path
```


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```


```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
```


```python
# Use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# seed random generator
_ = torch.manual_seed(42)
```

# Dataset


```python
QM9_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
df = pd.read_csv(QM9_CSV_URL)
```


```python
df.head()
```





  <div id="df-95294609-7e3b-4898-84cb-d8bc8f82e6e2" class="colab-df-container">
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
      <th>mol_id</th>
      <th>smiles</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>mu</th>
      <th>alpha</th>
      <th>homo</th>
      <th>lumo</th>
      <th>gap</th>
      <th>...</th>
      <th>zpve</th>
      <th>u0</th>
      <th>u298</th>
      <th>h298</th>
      <th>g298</th>
      <th>cv</th>
      <th>u0_atom</th>
      <th>u298_atom</th>
      <th>h298_atom</th>
      <th>g298_atom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>gdb_1</td>
      <td>C</td>
      <td>157.71180</td>
      <td>157.709970</td>
      <td>157.706990</td>
      <td>0.0000</td>
      <td>13.21</td>
      <td>-0.3877</td>
      <td>0.1171</td>
      <td>0.5048</td>
      <td>...</td>
      <td>0.044749</td>
      <td>-40.478930</td>
      <td>-40.476062</td>
      <td>-40.475117</td>
      <td>-40.498597</td>
      <td>6.469</td>
      <td>-395.999595</td>
      <td>-398.643290</td>
      <td>-401.014647</td>
      <td>-372.471772</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gdb_2</td>
      <td>N</td>
      <td>293.60975</td>
      <td>293.541110</td>
      <td>191.393970</td>
      <td>1.6256</td>
      <td>9.46</td>
      <td>-0.2570</td>
      <td>0.0829</td>
      <td>0.3399</td>
      <td>...</td>
      <td>0.034358</td>
      <td>-56.525887</td>
      <td>-56.523026</td>
      <td>-56.522082</td>
      <td>-56.544961</td>
      <td>6.316</td>
      <td>-276.861363</td>
      <td>-278.620271</td>
      <td>-280.399259</td>
      <td>-259.338802</td>
    </tr>
    <tr>
      <th>2</th>
      <td>gdb_3</td>
      <td>O</td>
      <td>799.58812</td>
      <td>437.903860</td>
      <td>282.945450</td>
      <td>1.8511</td>
      <td>6.31</td>
      <td>-0.2928</td>
      <td>0.0687</td>
      <td>0.3615</td>
      <td>...</td>
      <td>0.021375</td>
      <td>-76.404702</td>
      <td>-76.401867</td>
      <td>-76.400922</td>
      <td>-76.422349</td>
      <td>6.002</td>
      <td>-213.087624</td>
      <td>-213.974294</td>
      <td>-215.159658</td>
      <td>-201.407171</td>
    </tr>
    <tr>
      <th>3</th>
      <td>gdb_4</td>
      <td>C#C</td>
      <td>0.00000</td>
      <td>35.610036</td>
      <td>35.610036</td>
      <td>0.0000</td>
      <td>16.28</td>
      <td>-0.2845</td>
      <td>0.0506</td>
      <td>0.3351</td>
      <td>...</td>
      <td>0.026841</td>
      <td>-77.308427</td>
      <td>-77.305527</td>
      <td>-77.304583</td>
      <td>-77.327429</td>
      <td>8.574</td>
      <td>-385.501997</td>
      <td>-387.237686</td>
      <td>-389.016047</td>
      <td>-365.800724</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gdb_5</td>
      <td>C#N</td>
      <td>0.00000</td>
      <td>44.593883</td>
      <td>44.593883</td>
      <td>2.8937</td>
      <td>12.99</td>
      <td>-0.3604</td>
      <td>0.0191</td>
      <td>0.3796</td>
      <td>...</td>
      <td>0.016601</td>
      <td>-93.411888</td>
      <td>-93.409370</td>
      <td>-93.408425</td>
      <td>-93.431246</td>
      <td>6.278</td>
      <td>-301.820534</td>
      <td>-302.906752</td>
      <td>-304.091489</td>
      <td>-288.720028</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-95294609-7e3b-4898-84cb-d8bc8f82e6e2')"
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

    <script>
      const buttonEl =
        document.querySelector('#df-95294609-7e3b-4898-84cb-d8bc8f82e6e2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-95294609-7e3b-4898-84cb-d8bc8f82e6e2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-11efdd04-19aa-49ca-803e-49f53a162636">
  <button class="colab-df-quickchart" onclick="quickchart('df-11efdd04-19aa-49ca-803e-49f53a162636')"
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
        document.querySelector('#df-11efdd04-19aa-49ca-803e-49f53a162636 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
class SmilesTokenizer(object):
    """
    A simple regex-based tokenizer adapted from the deepchem smiles_tokenizer package.
    SMILES regex pattern for the tokenization is designed by Schwaller et. al., ACS Cent. Sci 5 (2019)
    """

    def __init__(self):
        self.regex_pattern = (
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\."
            r"|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        )
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, smiles):
        """
        Tokenizes SMILES string.
        """
        tokens = [token for token in self.regex.findall(smiles)]
        return tokens
```


```python
def build_vocab(smiles_list, tokenizer, max_vocab_size):
    """
    Builds a vocabulary of N=max_vocab_size most common tokens from list of SMILES strings.
    -------
    Dict[str, int]
        A dictionary that defines mapping of a token to its index in the vocabulary.
    """
    tokenized_smiles = [tokenizer.tokenize(s) for s in smiles_list]
    token_counter = Counter(c for s in tokenized_smiles for c in s)
    tokens = [token for token, _ in token_counter.most_common(max_vocab_size)]
    vocab = {token: idx for idx, token in enumerate(tokens)}
    return vocab


def smiles_to_ohe(smiles, tokenizer, vocab):
    """
    Transforms SMILES string to one-hot encoding representation.
    Returns - Tensor
    """
    unknown_token_id = len(vocab) - 1
    token_ids = [vocab.get(token, unknown_token_id) for token in tokenizer.tokenize(smiles)]
    ohe = torch.eye(len(vocab))[token_ids]
    return ohe
```


```python
# Test above functions
tokenizer = SmilesTokenizer()

smiles = "C=CS"
print("SMILES string:\n\t", smiles)
print("Tokens:\n\t", ", ".join(tokenizer.tokenize(smiles)))
vocab = build_vocab([smiles], tokenizer, 3)
print("Vocab:\n\t", vocab)
print("One-Hot-Enc:\n", np.array(smiles_to_ohe(smiles, tokenizer, vocab)).T)
```

    SMILES string:
    	 C=CS
    Tokens:
    	 C, =, C, S
    Vocab:
    	 {'C': 0, '=': 1, 'S': 2}
    One-Hot-Enc:
     [[1. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]]


**PreProcess Data**


```python

sample_size = 50000
n_train = 40000
n_test = n_val = 5000

# get a sample
df = df.sample(n=sample_size, axis=0, random_state=42)

# select columns from the data frame
smiles = df["smiles"].tolist()
y = df["mu"].to_numpy()

# build a vocab using the training data
max_vocab_size = 30
vocab = build_vocab(smiles[:n_train], tokenizer, max_vocab_size)
vocab_size = len(vocab)

# transform smiles to one-hot encoded tensors and apply padding
X = pad_sequence(
    sequences=[smiles_to_ohe(smi, tokenizer, vocab) for smi in smiles],
    batch_first=True,
    padding_value=0,
)
```


```python
# normalize the target using the training data
train_mean = y[:n_train].mean()
train_std = y[:n_train].std()
y = (y - train_mean) / train_std
```

# Build Dataset


```python
# build dataset
data = TensorDataset(X, torch.Tensor(y))

# define loaders
ids_train = np.arange(n_train)
ids_val = np.arange(n_val) + n_train
ids_test = np.arange(n_test) + n_train + n_val
train_loader = DataLoader(
    Subset(data, ids_train),
    batch_size=64,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
)
val_loader = DataLoader(
    Subset(data, ids_val), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42)
)
test_loader = DataLoader(
    Subset(data, ids_test),
    batch_size=1,
    shuffle=False,
    generator=torch.Generator().manual_seed(42),
)
```

# Build Model


```python
class GRURegressionModel(nn.Module):
    """GRU network with one recurrent layer"""

    def __init__(self, input_size, hidden_size=32, num_layers=1):
        """
        GRU network

        Parameters
        ----------
        input_size : int
            The number of expected features in the input vector
        hidden_size : int
            The number of features in the hidden state

        """
        super(GRURegressionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, hn = self.gru(x, h0)
        out = out[:, -1]
        out = self.dropout(out)
        out = self.fc(out)
        return out
```

# Training Class


```python
class ModelTrainer(object):
    """A class that provides training and validation infrastructure for the model and keeps track of training and validation metrics."""

    def __init__(self, model, lr, name=None, clip_gradients=False):
        """
        Initialization.

        Parameters
        ----------
        model : nn.Module
            a model
        lr : float
            learning rate for one training step

        """
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.clip_gradients = clip_gradients
        self.model.to(device)

        self.train_loss = []
        self.batch_loss = []
        self.val_loss = []

    def _train_epoch(self, loader):
        self.model.train()
        epoch_loss = 0
        batch_losses = []
        for i, (X_batch, y_batch) in enumerate(loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()

            if self.clip_gradients:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)

            self.optimizer.step()
            epoch_loss += loss.item()
            batch_losses.append(loss.item())

        return epoch_loss / len(loader), batch_losses

    def _eval_epoch(self, loader):
        self.model.eval()
        val_loss = 0
        predictions = []
        targets = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch.unsqueeze(1))
                val_loss += loss.item()
                predictions.append(y_pred.detach().numpy())
                targets.append(y_batch.unsqueeze(1).detach().numpy())

        predictions = np.concatenate(predictions).flatten()
        targets = np.concatenate(targets).flatten()
        return val_loss / len(loader), predictions, targets

    def train(self, train_loader, val_loader, n_epochs, print_every=10):
        """
        Train the model

        Parameters
        ----------
        train_loader :
            a dataloader with training data
        val_loader :
            a dataloader with training data
        n_epochs :
            number of epochs to train for
        """
        for e in range(n_epochs):
            train_loss, train_loss_batches = self._train_epoch(train_loader)
            val_loss, _, _ = self._eval_epoch(test_loader)
            self.batch_loss += train_loss_batches
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            if e % print_every == 0:
                print(f"Epoch {e+0:03} | train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f}")

    def validate(self, val_loader):
        """
        Validate the model

        Parameters
        ----------
        val_loader :
            a dataloader with training data

        Returns
        -------
        Tuple[list, list, list]
            Loss, y_predicted, y_target for each datapoint in val_loader.
        """
        loss, y_pred, y_targ = self._eval_epoch(val_loader)
        return loss, y_pred, y_targ
```


```python
model_gru = ModelTrainer(
    model=GRURegressionModel(vocab_size, hidden_size=32),
    lr=1e-3,
)
```


```python
model_gru.train(train_loader, val_loader, 51)
```

    Epoch 000 | train_loss: 0.71333 | val_loss: 0.55967
    Epoch 010 | train_loss: 0.44784 | val_loss: 0.43269
    Epoch 020 | train_loss: 0.40651 | val_loss: 0.39215
    Epoch 030 | train_loss: 0.37712 | val_loss: 0.37454
    Epoch 040 | train_loss: 0.35705 | val_loss: 0.36291
    Epoch 050 | train_loss: 0.33840 | val_loss: 0.35006


# Loss Checking and Evaluation


```python
_ = plt.plot(model_gru.train_loss, label=f"GRU train")
_ = plt.plot(model_gru.val_loss, label=f"GRU val")
_ = plt.xlabel("epoch")
_ = plt.ylabel("MSE")
_ = plt.legend()
```


    
![png](/gru022.png)
    


**Reference:**
Talktorial on Cheminformatics: T034 by Volkamer Lab


