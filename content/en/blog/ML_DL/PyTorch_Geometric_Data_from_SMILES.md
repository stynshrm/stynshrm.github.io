---
author: Satyan Sharma
title: Building PyTorch Geometric Data from SMILES
date: 2023-09-20
math: true
tags: ["Machine Learning", "Cheminformatics"]
thumbnail: /th/th_pg.png
---


```python
import numpy as np
```


```python
! pip install rdkit-pypi
```


```python
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
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


```python
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
```


```python
# define list of permitted atoms
permitted_list_of_atoms =  ['C','N','O','S','F','Si','P']
```


```python
# x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
x_smiles = ['c1ccccc1C(=O)O']

# convert SMILES to RDKit mol object
mol = Chem.MolFromSmiles(x_smiles[0])

# get feature dimensions
n_nodes = mol.GetNumAtoms()
n_edges = 2*mol.GetNumBonds()
```

**Node Features from SMILE**


```python
print(n_nodes, n_edges)
```

    9 18



```python
mol.GetAtoms()
```




    <rdkit.Chem.rdchem._ROAtomSeq at 0x7b773ad89c40>




```python
# construct node feature matrix X of shape (n_nodes, n_node_features)
n_nodes, n_node_features = n_nodes, len(permitted_list_of_atoms)
X = np.zeros((n_nodes, n_node_features))
```


```python
for atom in mol.GetAtoms():
  print(atom.GetSymbol())
```

    C
    C
    C
    C
    C
    C
    C
    O
    O



```python
symb = "Si"
atom_type_binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: symb == s, permitted_list_of_atoms))]
```


```python
print(permitted_list_of_atoms)
print(atom_type_binary_encoding)
```

    ['C', 'N', 'O', 'S', 'F', 'Si', 'P']
    [0, 0, 0, 0, 0, 1, 0]



```python
# Do it for entire SMILE
for atom in mol.GetAtoms():
  symb = str(atom.GetSymbol())
  atom_type_enc = [int(boolean_value) for boolean_value in list(map(lambda s: symb == s, permitted_list_of_atoms))]
  #print(atom_type_enc)
  he_neigh = atom.GetDegree()
  n_heavy_neighbors_enc = [int(boolean_value) for boolean_value in list(map(lambda s: he_neigh == s, [0, 1, 2, 3, 4, "MoreThanFour"]))]
  formal_charge_enc = [int(boolean_value) for boolean_value in list(map(lambda s: he_neigh == s, [-3, -2, -1, 0, 1, 2, 3, "Extreme"]))]
  atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc
  print(atom_feature_vector)
```

    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]



```python
for atom in mol.GetAtoms():
  ch = int(atom.GetFormalCharge())
  print(ch)
```

    0
    0
    0
    0
    0
    0
    0
    0
    0


**Edge Features**


```python
GetAdjacencyMatrix(mol)
```




    array([[0, 1, 0, 0, 0, 1, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 1, 0, 0, 0],
           [1, 0, 0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=int32)




```python
(rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
rows, cols
```




    (array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8]),
     array([1, 5, 0, 2, 1, 3, 2, 4, 3, 5, 0, 4, 6, 5, 7, 8, 6, 6]))




```python
#Lets look at last bond
i = 8
j = 6
mol.GetBondBetweenAtoms(i,j)
```




    <rdkit.Chem.rdchem.Bond at 0x7b773adc75a0>




```python
mol.GetBondBetweenAtoms(i,j).GetBondType()
```




    rdkit.Chem.rdchem.BondType.SINGLE




```python
mol.GetBondBetweenAtoms(i,j).IsInRing()
```




    False




```python
permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
```


```python
# Do it for entire SMILE
n_edge_features = 10
EF = np.zeros((n_edges, n_edge_features))

for (k, (i,j)) in enumerate(zip(rows, cols)):
  bond_type = mol.GetBondBetweenAtoms(int(i),int(j)).GetBondType()
  print(bond_type)
```

    AROMATIC
    AROMATIC
    AROMATIC
    AROMATIC
    AROMATIC
    AROMATIC
    AROMATIC
    AROMATIC
    AROMATIC
    AROMATIC
    AROMATIC
    AROMATIC
    SINGLE
    SINGLE
    DOUBLE
    SINGLE
    DOUBLE
    SINGLE



```python
for (k, (i,j)) in enumerate(zip(rows, cols)):
  bond_type = mol.GetBondBetweenAtoms(int(i),int(j)).GetBondType()
  bond_enc = [int(boolean_value) for boolean_value in list(map(lambda s: bond_type == s, permitted_list_of_bond_types))]
  print(bond_enc)
```

    [0, 0, 0, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 1]
    [1, 0, 0, 0]
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [1, 0, 0, 0]



```python
for (k, (i,j)) in enumerate(zip(rows, cols)):
  bond_type = mol.GetBondBetweenAtoms(int(i),int(j)).GetBondType()
  bond_type_enc = [int(boolean_value) for boolean_value in list(map(lambda s: bond_type == s, permitted_list_of_bond_types))]

  bond_type = (mol.GetBondBetweenAtoms(int(i),int(j)).GetIsConjugated())
  bond_is_conj_enc = [int(bond_type)]

  bond_type = (mol.GetBondBetweenAtoms(int(i),int(j)).IsInRing())
  bond_is_in_ring_enc = [int(bond_type)]

  bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

  print(bond_feature_vector)
```

    [0, 0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1, 1]
    [1, 0, 0, 0, 1, 0]
    [1, 0, 0, 0, 1, 0]
    [0, 1, 0, 0, 1, 0]
    [1, 0, 0, 0, 1, 0]
    [0, 1, 0, 0, 1, 0]
    [1, 0, 0, 0, 1, 0]



```python
torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
E = torch.stack([torch_rows, torch_cols], dim = 0)
E
```




    tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8],
            [1, 5, 0, 2, 1, 3, 2, 4, 3, 5, 0, 4, 6, 5, 7, 8, 6, 6]])



# Final Pytorch Geometric data object

will have,


* x = atom_feature_vector
*   edge_index : E
*   edge_attributes : bond_feature_vector



Ref : https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/


```python

```
