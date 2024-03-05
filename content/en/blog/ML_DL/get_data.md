---
author: Satyan Sharma
title: Hands on with ChEMBL API
date: 2023-11-22
math: true
tags: ["Cheminformatics"]
---

ChEMBL is a large-scale bioactivity database that collects information on the interactions between small molecules (such as drugs, compounds, or substances) and their biological targets. It stands for "Chemical Biology Database" and is maintained by the European Molecular Biology Laboratory (EMBL).


Researchers can use ChEMBL to search for specific compounds, investigate their interactions with biological targets (such as enzymes, receptors, or ion channels), and analyze bioactivity data to identify potential drug candidates or understand the mechanisms of action for existing drugs. Additionally, ChEMBL provides tools and APIs (Application Programming Interfaces) for accessing and querying the database programmatically, enabling integration with other bioinformatics and cheminformatics workflows.


```python
import math
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory
```


```python
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
```


```python
!pip install chembl-webresource-client
```


```python
from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm
```


```python
DATA = Path.cwd() / 'data'
DATA.mkdir()
```




```python
print(DATA)
```

    /Users/sasha/CADD/001_get_bioactivity_data_CHEMBL/data


 resource objects for API access.


```python
targets_api = new_client.target
compounds_api = new_client.molecule
bioactivities_api = new_client.activity
```

#### Get target data (EGFR kinase: UniProtID : P00533)¶


```python
uniprot_id = "P00533"

# Get target information from ChEMBL for specified values only
targets = targets_api.get(target_components__accession=uniprot_id).only(
    "target_chembl_id", "organism", "pref_name", "target_type"
)
print(f'The type of the targets is "{type(targets)}"')
```

    The type of the targets is "<class 'chembl_webresource_client.query_set.QuerySet'>"



```python
targets = pd.DataFrame.from_records(targets)
targets
```




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
      <th>organism</th>
      <th>pref_name</th>
      <th>target_chembl_id</th>
      <th>target_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Homo sapiens</td>
      <td>Epidermal growth factor receptor erbB1</td>
      <td>CHEMBL203</td>
      <td>SINGLE PROTEIN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Homo sapiens</td>
      <td>Epidermal growth factor receptor erbB1</td>
      <td>CHEMBL203</td>
      <td>SINGLE PROTEIN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Homo sapiens</td>
      <td>Epidermal growth factor receptor and ErbB2 (HE...</td>
      <td>CHEMBL2111431</td>
      <td>PROTEIN FAMILY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Homo sapiens</td>
      <td>Epidermal growth factor receptor</td>
      <td>CHEMBL2363049</td>
      <td>PROTEIN FAMILY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Homo sapiens</td>
      <td>MER intracellular domain/EGFR extracellular do...</td>
      <td>CHEMBL3137284</td>
      <td>CHIMERIC PROTEIN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Homo sapiens</td>
      <td>Protein cereblon/Epidermal growth factor receptor</td>
      <td>CHEMBL4523680</td>
      <td>PROTEIN-PROTEIN INTERACTION</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Homo sapiens</td>
      <td>EGFR/PPP1CA</td>
      <td>CHEMBL4523747</td>
      <td>PROTEIN-PROTEIN INTERACTION</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Homo sapiens</td>
      <td>VHL/EGFR</td>
      <td>CHEMBL4523998</td>
      <td>PROTEIN-PROTEIN INTERACTION</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Homo sapiens</td>
      <td>Baculoviral IAP repeat-containing protein 2/Ep...</td>
      <td>CHEMBL4802031</td>
      <td>PROTEIN-PROTEIN INTERACTION</td>
    </tr>
  </tbody>
</table>
</div>



### Restrict to first entry as our target of iterest


```python
chembl_id = targets.iloc[0].target_chembl_id
print(f"{chembl_id}")
```

    CHEMBL203


### Fetch bioactivty data for the target_chembl_id : CHEMBL_203 


```python
# fetch the bioactivity data and filter it to only human proteins, IC50, exact measurement, binding data

bioactivities = bioactivities_api.filter(target_chembl_id=chembl_id, type="IC50", relation="=", assay_type="B").only(
    "activity_id",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "molecule_chembl_id",
    "type",
    "standard_units",
    "relation",
    "standard_value",
    "target_chembl_id",
    "target_organism",
)

print(f"Length and type of bioactivities object: {len(bioactivities)}, {type(bioactivities)}")
```

    Length and type of bioactivities object: 10420, <class 'chembl_webresource_client.query_set.QuerySet'>



```python
# Whats in here, look at first entry

bioactivities[0]
```




    {'activity_id': 32260,
     'assay_chembl_id': 'CHEMBL674637',
     'assay_description': 'Inhibitory activity towards tyrosine phosphorylation for the epidermal growth factor-receptor kinase',
     'assay_type': 'B',
     'molecule_chembl_id': 'CHEMBL68920',
     'relation': '=',
     'standard_units': 'nM',
     'standard_value': '41.0',
     'target_chembl_id': 'CHEMBL203',
     'target_organism': 'Homo sapiens',
     'type': 'IC50',
     'units': 'uM',
     'value': '0.041'}




```python
# Download into a data frame
bioactivities_df = pd.DataFrame.from_dict(bioactivities)
print(f"DataFrame shape: {bioactivities_df.shape}")
bioactivities_df.head()
```

    DataFrame shape: (10420, 13)





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
      <th>activity_id</th>
      <th>assay_chembl_id</th>
      <th>assay_description</th>
      <th>assay_type</th>
      <th>molecule_chembl_id</th>
      <th>relation</th>
      <th>standard_units</th>
      <th>standard_value</th>
      <th>target_chembl_id</th>
      <th>target_organism</th>
      <th>type</th>
      <th>units</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32260</td>
      <td>CHEMBL674637</td>
      <td>Inhibitory activity towards tyrosine phosphory...</td>
      <td>B</td>
      <td>CHEMBL68920</td>
      <td>=</td>
      <td>nM</td>
      <td>41.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
      <td>uM</td>
      <td>0.041</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32267</td>
      <td>CHEMBL674637</td>
      <td>Inhibitory activity towards tyrosine phosphory...</td>
      <td>B</td>
      <td>CHEMBL69960</td>
      <td>=</td>
      <td>nM</td>
      <td>170.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
      <td>uM</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32680</td>
      <td>CHEMBL677833</td>
      <td>In vitro inhibition of Epidermal growth factor...</td>
      <td>B</td>
      <td>CHEMBL137635</td>
      <td>=</td>
      <td>nM</td>
      <td>9300.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
      <td>uM</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32770</td>
      <td>CHEMBL674643</td>
      <td>Inhibitory concentration of EGF dependent auto...</td>
      <td>B</td>
      <td>CHEMBL306988</td>
      <td>=</td>
      <td>nM</td>
      <td>500000.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
      <td>uM</td>
      <td>500.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32772</td>
      <td>CHEMBL674643</td>
      <td>Inhibitory concentration of EGF dependent auto...</td>
      <td>B</td>
      <td>CHEMBL66879</td>
      <td>=</td>
      <td>nM</td>
      <td>3000000.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
      <td>uM</td>
      <td>3000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# units has different values
bioactivities_df["units"].unique()
```




    array(['uM', 'nM', 'pM', 'M', "10'3 uM", "10'1 ug/ml", 'ug ml-1',
           "10'-1microM", "10'1 uM", "10'-1 ug/ml", "10'-2 ug/ml", "10'2 uM",
           "10'-3 ug/ml", "10'-2microM", '/uM', "10'-6g/ml", 'mM', 'umol/L',
           'nmol/L', "10'-10M", "10'-7M", 'nmol', '10^-8M', 'µM'],
          dtype=object)




```python
# drop 
bioactivities_df.drop(["units", "value"], axis=1, inplace=True)
bioactivities_df.head()
```




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
      <th>activity_id</th>
      <th>assay_chembl_id</th>
      <th>assay_description</th>
      <th>assay_type</th>
      <th>molecule_chembl_id</th>
      <th>relation</th>
      <th>standard_units</th>
      <th>standard_value</th>
      <th>target_chembl_id</th>
      <th>target_organism</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32260</td>
      <td>CHEMBL674637</td>
      <td>Inhibitory activity towards tyrosine phosphory...</td>
      <td>B</td>
      <td>CHEMBL68920</td>
      <td>=</td>
      <td>nM</td>
      <td>41.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32267</td>
      <td>CHEMBL674637</td>
      <td>Inhibitory activity towards tyrosine phosphory...</td>
      <td>B</td>
      <td>CHEMBL69960</td>
      <td>=</td>
      <td>nM</td>
      <td>170.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32680</td>
      <td>CHEMBL677833</td>
      <td>In vitro inhibition of Epidermal growth factor...</td>
      <td>B</td>
      <td>CHEMBL137635</td>
      <td>=</td>
      <td>nM</td>
      <td>9300.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32770</td>
      <td>CHEMBL674643</td>
      <td>Inhibitory concentration of EGF dependent auto...</td>
      <td>B</td>
      <td>CHEMBL306988</td>
      <td>=</td>
      <td>nM</td>
      <td>500000.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32772</td>
      <td>CHEMBL674643</td>
      <td>Inhibitory concentration of EGF dependent auto...</td>
      <td>B</td>
      <td>CHEMBL66879</td>
      <td>=</td>
      <td>nM</td>
      <td>3000000.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
    </tr>
  </tbody>
</table>
</div>




```python
bioactivities_df.dtypes
```




    activity_id            int64
    assay_chembl_id       object
    assay_description     object
    assay_type            object
    molecule_chembl_id    object
    relation              object
    standard_units        object
    standard_value        object
    target_chembl_id      object
    target_organism       object
    type                  object
    dtype: object




```python
bioactivities_df = bioactivities_df.astype({"standard_value": "float64"})
```

### Data cleaning
1. Delete missing entries
2. Keep only entries with “standard_unit == nM”
3. Delete duplicates in molecule_chembl_id
4. Rename Columns


```python
# Delete Missing values
bioactivities_df.dropna(axis=0, how="any", inplace=True)
print(f"DataFrame shape: {bioactivities_df.shape}")
```

    DataFrame shape: (10419, 11)



```python
bioactivities_df = bioactivities_df[bioactivities_df["standard_units"] == "nM"]
bioactivities_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
print(f"DataFrame shape: {bioactivities_df.shape}")
```

    DataFrame shape: (6823, 11)



```python
bioactivities_df.reset_index(drop=True, inplace=True)

bioactivities_df.rename(columns={"standard_value": "IC50", "standard_units": "units"}, inplace=True)
bioactivities_df.head()
```




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
      <th>activity_id</th>
      <th>assay_chembl_id</th>
      <th>assay_description</th>
      <th>assay_type</th>
      <th>molecule_chembl_id</th>
      <th>relation</th>
      <th>units</th>
      <th>IC50</th>
      <th>target_chembl_id</th>
      <th>target_organism</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32260</td>
      <td>CHEMBL674637</td>
      <td>Inhibitory activity towards tyrosine phosphory...</td>
      <td>B</td>
      <td>CHEMBL68920</td>
      <td>=</td>
      <td>nM</td>
      <td>41.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32267</td>
      <td>CHEMBL674637</td>
      <td>Inhibitory activity towards tyrosine phosphory...</td>
      <td>B</td>
      <td>CHEMBL69960</td>
      <td>=</td>
      <td>nM</td>
      <td>170.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32680</td>
      <td>CHEMBL677833</td>
      <td>In vitro inhibition of Epidermal growth factor...</td>
      <td>B</td>
      <td>CHEMBL137635</td>
      <td>=</td>
      <td>nM</td>
      <td>9300.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32770</td>
      <td>CHEMBL674643</td>
      <td>Inhibitory concentration of EGF dependent auto...</td>
      <td>B</td>
      <td>CHEMBL306988</td>
      <td>=</td>
      <td>nM</td>
      <td>500000.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32772</td>
      <td>CHEMBL674643</td>
      <td>Inhibitory concentration of EGF dependent auto...</td>
      <td>B</td>
      <td>CHEMBL66879</td>
      <td>=</td>
      <td>nM</td>
      <td>3000000.0</td>
      <td>CHEMBL203</td>
      <td>Homo sapiens</td>
      <td>IC50</td>
    </tr>
  </tbody>
</table>
</div>



### Fetch compound data (molecule_chembl_id) from ChEMBL 


```python
molecule_chembl_id = list(bioactivities_df["molecule_chembl_id"])

compounds_provider = compounds_api.filter(molecule_chembl_id__in = molecule_chembl_id).only("molecule_chembl_id", "molecule_structures")
```


```python
compounds = list(tqdm(compounds_provider))
compounds_df = pd.DataFrame.from_records(compounds)
print(f"DataFrame shape: {compounds_df.shape}")
```

    100%|██████████| 6823/6823 [06:32<00:00, 17.37it/s]

    DataFrame shape: (6823, 2)


    



```python
compounds_df.head()
```




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
      <th>molecule_chembl_id</th>
      <th>molecule_structures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CHEMBL6246</td>
      <td>{'canonical_smiles': 'O=c1oc2c(O)c(O)cc3c(=O)o...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CHEMBL10</td>
      <td>{'canonical_smiles': 'C[S+]([O-])c1ccc(-c2nc(-...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CHEMBL6976</td>
      <td>{'canonical_smiles': 'COc1cc2c(cc1OC)Nc1ncn(C)...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CHEMBL7002</td>
      <td>{'canonical_smiles': 'CC1(COc2ccc(CC3SC(=O)NC3...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHEMBL414013</td>
      <td>{'canonical_smiles': 'COc1cc2c(cc1OC)Nc1ncnc(O...</td>
    </tr>
  </tbody>
</table>
</div>



### Preprocess and filter compound data
1. Remove entries with missing entries
2. Delete duplicate molecules (by molecule_chembl_id)
3. Get molecules with canonical SMILES


```python
compounds_df.dropna(axis=0, how="any", inplace=True)
compounds_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
```


```python
# Check molecule_structures column
compounds_df.iloc[0].molecule_structures
```




    {'canonical_smiles': 'O=c1oc2c(O)c(O)cc3c(=O)oc4c(O)c(O)cc1c4c23',
     'molfile': '\n     RDKit          2D\n\n 22 25  0  0  0  0  0  0  0  0999 V2000\n   -0.4750   -0.2417    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.5750    0.3583    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.4750   -1.4792    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.5750    1.6000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.5333    0.3583    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.6292   -0.2417    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.5750   -2.0875    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.4750    2.2083    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.5333    1.6000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.6292   -1.4792    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.6292    2.2083    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.5333   -2.0875    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -2.5833   -0.2417    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.6792    0.3583    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -2.5833   -1.4500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.6792    1.5625    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -2.5625    2.2208    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    2.6625   -2.1042    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    1.6292    3.4000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.5333   -3.2917    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   -3.6375   -2.0125    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    3.7375    2.1208    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n  2  1  1  0\n  3  1  1  0\n  4  2  2  0\n  5  1  2  0\n  6  2  1  0\n  7  3  1  0\n  8  9  1  0\n  9  5  1  0\n 10  7  1  0\n 11  4  1  0\n 12  3  2  0\n 13  5  1  0\n 14  6  2  0\n 15 13  2  0\n 16 14  1  0\n 17  9  2  0\n 18 10  2  0\n 19 11  1  0\n 20 12  1  0\n 21 15  1  0\n 22 16  1  0\n 12 15  1  0\n  4  8  1  0\n  6 10  1  0\n 11 16  2  0\nM  END\n> <chembl_id>\nCHEMBL6246\n\n> <chembl_pref_name>\nundefined',
     'standard_inchi': 'InChI=1S/C14H6O8/c15-5-1-3-7-8-4(14(20)22-11(7)9(5)17)2-6(16)10(18)12(8)21-13(3)19/h1-2,15-18H',
     'standard_inchi_key': 'AFSDNFLWKVMVRB-UHFFFAOYSA-N'}




```python
compounds_df.iloc[0].molecule_structures.keys()
```




    dict_keys(['canonical_smiles', 'molfile', 'standard_inchi', 'standard_inchi_key'])




```python
# Keep only canonical_smiles
canonical_smiles = []

for i, compounds in compounds_df.iterrows():
    try:
        canonical_smiles.append(compounds["molecule_structures"]["canonical_smiles"])
    except KeyError:
        canonical_smiles.append(None)

compounds_df["smiles"] = canonical_smiles
compounds_df.shape
```




    (6816, 3)




```python
compounds_df.head()
```




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
      <th>molecule_chembl_id</th>
      <th>molecule_structures</th>
      <th>smiles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CHEMBL6246</td>
      <td>{'canonical_smiles': 'O=c1oc2c(O)c(O)cc3c(=O)o...</td>
      <td>O=c1oc2c(O)c(O)cc3c(=O)oc4c(O)c(O)cc1c4c23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CHEMBL10</td>
      <td>{'canonical_smiles': 'C[S+]([O-])c1ccc(-c2nc(-...</td>
      <td>C[S+]([O-])c1ccc(-c2nc(-c3ccc(F)cc3)c(-c3ccncc...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CHEMBL6976</td>
      <td>{'canonical_smiles': 'COc1cc2c(cc1OC)Nc1ncn(C)...</td>
      <td>COc1cc2c(cc1OC)Nc1ncn(C)c(=O)c1C2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CHEMBL7002</td>
      <td>{'canonical_smiles': 'CC1(COc2ccc(CC3SC(=O)NC3...</td>
      <td>CC1(COc2ccc(CC3SC(=O)NC3=O)cc2)CCCCC1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHEMBL414013</td>
      <td>{'canonical_smiles': 'COc1cc2c(cc1OC)Nc1ncnc(O...</td>
      <td>COc1cc2c(cc1OC)Nc1ncnc(O)c1C2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Are there missing smiles?
compounds_df[compounds_df["smiles"].isnull()]
```




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
      <th>molecule_chembl_id</th>
      <th>molecule_structures</th>
      <th>smiles</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
compounds_df.drop("molecule_structures", axis=1, inplace=True)
compounds_df.head()
```




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
      <th>molecule_chembl_id</th>
      <th>smiles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CHEMBL6246</td>
      <td>O=c1oc2c(O)c(O)cc3c(=O)oc4c(O)c(O)cc1c4c23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CHEMBL10</td>
      <td>C[S+]([O-])c1ccc(-c2nc(-c3ccc(F)cc3)c(-c3ccncc...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CHEMBL6976</td>
      <td>COc1cc2c(cc1OC)Nc1ncn(C)c(=O)c1C2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CHEMBL7002</td>
      <td>CC1(COc2ccc(CC3SC(=O)NC3=O)cc2)CCCCC1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHEMBL414013</td>
      <td>COc1cc2c(cc1OC)Nc1ncnc(O)c1C2</td>
    </tr>
  </tbody>
</table>
</div>



### Merge dataframes


```python
bioactivities_df.columns
```




    Index(['activity_id', 'assay_chembl_id', 'assay_description', 'assay_type',
           'molecule_chembl_id', 'relation', 'units', 'IC50', 'target_chembl_id',
           'target_organism', 'type'],
          dtype='object')




```python
compounds_df.columns
```




    Index(['molecule_chembl_id', 'smiles'], dtype='object')




```python
print(f"Bioactivities filtered: {bioactivities_df.shape[0]}")
print(f"Compounds filtered: {compounds_df.shape[0]}")
```

    Bioactivities filtered: 6823
    Compounds filtered: 6816



```python
# Merge DataFrames
output_df = pd.merge(
    bioactivities_df[["molecule_chembl_id", "IC50", "units"]],
    compounds_df,
    on="molecule_chembl_id",
)

# Reset row indices
output_df.reset_index(drop=True, inplace=True)

print(f"Dataset with {output_df.shape[0]} entries.")
```

    Dataset with 6816 entries.



```python
output_df.dtypes
```




    molecule_chembl_id     object
    IC50                  float64
    units                  object
    smiles                 object
    dtype: object




```python
output_df.head(10)
```




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
      <th>molecule_chembl_id</th>
      <th>IC50</th>
      <th>units</th>
      <th>smiles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CHEMBL68920</td>
      <td>41.0</td>
      <td>nM</td>
      <td>Cc1cc(C)c(/C=C2\C(=O)Nc3ncnc(Nc4ccc(F)c(Cl)c4)...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CHEMBL69960</td>
      <td>170.0</td>
      <td>nM</td>
      <td>Cc1cc(C(=O)N2CCOCC2)[nH]c1/C=C1\C(=O)Nc2ncnc(N...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CHEMBL137635</td>
      <td>9300.0</td>
      <td>nM</td>
      <td>CN(c1ccccc1)c1ncnc2ccc(N/N=N/Cc3ccccn3)cc12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CHEMBL306988</td>
      <td>500000.0</td>
      <td>nM</td>
      <td>CC(=C(C#N)C#N)c1ccc(NC(=O)CCC(=O)O)cc1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHEMBL66879</td>
      <td>3000000.0</td>
      <td>nM</td>
      <td>O=C(O)/C=C/c1ccc(O)cc1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CHEMBL77085</td>
      <td>96000.0</td>
      <td>nM</td>
      <td>N#CC(C#N)=Cc1cc(O)ccc1[N+](=O)[O-]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CHEMBL443268</td>
      <td>5310.0</td>
      <td>nM</td>
      <td>Cc1cc(C(=O)NCCN2CCOCC2)[nH]c1/C=C1\C(=O)N(C)c2...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CHEMBL76979</td>
      <td>264000.0</td>
      <td>nM</td>
      <td>COc1cc(/C=C(\C#N)C(=O)O)cc(OC)c1O</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CHEMBL76589</td>
      <td>125.0</td>
      <td>nM</td>
      <td>N#CC(C#N)=C(N)/C(C#N)=C/c1ccc(O)cc1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CHEMBL76904</td>
      <td>35000.0</td>
      <td>nM</td>
      <td>N#CC(C#N)=Cc1ccc(O)c(O)c1</td>
    </tr>
  </tbody>
</table>
</div>



### Convert IC50 to pIC50


```python
def convert_ic50_to_pic50(IC50_value):
    pIC50_value = 9 - math.log10(IC50_value)
    return pIC50_value

# Apply conversion to each row of the compounds DataFrame
output_df["pIC50"] = output_df.apply(lambda x: convert_ic50_to_pic50(x.IC50), axis=1)
```


```python
output_df.head()
```




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
      <th>molecule_chembl_id</th>
      <th>IC50</th>
      <th>units</th>
      <th>smiles</th>
      <th>pIC50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CHEMBL68920</td>
      <td>41.0</td>
      <td>nM</td>
      <td>Cc1cc(C)c(/C=C2\C(=O)Nc3ncnc(Nc4ccc(F)c(Cl)c4)...</td>
      <td>7.387216</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CHEMBL69960</td>
      <td>170.0</td>
      <td>nM</td>
      <td>Cc1cc(C(=O)N2CCOCC2)[nH]c1/C=C1\C(=O)Nc2ncnc(N...</td>
      <td>6.769551</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CHEMBL137635</td>
      <td>9300.0</td>
      <td>nM</td>
      <td>CN(c1ccccc1)c1ncnc2ccc(N/N=N/Cc3ccccn3)cc12</td>
      <td>5.031517</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CHEMBL306988</td>
      <td>500000.0</td>
      <td>nM</td>
      <td>CC(=C(C#N)C#N)c1ccc(NC(=O)CCC(=O)O)cc1</td>
      <td>3.301030</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHEMBL66879</td>
      <td>3000000.0</td>
      <td>nM</td>
      <td>O=C(O)/C=C/c1ccc(O)cc1</td>
      <td>2.522879</td>
    </tr>
  </tbody>
</table>
</div>




```python
output_df.hist(column="pIC50")
```




    array([[<Axes: title={'center': 'pIC50'}>]], dtype=object)




    
![png](/chembl.png)
    


### add a column for RDKit molecule objects


```python
# Add molecule column
PandasTools.AddMoleculeColumnToFrame(output_df, smilesCol="smiles")
```


```python
# Sort molecules by pIC50
output_df.sort_values(by="pIC50", ascending=False, inplace=True)

# Reset index
output_df.reset_index(drop=True, inplace=True)

#output_df.drop("smiles", axis=1).head(10)
```


```python
output_df.head(10)
```




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
      <th>molecule_chembl_id</th>
      <th>IC50</th>
      <th>units</th>
      <th>smiles</th>
      <th>pIC50</th>
      <th>ROMol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CHEMBL63786</td>
      <td>0.003</td>
      <td>nM</td>
      <td>Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1</td>
      <td>11.522879</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x1769d6810&gt;</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CHEMBL53711</td>
      <td>0.006</td>
      <td>nM</td>
      <td>CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1</td>
      <td>11.221849</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x1769cdfc0&gt;</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CHEMBL35820</td>
      <td>0.006</td>
      <td>nM</td>
      <td>CCOc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OCC</td>
      <td>11.221849</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x1769c1150&gt;</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CHEMBL53753</td>
      <td>0.008</td>
      <td>nM</td>
      <td>CNc1cc2c(Nc3cccc(Br)c3)ncnc2cn1</td>
      <td>11.096910</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x1769c6dc0&gt;</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHEMBL66031</td>
      <td>0.008</td>
      <td>nM</td>
      <td>Brc1cccc(Nc2ncnc3cc4[nH]cnc4cc23)c1</td>
      <td>11.096910</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x1769c8350&gt;</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CHEMBL176582</td>
      <td>0.010</td>
      <td>nM</td>
      <td>Cn1cnc2cc3ncnc(Nc4cccc(Br)c4)c3cc21</td>
      <td>11.000000</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x1769d66c0&gt;</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CHEMBL174426</td>
      <td>0.025</td>
      <td>nM</td>
      <td>Cn1cnc2cc3c(Nc4cccc(Br)c4)ncnc3cc21</td>
      <td>10.602060</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x1769d6e30&gt;</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CHEMBL29197</td>
      <td>0.025</td>
      <td>nM</td>
      <td>COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC</td>
      <td>10.602060</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x1769c0dd0&gt;</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CHEMBL1243316</td>
      <td>0.030</td>
      <td>nM</td>
      <td>C#CCNC/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)c(C#N...</td>
      <td>10.522879</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x1769fa7a0&gt;</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CHEMBL363815</td>
      <td>0.037</td>
      <td>nM</td>
      <td>C=CC(=O)Nc1ccc2ncnc(Nc3cc(Cl)c(Cl)cc3F)c2c1</td>
      <td>10.431798</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x1769dcb30&gt;</td>
    </tr>
  </tbody>
</table>
</div>




```python
output_df.to_csv(DATA / "EGFR_compounds.csv")
```


```python

```
