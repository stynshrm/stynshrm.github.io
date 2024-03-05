---
author: Satyan Sharma
title: ML model Interpretation with Molecule Similarity Maps
date: 2022-09-20
math: true
tags: ["Cheminformatics"]
---

RDKit's similarity map functionality (https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-5-43).
Can be used to interpret a machine learning model which is very important for designing new molecules.

## Similarity Maps - Intro


```python
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
```

#### Similarity map functionality using the Morgan fingerprint.

for similarity between atorvastatin (Lipitor) and rosuvastatin (Crestor)


```python
atorvastatin = Chem.MolFromSmiles('O=C(O)C[C@H](O)C[C@H](O)CCn2c(c(c(c2c1ccc(F)cc1)c3ccccc3)C(=O)Nc4ccccc4)C(C)C')
rosuvastatin = Chem.MolFromSmiles('OC(=O)C[C@H](O)C[C@H](O)\C=C\c1c(C(C)C)nc(N(C)S(=O)(=O)C)nc1c2ccc(F)cc2')
Draw.MolsToGridImage((atorvastatin,rosuvastatin))
```




    
![png](/simMaps_3_0.png)
    




```python
import io
from PIL import Image
import numpy as np
```


```python
def show_png(data):
    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img
```


```python
d = Draw.MolDraw2DCairo(400, 400)
_, maxWeight = SimilarityMaps.GetSimilarityMapForFingerprint(atorvastatin, rosuvastatin, 
                                        lambda m, i: SimilarityMaps.GetMorganFingerprint(m, i, radius=2, fpType='bv'), 
                                        draw2d=d)
d.FinishDrawing()
show_png(d.GetDrawingText())
```




    
![png](/simMaps_6_0.png)
    



#### Similarity map functionality using the count-based fingerprints.


```python
d = Draw.MolDraw2DCairo(400, 400)
_, maxWeight = SimilarityMaps.GetSimilarityMapForFingerprint(atorvastatin, rosuvastatin, 
                                        lambda m, i: SimilarityMaps.GetMorganFingerprint(m, i, radius=2, fpType='count'), 
                                        draw2d=d)
d.FinishDrawing()
show_png(d.GetDrawingText())
```




    
![png](/simMaps_8_0.png)
    



### Viewing partial charges

the partial charges calculated with extended Hueckel theory (eHT) using Mulliken analysis


```python
from rdkit.Chem import rdEHTTools
from rdkit.Chem import rdDistGeom
mh = Chem.AddHs(atorvastatin)
rdDistGeom.EmbedMolecule(mh)
_,res = rdEHTTools.RunMol(mh)
static_chgs = res.GetAtomicCharges()[:atorvastatin.GetNumAtoms()]
d = Draw.MolDraw2DCairo(400, 400)
SimilarityMaps.GetSimilarityMapFromWeights(atorvastatin,list(static_chgs),draw2d=d)
d.FinishDrawing()
show_png(d.GetDrawingText())
```

    !!! Warning !!! Distance between atoms 46 and 6 (0.979791 A) is suspicious.
    !!! Warning !!! Distance between atoms 50 and 9 (0.996861 A) is suspicious.





    
![png](/simMaps_10_1.png)
    



Start by generating 10 diverse conformers, calculating the charges for each, and plotting the average:


```python
mh = Chem.AddHs(atorvastatin)
ps = rdDistGeom.ETKDGv2()
ps.pruneRmsThresh = 0.5
ps.randomSeed = 0xf00d
rdDistGeom.EmbedMultipleConfs(mh,10,ps)
print(f'Found {mh.GetNumConformers()} conformers')
chgs = []
for conf in mh.GetConformers():
    _,res = rdEHTTools.RunMol(mh,confId=conf.GetId())
    chgs.append(res.GetAtomicCharges()[:atorvastatin.GetNumAtoms()])
chgs = np.array(chgs)
mean_chgs = np.mean(chgs,axis=0)
std_chgs = np.std(chgs,axis=0)
d = Draw.MolDraw2DCairo(400, 400)
SimilarityMaps.GetSimilarityMapFromWeights(atorvastatin,list(mean_chgs),draw2d=d)
d.FinishDrawing()
show_png(d.GetDrawingText())
```

    Found 10 conformers


    !!! Warning !!! Distance between atoms 46 and 6 (0.970024 A) is suspicious.
    !!! Warning !!! Distance between atoms 50 and 9 (0.981406 A) is suspicious.
    !!! Warning !!! Distance between atoms 46 and 6 (0.981542 A) is suspicious.
    !!! Warning !!! Distance between atoms 50 and 9 (0.979128 A) is suspicious.
    !!! Warning !!! Distance between atoms 46 and 6 (0.990523 A) is suspicious.
    !!! Warning !!! Distance between atoms 46 and 6 (0.988965 A) is suspicious.
    !!! Warning !!! Distance between atoms 46 and 6 (0.996235 A) is suspicious.
    !!! Warning !!! Distance between atoms 46 and 6 (0.971296 A) is suspicious.
    !!! Warning !!! Distance between atoms 50 and 9 (0.991920 A) is suspicious.
    !!! Warning !!! Distance between atoms 46 and 6 (0.977223 A) is suspicious.
    !!! Warning !!! Distance between atoms 46 and 6 (0.984263 A) is suspicious.
    !!! Warning !!! Distance between atoms 46 and 6 (0.987732 A) is suspicious.
    !!! Warning !!! Distance between atoms 50 and 9 (0.989019 A) is suspicious.
    !!! Warning !!! Distance between atoms 46 and 6 (0.970647 A) is suspicious.
    !!! Warning !!! Distance between atoms 50 and 9 (0.989967 A) is suspicious.





    
![png](/simMaps_12_2.png)
    



## Random Forest Classifier


```python
%matplotlib inline
import matplotlib.pyplot as plt
import os
from functools import partial
from rdkit import Chem
from rdkit import rdBase
from rdkit import RDPaths
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem.Draw import SimilarityMaps, IPythonConsole
from IPython.display import SVG
import io
from PIL import Image
import rdkit
import numpy as np
from sklearn.ensemble import RandomForestClassifier
```


```python
# Load solubility data which is provided by rdkit data.
trainpath = os.path.join('./solubility.train.sdf')
testpath =  os.path.join('./solubility.test.sdf')
```


```python
train_mols = [m for m in Chem.SDMolSupplier(trainpath) if m is not None]
test_mols = [m for m in Chem.SDMolSupplier(testpath) if m is not None]
val_dict = {'(A) low':0,
           '(B) medium':1,
           '(C) high':2}
```


```python
def mol2arr(mol, fpfunc):
    fp = fpfunc(mol)
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
# https://rdkit.blogspot.com/2020/01/similarity-maps-with-new-drawing-code.html
def show_png(data):
    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img
fpfunc = partial(SimilarityMaps.GetMorganFingerprint, nBits=1024, radius=2)
```


```python
# calculate fingerprint and get solubility class.
trainX = [fpfunc(m) for m in train_mols]
trainY = [val_dict[m.GetProp('SOL_classification')] for m in train_mols]

testX = [fpfunc(m) for m in test_mols]
testY = [val_dict[m.GetProp('SOL_classification')] for m in test_mols]
rfc = RandomForestClassifier(random_state=794)
rfc.fit(trainX, trainY)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=794)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=794)</pre></div></div></div></div></div>




```python
# https://www.rdkit.org/docs/Cookbook.html
# to get probability of High solubility
def getProba(fp, predctionFunction):
    return predctionFunction((fp,))[0][2]
def drawmol(mols, idx):
    d = Draw.MolDraw2DCairo(1,1)
    _, maxWeight = SimilarityMaps.GetSimilarityMapForModel(mols[idx],fpfunc, 
                                                           lambda x: getProba(x, rfc.predict_proba),
                                                           colorMap='coolwarm',
                                                          size=(200,200),
                                                          step=0.001,
                                                          alpha=0.2)
    d.FinishDrawing()
    show_png(d.GetDrawingText())
    print(mols[idx].GetProp('SOL_classification'))

```


```python
# Lets view 
high_test_mols = [mol for mol in test_mols if mol.GetProp('SOL_classification') == '(C) high']
low_test_mols = [mol for mol in test_mols if mol.GetProp('SOL_classification') == '(A) low']

```


```python
drawmol(high_test_mols, 1)
```

    (C) high



    
![png](/simMaps_21_1.png)
    



```python
drawmol(low_test_mols, 5)
```

    (A) low



    
![png](/simMaps_22_1.png)
    


### Reference:

http://rdkit.blogspot.com/2020/01/similarity-maps-with-new-drawing-code.html


