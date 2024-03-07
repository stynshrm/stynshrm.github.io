---
author: Satyan Sharma
title: Molecular filtering - Visualising Lipinskis Rule of Five
date: 2022-09-20
math: true
tags: ["Cheminformatics"]
thumbnail: /th/th_5.png
---

 Here we filter candidate drug molecules by Lipinsik’s rule of five to keep only orally bioavailable compounds.
 
 ### ADME - absorption, distribution, metabolism, and excretion
 
ADMET is an acronym in pharmacology that stands for Absorption, Distribution, Metabolism, Excretion, and Toxicity. 
Pharmacokinetic studies and in vitro assays are commonly used to evaluate ADMET properties of new drug candidates. These are essential factors considered in drug development and pharmacokinetics:

* Absorption: Refers to the process by which a drug enters the bloodstream. It can occur through various routes such as oral ingestion, inhalation, injection, or topical application.
* Distribution: Once in the bloodstream, drugs distribute throughout the body. This involves how the drug is transported to different tissues and organs, including crossing barriers such as the blood-brain barrier.
* Metabolism: Metabolism involves the chemical transformation of the drug within the body, typically in the liver. Metabolism can alter the activity and toxicity of a drug and often results in the formation of metabolites, which may be pharmacologically active or inactive.
* Excretion: Excretion is the removal of drugs and their metabolites from the body, primarily through urine or feces via the kidneys and liver, respectively. Other routes of excretion include sweat, saliva, and breath.
* Toxicity: Toxicity refers to the potential of a drug to cause harm or adverse effects to the body. This can include acute toxicity (immediate adverse effects) or chronic toxicity (long-term adverse effects), and it is crucial to assess and mitigate potential risks during drug development.

### Lipinski's Rule of Five (Ro5)

Ro5 predicts the likelihood of a compound being orally bioavailable, which is closely related to its ADMET properties.

The rule was proposed by Christopher A. Lipinski in 1997 and states that:

1. Molecular weight should be less than 500 daltons.
2. The octanol-water partition coefficient (log P) should be less than 5.
3. There should be no more than 5 hydrogen bond donors (e.g., OH or NH groups).
4. There should be no more than 10 hydrogen bond acceptors (e.g., N or O atoms).

Compounds that violate more than one of these rules are less likely to be orally bioavailable, although there are exceptions and refinements to the rule.


### Check the Ro5 validity for some compunds that are potent against EGFR


```python
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, PandasTools
```

### Load Data


```python
molecules = pd.read_csv("./EGFR_compounds.csv", index_col=0)
```


```python
molecules.head()
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
      <td>CHEMBL63786</td>
      <td>0.003</td>
      <td>nM</td>
      <td>Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1</td>
      <td>11.522879</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CHEMBL35820</td>
      <td>0.006</td>
      <td>nM</td>
      <td>CCOc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OCC</td>
      <td>11.221849</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CHEMBL53711</td>
      <td>0.006</td>
      <td>nM</td>
      <td>CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1</td>
      <td>11.221849</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CHEMBL66031</td>
      <td>0.008</td>
      <td>nM</td>
      <td>Brc1cccc(Nc2ncnc3cc4[nH]cnc4cc23)c1</td>
      <td>11.096910</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHEMBL53753</td>
      <td>0.008</td>
      <td>nM</td>
      <td>CNc1cc2c(Nc3cccc(Br)c3)ncnc2cn1</td>
      <td>11.096910</td>
    </tr>
  </tbody>
</table>
</div>



### Function to calculate Ro5


```python
def calculate_ro5_properties(smiles):
    """
    Test if input molecule (SMILES) fulfills Lipinski's rule of five.

    Returns:    pandas.Series
        Molecular weight, number of hydrogen bond acceptors/donor and logP value
        and Lipinski's rule of five compliance for input molecule.
    """
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate Ro5-relevant chemical properties
    molecular_weight = Descriptors.ExactMolWt(molecule)
    n_hba = Descriptors.NumHAcceptors(molecule)
    n_hbd = Descriptors.NumHDonors(molecule)
    logp = Descriptors.MolLogP(molecule)
    # Check if Ro5 conditions fulfilled
    conditions = [molecular_weight <= 500, n_hba <= 10, n_hbd <= 5, logp <= 5]
    ro5_fulfilled = sum(conditions) >= 3
    # Return True if no more than one out of four conditions is violated
    return pd.Series(
        [molecular_weight, n_hba, n_hbd, logp, ro5_fulfilled],
        index=["molecular_weight", "n_hba", "n_hbd", "logp", "ro5_fulfilled"],
    )
```


```python
# apply to Smiles 
ro5_properties = molecules["smiles"].apply(calculate_ro5_properties)
ro5_properties.head()
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
      <th>molecular_weight</th>
      <th>n_hba</th>
      <th>n_hbd</th>
      <th>logp</th>
      <th>ro5_fulfilled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>349.021459</td>
      <td>3</td>
      <td>1</td>
      <td>5.2891</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>387.058239</td>
      <td>5</td>
      <td>1</td>
      <td>4.9333</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>343.043258</td>
      <td>5</td>
      <td>1</td>
      <td>3.5969</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>339.011957</td>
      <td>4</td>
      <td>2</td>
      <td>4.0122</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>329.027607</td>
      <td>5</td>
      <td>2</td>
      <td>3.5726</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



### Split dataframe on Molecules that fulfil or violate Ro5


```python
# Concatenate molecules with Ro5 data
molecules = pd.concat([molecules, ro5_properties], axis=1)
molecules.loc[:,['molecule_chembl_id', 'smiles', 'ro5_fulfilled']].head()
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
      <th>ro5_fulfilled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CHEMBL63786</td>
      <td>Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CHEMBL35820</td>
      <td>CCOc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OCC</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CHEMBL53711</td>
      <td>CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CHEMBL66031</td>
      <td>Brc1cccc(Nc2ncnc3cc4[nH]cnc4cc23)c1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHEMBL53753</td>
      <td>CNc1cc2c(Nc3cccc(Br)c3)ncnc2cn1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
molecules_ro5_fulfilled = molecules[molecules["ro5_fulfilled"]]
molecules_ro5_violated = molecules[~molecules["ro5_fulfilled"]]
```


```python
molecules_ro5_violated.head()
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
      <th>molecular_weight</th>
      <th>n_hba</th>
      <th>n_hbd</th>
      <th>logp</th>
      <th>ro5_fulfilled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>CHEMBL180022</td>
      <td>0.08</td>
      <td>nM</td>
      <td>CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4ccccn4)c(Cl)c3)c2...</td>
      <td>10.096910</td>
      <td>556.198966</td>
      <td>8</td>
      <td>2</td>
      <td>5.93248</td>
      <td>False</td>
    </tr>
    <tr>
      <th>32</th>
      <td>CHEMBL3753235</td>
      <td>0.20</td>
      <td>nM</td>
      <td>C=CC(=O)Nc1cccc(-n2c(=O)c(-c3ccccc3)cc3cnc(Nc4...</td>
      <td>9.698970</td>
      <td>587.264488</td>
      <td>9</td>
      <td>2</td>
      <td>5.07620</td>
      <td>False</td>
    </tr>
    <tr>
      <th>49</th>
      <td>CHEMBL2029438</td>
      <td>0.29</td>
      <td>nM</td>
      <td>C=CC(=O)Nc1cccc(N2C(=O)N(Cc3ccccc3)Cc3cnc(Nc4c...</td>
      <td>9.537602</td>
      <td>604.291037</td>
      <td>8</td>
      <td>2</td>
      <td>5.37900</td>
      <td>False</td>
    </tr>
    <tr>
      <th>55</th>
      <td>CHEMBL4078431</td>
      <td>0.30</td>
      <td>nM</td>
      <td>COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OCCCOP(N)(=O)N(...</td>
      <td>9.522879</td>
      <td>605.036108</td>
      <td>7</td>
      <td>2</td>
      <td>5.77640</td>
      <td>False</td>
    </tr>
    <tr>
      <th>56</th>
      <td>CHEMBL4127317</td>
      <td>0.30</td>
      <td>nM</td>
      <td>C=CC(=O)Nc1cccc(-n2c(=O)n(C(C)C)c(=O)c3cnc(Nc4...</td>
      <td>9.522879</td>
      <td>570.270302</td>
      <td>11</td>
      <td>2</td>
      <td>3.15180</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Calculate Stats for the df that obey or violate Ro5


```python
def calculate_mean_std(dataframe):
    """
    Calculate the mean and standard deviation of a dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Properties (columns) for a set of items (rows).

    Returns
    -------
    pd.DataFrame
        Mean and standard deviation (columns) for different properties (rows).
    """
    # Generate descriptive statistics for property columns
    stats = dataframe.describe()
    # Transpose DataFrame (statistical measures = columns)
    stats = stats.T
    # Select mean and standard deviation
    stats = stats[["mean", "std"]]
    return stats
```


```python
# calculate the statistic for the dataset of compounds
molecules_ro5_fulfilled_stats = calculate_mean_std(
    molecules_ro5_fulfilled[["molecular_weight", "n_hba", "n_hbd", "logp"]]
)
molecules_ro5_violated_stats = calculate_mean_std(
    molecules_ro5_violated[["molecular_weight", "n_hba", "n_hbd", "logp"]]
)
```


```python
molecules_ro5_violated_stats
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
      <th>mean</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>molecular_weight</th>
      <td>587.961963</td>
      <td>101.999229</td>
    </tr>
    <tr>
      <th>n_hba</th>
      <td>7.963558</td>
      <td>2.373576</td>
    </tr>
    <tr>
      <th>n_hbd</th>
      <td>2.301179</td>
      <td>1.719732</td>
    </tr>
    <tr>
      <th>logp</th>
      <td>5.973461</td>
      <td>1.430636</td>
    </tr>
  </tbody>
</table>
</div>




```python
molecules_ro5_fulfilled_stats
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
      <th>mean</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>molecular_weight</th>
      <td>414.439011</td>
      <td>87.985100</td>
    </tr>
    <tr>
      <th>n_hba</th>
      <td>5.996548</td>
      <td>1.875491</td>
    </tr>
    <tr>
      <th>n_hbd</th>
      <td>1.889968</td>
      <td>1.008368</td>
    </tr>
    <tr>
      <th>logp</th>
      <td>4.070568</td>
      <td>1.193034</td>
    </tr>
  </tbody>
</table>
</div>



###  Visualize Ro5 properties (radar plot)

#### Scale Y-values


```python
# Scale the values for easy plotting.
# The MWT has a threshold of 500, 
# whereas the number of HBAs and HBDs and the LogP have thresholds of only 10, 5, and 5.
# In order to visualize these different scales most simplistically, 
# we will scale all property values to a scaled threshold = 5:

# scaled property value = property value / property threshold * scaled property threshold

def _scale_by_thresholds(stats, thresholds, scaled_threshold):
    """
    Scale values for different properties that have each an individually defined threshold.

    Parameters
    ----------
    stats : pd.DataFrame
        Dataframe with "mean" and "std" (columns) for each physicochemical property (rows).
    thresholds : dict of str: int
        Thresholds defined for each property.
    scaled_threshold : int or float
        Scaled thresholds across all properties.

    Returns
    -------
    pd.DataFrame
        DataFrame with scaled means and standard deviations for each physiochemical property.
    """
    # Raise error if scaling keys and data_stats indicies are not matching
    for property_name in stats.index:
        if property_name not in thresholds.keys():
            raise KeyError(f"Add property '{property_name}' to scaling variable.")
    # Scale property data
    stats_scaled = stats.apply(lambda x: x / thresholds[x.name] * scaled_threshold, axis=1)
    return stats_scaled
```

#### Scale Xvalues to radians


```python
def _define_radial_axes_angles(n_axes):
    """Define angles (radians) for radial (x-)axes depending on the number of axes."""
    x_angles = [i / float(n_axes) * 2 * math.pi for i in range(n_axes)]
    x_angles += x_angles[:1]
    return x_angles
```


```python
thresholds = {"molecular_weight": 500, "n_hba": 10, "n_hbd": 5, "logp": 5}
scaled_threshold = 5
properties_labels = [
    "Molecular weight (Da) / 100",
    "# HBA / 2",
    "# HBD",
    "LogP",
]
y_max = 8
```

### Plotter function


```python
def plot_radar(
    y,
    thresholds,
    scaled_threshold,
    properties_labels,
    y_max=None,
    output_path=None,
):
    """
    Plot a radar chart based on the mean and standard deviation of a data set's properties.

    Parameters
    ----------
    y : pd.DataFrame
        Dataframe with "mean" and "std" (columns) for each physicochemical property (rows).
    thresholds : dict of str: int
        Thresholds defined for each property.
    scaled_threshold : int or float
        Scaled thresholds across all properties.
    properties_labels : list of str
        List of property names to be used as labels in the plot.
    y_max : None or int or float
        Set maximum y value. If None, let matplotlib decide.
    output_path : None or pathlib.Path
        If not None, save plot to file.
    """

    # Define radial x-axes angles -- uses our helper function!
    x = _define_radial_axes_angles(len(y))
    # Scale y-axis values with respect to a defined threshold -- uses our helper function!
    y = _scale_by_thresholds(y, thresholds, scaled_threshold)
    # Since our chart will be circular we append the first value of each property to the end
    y = pd.concat([y, y.head(1)])

    # Set figure and subplot axis
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    # Plot data
    ax.fill(x, [scaled_threshold] * len(x), "cornflowerblue", alpha=0.2)
    ax.plot(x, y["mean"], "b", lw=3, ls="-")
    ax.plot(x, y["mean"] + y["std"], "orange", lw=2, ls="--")
    ax.plot(x, y["mean"] - y["std"], "orange", lw=2, ls="-.")

    # From here on, we only do plot cosmetics
    # Set 0° to 12 o'clock
    ax.set_theta_offset(math.pi / 2)
    # Set clockwise rotation
    ax.set_theta_direction(-1)

    # Set y-labels next to 180° radius axis
    ax.set_rlabel_position(180)
    # Set number of radial axes' ticks and remove labels
    plt.xticks(x, [])
    # Get maximal y-ticks value
    if not y_max:
        y_max = int(ax.get_yticks()[-1])
    # Set axes limits
    plt.ylim(0, y_max)
    # Set number and labels of y axis ticks
    plt.yticks(
        range(1, y_max),
        ["5" if i == scaled_threshold else "" for i in range(1, y_max)],
        fontsize=16,
    )

    # Draw ytick labels to make sure they fit properly
    # Note that we use [:1] to exclude the last element which equals the first element (not needed here)
    for i, (angle, label) in enumerate(zip(x[:-1], properties_labels)):
        if angle == 0:
            ha = "center"
        elif 0 < angle < math.pi:
            ha = "left"
        elif angle == math.pi:
            ha = "center"
        else:
            ha = "right"
        ax.text(
            x=angle,
            y=y_max + 1,
            s=label,
            size=16,
            horizontalalignment=ha,
            verticalalignment="center",
        )

    # Add legend relative to top-left plot
    labels = ("rule of five area", "mean", "mean - std", "mean + std")
    ax.legend(labels, loc=(1.1, 0.7), labelspacing=0.3, fontsize=16)

    # Save plot - use bbox_inches to include text boxes
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight", transparent=True)

    plt.show()
```

### Compounds that obey Ro5 


```python
plot_radar(
    molecules_ro5_fulfilled_stats,
    thresholds,
    scaled_threshold,
    properties_labels,
    y_max,
)
# The light blue shaded area is region where a molecule’s physicochemical properties are compliant with the Ro5. 
# For Ro5 compliant mean is in area, while some have std out of shaded region.
```


    
![png](/ro5_27_0.png)
    


### Compounds that violate Ro5 


```python
plot_radar(
    molecules_ro5_violated_stats,
    thresholds,
    scaled_threshold,
    properties_labels,
    y_max,
)
```


    
![png](/ro5_29_0.png)
    


**Reference**
Talktorial - T002 by Volkamers Lab

