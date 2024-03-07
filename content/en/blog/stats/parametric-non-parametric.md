---
author: Satyan Sharma
title: Parametric vs Nonparametric
date: 2021-07-14
math: true
tags: ["Statistcs", "Data Science"]
thumbnail: /th/th_para.png
---
## Parametric Methods:

- Rely on assumptions about the shape of the distribution.
- Assume that in some way the underlying population and its parameters
(means and variances) follow a normal distribution.

## Nonparmetric Methods:

- Rely on no or few assumptions about the shape of the underlying distribution.
- Most often used to analyse data which do not meet the distributional
requirements of parametric methods - skewed data.

### Examples:

| Analysis                                   |              Parametric             |               Nonparametric |
|:-|:-|:-|
| Comparing two independent groups           |                t-test               |      Wilcoxon rank-sum test | 
| Comparing measurements taken twice <br>(case control)         |            Paired t-test            |   Wilcoxon signed-rank test |
| Comparing three or more independent groups &nbsp; &nbsp;|                ANOVA               |         Kurskal-Wallis test |
| Degree of association between variables    | Pearson coefficient of &nbsp; &nbsp;<br>correlation | Spearman's rank correlation |

