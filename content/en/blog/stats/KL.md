---
author: Satyan Sharma
title: KL Divergence and Cross Entropy
date: 2022-06-01
math: true
tags: ["Statistcs", "Data Science"]
---
## KL Divergence - Derivation by Intution

KL divergence, also known _relative entropy_, is a measure of how one probability distribution (say $Q$) differs from a reference probability distribution ($P$). It is a measure of the information lost when $Q(x)$ is used to approximate $P(x)$.
The log can be base-2 to give units in “bits,” or the natural logarithm base-e with units in “nats.” When the divergence is 0, both distributions are identical.

- The Least Square approach is optimal when the approximation error is normally distributed but can lead to wrong results when not. KL divergence is a more robust solution.

- Important to note that, $P$ and $Q$ can be totally different distributions with different parameters ($\theta$ and $\phi$). 

if random variable $x \in X$, and the probabilities are $P(x)$ and $Q(x)$, one could naively write the difference as:

$$ P_\theta(x) - Q_\phi(x)$$

Taking log, to get log likelihood ratio:

$$log P_\theta(x) - log Q_\phi(x) = log \left( \frac{P_\theta(x)}{Q_\phi(x)} \right)$$

$\frac{P_\theta(x)}{Q_\phi(x)}$ is the likelihood ratio. 

So, the Expected value of log likelihood ratio, which is the KL-Divergence, we weight it by the probability of reference distribution:

$$ 
D_{KL}[P_\theta || Q_\phi] = \sum_{x \in X} P_\theta(x)log \left( \frac{P_\theta(x)}{Q_\phi(x)} \right)
$$

And it is straightforward to implement in python
```python
def kl_divergence(p_probs, q_probs):
    """"KL (p || q)"""
    kl_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(kl_div)

#test
n = 15
uniform_data = np.full(n, 1.0 / n)

index = np.arange(n)
p = uniform_data.dot(index)/n
binom_data = binom.pmf(index, n, p)

print("KL- Divergence is: ", kl_divergence(binom_data, uniform_data))
```
```
    KL- Divergence is :  0.6311103254079738
```

### Forward KL
The difference between $P_\theta(x)$ and $Q_\phi(x)$ is weighted by $P_\theta(x)$. Generally, the Forward KL has a **mean seeking** behavior. It is used, indirectly, in cross entropy.


### Reverse KL
The difference between $P_\theta(x)$ and $Q_\phi(x)$ is weighted by  $Q_\phi(x)$. It has **mode seeking** behaviour and is used mainly in density estimation and variational inference.

Note:
- KL divergence is always positiove.
- $D_{KL}[P||Q] \neq D_{KL}[Q||P]$
- Hence, is not a metric.

## JS Divergence
Jensen-Shannon Divergence  is the weighted sum of KL Divergence - Forward and backward KL. It thus calculates a normalized score that is symmetrical. Thus, JS divergence is a metric and therefore is sometimes called Jenson-Shannon distance.

$$
D_{JS}(P || Q) = \frac{1}{2} D_{KL} \left(P || \frac{P+Q}{2} \right) + \frac{1}{2} D_{KL} \left(Q || \frac{P+Q}{2} \right)
$$

It is not that popular as it requires to first calculate KL divergence. JS distance is used to solve the issues of KL divergence in high dimensional spaces. JS divergence is better behaved in the sense that it doesn't become infinite when 
$P_\theta(x) = 0$.

## Cross-Entropy
When we expand Forward KL-Divergnece, we could write is as:
$$
D_{KL}[P || Q] = \sum P(x)log(P(x)) - \sum P(x)log(Q(x))
$$
with the entropy term
$$
H[P] = - \sum P(x)log(P(x))
$$
and cross entropy term
$$
H[P,Q] = - \sum P(x)log(Q(x))
$$

So, we can summarize the relationship as
$$
D_{KL}[P || Q] = H[P,Q] - H[P]
$$ 

In supervised classification problems, lets say there are $K$ classes. With $y_i$ as the ground truth and $\hat{y}_i$ be the predicted class probabilties. The cross-entropy is
$$
H[y_i,\hat{y}_i] = - \sum_{i=1}^{K}y_ilog\hat{y}_i
$$