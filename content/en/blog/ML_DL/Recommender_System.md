
---
author: Satyan Sharma
title: Recommender System with Collaborative Filtering
date: 2021-12-29
math: true
tags: ["Machine Learning"]
thumbnail: /th/th_reco.png
---

A recommender system is a type of information filtering system that predicts a user's preferences for items (such as movies, books, music, products, etc.) and suggests relevant items to the user. These systems are widely used in e-commerce platforms, streaming services, social media platforms, and many other applications where personalized recommendations can enhance user experience and engagement.

There are several types of recommender systems, including:

* Collaborative Filtering: This approach recommends items based on the preferences of users who have similar tastes. It doesn't require explicit knowledge about the items or users, but rather relies on the patterns and similarities in user behavior.
* Content-Based Filtering: Content-based filtering recommends items based on their attributes and features. It analyzes the characteristics of both the items and the user's preferences to make recommendations. For example, recommending movies based on their genre, actors, directors, etc., and matching them with the user's historical preferences.
* Hybrid Recommender Systems: Hybrid systems combine multiple recommendation techniques to provide more accurate and diverse recommendations. For instance, combining collaborative filtering and content-based filtering to leverage the strengths of both approaches.
* Knowledge-Based Recommender Systems: Knowledge-based systems recommend items based on explicit knowledge about user preferences, domain-specific rules, or constraints. These systems are often used in domains where there is rich domain knowledge available.
* Context-Aware Recommender Systems: Context-aware systems take into account contextual information such as time, location, and device used when making recommendations. For example, recommending nearby restaurants based on a user's current location and time of day.


## Collaborative Filtering

```python
import numpy as np
import pandas as pd
import sys
```


```python
movies = pd.read_csv("./ml-20m/movies.csv")
tags = pd.read_csv("./ml-20m/tags.csv")
ratings = pd.read_csv("./ml-20m/ratings.csv", nrows=16000000) 
```


```python
movies.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
tags.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>tag</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18</td>
      <td>4141</td>
      <td>Mark Waters</td>
      <td>1240597180</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65</td>
      <td>208</td>
      <td>dark hero</td>
      <td>1368150078</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65</td>
      <td>353</td>
      <td>dark hero</td>
      <td>1368150079</td>
    </tr>
    <tr>
      <th>3</th>
      <td>65</td>
      <td>521</td>
      <td>noir thriller</td>
      <td>1368149983</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65</td>
      <td>592</td>
      <td>dark hero</td>
      <td>1368150078</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3.5</td>
      <td>1112486027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29</td>
      <td>3.5</td>
      <td>1112484676</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>32</td>
      <td>3.5</td>
      <td>1112484819</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>3.5</td>
      <td>1112484727</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>3.5</td>
      <td>1112484580</td>
    </tr>
  </tbody>
</table>
</div>




```python
tags.drop(['timestamp'], axis=1, inplace=True)
ratings.drop(['timestamp'], axis=1, inplace=True)
```


```python
len(ratings.movieId.unique())
```




    25164




```python
movies['genres'] = movies['genres'].str.replace('|', ' ')
```


```python
# Restrict to users that have rated atleast 60 movies
ratings_df = ratings.groupby('userId').filter(lambda x: len(x) >= 60)
```


```python
ratings_df.shape
```




    (14248972, 3)




```python
ratings.shape
```




    (16000000, 3)




```python
len(ratings.userId.unique())
```




    110725




```python
len(ratings_df.userId.unique())
```




    60448




```python
# whihc all  movies are there in ratings_df, keep only those in movies 
ratings_movie_list = ratings_df['movieId'].unique().tolist()
movies = movies[movies['movieId'].isin(ratings_movie_list)]
movies.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure Animation Children Comedy Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure Children Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy Drama Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_df = pd.merge(movies, tags, on='movieId', how='left')
merged_df.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure Animation Children Comedy Fantasy</td>
      <td>1644.0</td>
      <td>Watched</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure Animation Children Comedy Fantasy</td>
      <td>1741.0</td>
      <td>computer animation</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure Animation Children Comedy Fantasy</td>
      <td>1741.0</td>
      <td>Disney animated feature</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure Animation Children Comedy Fantasy</td>
      <td>1741.0</td>
      <td>Pixar animation</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure Animation Children Comedy Fantasy</td>
      <td>1741.0</td>
      <td>TÃ©a Leoni does not star in this movie</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_df.fillna("", inplace=True)
merged_df = pd.DataFrame(merged_df.groupby('movieId')['tag'].apply(' '.join))
```


```python
merged_df.head()
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
      <th>tag</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Watched computer animation Disney animated fea...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>time travel adapted from:book board game child...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>old people that is actually funny sequel fever...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>chick flick revenge characters chick flick cha...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Diane Keaton family sequel Steve Martin weddin...</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_df = pd.merge(movies, merged_df, on='movieId', how='left')
```


```python
final_df.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure Animation Children Comedy Fantasy</td>
      <td>Watched computer animation Disney animated fea...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure Children Fantasy</td>
      <td>time travel adapted from:book board game child...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy Romance</td>
      <td>old people that is actually funny sequel fever...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy Drama Romance</td>
      <td>chick flick revenge characters chick flick cha...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>Diane Keaton family sequel Steve Martin weddin...</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_df['metadata'] = final_df[['tag', 'genres']].apply(' '.join, axis=1)
```


```python
final_df.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>tag</th>
      <th>metadata</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure Animation Children Comedy Fantasy</td>
      <td>Watched computer animation Disney animated fea...</td>
      <td>Watched computer animation Disney animated fea...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure Children Fantasy</td>
      <td>time travel adapted from:book board game child...</td>
      <td>time travel adapted from:book board game child...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy Romance</td>
      <td>old people that is actually funny sequel fever...</td>
      <td>old people that is actually funny sequel fever...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy Drama Romance</td>
      <td>chick flick revenge characters chick flick cha...</td>
      <td>chick flick revenge characters chick flick cha...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>Diane Keaton family sequel Steve Martin weddin...</td>
      <td>Diane Keaton family sequel Steve Martin weddin...</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_df.shape
```




    (25093, 5)




```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_mat = tfidf.fit_transform(final_df['metadata'])

tfidf_df = pd.DataFrame(tfidf_mat.toarray(), index=final_df.index.tolist()) 
tfidf_df.head() 
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>23519</th>
      <th>23520</th>
      <th>23521</th>
      <th>23522</th>
      <th>23523</th>
      <th>23524</th>
      <th>23525</th>
      <th>23526</th>
      <th>23527</th>
      <th>23528</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23529 columns</p>
</div>




```python
tfidf_df.shape  #each row is a movie
```




    (25093, 23529)




```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=200)
latent_matrix = svd.fit_transform(tfidf_df)
```


```python
latent_matrix_df1 = pd.DataFrame(latent_matrix[:,0:200], index = final_df['title'].tolist())
```


```python
import matplotlib.pyplot as plt
%matplotlib inline

explained_var = svd.explained_variance_ratio_.cumsum()
plt.plot(explained_var, '.-')
plt.xlabel("SVD components")
plt.ylabel("cumulative var explained(%)")
plt.show()
```


    
![png](reco_26_0.png)
    



```python
latent_matrix.shape
```




    (25093, 200)




```python
ratings_df.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>32</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>3.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings_df1 = pd.merge(movies[["movieId"]], ratings_df, on="movieId", how = "right")
```


```python
ratings_df1.head()
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
      <th>movieId</th>
      <th>userId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>8</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>11</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>13</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>14</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings_df2 = ratings_df1.pivot(index='movieId', columns='userId', values = 'rating').fillna(0)
ratings_df2.head()
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
      <th>userId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>5</th>
      <th>7</th>
      <th>8</th>
      <th>11</th>
      <th>13</th>
      <th>14</th>
      <th>16</th>
      <th>...</th>
      <th>110703</th>
      <th>110706</th>
      <th>110707</th>
      <th>110708</th>
      <th>110710</th>
      <th>110711</th>
      <th>110712</th>
      <th>110714</th>
      <th>110722</th>
      <th>110724</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60448 columns</p>
</div>




```python
ratings_df2.shape
```




    (25093, 60448)




```python
svd = TruncatedSVD(n_components=200)
latent_matrix = svd.fit_transform(ratings_df2)
latent_matrix_df2 = pd.DataFrame(latent_matrix, index = final_df['title'].tolist())
```


```python
latent_matrix_df2.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>190</th>
      <th>191</th>
      <th>192</th>
      <th>193</th>
      <th>194</th>
      <th>195</th>
      <th>196</th>
      <th>197</th>
      <th>198</th>
      <th>199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Toy Story (1995)</th>
      <td>503.065269</td>
      <td>-10.274285</td>
      <td>118.147003</td>
      <td>63.001323</td>
      <td>33.324811</td>
      <td>144.449895</td>
      <td>-58.370170</td>
      <td>58.839586</td>
      <td>-50.801258</td>
      <td>7.799550</td>
      <td>...</td>
      <td>-17.939855</td>
      <td>-6.105493</td>
      <td>13.302291</td>
      <td>10.795169</td>
      <td>-6.692900</td>
      <td>-19.925751</td>
      <td>8.026037</td>
      <td>-17.779819</td>
      <td>-3.730701</td>
      <td>-11.210057</td>
    </tr>
    <tr>
      <th>Jumanji (1995)</th>
      <td>226.951509</td>
      <td>-6.880712</td>
      <td>142.044769</td>
      <td>-38.659399</td>
      <td>-34.455545</td>
      <td>9.189622</td>
      <td>-59.369870</td>
      <td>43.351875</td>
      <td>19.546100</td>
      <td>-23.091564</td>
      <td>...</td>
      <td>-5.759097</td>
      <td>15.838553</td>
      <td>-4.651140</td>
      <td>-7.410019</td>
      <td>3.693425</td>
      <td>12.570774</td>
      <td>-3.953578</td>
      <td>24.891350</td>
      <td>10.353275</td>
      <td>3.837859</td>
    </tr>
    <tr>
      <th>Grumpier Old Men (1995)</th>
      <td>94.293293</td>
      <td>-45.533961</td>
      <td>61.644364</td>
      <td>-38.511678</td>
      <td>-28.622765</td>
      <td>-0.040053</td>
      <td>-3.608147</td>
      <td>1.412690</td>
      <td>-17.248191</td>
      <td>-29.525224</td>
      <td>...</td>
      <td>-0.670059</td>
      <td>1.097869</td>
      <td>-2.833690</td>
      <td>-2.598111</td>
      <td>-3.171674</td>
      <td>-3.332366</td>
      <td>4.101573</td>
      <td>-3.049846</td>
      <td>4.151136</td>
      <td>-5.025944</td>
    </tr>
    <tr>
      <th>Waiting to Exhale (1995)</th>
      <td>23.234759</td>
      <td>-25.256543</td>
      <td>18.811186</td>
      <td>-7.308903</td>
      <td>-25.304709</td>
      <td>0.539134</td>
      <td>-0.387492</td>
      <td>3.263833</td>
      <td>-4.996824</td>
      <td>2.722049</td>
      <td>...</td>
      <td>0.020090</td>
      <td>-1.532013</td>
      <td>-2.560055</td>
      <td>-0.007939</td>
      <td>-1.131231</td>
      <td>0.192202</td>
      <td>0.356880</td>
      <td>-3.906472</td>
      <td>-0.169543</td>
      <td>-1.063388</td>
    </tr>
    <tr>
      <th>Father of the Bride Part II (1995)</th>
      <td>80.873515</td>
      <td>-40.008944</td>
      <td>67.371184</td>
      <td>-34.720191</td>
      <td>-44.404007</td>
      <td>13.227772</td>
      <td>-10.653996</td>
      <td>8.077569</td>
      <td>-11.464432</td>
      <td>-18.275107</td>
      <td>...</td>
      <td>-0.786211</td>
      <td>-0.501860</td>
      <td>-4.795863</td>
      <td>-3.421932</td>
      <td>-3.465008</td>
      <td>3.484481</td>
      <td>0.667434</td>
      <td>-0.128325</td>
      <td>3.316472</td>
      <td>-3.240816</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 200 columns</p>
</div>




```python
explained_var = svd.explained_variance_ratio_.cumsum()
plt.plot(explained_var, '.-')
plt.xlabel("SVD components")
plt.ylabel("cumulative var explained(%)")
plt.show()
```


    
![png](reco_35_0.png)
    



```python
from sklearn.metrics.pairwise import cosine_similarity
```


```python
latent_matrix_df1.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>190</th>
      <th>191</th>
      <th>192</th>
      <th>193</th>
      <th>194</th>
      <th>195</th>
      <th>196</th>
      <th>197</th>
      <th>198</th>
      <th>199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Toy Story (1995)</th>
      <td>0.027829</td>
      <td>0.053107</td>
      <td>0.019021</td>
      <td>0.003907</td>
      <td>0.005143</td>
      <td>-0.027165</td>
      <td>0.117464</td>
      <td>-0.000203</td>
      <td>0.000751</td>
      <td>0.073886</td>
      <td>...</td>
      <td>-0.072060</td>
      <td>0.014005</td>
      <td>-0.030398</td>
      <td>0.085778</td>
      <td>0.211589</td>
      <td>-0.046935</td>
      <td>0.008914</td>
      <td>0.015430</td>
      <td>0.035303</td>
      <td>-0.034852</td>
    </tr>
    <tr>
      <th>Jumanji (1995)</th>
      <td>0.011114</td>
      <td>0.011237</td>
      <td>0.025765</td>
      <td>0.002484</td>
      <td>0.014320</td>
      <td>-0.001906</td>
      <td>0.070994</td>
      <td>-0.001397</td>
      <td>0.008939</td>
      <td>0.040755</td>
      <td>...</td>
      <td>0.017030</td>
      <td>0.026041</td>
      <td>0.014499</td>
      <td>0.021231</td>
      <td>-0.076609</td>
      <td>-0.022787</td>
      <td>0.054812</td>
      <td>-0.017474</td>
      <td>0.044199</td>
      <td>-0.005890</td>
    </tr>
    <tr>
      <th>Grumpier Old Men (1995)</th>
      <td>0.040006</td>
      <td>0.073972</td>
      <td>-0.004636</td>
      <td>-0.001118</td>
      <td>0.031234</td>
      <td>0.002447</td>
      <td>-0.003453</td>
      <td>0.000312</td>
      <td>-0.001469</td>
      <td>0.000665</td>
      <td>...</td>
      <td>0.021166</td>
      <td>-0.003256</td>
      <td>-0.014490</td>
      <td>0.009730</td>
      <td>-0.000665</td>
      <td>-0.009319</td>
      <td>0.000651</td>
      <td>0.005782</td>
      <td>-0.006482</td>
      <td>0.036551</td>
    </tr>
    <tr>
      <th>Waiting to Exhale (1995)</th>
      <td>0.138340</td>
      <td>0.076832</td>
      <td>-0.021021</td>
      <td>-0.002120</td>
      <td>0.100808</td>
      <td>0.013420</td>
      <td>-0.012406</td>
      <td>-0.003615</td>
      <td>-0.006283</td>
      <td>-0.002056</td>
      <td>...</td>
      <td>0.037234</td>
      <td>0.028600</td>
      <td>0.028611</td>
      <td>-0.039458</td>
      <td>-0.026834</td>
      <td>-0.080583</td>
      <td>0.047081</td>
      <td>-0.010145</td>
      <td>-0.005737</td>
      <td>0.013725</td>
    </tr>
    <tr>
      <th>Father of the Bride Part II (1995)</th>
      <td>0.040096</td>
      <td>0.084344</td>
      <td>0.000854</td>
      <td>0.000621</td>
      <td>-0.013870</td>
      <td>-0.000925</td>
      <td>0.013931</td>
      <td>0.006003</td>
      <td>0.006188</td>
      <td>0.011663</td>
      <td>...</td>
      <td>-0.029974</td>
      <td>-0.014503</td>
      <td>0.007450</td>
      <td>0.044511</td>
      <td>-0.000263</td>
      <td>0.021819</td>
      <td>0.004452</td>
      <td>-0.039057</td>
      <td>0.003836</td>
      <td>0.003095</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 200 columns</p>
</div>




```python
latent_matrix_df2.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>190</th>
      <th>191</th>
      <th>192</th>
      <th>193</th>
      <th>194</th>
      <th>195</th>
      <th>196</th>
      <th>197</th>
      <th>198</th>
      <th>199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Toy Story (1995)</th>
      <td>503.065269</td>
      <td>-10.274285</td>
      <td>118.147003</td>
      <td>63.001323</td>
      <td>33.324811</td>
      <td>144.449895</td>
      <td>-58.370170</td>
      <td>58.839586</td>
      <td>-50.801258</td>
      <td>7.799550</td>
      <td>...</td>
      <td>-17.939855</td>
      <td>-6.105493</td>
      <td>13.302291</td>
      <td>10.795169</td>
      <td>-6.692900</td>
      <td>-19.925751</td>
      <td>8.026037</td>
      <td>-17.779819</td>
      <td>-3.730701</td>
      <td>-11.210057</td>
    </tr>
    <tr>
      <th>Jumanji (1995)</th>
      <td>226.951509</td>
      <td>-6.880712</td>
      <td>142.044769</td>
      <td>-38.659399</td>
      <td>-34.455545</td>
      <td>9.189622</td>
      <td>-59.369870</td>
      <td>43.351875</td>
      <td>19.546100</td>
      <td>-23.091564</td>
      <td>...</td>
      <td>-5.759097</td>
      <td>15.838553</td>
      <td>-4.651140</td>
      <td>-7.410019</td>
      <td>3.693425</td>
      <td>12.570774</td>
      <td>-3.953578</td>
      <td>24.891350</td>
      <td>10.353275</td>
      <td>3.837859</td>
    </tr>
    <tr>
      <th>Grumpier Old Men (1995)</th>
      <td>94.293293</td>
      <td>-45.533961</td>
      <td>61.644364</td>
      <td>-38.511678</td>
      <td>-28.622765</td>
      <td>-0.040053</td>
      <td>-3.608147</td>
      <td>1.412690</td>
      <td>-17.248191</td>
      <td>-29.525224</td>
      <td>...</td>
      <td>-0.670059</td>
      <td>1.097869</td>
      <td>-2.833690</td>
      <td>-2.598111</td>
      <td>-3.171674</td>
      <td>-3.332366</td>
      <td>4.101573</td>
      <td>-3.049846</td>
      <td>4.151136</td>
      <td>-5.025944</td>
    </tr>
    <tr>
      <th>Waiting to Exhale (1995)</th>
      <td>23.234759</td>
      <td>-25.256543</td>
      <td>18.811186</td>
      <td>-7.308903</td>
      <td>-25.304709</td>
      <td>0.539134</td>
      <td>-0.387492</td>
      <td>3.263833</td>
      <td>-4.996824</td>
      <td>2.722049</td>
      <td>...</td>
      <td>0.020090</td>
      <td>-1.532013</td>
      <td>-2.560055</td>
      <td>-0.007939</td>
      <td>-1.131231</td>
      <td>0.192202</td>
      <td>0.356880</td>
      <td>-3.906472</td>
      <td>-0.169543</td>
      <td>-1.063388</td>
    </tr>
    <tr>
      <th>Father of the Bride Part II (1995)</th>
      <td>80.873515</td>
      <td>-40.008944</td>
      <td>67.371184</td>
      <td>-34.720191</td>
      <td>-44.404007</td>
      <td>13.227772</td>
      <td>-10.653996</td>
      <td>8.077569</td>
      <td>-11.464432</td>
      <td>-18.275107</td>
      <td>...</td>
      <td>-0.786211</td>
      <td>-0.501860</td>
      <td>-4.795863</td>
      <td>-3.421932</td>
      <td>-3.465008</td>
      <td>3.484481</td>
      <td>0.667434</td>
      <td>-0.128325</td>
      <td>3.316472</td>
      <td>-3.240816</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 200 columns</p>
</div>




```python
# Check similaruty of movie with content and collaboratice matricess
movie_content_vector = np.array(latent_matrix_df1.loc['Toy Story (1995)']).reshape(1,-1)
movie_collab_vector = np.array(latent_matrix_df2.loc['Toy Story (1995)']).reshape(1,-1)
```


```python
score_1 = cosine_similarity(latent_matrix_df1, movie_content_vector).reshape(-1)
score_2 = cosine_similarity(latent_matrix_df2, movie_collab_vector).reshape(-1)

#average score 
av_score = (score_1 + score_2)/2.0
```


```python
movie_sim = {'content':score_1, 'collab': score_2, 'hybrid':av_score}
simil_df = pd.DataFrame(movie_sim, index=latent_matrix_df1.index)
```


```python
simil_df.head()
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
      <th>content</th>
      <th>collab</th>
      <th>hybrid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Toy Story (1995)</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Jumanji (1995)</th>
      <td>0.073692</td>
      <td>0.570838</td>
      <td>0.322265</td>
    </tr>
    <tr>
      <th>Grumpier Old Men (1995)</th>
      <td>0.058560</td>
      <td>0.460268</td>
      <td>0.259414</td>
    </tr>
    <tr>
      <th>Waiting to Exhale (1995)</th>
      <td>0.029523</td>
      <td>0.277400</td>
      <td>0.153462</td>
    </tr>
    <tr>
      <th>Father of the Bride Part II (1995)</th>
      <td>0.052702</td>
      <td>0.450922</td>
      <td>0.251812</td>
    </tr>
  </tbody>
</table>
</div>




```python
simil_df.sort_values('content', ascending=False) # based on movie content toystory is similar to toy story2
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
      <th>content</th>
      <th>collab</th>
      <th>hybrid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Toy Story (1995)</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Toy Story 2 (1999)</th>
      <td>0.960975</td>
      <td>0.765740</td>
      <td>0.863358</td>
    </tr>
    <tr>
      <th>Bug's Life, A (1998)</th>
      <td>0.905825</td>
      <td>0.654965</td>
      <td>0.780395</td>
    </tr>
    <tr>
      <th>Ratatouille (2007)</th>
      <td>0.898999</td>
      <td>0.429254</td>
      <td>0.664126</td>
    </tr>
    <tr>
      <th>Monsters, Inc. (2001)</th>
      <td>0.883235</td>
      <td>0.621118</td>
      <td>0.752176</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Life, Above All (2010)</th>
      <td>-0.113142</td>
      <td>0.073144</td>
      <td>-0.019999</td>
    </tr>
    <tr>
      <th>Nell (1994)</th>
      <td>-0.117324</td>
      <td>0.341333</td>
      <td>0.112004</td>
    </tr>
    <tr>
      <th>Stevie (2002)</th>
      <td>-0.117623</td>
      <td>0.174810</td>
      <td>0.028593</td>
    </tr>
    <tr>
      <th>Newsfront (1978)</th>
      <td>-0.123699</td>
      <td>0.029111</td>
      <td>-0.047294</td>
    </tr>
    <tr>
      <th>Samson and Delilah (2009)</th>
      <td>-0.124658</td>
      <td>0.108386</td>
      <td>-0.008136</td>
    </tr>
  </tbody>
</table>
<p>25093 rows × 3 columns</p>
</div>




```python
simil_df.sort_values('collab', ascending=False) # based on user who likes Toystory would like Toystory2 
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
      <th>content</th>
      <th>collab</th>
      <th>hybrid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Toy Story (1995)</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Toy Story 2 (1999)</th>
      <td>0.960975</td>
      <td>0.765740</td>
      <td>0.863358</td>
    </tr>
    <tr>
      <th>Aladdin (1992)</th>
      <td>0.413372</td>
      <td>0.686649</td>
      <td>0.550010</td>
    </tr>
    <tr>
      <th>Lion King, The (1994)</th>
      <td>0.456150</td>
      <td>0.675752</td>
      <td>0.565951</td>
    </tr>
    <tr>
      <th>Star Wars: Episode IV - A New Hope (1977)</th>
      <td>0.020970</td>
      <td>0.675528</td>
      <td>0.348249</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Koumiko Mystery, The (Mystère Koumiko, Le) (1967)</th>
      <td>0.031572</td>
      <td>-0.021734</td>
      <td>0.004919</td>
    </tr>
    <tr>
      <th>Steep (2007)</th>
      <td>-0.001602</td>
      <td>-0.021942</td>
      <td>-0.011772</td>
    </tr>
    <tr>
      <th>Cheers for Miss Bishop (1941)</th>
      <td>-0.000040</td>
      <td>-0.023719</td>
      <td>-0.011880</td>
    </tr>
    <tr>
      <th>Stranger, The (Agantuk) (Visitor, The) (1991)</th>
      <td>0.009928</td>
      <td>-0.025073</td>
      <td>-0.007573</td>
    </tr>
    <tr>
      <th>Happy End (1967)</th>
      <td>0.047334</td>
      <td>-0.029115</td>
      <td>0.009110</td>
    </tr>
  </tbody>
</table>
<p>25093 rows × 3 columns</p>
</div>




```python
simil_df.sort_values('hybrid', ascending=False)
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
      <th>content</th>
      <th>collab</th>
      <th>hybrid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Toy Story (1995)</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Toy Story 2 (1999)</th>
      <td>0.960975</td>
      <td>0.765740</td>
      <td>0.863358</td>
    </tr>
    <tr>
      <th>Bug's Life, A (1998)</th>
      <td>0.905825</td>
      <td>0.654965</td>
      <td>0.780395</td>
    </tr>
    <tr>
      <th>Monsters, Inc. (2001)</th>
      <td>0.883235</td>
      <td>0.621118</td>
      <td>0.752176</td>
    </tr>
    <tr>
      <th>Finding Nemo (2003)</th>
      <td>0.869589</td>
      <td>0.603694</td>
      <td>0.736641</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Love unto Death (L'amour a mort) (1984)</th>
      <td>-0.096918</td>
      <td>0.016217</td>
      <td>-0.040350</td>
    </tr>
    <tr>
      <th>Cane Toads: The Conquest (2010)</th>
      <td>-0.092350</td>
      <td>0.002899</td>
      <td>-0.044726</td>
    </tr>
    <tr>
      <th>Amish Murder, An (2013)</th>
      <td>-0.090066</td>
      <td>-0.000880</td>
      <td>-0.045473</td>
    </tr>
    <tr>
      <th>Newsfront (1978)</th>
      <td>-0.123699</td>
      <td>0.029111</td>
      <td>-0.047294</td>
    </tr>
    <tr>
      <th>You Ain't Seen Nothin' Yet (Vous n'avez encore rien vu) (2012)</th>
      <td>-0.096918</td>
      <td>-0.021501</td>
      <td>-0.059209</td>
    </tr>
  </tbody>
</table>
<p>25093 rows × 3 columns</p>
</div>




```python

```
