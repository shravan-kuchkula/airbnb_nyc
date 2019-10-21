
## Dimensionality Reduction using t-SNE
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets. t-SNE minimizes the divergence between two distributions: a distribution that measures pairwise similarities of the input objects and a distribution that measures pairwise similarities of the corresponding low-dimensional points in the embedding.

In this way, t-SNE maps the multi-dimensional data to a lower dimensional space and attempts to find patterns in the data by identifying observed clusters based on similarity of data points with multiple features. However, after this process, the input features are no longer identifiable, and you cannot make any inference based only on the output of t-SNE. Hence it is mainly **a data exploration and visualization technique.**

We will start by taking the 50 principal components that we created in the earlier post [New York City Airbnb PCA](https://shravan-kuchkula.github.io/nyc-airbnb-pca/), and apply the t-SNE with 3 components which we can use to create a 3D scatter plot of the data points.

## Get the data
The principal components created earlier are stored in `airbnb_final.csv` file, which we will load in to begin our analysis.


```python
import pandas as pd
data = pd.read_csv('airbnb_final.csv')
data.head()
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
      <th>price_category</th>
      <th>name</th>
      <th>id</th>
      <th>price</th>
      <th>adjusted_price</th>
      <th>minimum_nights</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>neighbourhood_group_cleansed</th>
      <th>neighbourhood_cleansed</th>
      <th>...</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
      <th>49</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>high</td>
      <td>Skylit Midtown Castle</td>
      <td>2595</td>
      <td>225.0</td>
      <td>225.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>...</td>
      <td>0.576904</td>
      <td>0.466956</td>
      <td>0.331311</td>
      <td>0.261779</td>
      <td>-0.330193</td>
      <td>1.620287</td>
      <td>0.867739</td>
      <td>-0.798060</td>
      <td>-0.576860</td>
      <td>-0.254925</td>
    </tr>
    <tr>
      <th>1</th>
      <td>medium</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>3647</td>
      <td>150.0</td>
      <td>50.000000</td>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>...</td>
      <td>-0.252328</td>
      <td>0.226731</td>
      <td>0.269839</td>
      <td>-0.211928</td>
      <td>0.147831</td>
      <td>1.354929</td>
      <td>0.801862</td>
      <td>-0.292820</td>
      <td>-0.804985</td>
      <td>-0.202175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>5022</td>
      <td>80.0</td>
      <td>8.000000</td>
      <td>10</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>...</td>
      <td>0.193168</td>
      <td>0.068044</td>
      <td>0.015844</td>
      <td>0.197295</td>
      <td>-0.167786</td>
      <td>1.117572</td>
      <td>0.749340</td>
      <td>0.109282</td>
      <td>-0.870940</td>
      <td>-0.278948</td>
    </tr>
    <tr>
      <th>3</th>
      <td>medium</td>
      <td>Large Cozy 1 BR Apartment In Midtown East</td>
      <td>5099</td>
      <td>200.0</td>
      <td>66.666667</td>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>Murray Hill</td>
      <td>...</td>
      <td>0.416250</td>
      <td>-0.057388</td>
      <td>0.073780</td>
      <td>0.143102</td>
      <td>0.198163</td>
      <td>1.440193</td>
      <td>0.641507</td>
      <td>-0.021390</td>
      <td>-0.912154</td>
      <td>-0.285108</td>
    </tr>
    <tr>
      <th>4</th>
      <td>low</td>
      <td>BlissArtsSpace!</td>
      <td>5121</td>
      <td>60.0</td>
      <td>1.333333</td>
      <td>45</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Brooklyn</td>
      <td>Bedford-Stuyvesant</td>
      <td>...</td>
      <td>-0.214280</td>
      <td>-0.095810</td>
      <td>0.224063</td>
      <td>0.083689</td>
      <td>-0.192449</td>
      <td>0.410582</td>
      <td>0.017357</td>
      <td>0.246550</td>
      <td>0.796558</td>
      <td>0.307420</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60 columns</p>
</div>




```python
# rename the PC columns
pc_col_names = ["pc_" + item for item in list(data.columns[10:])]
other_col_names = list(data.columns[:10])
data.columns = other_col_names + pc_col_names
```

## Apply t-SNE


```python
from sklearn.manifold import TSNE

# extract the 50 principal components
A = data.iloc[:,10:].values
type(A)
```




    numpy.ndarray




```python
# Dimension reduction with t-SNE
model = TSNE(n_components=3, learning_rate=100, random_state=42)
tsne_features = model.fit_transform(A)

# Construct a t-SNE dataframe
tsne_df = pd.DataFrame({'TSNE1': tsne_features[:,0], 
              'TSNE2': tsne_features[:,1],
              'TSNE3': tsne_features[:,2]
             })
```


```python
tsne_df.shape
```




    (45605, 3)



The `tsne_df` dataframe contains the  3 tsne features for all 45,605 airbnb listings. We can now use this data along with other columns of the airbnb dataset to build a 3D scatterplot.


```python
data_tsne = data[other_col_names]
tsne_final= pd.concat([tsne_df, data_tsne], axis=1)

# save this as tsne takes extremely long to run
tsne_final.to_csv('tsne_final.csv', index=False)

tsne_final.head()
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
      <th>TSNE1</th>
      <th>TSNE2</th>
      <th>TSNE3</th>
      <th>price_category</th>
      <th>name</th>
      <th>id</th>
      <th>price</th>
      <th>adjusted_price</th>
      <th>minimum_nights</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>neighbourhood_group_cleansed</th>
      <th>neighbourhood_cleansed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.618355</td>
      <td>18.307888</td>
      <td>4.037642</td>
      <td>high</td>
      <td>Skylit Midtown Castle</td>
      <td>2595</td>
      <td>225.0</td>
      <td>225.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>Midtown</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-20.100536</td>
      <td>8.020902</td>
      <td>-1.968155</td>
      <td>medium</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>3647</td>
      <td>150.0</td>
      <td>50.000000</td>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>Harlem</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-9.849981</td>
      <td>16.748266</td>
      <td>2.556231</td>
      <td>low</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>5022</td>
      <td>80.0</td>
      <td>8.000000</td>
      <td>10</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.867686</td>
      <td>1.036031</td>
      <td>15.170166</td>
      <td>medium</td>
      <td>Large Cozy 1 BR Apartment In Midtown East</td>
      <td>5099</td>
      <td>200.0</td>
      <td>66.666667</td>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>Murray Hill</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-8.865001</td>
      <td>-15.556909</td>
      <td>-7.953006</td>
      <td>low</td>
      <td>BlissArtsSpace!</td>
      <td>5121</td>
      <td>60.0</td>
      <td>1.333333</td>
      <td>45</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Brooklyn</td>
      <td>Bedford-Stuyvesant</td>
    </tr>
  </tbody>
</table>
</div>



## Plotly express to visualize the data


```python
import pandas as pd
import plotly.express as px
tsne_final = pd.read_csv('../data/raw/tsne_final.csv')
```


```python
plotly_data = tsne_final[(tsne_final.neighbourhood_cleansed == 'Chelsea') & 
                         (tsne_final.minimum_nights <= 3) &
                         (tsne_final.bedrooms == 0)
                        ]
```


```python
plotly_data.shape
```




    (107, 13)




```python
plotly_data.head()
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
      <th>TSNE1</th>
      <th>TSNE2</th>
      <th>TSNE3</th>
      <th>price_category</th>
      <th>name</th>
      <th>id</th>
      <th>price</th>
      <th>adjusted_price</th>
      <th>minimum_nights</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>neighbourhood_group_cleansed</th>
      <th>neighbourhood_cleansed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>173</th>
      <td>-8.011493</td>
      <td>-10.242352</td>
      <td>-9.682402</td>
      <td>low</td>
      <td>Chelsea Studio sublet 1 - 2 months</td>
      <td>47370</td>
      <td>125.0</td>
      <td>41.666667</td>
      <td>3</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>Chelsea</td>
    </tr>
    <tr>
      <th>1082</th>
      <td>-11.707626</td>
      <td>13.309287</td>
      <td>1.150806</td>
      <td>high</td>
      <td>Beautiful Brand New Chelsea Studio</td>
      <td>515392</td>
      <td>200.0</td>
      <td>200.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>Chelsea</td>
    </tr>
    <tr>
      <th>2758</th>
      <td>-11.331677</td>
      <td>16.121357</td>
      <td>1.707191</td>
      <td>medium</td>
      <td>Large Comfortable Studio in Chelsea</td>
      <td>1820858</td>
      <td>161.0</td>
      <td>80.500000</td>
      <td>2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>Chelsea</td>
    </tr>
    <tr>
      <th>2838</th>
      <td>-10.474242</td>
      <td>13.188574</td>
      <td>-4.580487</td>
      <td>medium</td>
      <td>Awesome Huge Studio - NYC Center</td>
      <td>1891017</td>
      <td>200.0</td>
      <td>66.666667</td>
      <td>3</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>Chelsea</td>
    </tr>
    <tr>
      <th>2957</th>
      <td>-12.070560</td>
      <td>6.332698</td>
      <td>3.506707</td>
      <td>medium</td>
      <td>Luxury studio</td>
      <td>1975999</td>
      <td>189.0</td>
      <td>63.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Manhattan</td>
      <td>Chelsea</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.scatter_3d(plotly_data, x='TSNE1', y='TSNE2', z='TSNE3', color='price_category', 
                    hover_name='name', hover_data=['price', 'minimum_nights', 'id'], 
                    template='plotly_dark', opacity=0.9, title='Visualizing airbnb locations in feature space',
                    labels={'TSNE1': 'X', 'TSNE2': 'Y', 'TSNE3':'Z'}, )

fig.write_html('scatter-3d.html')
```


```python

```


```python

```
