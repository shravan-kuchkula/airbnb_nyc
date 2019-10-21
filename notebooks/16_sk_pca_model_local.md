
## New York City Airbnb Dimensionality Reduction using PCA
When working with a dataset with many features it is extremely difficult to visualize/explore the relationships between features. Not only it makes the EDA process difficult but also affects the machine learning model’s performance since the chances are that you might overfit your model or violate some of the assumptions of the algorithm, like the independence of features in linear regression. This is where dimensionality reduction comes in.

In machine learning, dimensionality reduction is the process of reducing the number of random variables under consideration by obtaining a set of principal variables. By reducing the dimension of your feature space, you have fewer relationships between features to consider which can be explored and visualized easily and also you are less likely to overfit your model. One way to perform Dimensionality reduction is by using Principal Components Analysis.

Principal Component Analysis or PCA is a linear feature extraction technique. It performs a linear mapping of the data to a lower-dimensional space in such a way that the variance of the data in the low-dimensional representation is maximized. It does so by calculating the eigenvectors from the covariance matrix. The eigenvectors that correspond to the largest eigenvalues (the principal components) are used to reconstruct a significant fraction of the variance of the original data. For more details read my post on PCA here: [Intro-to-pca](https://shravan-kuchkula.github.io/PCA-in-R/)

In simpler terms, PCA combines your input features in a specific way that you can drop the least important feature while still retaining the most valuable parts of all of the features. As an added benefit, each of the new features or components created after PCA are all independent of one another.

## Get the data


```python
# data managing and display libs
import pandas as pd
import numpy as np
import os
import io

import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline 

# sagemaker libraries
import boto3
```


```python
# boto3 client to get S3 data
s3_client = boto3.client('s3')
bucket_name='skuchkula-sagemaker-airbnb'
```


```python
# list the bucket objects
response = s3_client.list_objects(Bucket=bucket_name)

# get list of objects inside the bucket
files = [file['Key'] for file in response['Contents']]
files
```




    ['clean/airbnb_clean.csv',
     'detailed_listings.csv',
     'feature_eng/amenities_features.csv',
     'feature_eng/description_features.csv',
     'feature_eng/host_verification_features.csv',
     'feature_eng/merged_features.csv',
     'feature_eng/min_max_scaled_final_df.csv',
     'feature_eng/scaled_final_df.csv',
     'summary_listings.csv']




```python
# download the file from s3
def get_data_frame(bucket_name, file_name):
    '''
    Takes the location of the dataset on S3 and returns a dataframe.
    arguments:
            bucket_name: the name of the bucket
            file_name: the key inside the bucket
    returns:
            dataframe
    '''
    # get an S3 object by passing in the bucket and file name
    data_object = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    
    # information is in the "Body" of the object
    data_body = data_object["Body"].read()
    
    # read in bytes data
    data_stream = io.BytesIO(data_body)
    
    # create a dataframe
    df = pd.read_csv(data_stream, header=0, delimiter=",", low_memory=False, keep_default_na=False)
    
    return df
```


```python
airbnb_file='feature_eng/min_max_scaled_final_df.csv'
df_airbnb = get_data_frame(bucket_name, airbnb_file)
df_airbnb.head()
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
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>guests_included</th>
      <th>extra_people</th>
      <th>availability_30</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>number_of_reviews</th>
      <th>...</th>
      <th>description_contains_yankee</th>
      <th>description_contains_yard</th>
      <th>description_contains_year</th>
      <th>description_contains_yellow</th>
      <th>description_contains_yoga</th>
      <th>description_contains_york</th>
      <th>description_contains_young</th>
      <th>description_contains_yummy</th>
      <th>description_contains_zero</th>
      <th>description_contains_zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.04</td>
      <td>0.064516</td>
      <td>0.000000</td>
      <td>0.025</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.433333</td>
      <td>0.283333</td>
      <td>0.344444</td>
      <td>0.071987</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.04</td>
      <td>0.064516</td>
      <td>0.071429</td>
      <td>0.025</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00</td>
      <td>0.064516</td>
      <td>0.071429</td>
      <td>0.025</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.014085</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.137645</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.04</td>
      <td>0.064516</td>
      <td>0.071429</td>
      <td>0.025</td>
      <td>0.066667</td>
      <td>0.333333</td>
      <td>0.800000</td>
      <td>0.550000</td>
      <td>0.700000</td>
      <td>0.117371</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.074083</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04</td>
      <td>0.064516</td>
      <td>0.071429</td>
      <td>0.025</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076682</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.18384</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2201 columns</p>
</div>



## PCA analysis
PCA attempts to reduce the number of features within a dataset while retaining the “principal components”, which are defined as weighted, linear combinations of existing features that are designed to be linearly independent and account for the largest possible variability in the data! You can think of this method as taking many features and combining similar or redundant features together to form a new, smaller feature set.

Using sklearn's PCA implementation, we pass in n_components=50 to produce 50 principal components.

n_components: An integer that defines the number of PCA components to produce. 


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(df_airbnb)
```




    PCA(copy=True, iterated_power='auto', n_components=50, random_state=None,
        svd_solver='auto', tol=0.0, whiten=False)




```python
features = range(pca.n_components_)
plt.figure(figsize=(18,10))
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()
```


![png](16_sk_pca_model_local_files/16_sk_pca_model_local_9_0.png)


The breakdown of how much variance explained by each PCA feature is shown in the above plot.


```python
pca_features = pca.transform(df_airbnb)
pca_features.shape
```




    (45605, 50)




```python
pca_df = pd.DataFrame(pca_features)
pca_df.head()
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
      <td>2.658552</td>
      <td>-1.455526</td>
      <td>0.699093</td>
      <td>-0.553412</td>
      <td>-0.125384</td>
      <td>-0.654187</td>
      <td>-0.044713</td>
      <td>-0.001357</td>
      <td>0.108582</td>
      <td>1.333186</td>
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
      <td>-1.644174</td>
      <td>-0.715321</td>
      <td>1.097099</td>
      <td>0.893833</td>
      <td>0.399013</td>
      <td>-0.139751</td>
      <td>-1.183842</td>
      <td>-1.132572</td>
      <td>0.996817</td>
      <td>0.151908</td>
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
      <td>-1.012114</td>
      <td>-1.718006</td>
      <td>-0.320889</td>
      <td>-0.356113</td>
      <td>0.113797</td>
      <td>0.392780</td>
      <td>-0.399569</td>
      <td>-0.819086</td>
      <td>-0.133456</td>
      <td>-0.200684</td>
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
      <td>0.067402</td>
      <td>-0.520919</td>
      <td>-0.263865</td>
      <td>0.353879</td>
      <td>-0.907122</td>
      <td>-0.906052</td>
      <td>-1.158888</td>
      <td>-0.624668</td>
      <td>-0.071571</td>
      <td>0.780228</td>
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
      <td>-1.666907</td>
      <td>1.296202</td>
      <td>0.520533</td>
      <td>0.659033</td>
      <td>1.262238</td>
      <td>-0.516450</td>
      <td>-0.827510</td>
      <td>-0.678685</td>
      <td>0.010626</td>
      <td>0.180606</td>
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
<p>5 rows × 50 columns</p>
</div>

## Component make-up

```python
# display makeup of first component
num=1
display_component(v, df_airbnb.columns.values, component_num=num, n_weights=20)
```


![png](14_sk_PCA_files/14_sk_PCA_20_0.png)



```python
# display makeup of first component
num=2
display_component(v, df_airbnb.columns.values, component_num=num, n_weights=20)
```


![png](14_sk_PCA_files/14_sk_PCA_21_0.png)



```python
# display makeup of first component
num=3
display_component(v, df_airbnb.columns.values, component_num=num, n_weights=20)
```


![png](14_sk_PCA_files/14_sk_PCA_22_0.png)



```python
# display makeup of first component
num=4
display_component(v, df_airbnb.columns.values, component_num=num, n_weights=20)
```


![png](14_sk_PCA_files/14_sk_PCA_23_0.png)



## Create price_category to aid in visualization
As of now, the price feature is a continuous variable. We can apply binning to create a discreatized version of the price column. Doing so, we can filter locations based on price category. Ideally, this should be inside the Feature Engineering step, but for now, we will go with it.


```python
# Read in the cleaned dataset
airbnb_detailed = pd.read_csv('airbnb_clean.csv')

# calculate the adjusted_price
airbnb_detailed['adjusted_price'] = airbnb_detailed.price / airbnb_detailed.minimum_nights
```


```python
# let pandas know that you are working with a copy
airbnb_temp = airbnb_detailed.copy()

# get the indices of low, med and high rows
low_indexes = airbnb_temp[airbnb_temp.adjusted_price < 50].index
med_indexes = airbnb_temp[(airbnb_temp.adjusted_price >= 50) &
                         (airbnb_temp.adjusted_price < 200)].index
high_indexes = airbnb_temp[(airbnb_temp.adjusted_price >= 200)].index
```


```python
# create a new column called 'price_category'
airbnb_temp.loc[low_indexes, 'price_category'] = 'low'
airbnb_temp.loc[med_indexes, 'price_category'] = 'medium'
airbnb_temp.loc[high_indexes, 'price_category'] = 'high'
airbnb_temp.head()
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
      <th>id</th>
      <th>name</th>
      <th>summary</th>
      <th>description</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>host_verifications</th>
      <th>neighbourhood_cleansed</th>
      <th>neighbourhood_group_cleansed</th>
      <th>latitude</th>
      <th>...</th>
      <th>cancellation_policy_strict</th>
      <th>cancellation_policy_strict_14_with_grace_period</th>
      <th>cancellation_policy_super_strict_30</th>
      <th>cancellation_policy_super_strict_60</th>
      <th>require_guest_profile_picture_f</th>
      <th>require_guest_profile_picture_t</th>
      <th>require_guest_phone_verification_f</th>
      <th>require_guest_phone_verification_t</th>
      <th>adjusted_price</th>
      <th>price_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>Skylit Midtown Castle</td>
      <td>Find your romantic getaway to this beautiful, ...</td>
      <td>Find your romantic getaway to this beautiful, ...</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>['email', 'phone', 'reviews', 'kba', 'work_ema...</td>
      <td>Midtown</td>
      <td>Manhattan</td>
      <td>40.75362</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>225.000000</td>
      <td>high</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3647</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>NaN</td>
      <td>WELCOME TO OUR INTERNATIONAL URBAN COMMUNITY T...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>['email', 'phone', 'google', 'reviews', 'jumio...</td>
      <td>Harlem</td>
      <td>Manhattan</td>
      <td>40.80902</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>50.000000</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5022</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>NaN</td>
      <td>Loft apartment with high ceiling and wood floo...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>['email', 'phone', 'facebook', 'reviews', 'kba']</td>
      <td>East Harlem</td>
      <td>Manhattan</td>
      <td>40.79851</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8.000000</td>
      <td>low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5099</td>
      <td>Large Cozy 1 BR Apartment In Midtown East</td>
      <td>My large 1 bedroom apartment is true New York ...</td>
      <td>My large 1 bedroom apartment is true New York ...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>['email', 'phone', 'reviews', 'jumio', 'govern...</td>
      <td>Murray Hill</td>
      <td>Manhattan</td>
      <td>40.74767</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>66.666667</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5121</td>
      <td>BlissArtsSpace!</td>
      <td>NaN</td>
      <td>HELLO EVERYONE AND THANKS FOR VISITING BLISS A...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>['email', 'phone', 'facebook', 'reviews', 'off...</td>
      <td>Bedford-Stuyvesant</td>
      <td>Brooklyn</td>
      <td>40.68688</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.333333</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 69 columns</p>
</div>




```python
airbnb_temp.price_category.value_counts()
```




    low       25133
    medium    18484
    high       1988
    Name: price_category, dtype: int64



## Merge PCA features with the original dataset


```python
# take only cols that you want to display in visualization from main dataset
cols = ['price_category', 'name', 'id', 'price', 'adjusted_price', 
        'minimum_nights', 'bedrooms', 'bathrooms',
        'neighbourhood_group_cleansed', 'neighbourhood_cleansed']

airbnb_reduced = airbnb_temp[cols]

# merge this with PCA features
airbnb_final = pd.concat([airbnb_reduced, pca_df], axis=1)

airbnb_final.head()
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



## Save and upload to S3


```python
airbnb_final.to_csv('airbnb_final.csv', index=False)
```


```python
import configparser
config = configparser.ConfigParser()
config.read_file(open('credentials.cfg'))

KEY = config.get('AWS','KEY')
SECRET = config.get('AWS','SECRET')
```


```python
import boto3

# Generate the boto3 client for interacting with S3
s3 = boto3.client('s3', region_name='us-east-1', 
                        # Set up AWS credentials 
                        aws_access_key_id=KEY, 
                        aws_secret_access_key=SECRET)
```


```python
s3.upload_file(Bucket='skuchkula-sagemaker-airbnb',
              Filename='airbnb_final.csv',
              Key='feature/airbnb_final.csv')
```
