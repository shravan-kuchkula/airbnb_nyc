
## Feature Engineering: Turn raw text into numeric features
Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the machine learning models that result in better results. It is one of the key Data Prepration steps in the overall machine learning pipeline (aka CRISP-DM process) as depicted in the figure:

![crisp-dm](../reports/figures/crisp-dm.png)
> ***Figure: CRISP-DM process for a typical Machine Learning problem***


A wealth of information is hidden in the description, amenities and other text columns of the airbnb dataset. It is possible to extract new features from these columns and enrich our existing dataset.

Specifically, we will create new features from the cleaned dataset that was generated as described here [New York City Airbnb Data Cleaning](https://shravan-kuchkula.github.io/nyc-airbnb-data-cleaning/). The following are the goals of this step:
- For the amenities column, we will be creating a binary bag-of-words representation, that is, we build a document-term-matrix with a 1 indicating the presence of amenity and 0 otherwise.
- For the host verifications column, create a similar Document-Term matrix with 1 indicating the presence of verification method and 0 otherwise.
- For the description column, create a TF-IDF representation.
- Merge all these features into one dataframe.

## Get the data from S3
Read in the data that was cleaned as part of this blog post: [New York City Airbnb Data Cleaning](https://shravan-kuchkula.github.io/nyc-airbnb-data-cleaning/) . The cleansed dataset is loaded from `s3://skuchkula-sagemaker-airbnb/airbnb_clean.csv`.


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
#import sagemaker
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
     'summary_listings.csv']




```python
airbnb_file = files[0]
```


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
df_airbnb = get_data_frame(bucket_name, airbnb_file)
```


```python
df_airbnb.shape
```




    (45605, 67)



After dealing with missing values and dropping columns that are not relevant, we are left with 67 features. Out of these we will be extracting the amenities, host verifications and description columns.

## Create features from amenities values
To get to our end goal of comparing airbnb listings, we first need to do some pre-processing tasks and book-keeping of the actual words in each listing's amenities column. Shown below is the list of amenities offered by the first listing in our dataset:

```
{TV,Wifi,"Air conditioning",Kitchen,"Paid parking", ..}
```

The first step here would be to tokenize the list of amenities in the amenities column. After splitting them into tokens, we will make a `binary bag of words`, then we will create a dictionary with the tokens, amenities_idx, which will have the following format:

```
{'amenity': index value, ..}
```


```python
df_airbnb.amenities.head()
```




    0    {TV,Wifi,"Air conditioning",Kitchen,"Paid park...
    1    {"Cable TV",Internet,Wifi,"Air conditioning",K...
    2    {Internet,Wifi,"Air conditioning",Kitchen,Elev...
    3    {TV,"Cable TV",Internet,Wifi,Kitchen,"Buzzer/w...
    4    {Wifi,"Air conditioning",Kitchen,"Pets live on...
    Name: amenities, dtype: object




```python
# remove the curly brackets
df_airbnb['amenities'] =  df_airbnb['amenities'].apply(lambda x: x[1:-1])
```


```python
df_airbnb.amenities.head()
```




    0    TV,Wifi,"Air conditioning",Kitchen,"Paid parki...
    1    "Cable TV",Internet,Wifi,"Air conditioning",Ki...
    2    Internet,Wifi,"Air conditioning",Kitchen,Eleva...
    3    TV,"Cable TV",Internet,Wifi,Kitchen,"Buzzer/wi...
    4    Wifi,"Air conditioning",Kitchen,"Pets live on ...
    Name: amenities, dtype: object




```python
amenities = df_airbnb.amenities
```


```python
# create a dictionary of terms
amenities_idx ={}
idx = 0
corpus = []
for i in range(len(amenities)):
    items = amenities[i]
    items_lower = items.lower()
    tokens = items_lower.split(',')
    corpus.append(tokens)
    for token in tokens:
        if token not in amenities_idx:
            amenities_idx[token] = idx
            idx += 1
```


```python
print("Total number of terms in the corpus: ", len(amenities_idx))
print("Total number of documents in the corpus: ", len(corpus))
```

    Total number of terms in the corpus:  131
    Total number of documents in the corpus:  45605


There are **131** unique amenities offered across all the **45,605** airbnb listings in New york city. With this, we can now initialize a document-term matrix. Here, a document is an airbnb listing and a term is an amenity. The size of the matrix should be dipicted:

![dtm](../reports/figures/dtm.png)
> *Figure:* **Document-term-matrix (DTM) of airbnb listings and amenities**


```python
# Get the number of items and tokens 
M = len(amenities)
N = len(amenities_idx)

# Initialize a matrix of zeros
A = np.zeros((M, N))
```

We can then define an encoder function that will put a 1 in the corresponding index of the matrix. And finally, we will create the document-term matrix by applying this encoder function to all the documents in the corpus.


```python
# Define the amenity_encoder function
def amenity_encoder(tokens):
    x = np.zeros(N)
    for token in tokens:
        # Get the index for each amenity
        idx = amenities_idx[token]
        # Put 1 at the corresponding indices
        x[idx] = 1
    return x
```


```python
# Make a document-term matrix
i = 0
for tokens in corpus:
    A[i, :] = amenity_encoder(tokens)
    i = i + 1
```


```python
A.shape
```




    (45605, 131)




```python
type(A)
```




    numpy.ndarray



In this manner, we created a binary bag-of-words representation for the amenities of each airbnb listing. The DTM matrix is converted into a pandas Dataframe by making use of the *amenities_idx* dictionary of terms as columns of the dataframe.


```python
amenities_features = pd.DataFrame(A, columns=list(amenities_idx.keys()))
amenities_features.head()
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
      <th>tv</th>
      <th>wifi</th>
      <th>"air conditioning"</th>
      <th>kitchen</th>
      <th>"paid parking off premises"</th>
      <th>"free street parking"</th>
      <th>"indoor fireplace"</th>
      <th>heating</th>
      <th>"family/kid friendly"</th>
      <th>"smoke detector"</th>
      <th>...</th>
      <th>"lake access"</th>
      <th>"pool with pool hoist"</th>
      <th>"full kitchen"</th>
      <th>"electric profiling bed"</th>
      <th>"ground floor access"</th>
      <th>"air purifier"</th>
      <th>"mobile hoist"</th>
      <th>kitchenette</th>
      <th>"fixed grab bars for shower</th>
      <th>"ceiling hoist"</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
<p>5 rows × 131 columns</p>
</div>




```python
# clean the column names and prefix them with amenities
amenities_features.columns
```




    Index(['tv', 'wifi', '"air conditioning"', 'kitchen',
           '"paid parking off premises"', '"free street parking"',
           '"indoor fireplace"', 'heating', '"family/kid friendly"',
           '"smoke detector"',
           ...
           '"lake access"', '"pool with pool hoist"', '"full kitchen"',
           '"electric profiling bed"', '"ground floor access"', '"air purifier"',
           '"mobile hoist"', 'kitchenette', '"fixed grab bars for shower',
           '"ceiling hoist"'],
          dtype='object', length=131)



As some of the columns have a space in between, and some have a / to indicate an alternative, I have cleaned this up so that our columns are neatly represented in the final dataset.


```python
import re
def clean_column(text):
    # sub spaces with underscore
    text = re.sub(r'[\s+]', '_', text)
    # remove ""
    text = re.sub(r'[\"]', '', text)
    
    return text
```


```python
amenities_features.columns = ["amenities_" + clean_column(item) for item in list(amenities_features.columns)]
```


```python
amenities_features.head()
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
      <th>amenities_tv</th>
      <th>amenities_wifi</th>
      <th>amenities_air_conditioning</th>
      <th>amenities_kitchen</th>
      <th>amenities_paid_parking_off_premises</th>
      <th>amenities_free_street_parking</th>
      <th>amenities_indoor_fireplace</th>
      <th>amenities_heating</th>
      <th>amenities_family/kid_friendly</th>
      <th>amenities_smoke_detector</th>
      <th>...</th>
      <th>amenities_lake_access</th>
      <th>amenities_pool_with_pool_hoist</th>
      <th>amenities_full_kitchen</th>
      <th>amenities_electric_profiling_bed</th>
      <th>amenities_ground_floor_access</th>
      <th>amenities_air_purifier</th>
      <th>amenities_mobile_hoist</th>
      <th>amenities_kitchenette</th>
      <th>amenities_fixed_grab_bars_for_shower</th>
      <th>amenities_ceiling_hoist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
<p>5 rows × 131 columns</p>
</div>



Save a copy of the amenities features into S3 as an intermediary dataset.


```python
# create a csv file and store it in S3
amenities_features.to_csv('amenities_features.csv', index=False)
```


```python
# upload it to S3
s3_client.upload_file(Bucket=bucket_name, 
                      Filename='amenities_features.csv', 
                      Key='feature_eng/amenities_features.csv')
```

## Create features from host_verifications
The host_verifications column contains a list of host verification types for each of the airbnb listing. Using the same concept of creating a binary bag of words representation by constructing a DTM, we proceed with our feature engineering task.


```python
df_airbnb.host_verifications[:10]
```




    0    ['email', 'phone', 'reviews', 'kba', 'work_ema...
    1    ['email', 'phone', 'google', 'reviews', 'jumio...
    2     ['email', 'phone', 'facebook', 'reviews', 'kba']
    3    ['email', 'phone', 'reviews', 'jumio', 'govern...
    4    ['email', 'phone', 'facebook', 'reviews', 'off...
    5            ['email', 'phone', 'facebook', 'reviews']
    6    ['email', 'phone', 'facebook', 'google', 'revi...
    7                 ['email', 'phone', 'reviews', 'kba']
    8    ['email', 'phone', 'manual_online', 'reviews',...
    9    ['email', 'phone', 'reviews', 'jumio', 'govern...
    Name: host_verifications, dtype: object




```python
import re
re.findall(r'\w+', df_airbnb.host_verifications[0])
```




    ['email', 'phone', 'reviews', 'kba', 'work_email']




```python
df_airbnb.loc[:, 'host_verifications'] = df_airbnb.host_verifications.apply(lambda x: re.findall(r'\w+', x))
```


```python
verifications = df_airbnb.host_verifications
```


```python
verification_idx = {}
idx = 0
corpus = []
for i in range(len(verifications)):
    items = verifications[i]
    corpus.append(items)
    for item in items:
        if item not in verification_idx:
            verification_idx[item] = idx
            idx += 1
```


```python
verification_idx
```




    {'email': 0,
     'phone': 1,
     'reviews': 2,
     'kba': 3,
     'work_email': 4,
     'google': 5,
     'jumio': 6,
     'government_id': 7,
     'facebook': 8,
     'offline_government_id': 9,
     'selfie': 10,
     'identity_manual': 11,
     'manual_online': 12,
     'sent_id': 13,
     'manual_offline': 14,
     'None': 15,
     'weibo': 16,
     'sesame': 17,
     'sesame_offline': 18,
     'zhima_selfie': 19}




```python
print("Total number of terms in the corpus: ", len(verification_idx))
print("Total number of documents in the corpus: ", len(corpus))
```

    Total number of terms in the corpus:  20
    Total number of documents in the corpus:  45605



```python
# Get the number of items and tokens 
M = len(verifications)
N = len(verification_idx)

# Initialize a matrix of zeros
B = np.zeros((M, N))
```


```python
# Define the verification_encoder function
def verification_encoder(tokens):
    x = np.zeros(N)
    for token in tokens:
        # Get the index for each verification
        idx = verification_idx[token]
        # Put 1 at the corresponding indices
        x[idx] = 1
    return x
```


```python
# Make a document-term matrix
i = 0
for tokens in corpus:
    B[i, :] = verification_encoder(tokens)
    i = i + 1
```


```python
B.shape
```




    (45605, 20)




```python
type(B)
```




    numpy.ndarray




```python
list(verification_idx.keys())
```




    ['email',
     'phone',
     'reviews',
     'kba',
     'work_email',
     'google',
     'jumio',
     'government_id',
     'facebook',
     'offline_government_id',
     'selfie',
     'identity_manual',
     'manual_online',
     'sent_id',
     'manual_offline',
     'None',
     'weibo',
     'sesame',
     'sesame_offline',
     'zhima_selfie']




```python
verification_features = pd.DataFrame(B, columns=list(verification_idx.keys()))
```


```python
verification_features.columns = verification_features.add_prefix('host_verification_by_').columns
```


```python
verification_features.shape
```




    (45605, 20)




```python
verification_features.head()
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
      <th>host_verification_by_email</th>
      <th>host_verification_by_phone</th>
      <th>host_verification_by_reviews</th>
      <th>host_verification_by_kba</th>
      <th>host_verification_by_work_email</th>
      <th>host_verification_by_google</th>
      <th>host_verification_by_jumio</th>
      <th>host_verification_by_government_id</th>
      <th>host_verification_by_facebook</th>
      <th>host_verification_by_offline_government_id</th>
      <th>host_verification_by_selfie</th>
      <th>host_verification_by_identity_manual</th>
      <th>host_verification_by_manual_online</th>
      <th>host_verification_by_sent_id</th>
      <th>host_verification_by_manual_offline</th>
      <th>host_verification_by_None</th>
      <th>host_verification_by_weibo</th>
      <th>host_verification_by_sesame</th>
      <th>host_verification_by_sesame_offline</th>
      <th>host_verification_by_zhima_selfie</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
</div>




```python
# create a csv file and store it in S3
verification_features.to_csv('host_verification_features.csv', index=False)
```


```python
# upload it to S3
s3_client.upload_file(Bucket=bucket_name, 
                      Filename='host_verification_features.csv', 
                      Key='feature_eng/host_verification_features.csv')
```

## Create features from text columns
### Tokenize and normalize the text columns
Now we arrive at the two main text columns present within our dataset. Starting off by displaying the first few columns of the description and summary fields reveals that keeping only the description for the feature building phase would make more sense. Since, summary is just a shortened version of the description and moreover not every listing contains a summary, I have chosen to consider only the description field.


```python
# Set the display properties so that we can inspect the data
pd.set_option("display.max_colwidth", 1000)
```


```python
TEXT_COLUMNS = ['description', 'summary']
df_airbnb[TEXT_COLUMNS].head()
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
      <th>description</th>
      <th>summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Find your romantic getaway to this beautiful, spacious skylit studio in the heart of Midtown, Manhattan.  STUNNING SKYLIT STUDIO / 1 BED + SINGLE / FULL BATH / FULL KITCHEN / FIREPLACE / CENTRALLY LOCATED / WiFi + APPLE TV / SHEETS + TOWELS - Spacious (500+ft²), immaculate and nicely furnished &amp; designed studio. - Tuck yourself into the ultra comfortable bed under the skylight. Fall in love with a myriad of bright lights in the city night sky.  - Single-sized bed/convertible floor mattress with luxury bedding (available upon request). - Gorgeous pyramid skylight with amazing diffused natural light, stunning architectural details, soaring high vaulted ceilings, exposed brick, wood burning fireplace, floor seating area with natural zafu cushions, modern style mixed with eclectic art &amp; antique treasures, large full bath, newly renovated kitchen, air conditioning/heat, high speed WiFi Internet, and Apple TV. - Centrally located in the heart of Midtown Manhattan just a few blocks from a...</td>
      <td>Find your romantic getaway to this beautiful, spacious skylit studio in the heart of Midtown, Manhattan.  STUNNING SKYLIT STUDIO / 1 BED + SINGLE / FULL BATH / FULL KITCHEN / FIREPLACE / CENTRALLY LOCATED / WiFi + APPLE TV / SHEETS + TOWELS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WELCOME TO OUR INTERNATIONAL URBAN COMMUNITY This Spacious 1 bedroom  is with Plenty of Windows with a View....... Sleeps.....Four Adults.....two in the Livingrm. with (2) Sofa-beds.  (Website hidden by Airbnb) two in the Bedrm.on a very Comfortable Queen Size Bed... A Complete Bathrm.....With Shower and Bathtub....... Fully Equipped with Linens &amp; Towels........ Spacious Living Room......Flat ScreenTelevision.....DVD Player with Movies available for your viewing during your stay............................................................................. Dining Area.....for Morning Coffee or Tea..................................................... The Kitchen Area is Modern with Granite Counter Top... includes the use of a Coffee Maker...Microwave to Heat up a Carry Out/In Meal.... Not suited for a Gourmet Cook...or Top Chef......Sorry!!!! . This Flat is located in HISTORIC HARLEM.... near the Appollo Theater and The Museum Mile...on Fifth Avenue. Sylvia's World Famous Resturant......</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>Loft apartment with high ceiling and wood flooring located 10 minutes away from Central Park in Harlem - 1 block away from 6 train and 3 blocks from 2 &amp; 3 line. This is in a recently renovated building which includes elevator, trash shoot. marble entrance and laundromat in the basement.  The apartment is a spacious loft studio. The seating area and sleeping area is divided by a bookcase. There is a long hallway entrance where the bathroom and closet for your clothes is situated. The apartment is in mint condition, the walls have been freshly painted a few months ago. Supermarket, and 24 hour convenience store less than 1 block away.  1 block away from Hot Yoga Studio and NY Sports club facility.  Perfect for anyone wanting to stay in Manhattan but get more space.  10 minutes away from midtown and 15 minutes away from downtown. The neighborhood is lively and diverse. You will need to travel at least 10 blocks to find cafe's, restaurants etc.. There are a few restaurants on 100 stree...</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>My large 1 bedroom apartment is true New York City living.  The apt is in midtown on the east side and centrally located, just a 10-minute walk from Grand Central Station, Empire State Building, Times Square. The kitchen and living room are large and bright with Apple TV. I have a new Queen Bed that sleeps 2 people, and a Queen Aero Bed that can sleep 2 people in the living room. The apartment is located on the 5th floor of a walk up - no elevator (lift). I have a large 1 bedroom apartment centrally located in Midtown East.  A 10 minute walk from Grand Central Station, Times Square, Empire State Building and all major subway and bus lines. The apartment is located on the 5th floor of a pre-war walk up building-no elevator/lift.  The apartment is bright with has high ceilings and flow through rooms. A spacious, cozy living room with Netflix and Apple TV.  A large bright kitchen to sit and enjoy coffee or tea.  The bedroom is spacious with a comfortable queen size bed that sleeps 2. ...</td>
      <td>My large 1 bedroom apartment is true New York City living.  The apt is in midtown on the east side and centrally located, just a 10-minute walk from Grand Central Station, Empire State Building, Times Square. The kitchen and living room are large and bright with Apple TV. I have a new Queen Bed that sleeps 2 people, and a Queen Aero Bed that can sleep 2 people in the living room. The apartment is located on the 5th floor of a walk up - no elevator (lift).</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HELLO EVERYONE AND THANKS FOR VISITING BLISS ART SPACE!  Thank you all for your support. I've traveled a lot in the last year few years, to the  U.K. Germany, Italy and France! Loved Paris, Berlin and Calabria! Highly recommend all these places.  One room available for rent in a 2 bedroom apt in Bklyn. We share a common space with kitchen. I am an artist(painter, filmmaker) and curator who is working in the film industry while I'm building my art event production businesses. Price above is nightly for one person. Monthly rates available.  Price is $900 per month for one person. Utilities not included, they are about 50 bucks, payable when the bill arrives mid month.   Couples rates are slightly more for monthly and 90$ per night short term. If you are a couple please Iet me know and I’ll give you the monthly rate for that. Room rental is on a temporary basis, perfect from 2- 6 months - no long term requests please! At the moment I AM ONLY TAKING BOOKINGS OF AT LEAST ONE OR ONE AND ...</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



Next, I send each of these descriptions through my `nlp_pipeline` function (code is in the appendix), which will basically clean the text, remove stop words and returns a clean string of the description. Shown below are the clean version of the first 10 descriptions.


```python
# use descriptions column
descriptions = list(df_airbnb.description)

# send this list of descriptions through my nlp pipeline
clean_descriptions = nlp_pipeline(descriptions)
```


```python
# rejoin the tokens to form strings which will be used to vectorize
clean_descriptions_text = [' '.join(item) for item in clean_descriptions]

clean_descriptions_text[:10]
```




    ['find romantic getaway beautiful spacious studio heart manhattan stun studio single full bath full kitchen fireplace centrally locate wifi apple sheet towel spacious immaculate nicely furnish design studio tuck ultra comfortable skylight fall love myriad bright light city night single size floor mattress luxury bedding available request gorgeous pyramid skylight amaze diffuse natural light stun architectural detail soaring high vault ceiling expose brick wood burning fireplace floor seating area natural cushion modern style eclectic antique treasure large full bath newly renovate kitchen high speed wifi internet apple centrally locate heart manhattan block',
     'welcome international urban community spacious plenty window view sofa website hide comfortable queen size complete shower bathtub fully equip linen towel spacious living player movie available viewing stay dining morning coffee kitchen area modern granite counter include coffee heat carry meal suit gourmet flat locate historic harlem near theater museum fifth avenue world famous',
     'loft high ceiling wood flooring locate minute away central park harlem block away train block line recently renovate building include elevator trash shoot marble entrance laundromat basement spacious loft studio seating area sleeping area divide bookcase long hallway entrance closet clothes situate mint condition wall freshly paint month supermarket hour convenience store block away block away yoga studio sport club facility perfect want stay manhattan space minute away minute away downtown neighborhood lively diverse need travel least block find cafe restaurant restaurant street',
     'large true york city living east side centrally locate minute walk grand central station empire state building time square kitchen living room large bright apple queen sleep people queen sleep people living room locate floor walk elevator lift large centrally locate east minute walk grand central station time square empire state building major subway line locate floor walk building bright high ceiling flow room spacious cozy living room apple large bright kitchen enjoy coffee spacious comfortable queen size sleep',
     'hello thanks visiting bliss space thank support travel last year year germany italy france love paris berlin calabria highly recommend place room available rent share common space kitchen filmmaker curator working film industry building event production business price nightly person monthly rate available price month person utility include buck payable bill arrive month couple rate slightly monthly night short term couple please know give monthly rate room rental temporary basis perfect month long term request please moment taking booking least',
     'please expect luxury basic room center manhattan large furnish private room share host locate block away central park avenue close subway station columbus circle street great restaurant broadway transportation easily accessible cost room night weekly rate available guest also feature hardwood floor second floor walk full size microwave small refrigerator well appliance wire internet wifi electric heat sheet towel include kitchen available living room come time except midnight basic check time flexible schedule please please living room private place host',
     'best guest seeking safe clean spare room family comfortable independent accommodate family noise quiet hour afraid friendly year golden guest perfectly clean peeling paint short guest want feel like stay sister visiting city sister change sheet clean stay family little guest room enjoy privacy warm welcome security guest room comfortable clean small well outfit single fabulous mattress firm time share immediately across hall share sense suite family second bath stay fully supply shampoo conditioner scrub soap',
     'sunny quiet heart east village hardwood floor window every room beautiful retro furnishing huge flat screen cable separate full size ceiling kitchen living room make room total small clean elevator building require deposit cleaning schedule handy deposit return departure original condition minimum stay month ideal someone stay year east village romantic artist neighborhood best restaurant bar shop city train second avenue minute walk train street minute walk avenue right outside hope enjoy using apple please make sure sign check otherwise charge show',
     'live like instead stuffy tiny overprice hotel small cozy nestle southern part historic lower east side step subway walking distance cool neighborhood please note automatically book messaging first thanks cute cozy lower east side adjacent lower manhattan step train east broadway stop close street many bar restaurant shop clean fully equip kitchen cable heat wireless internet locate floor well maintain walk building walking distance soho east village little italy close brooklyn place better hotel price list people possible accommodate extra charge ideal',
     'beautiful upper west side great location close columbia university beautiful cathedral block central park riverside park morning side park many amaze restaurant step away small block train block train charm sunny comfortable full size fold couch living room brand kitchen brand internet floor elevator building evening doorman cant wait hear deposit require']



### Vectorize the corpus
Now that we have the cleaned corpus, we can make use of `TfidfVectorizer` to convert the text into vectorized format. Due to memory limitations, I chose to keep *max_features* to 2000.


```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=10, max_df=0.95, max_features=2000,
                                   ngram_range=(1,1), stop_words='english')

tfidf_feature_matrix = tfidf_vectorizer.fit_transform(clean_descriptions_text)

tfidf_feature_matrix.shape
```




    (45605, 2000)




```python
type(tfidf_feature_matrix)
```




    scipy.sparse.csr.csr_matrix




```python
display(tfidf_vectorizer.get_feature_names()[:10])
display(tfidf_vectorizer.get_feature_names()[-10:])
```


    ['able',
     'abode',
     'abound',
     'absolute',
     'absolutely',
     'abundance',
     'abundant',
     'academy',
     'accent',
     'accept']



    ['yankee',
     'yard',
     'year',
     'yellow',
     'yoga',
     'york',
     'young',
     'yummy',
     'zero',
     'zone']



```python
# create a dataframe from feature matrix
feature_matrix_df = pd.DataFrame(tfidf_feature_matrix.toarray(), 
                                 columns=tfidf_vectorizer.get_feature_names())

feature_matrix_df.head()
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
      <th>able</th>
      <th>abode</th>
      <th>abound</th>
      <th>absolute</th>
      <th>absolutely</th>
      <th>abundance</th>
      <th>abundant</th>
      <th>academy</th>
      <th>accent</th>
      <th>accept</th>
      <th>...</th>
      <th>yankee</th>
      <th>yard</th>
      <th>year</th>
      <th>yellow</th>
      <th>yoga</th>
      <th>york</th>
      <th>young</th>
      <th>yummy</th>
      <th>zero</th>
      <th>zone</th>
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
<p>5 rows × 2000 columns</p>
</div>




```python
feature_matrix_df.shape
```




    (45605, 2000)




```python
feature_matrix_df.columns = feature_matrix_df.add_prefix('description_contains_').columns

# create a csv file and store it in S3
feature_matrix_df.to_csv('description_features.csv', index=False)
```

Upload the file to S3 as an intermediary dataset.


```python
# upload it to S3
s3_client.upload_file(Bucket=bucket_name, 
                      Filename='description_features.csv', 
                      Key='feature_eng/description_features.csv')
```

## Merge all the dataframes


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
     'summary_listings.csv']




```python
amenities_df = get_data_frame(bucket_name, 'feature_eng/amenities_features.csv')
host_verification_df = get_data_frame(bucket_name, 'feature_eng/host_verification_features.csv')
description_df = get_data_frame(bucket_name, 'feature_eng/description_features.csv')
```


```python
print("Amenities dataframe shape: ", amenities_df.shape)
print("Host Verifications dataframe shape: ", host_verification_df.shape)
print("Descriptions dataframe shape: ", description_df.shape)
```

    Amenities dataframe shape:  (45605, 131)
    Host Verifications dataframe shape:  (45605, 20)
    Descriptions dataframe shape:  (45605, 2000)



```python
merged_df = pd.concat([amenities_df, host_verification_df, description_df], axis='columns')
```


```python
merged_df.shape
```




    (45605, 2151)




```python
# create a csv file and store it in S3
merged_df.to_csv('merged_df.csv', index=False)
```


```python
# upload it to S3
s3_client.upload_file(Bucket=bucket_name, 
                      Filename='merged_df.csv', 
                      Key='feature_eng/merged_features.csv')
```

## Final list of 2151 features


```python
merged_df.info(verbose=True)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45605 entries, 0 to 45604
    Data columns (total 2151 columns):
    amenities_tv                                            float64
    amenities_wifi                                          float64
    amenities_air_conditioning                              float64
    amenities_kitchen                                       float64
    amenities_paid_parking_off_premises                     float64
    amenities_free_street_parking                           float64
    amenities_indoor_fireplace                              float64
    amenities_heating                                       float64
    amenities_family/kid_friendly                           float64
    amenities_smoke_detector                                float64
    amenities_carbon_monoxide_detector                      float64
    amenities_fire_extinguisher                             float64
    amenities_essentials                                    float64
    amenities_shampoo                                       float64
    amenities_lock_on_bedroom_door                          float64
    amenities_hangers                                       float64
    amenities_hair_dryer                                    float64
    amenities_iron                                          float64
    amenities_laptop_friendly_workspace                     float64
    amenities_self_check-in                                 float64
    amenities_keypad                                        float64
    amenities_private_living_room                           float64
    amenities_bathtub                                       float64
    amenities_hot_water                                     float64
    amenities_bed_linens                                    float64
    amenities_extra_pillows_and_blankets                    float64
    amenities_ethernet_connection                           float64
    amenities_coffee_maker                                  float64
    amenities_refrigerator                                  float64
    amenities_dishes_and_silverware                         float64
    amenities_cooking_basics                                float64
    amenities_oven                                          float64
    amenities_stove                                         float64
    amenities_luggage_dropoff_allowed                       float64
    amenities_long_term_stays_allowed                       float64
    amenities_cleaning_before_checkout                      float64
    amenities_wide_entrance_for_guests                      float64
    amenities_flat_path_to_guest_entrance                   float64
    amenities_well-lit_path_to_entrance                     float64
    amenities_no_stairs_or_steps_to_enter                   float64
    amenities_cable_tv                                      float64
    amenities_internet                                      float64
    amenities_buzzer/wireless_intercom                      float64
    amenities_translation_missing:_en.hosting_amenity_49    float64
    amenities_translation_missing:_en.hosting_amenity_50    float64
    amenities_elevator                                      float64
    amenities_washer                                        float64
    amenities_dryer                                         float64
    amenities_host_greets_you                               float64
    amenities_pets_live_on_this_property                    float64
    amenities_cat(s)                                        float64
    amenities_microwave                                     float64
    amenities_doorman                                       float64
    amenities_breakfast                                     float64
    amenities_dog(s)                                        float64
    amenities_24-hour_check-in                              float64
    amenities_lockbox                                       float64
    amenities_suitable_for_events                           float64
    amenities_first_aid_kit                                 float64
    amenities_safety_card                                   float64
    amenities_patio_or_balcony                              float64
    amenities_garden_or_backyard                            float64
    amenities_beach_essentials                              float64
    amenities_dishwasher                                    float64
    amenities_other                                         float64
    amenities_building_staff                                float64
    amenities_private_entrance                              float64
    amenities_high_chair                                    float64
    amenities_stair_gates                                   float64
    amenities_children’s_books_and_toys                     float64
    amenities_pack_’n_play/travel_crib                      float64
    amenities_room-darkening_shades                         float64
    amenities_crib                                          float64
    amenities_bbq_grill                                     float64
    amenities_pets_allowed                                  float64
    amenities_pool                                          float64
    amenities_single_level_home                             float64
    amenities_free_parking_on_premises                      float64
    amenities_washer_/_dryer                                float64
    amenities_paid_parking_on_premises                      float64
    amenities_hot_tub                                       float64
    amenities_babysitter_recommendations                    float64
    amenities_children’s_dinnerware                         float64
    amenities_other_pet(s)                                  float64
    amenities_smart_lock                                    float64
    amenities_wide_hallways                                 float64
    amenities_wide_entrance                                 float64
    amenities_extra_space_around_bed                        float64
    amenities_accessible-height_bed                         float64
    amenities_gym                                           float64
    amenities_wheelchair_accessible                         float64
    amenities_outlet_covers                                 float64
    amenities_baby_bath                                     float64
    amenities_changing_table                                float64
    amenities_table_corner_guards                           float64
    amenities_window_guards                                 float64
    amenities_firm_mattress                                 float64
    amenities_smoking_allowed                               float64
    amenities_game_console                                  float64
    amenities_fireplace_guards                              float64
    amenities_wide_doorway_to_guest_bathroom                float64
    amenities_accessible-height_toilet                      float64
    amenities_pocket_wifi                                   float64
    amenities_handheld_shower_head                          float64
    amenities_fixed_grab_bars_for_shower                    float64
    amenities_waterfront                                    float64
    amenities_wide_clearance_to_shower                      float64
    amenities__toilet                                       float64
    amenities_wide_entryway                                 float64
    amenities_ski-in/ski-out                                float64
    amenities_bathtub_with_bath_chair                       float64
    amenities_private_bathroom                              float64
    amenities_baby_monitor                                  float64
    amenities_ev_charger                                    float64
    amenities_disabled_parking_spot                         float64
    amenities_shower_chair                                  float64
    amenities_beachfront                                    float64
    amenities_                                              float64
    amenities_roll-in_shower                                float64
    amenities_fixed_grab_bars_for_toilet                    float64
    amenities_hot_water_kettle                              float64
    amenities_lake_access                                   float64
    amenities_pool_with_pool_hoist                          float64
    amenities_full_kitchen                                  float64
    amenities_electric_profiling_bed                        float64
    amenities_ground_floor_access                           float64
    amenities_air_purifier                                  float64
    amenities_mobile_hoist                                  float64
    amenities_kitchenette                                   float64
    amenities_fixed_grab_bars_for_shower.1                  float64
    amenities_ceiling_hoist                                 float64
    host_verification_by_email                              float64
    host_verification_by_phone                              float64
    host_verification_by_reviews                            float64
    host_verification_by_kba                                float64
    host_verification_by_work_email                         float64
    host_verification_by_google                             float64
    host_verification_by_jumio                              float64
    host_verification_by_government_id                      float64
    host_verification_by_facebook                           float64
    host_verification_by_offline_government_id              float64
    host_verification_by_selfie                             float64
    host_verification_by_identity_manual                    float64
    host_verification_by_manual_online                      float64
    host_verification_by_sent_id                            float64
    host_verification_by_manual_offline                     float64
    host_verification_by_None                               float64
    host_verification_by_weibo                              float64
    host_verification_by_sesame                             float64
    host_verification_by_sesame_offline                     float64
    host_verification_by_zhima_selfie                       float64
    description_contains_able                               float64
    description_contains_abode                              float64
    description_contains_abound                             float64
    description_contains_absolute                           float64
    description_contains_absolutely                         float64
    description_contains_abundance                          float64
    description_contains_abundant                           float64
    description_contains_academy                            float64
    description_contains_accent                             float64
    description_contains_accept                             float64
    description_contains_access                             float64
    description_contains_accessibility                      float64
    description_contains_accessible                         float64
    description_contains_accommodate                        float64
    description_contains_accommodation                      float64
    description_contains_account                            float64
    description_contains_acre                               float64
    description_contains_action                             float64
    description_contains_active                             float64
    description_contains_activity                           float64
    description_contains_actual                             float64
    description_contains_actually                           float64
    description_contains_addition                           float64
    description_contains_additional                         float64
    description_contains_additionally                       float64
    description_contains_address                            float64
    description_contains_adjacent                           float64
    description_contains_adjoin                             float64
    description_contains_adjust                             float64
    description_contains_adjustable                         float64
    description_contains_adorable                           float64
    description_contains_adult                              float64
    description_contains_advance                            float64
    description_contains_advantage                          float64
    description_contains_adventure                          float64
    description_contains_adventurer                         float64
    description_contains_advice                             float64
    description_contains_advise                             float64
    description_contains_aesthetic                          float64
    description_contains_affordable                         float64
    description_contains_african                            float64
    description_contains_afternoon                          float64
    description_contains_ahead                              float64
    description_contains_airport                            float64
    description_contains_airy                               float64
    description_contains_alarm                              float64
    description_contains_alcove                             float64
    description_contains_alike                              float64
    description_contains_alive                              float64
    description_contains_allergic                           float64
    description_contains_allergy                            float64
    description_contains_alley                              float64
    description_contains_allow                              float64
    description_contains_alphabet                           float64
    description_contains_alternate                          float64
    description_contains_alternative                        float64
    description_contains_amaze                              float64
    description_contains_amazingly                          float64
    description_contains_amazon                             float64
    description_contains_ambiance                           float64
    description_contains_amenity                            float64
    description_contains_america                            float64
    description_contains_american                           float64
    description_contains_ample                              float64
    description_contains_amsterdam                          float64
    description_contains_animal                             float64
    description_contains_answer                             float64
    description_contains_antique                            float64
    description_contains_apart                              float64
    description_contains_apartment                          float64
    description_contains_apollo                             float64
    description_contains_apple                              float64
    description_contains_appliance                          float64
    description_contains_apply                              float64
    description_contains_appoint                            float64
    description_contains_appreciate                         float64
    description_contains_approximately                      float64
    description_contains_april                              float64
    description_contains_architect                          float64
    description_contains_architectural                      float64
    description_contains_architecture                       float64
    description_contains_area                               float64
    description_contains_arise                              float64
    description_contains_armchair                           float64
    description_contains_armoire                            float64
    description_contains_army                               float64
    description_contains_arrange                            float64
    description_contains_arrangement                        float64
    description_contains_array                              float64
    description_contains_arrival                            float64
    description_contains_arrive                             float64
    description_contains_art                                float64
    description_contains_arthur                             float64
    description_contains_artist                             float64
    description_contains_artistic                           float64
    description_contains_artwork                            float64
    description_contains_asian                              float64
    description_contains_aside                              float64
    description_contains_assist                             float64
    description_contains_assistance                         float64
    description_contains_assure                             float64
    description_contains_astor                              float64
    description_contains_atlantic                           float64
    description_contains_atmosphere                         float64
    description_contains_attach                             float64
    description_contains_attention                          float64
    description_contains_attic                              float64
    description_contains_attraction                         float64
    description_contains_attractive                         float64
    description_contains_august                             float64
    description_contains_authentic                          float64
    description_contains_authority                          float64
    description_contains_avail                              float64
    description_contains_availability                       float64
    description_contains_available                          float64
    description_contains_avenue                             float64
    description_contains_average                            float64
    description_contains_avoid                              float64
    description_contains_await                              float64
    description_contains_award                              float64
    description_contains_aware                              float64
    description_contains_away                               float64
    description_contains_awesome                            float64
    description_contains_baby                               float64
    description_contains_background                         float64
    description_contains_backyard                           float64
    description_contains_bagel                              float64
    description_contains_bakery                             float64
    description_contains_balance                            float64
    description_contains_balcony                            float64
    description_contains_bamboo                             float64
    description_contains_bank                               float64
    description_contains_bar                                float64
    description_contains_barbecue                           float64
    description_contains_barrio                             float64
    description_contains_base                               float64
    description_contains_baseball                           float64
    description_contains_basement                           float64
    description_contains_basic                              float64
    description_contains_basically                          float64
    description_contains_basis                              float64
    description_contains_basketball                         float64
    description_contains_bath                               float64
    description_contains_bathroom                           float64
    description_contains_bathtub                            float64
    description_contains_battery                            float64
    description_contains_beach                              float64
    description_contains_beam                               float64
    description_contains_beat                               float64
    description_contains_beautiful                          float64
    description_contains_beautifully                        float64
    description_contains_beauty                             float64
    description_contains_bedding                            float64
    description_contains_bedroom                            float64
    description_contains_bedside                            float64
    description_contains_beer                               float64
    description_contains_beginning                          float64
    description_contains_believe                            float64
    description_contains_bell                               float64
    description_contains_belonging                          float64
    description_contains_beloved                            float64
    description_contains_bench                              float64
    description_contains_benefit                            float64
    description_contains_bespeak                            float64
    description_contains_best                               float64
    description_contains_better                             float64
    description_contains_bicycle                            float64
    description_contains_bigger                             float64
    description_contains_bike                               float64
    description_contains_bird                               float64
    description_contains_bistro                             float64
    description_contains_bite                               float64
    description_contains_black                              float64
    description_contains_blackout                           float64
    description_contains_blanket                            float64
    description_contains_blend                              float64
    description_contains_blender                            float64
    description_contains_blind                              float64
    description_contains_block                              float64
    description_contains_blow                               float64
    description_contains_blue                               float64
    description_contains_board                              float64
    description_contains_boardwalk                          float64
    description_contains_boast                              float64
    description_contains_boat                               float64
    description_contains_bodega                             float64
    description_contains_body                               float64
    description_contains_bohemian                           float64
    description_contains_bonus                              float64
    description_contains_book                               float64
    description_contains_booking                            float64
    description_contains_bookshelf                          float64
    description_contains_bookstore                          float64
    description_contains_border                             float64
    description_contains_borough                            float64
    description_contains_bosch                              float64
    description_contains_bose                               float64
    description_contains_botanic                            float64
    description_contains_botanical                          float64
    description_contains_bother                             float64
    description_contains_bottle                             float64
    description_contains_boulevard                          float64
    description_contains_boutique                           float64
    description_contains_bowery                             float64
    description_contains_bowl                               float64
    description_contains_boyfriend                          float64
    description_contains_brand                              float64
    description_contains_break                              float64
    description_contains_breakfast                          float64
    description_contains_breath                             float64
    description_contains_breathtaking                       float64
    description_contains_breeze                             float64
    description_contains_brewery                            float64
    description_contains_brick                              float64
    description_contains_bridge                             float64
    description_contains_bright                             float64
    description_contains_brighton                           float64
    description_contains_bring                              float64
    description_contains_broad                              float64
    description_contains_broadway                           float64
    description_contains_bronx                              float64
    description_contains_brooklyn                           float64
    description_contains_brother                            float64
    description_contains_brown                              float64
    description_contains_brownstone                         float64
    description_contains_brunch                             float64
    description_contains_brush                              float64
    description_contains_budget                             float64
    description_contains_build                              float64
    description_contains_building                           float64
    description_contains_bunch                              float64
    description_contains_bunk                               float64
    description_contains_burger                             float64
    description_contains_burner                             float64
    description_contains_burning                            float64
    description_contains_business                           float64
    description_contains_bustle                             float64
    description_contains_busy                               float64
    description_contains_buzz                               float64
    description_contains_cabinet                            float64
    description_contains_cabinetry                          float64
    description_contains_cable                              float64
    description_contains_cafe                               float64
    description_contains_calendar                           float64
    description_contains_california                         float64
    description_contains_calm                               float64
    description_contains_camera                             float64
    description_contains_campus                             float64
    description_contains_canal                              float64
    description_contains_candle                             float64
    description_contains_card                               float64
    description_contains_care                               float64
    description_contains_carefully                          float64
    description_contains_caribbean                          float64
    description_contains_carnegie                           float64
    description_contains_carpet                             float64
    description_contains_carroll                            float64
    description_contains_carry                              float64
    description_contains_case                               float64
    description_contains_cash                               float64
    description_contains_casino                             float64
    description_contains_casper                             float64
    description_contains_cast                               float64
    description_contains_casual                             float64
    description_contains_catch                              float64
    description_contains_cater                              float64
    description_contains_cathedral                          float64
    description_contains_ceiling                            float64
    description_contains_celebrity                          float64
    description_contains_cell                               float64
    description_contains_cemetery                           float64
    description_contains_center                             float64
    description_contains_central                            float64
    description_contains_centrally                          float64
    description_contains_centre                             float64
    description_contains_century                            float64
    description_contains_cereal                             float64
    description_contains_certain                            float64
    description_contains_certainly                          float64
    description_contains_chain                              float64
    description_contains_chair                              float64
    description_contains_chaise                             float64
    description_contains_chance                             float64
    description_contains_change                             float64
    description_contains_channel                            float64
    description_contains_character                          float64
    description_contains_charge                             float64
    description_contains_charm                              float64
    description_contains_chase                              float64
    description_contains_chat                               float64
    description_contains_cheap                              float64
    description_contains_check                              float64
    description_contains_checkout                           float64
    description_contains_cheerful                           float64
    description_contains_cheery                             float64
    description_contains_cheese                             float64
    description_contains_chef                               float64
    description_contains_cherry                             float64
    description_contains_chest                              float64
    description_contains_chic                               float64
    description_contains_chicken                            float64
    description_contains_child                              float64
    description_contains_chill                              float64
    description_contains_china                              float64
    description_contains_chinese                            float64
    description_contains_chirp                              float64
    description_contains_choice                             float64
    description_contains_choose                             float64
    description_contains_christmas                          float64
    description_contains_christopher                        float64
    description_contains_church                             float64
    description_contains_cinema                             float64
    description_contains_circle                             float64
    description_contains_city                               float64
    description_contains_class                              float64
    description_contains_classic                            float64
    description_contains_clean                              float64
    description_contains_cleaner                            float64
    description_contains_cleaning                           float64
    description_contains_cleanliness                        float64
    description_contains_clear                              float64
    description_contains_click                              float64
    description_contains_climb                              float64
    description_contains_clinton                            float64
    description_contains_clock                              float64
    description_contains_cloister                           float64
    description_contains_close                              float64
    description_contains_closer                             float64
    description_contains_closet                             float64
    description_contains_clothes                            float64
    description_contains_clothing                           float64
    description_contains_club                               float64
    description_contains_clutter                            float64
    description_contains_coat                               float64
    description_contains_cobble                             float64
    description_contains_cobblestone                        float64
    description_contains_cocktail                           float64
    description_contains_code                               float64
    description_contains_coffee                             float64
    description_contains_coin                               float64
    description_contains_cold                               float64
    description_contains_collect                            float64
    description_contains_collection                         float64
    description_contains_college                            float64
    description_contains_color                              float64
    description_contains_colorful                           float64
    description_contains_columbia                           float64
    description_contains_columbus                           float64
    description_contains_combination                        float64
    description_contains_combine                            float64
    description_contains_combo                              float64
    description_contains_come                               float64
    description_contains_comedy                             float64
    description_contains_comfort                            float64
    description_contains_comfortable                        float64
    description_contains_comfortably                        float64
    description_contains_comforter                          float64
    description_contains_comfy                              float64
    description_contains_coming                             float64
    description_contains_commercial                         float64
    description_contains_commitment                         float64
    description_contains_common                             float64
    description_contains_communal                           float64
    description_contains_communicate                        float64
    description_contains_communication                      float64
    description_contains_community                          float64
    description_contains_commute                            float64
    description_contains_commuting                          float64
    description_contains_compact                            float64
    description_contains_company                            float64
    description_contains_compare                            float64
    description_contains_complementary                      float64
    description_contains_complete                           float64
    description_contains_completely                         float64
    description_contains_complex                            float64
    description_contains_complimentary                      float64
    description_contains_comprise                           float64
    description_contains_computer                           float64
    description_contains_concept                            float64
    description_contains_concern                            float64
    description_contains_concert                            float64
    description_contains_concierge                          float64
    description_contains_concrete                           float64
    description_contains_condiment                          float64
    description_contains_condition                          float64
    description_contains_conditioner                        float64
    description_contains_conditioning                       float64
    description_contains_condo                              float64
    description_contains_condominium                        float64
    description_contains_coney                              float64
    description_contains_confirm                            float64
    description_contains_connect                            float64
    description_contains_connection                         float64
    description_contains_consider                           float64
    description_contains_considerate                        float64
    description_contains_consist                            float64
    description_contains_construct                          float64
    description_contains_construction                       float64
    description_contains_contact                            float64
    description_contains_contain                            float64
    description_contains_contemporary                       float64
    description_contains_continue                           float64
    description_contains_control                            float64
    description_contains_convenience                        float64
    description_contains_convenient                         float64
    description_contains_conveniently                       float64
    description_contains_conversation                       float64
    description_contains_convert                            float64
    description_contains_convertible                        float64
    description_contains_cook                               float64
    description_contains_cooker                             float64
    description_contains_cooking                            float64
    description_contains_cookware                           float64
    description_contains_cool                               float64
    description_contains_cooler                             float64
    description_contains_cooling                            float64
    description_contains_coordinate                         float64
    description_contains_corner                             float64
    description_contains_cost                               float64
    description_contains_cosy                               float64
    description_contains_cotton                             float64
    description_contains_couch                              float64
    description_contains_count                              float64
    description_contains_counter                            float64
    description_contains_countertop                         float64
    description_contains_countless                          float64
    description_contains_country                            float64
    description_contains_county                             float64
    description_contains_couple                             float64
    description_contains_course                             float64
    description_contains_court                              float64
    description_contains_courtyard                          float64
    description_contains_cover                              float64
    description_contains_covet                              float64
    description_contains_coziness                           float64
    description_contains_cozy                               float64
    description_contains_craft                              float64
    description_contains_crash                              float64
    description_contains_crazy                              float64
    description_contains_cream                              float64
    description_contains_create                             float64
    description_contains_creative                           float64
    description_contains_crib                               float64
    description_contains_cross                              float64
    description_contains_crosstown                          float64
    description_contains_crowd                              float64
    description_contains_crown                              float64
    description_contains_cuisine                            float64
    description_contains_culinary                           float64
    description_contains_cultural                           float64
    description_contains_culturally                         float64
    description_contains_culture                            float64
    description_contains_cupboard                           float64
    description_contains_current                            float64
    description_contains_currently                          float64
    description_contains_curtain                            float64
    description_contains_custom                             float64
    description_contains_cute                               float64
    description_contains_cutlery                            float64
    description_contains_daily                              float64
    description_contains_dance                              float64
    description_contains_dark                               float64
    description_contains_date                               float64
    description_contains_daughter                           float64
    description_contains_day                                float64
    description_contains_daybed                             float64
    description_contains_daylight                           float64
    description_contains_daytime                            float64
    description_contains_dead                               float64
    description_contains_deal                               float64
    description_contains_dear                               float64
    description_contains_december                           float64
    description_contains_decent                             float64
    description_contains_decide                             float64
    description_contains_deck                               float64
    description_contains_deco                               float64
    description_contains_decor                              float64
    description_contains_decorate                           float64
    description_contains_decoration                         float64
    description_contains_decorative                         float64
    description_contains_dedicate                           float64
    description_contains_deep                               float64
    description_contains_definitely                         float64
    description_contains_definition                         float64
    description_contains_degree                             float64
    description_contains_deli                               float64
    description_contains_delicious                          float64
    description_contains_delight                            float64
    description_contains_delightful                         float64
    description_contains_deliver                            float64
    description_contains_delivery                           float64
    description_contains_deluxe                             float64
    description_contains_demand                             float64
    description_contains_department                         float64
    description_contains_departure                          float64
    description_contains_depend                             float64
    description_contains_deposit                            float64
    description_contains_description                        float64
    description_contains_design                             float64
    description_contains_designate                          float64
    description_contains_designer                           float64
    description_contains_desirable                          float64
    description_contains_desire                             float64
    description_contains_desk                               float64
    description_contains_despite                            float64
    description_contains_destination                        float64
    description_contains_detergent                          float64
    description_contains_device                             float64
    description_contains_different                          float64
    description_contains_difficult                          float64
    description_contains_digital                            float64
    description_contains_dine                               float64
    description_contains_diner                              float64
    description_contains_dining                             float64
    description_contains_dinner                             float64
    description_contains_direct                             float64
    description_contains_direction                          float64
    description_contains_directly                           float64
    description_contains_discount                           float64
    description_contains_discover                           float64
    description_contains_discus                             float64
    description_contains_dish                               float64
    description_contains_dishwasher                         float64
    description_contains_disposal                           float64
    description_contains_distance                           float64
    description_contains_district                           float64
    description_contains_dive                               float64
    description_contains_diverse                            float64
    description_contains_diversity                          float64
    description_contains_divide                             float64
    description_contains_divine                             float64
    description_contains_dock                               float64
    description_contains_dollar                             float64
    description_contains_dominican                          float64
    description_contains_domino                             float64
    description_contains_donut                              float64
    description_contains_door                               float64
    description_contains_doorman                            float64
    description_contains_doorstep                           float64
    description_contains_double                             float64
    description_contains_downstairs                         float64
    description_contains_downtown                           float64
    description_contains_dozen                              float64
    description_contains_draw                               float64
    description_contains_drawer                             float64
    description_contains_dream                              float64
    description_contains_drench                             float64
    description_contains_dresser                            float64
    description_contains_dressing                           float64
    description_contains_drier                              float64
    description_contains_drink                              float64
    description_contains_drinking                           float64
    description_contains_drive                              float64
    description_contains_driveway                           float64
    description_contains_driving                            float64
    description_contains_drop                               float64
    description_contains_drug                               float64
    description_contains_drugstore                          float64
    description_contains_dryer                              float64
    description_contains_duplex                             float64
    description_contains_duration                           float64
    description_contains_duvet                              float64
    description_contains_dynamic                            float64
    description_contains_early                              float64
    description_contains_earn                               float64
    description_contains_earth                              float64
    description_contains_ease                               float64
    description_contains_easily                             float64
    description_contains_east                               float64
    description_contains_eastern                            float64
    description_contains_easy                               float64
    description_contains_eatery                             float64
    description_contains_eating                             float64
    description_contains_eats                               float64
    description_contains_eclectic                           float64
    description_contains_edge                               float64
    description_contains_efficient                          float64
    description_contains_electric                           float64
    description_contains_electricity                        float64
    description_contains_electronic                         float64
    description_contains_elegant                            float64
    description_contains_elegantly                          float64
    description_contains_elevated                           float64
    description_contains_elevator                           float64
    description_contains_email                              float64
    description_contains_emergency                          float64
    description_contains_empire                             float64
    description_contains_enclave                            float64
    description_contains_enclose                            float64
    description_contains_encourage                          float64
    description_contains_endless                            float64
    description_contains_energy                             float64
    description_contains_english                            float64
    description_contains_enjoy                              float64
    description_contains_enjoyable                          float64
    description_contains_enjoyment                          float64
    description_contains_enormous                           float64
    description_contains_ensure                             float64
    description_contains_enter                              float64
    description_contains_entering                           float64
    description_contains_entertain                          float64
    description_contains_entertainment                      float64
    description_contains_entire                             float64
    description_contains_entirely                           float64
    description_contains_entrance                           float64
    description_contains_entrepreneur                       float64
    description_contains_entry                              float64
    description_contains_environment                        float64
    description_contains_epicenter                          float64
    description_contains_equinox                            float64
    description_contains_equip                              float64
    description_contains_equipment                          float64
    description_contains_escape                             float64
    description_contains_especially                         float64
    description_contains_espresso                           float64
    description_contains_essential                          float64
    description_contains_essex                              float64
    description_contains_establishment                      float64
    description_contains_estate                             float64
    description_contains_ethnic                             float64
    description_contains_europe                             float64
    description_contains_european                           float64
    description_contains_evening                            float64
    description_contains_event                              float64
    description_contains_everyday                           float64
    description_contains_exact                              float64
    description_contains_exactly                            float64
    description_contains_excellent                          float64
    description_contains_exception                          float64
    description_contains_exceptional                        float64
    description_contains_exceptionally                      float64
    description_contains_exchange                           float64
    description_contains_excite                             float64
    description_contains_excitement                         float64
    description_contains_exclusive                          float64
    description_contains_exclusively                        float64
    description_contains_exercise                           float64
    description_contains_exit                               float64
    description_contains_expansive                          float64
    description_contains_expect                             float64
    description_contains_expensive                          float64
    description_contains_experience                         float64
    description_contains_explore                            float64
    description_contains_expose                             float64
    description_contains_exposure                           float64
    description_contains_express                            float64
    description_contains_expressway                         float64
    description_contains_exquisite                          float64
    description_contains_extend                             float64
    description_contains_extensive                          float64
    description_contains_extra                              float64
    description_contains_extremely                          float64
    description_contains_eye                                float64
    description_contains_fabulous                           float64
    description_contains_face                               float64
    description_contains_facility                           float64
    description_contains_facing                             float64
    description_contains_fact                               float64
    description_contains_factory                            float64
    description_contains_fair                               float64
    description_contains_fairly                             float64
    description_contains_fairway                            float64
    description_contains_fall                               float64
    description_contains_famed                              float64
    description_contains_familiar                           float64
    description_contains_family                             float64
    description_contains_famous                             float64
    description_contains_fancy                              float64
    description_contains_fantastic                          float64
    description_contains_farm                               float64
    description_contains_farmer                             float64
    description_contains_fashion                            float64
    description_contains_fashionable                        float64
    description_contains_fast                               float64
    description_contains_favorite                           float64
    description_contains_feature                            float64
    description_contains_feed                               float64
    description_contains_feel                               float64
    description_contains_feeling                            float64
    description_contains_fellow                             float64
    description_contains_female                             float64
    description_contains_ferry                              float64
    description_contains_field                              float64
    description_contains_fifth                              float64
    description_contains_film                               float64
    description_contains_filter                             float64
    description_contains_financial                          float64
    description_contains_fine                               float64
    description_contains_fingertip                          float64
    description_contains_finish                             float64
    description_contains_fireplace                          float64
    description_contains_firm                               float64
    description_contains_fish                               float64
    description_contains_fitness                            float64
    description_contains_fixture                            float64
    description_contains_flat                               float64
    description_contains_flatiron                           float64
    description_contains_flavor                             float64
    description_contains_flea                               float64
    description_contains_flexible                           float64
    description_contains_flight                             float64
    description_contains_flood                              float64
    description_contains_floor                              float64
    description_contains_flooring                           float64
    description_contains_flower                             float64
    description_contains_flush                              float64
    description_contains_foam                               float64
    description_contains_focus                              float64
    description_contains_fold                               float64
    description_contains_folding                            float64
    description_contains_folk                               float64
    description_contains_follow                             float64
    description_contains_following                          float64
    description_contains_food                               float64
    description_contains_foodie                             float64
    description_contains_foot                               float64
    description_contains_footstep                           float64
    description_contains_forest                             float64
    description_contains_forget                             float64
    description_contains_form                               float64
    description_contains_formal                             float64
    description_contains_fort                               float64
    description_contains_forward                            float64
    description_contains_fourth                             float64
    description_contains_foyer                              float64
    description_contains_frame                              float64
    description_contains_franklin                           float64
    description_contains_free                               float64
    description_contains_freedom                            float64
    description_contains_freezer                            float64
    description_contains_french                             float64
    description_contains_frequently                         float64
    description_contains_fresh                              float64
    description_contains_freshly                            float64
    description_contains_friday                             float64
    description_contains_fridge                             float64
    description_contains_friend                             float64
    description_contains_friendly                           float64
    description_contains_fruit                              float64
    description_contains_fully                              float64
    description_contains_fulton                             float64
    description_contains_functional                         float64
    description_contains_functioning                        float64
    description_contains_funky                              float64
    description_contains_furnish                            float64
    description_contains_furnishing                         float64
    description_contains_furniture                          float64
    description_contains_furry                              float64
    description_contains_futon                              float64
    description_contains_gallery                            float64
    description_contains_galore                             float64
    description_contains_game                               float64
    description_contains_garage                             float64
    description_contains_garbage                            float64
    description_contains_garden                             float64
    description_contains_garment                            float64
    description_contains_gate                               float64
    description_contains_gateway                            float64
    description_contains_gathering                          float64
    description_contains_general                            float64
    description_contains_generally                          float64
    description_contains_generous                           float64
    description_contains_george                             float64
    description_contains_getaway                            float64
    description_contains_getting                            float64
    description_contains_giant                              float64
    description_contains_gigantic                           float64
    description_contains_girl                               float64
    description_contains_girlfriend                         float64
    description_contains_given                              float64
    description_contains_giving                             float64
    description_contains_glad                               float64
    description_contains_gladly                             float64
    description_contains_glass                              float64
    description_contains_glassware                          float64
    description_contains_globe                              float64
    description_contains_goal                               float64
    description_contains_going                              float64
    description_contains_good                               float64
    description_contains_gorgeous                           float64
    description_contains_gourmet                            float64
    description_contains_grab                               float64
    description_contains_graffiti                           float64
    description_contains_graham                             float64
    description_contains_grand                              float64
    description_contains_granite                            float64
    description_contains_grant                              float64
    description_contains_great                              float64
    description_contains_greatest                           float64
    description_contains_greek                              float64
    description_contains_green                              float64
    description_contains_greene                             float64
    description_contains_greenery                           float64
    description_contains_greenwich                          float64
    description_contains_greenwood                          float64
    description_contains_greet                              float64
    description_contains_greeting                           float64
    description_contains_grill                              float64
    description_contains_grilling                           float64
    description_contains_grocery                            float64
    description_contains_grog                               float64
    description_contains_ground                             float64
    description_contains_group                              float64
    description_contains_grow                               float64
    description_contains_growing                            float64
    description_contains_guarantee                          float64
    description_contains_guess                              float64
    description_contains_guest                              float64
    description_contains_guggenheim                         float64
    description_contains_guide                              float64
    description_contains_guidebook                          float64
    description_contains_guitar                             float64
    description_contains_hair                               float64
    description_contains_half                               float64
    description_contains_hall                               float64
    description_contains_hallway                            float64
    description_contains_hamilton                           float64
    description_contains_hammock                            float64
    description_contains_hand                               float64
    description_contains_handle                             float64
    description_contains_handmade                           float64
    description_contains_hang                               float64
    description_contains_hanger                             float64
    description_contains_hanging                            float64
    description_contains_hangout                            float64
    description_contains_happen                             float64
    description_contains_happening                          float64
    description_contains_happily                            float64
    description_contains_happy                              float64
    description_contains_harbor                             float64
    description_contains_hard                               float64
    description_contains_hardly                             float64
    description_contains_hardware                           float64
    description_contains_hardwood                           float64
    description_contains_harlem                             float64
    description_contains_hassle                             float64
    description_contains_hdtv                               float64
    description_contains_head                               float64
    description_contains_health                             float64
    description_contains_healthy                            float64
    description_contains_hear                               float64
    description_contains_heart                              float64
    description_contains_heat                               float64
    description_contains_heater                             float64
    description_contains_heating                            float64
    description_contains_heavy                              float64
    description_contains_hectic                             float64
    description_contains_height                             float64
    description_contains_hell                               float64
    description_contains_hello                              float64
    description_contains_help                               float64
    description_contains_helpful                            float64
    description_contains_helping                            float64
    description_contains_herald                             float64
    description_contains_herb                               float64
    description_contains_hesitate                           float64
    description_contains_hide                               float64
    description_contains_hideaway                           float64
    description_contains_high                               float64
    description_contains_highlight                          float64
    description_contains_highly                             float64
    description_contains_highway                            float64
    description_contains_hill                               float64
    description_contains_hipster                            float64
    description_contains_historic                           float64
    description_contains_historical                         float64
    description_contains_history                            float64
    description_contains_hold                               float64
    description_contains_holiday                            float64
    description_contains_home                               float64
    description_contains_homely                             float64
    description_contains_homey                              float64
    description_contains_hood                               float64
    description_contains_hook                               float64
    description_contains_hope                               float64
    description_contains_hospital                           float64
    description_contains_hospitality                        float64
    description_contains_host                               float64
    description_contains_hostel                             float64
    description_contains_hotel                              float64
    description_contains_hotspot                            float64
    description_contains_hour                               float64
    description_contains_house                              float64
    description_contains_household                          float64
    description_contains_housekeeping                       float64
    description_contains_housemate                          float64
    description_contains_housing                            float64
    description_contains_houston                            float64
    description_contains_hudson                             float64
    description_contains_huge                               float64
    description_contains_hunt                               float64
    description_contains_husband                            float64
    description_contains_hustle                             float64
    description_contains_iconic                             float64
    description_contains_idea                               float64
    description_contains_ideal                              float64
    description_contains_ideally                            float64
    description_contains_image                              float64
    description_contains_imagine                            float64
    description_contains_immaculate                         float64
    description_contains_immediate                          float64
    description_contains_immediately                        float64
    description_contains_important                          float64
    description_contains_importantly                        float64
    description_contains_inch                               float64
    description_contains_include                            float64
    description_contains_incredible                         float64
    description_contains_incredibly                         float64
    description_contains_independent                        float64
    description_contains_indian                             float64
    description_contains_individual                         float64
    description_contains_indoor                             float64
    description_contains_industrial                         float64
    description_contains_industry                           float64
    description_contains_inexpensive                        float64
    description_contains_inflatable                         float64
    description_contains_info                               float64
    description_contains_information                        float64
    description_contains_inquire                            float64
    description_contains_inquiry                            float64
    description_contains_inside                             float64
    description_contains_inspire                            float64
    description_contains_instal                             float64
    description_contains_instant                            float64
    description_contains_instead                            float64
    description_contains_institute                          float64
    description_contains_instruction                        float64
    description_contains_interact                           float64
    description_contains_interaction                        float64
    description_contains_intercom                           float64
    description_contains_interior                           float64
    description_contains_intern                             float64
    description_contains_international                      float64
    description_contains_internet                           float64
    description_contains_intersection                       float64
    description_contains_intimate                           float64
    description_contains_invite                             float64
    description_contains_irish                              float64
    description_contains_iron                               float64
    description_contains_ironing                            float64
    description_contains_island                             float64
    description_contains_issue                              float64
    description_contains_italian                            float64
    description_contains_italy                              float64
    description_contains_item                               float64
    description_contains_jackson                            float64
    description_contains_jamaica                            float64
    description_contains_jamaican                           float64
    description_contains_january                            float64
    description_contains_japanese                           float64
    description_contains_jazz                               float64
    description_contains_jefferson                          float64
    description_contains_jersey                             float64
    description_contains_jewish                             float64
    description_contains_jogging                            float64
    description_contains_john                               float64
    description_contains_join                               float64
    description_contains_journal                            float64
    description_contains_juice                              float64
    description_contains_juicer                             float64
    description_contains_july                               float64
    description_contains_jump                               float64
    description_contains_junction                           float64
    description_contains_june                               float64
    description_contains_jungle                             float64
    description_contains_junior                             float64
    description_contains_keeping                            float64
    description_contains_kennedy                            float64
    description_contains_kettle                             float64
    description_contains_keyboard                           float64
    description_contains_keyless                            float64
    description_contains_keypad                             float64
    description_contains_kick                               float64
    description_contains_killer                             float64
    description_contains_kind                               float64
    description_contains_kindly                             float64
    description_contains_king                               float64
    description_contains_kingston                           float64
    description_contains_kitchen                            float64
    description_contains_kitchenette                        float64
    description_contains_kitchenware                        float64
    description_contains_kitty                              float64
    description_contains_know                               float64
    description_contains_knowing                            float64
    description_contains_knowledge                          float64
    description_contains_korean                             float64
    description_contains_kosher                             float64
    description_contains_lady                               float64
    description_contains_lafayette                          float64
    description_contains_lake                               float64
    description_contains_lamp                               float64
    description_contains_land                               float64
    description_contains_landlord                           float64
    description_contains_landmark                           float64
    description_contains_landscape                          float64
    description_contains_laptop                             float64
    description_contains_large                              float64
    description_contains_larger                             float64
    description_contains_late                               float64
    description_contains_later                              float64
    description_contains_latest                             float64
    description_contains_latin                              float64
    description_contains_launder                            float64
    description_contains_laundromat                         float64
    description_contains_laundry                            float64
    description_contains_layout                             float64
    description_contains_lead                               float64
    description_contains_leading                            float64
    description_contains_leaf                               float64
    description_contains_leafy                              float64
    description_contains_lease                              float64
    description_contains_leather                            float64
    description_contains_leave                              float64
    description_contains_leaving                            float64
    description_contains_left                               float64
    description_contains_legal                              float64
    description_contains_legendary                          float64
    description_contains_leisure                            float64
    description_contains_length                             float64
    description_contains_level                              float64
    description_contains_lexington                          float64
    description_contains_liberty                            float64
    description_contains_library                            float64
    description_contains_life                               float64
    description_contains_lifestyle                          float64
    description_contains_light                              float64
    description_contains_lighting                           float64
    description_contains_like                               float64
    description_contains_likely                             float64
    description_contains_limestone                          float64
    description_contains_limit                              float64
    description_contains_limited                            float64
    description_contains_lincoln                            float64
    description_contains_line                               float64
    description_contains_linen                              float64
    description_contains_link                               float64
    description_contains_liquor                             float64
    description_contains_list                               float64
    description_contains_listen                             float64
    description_contains_listing                            float64
    description_contains_literally                          float64
    description_contains_little                             float64
    description_contains_live                               float64
    description_contains_lively                             float64
    description_contains_living                             float64
    description_contains_load                               float64
    description_contains_lobby                              float64
    description_contains_local                              float64
    description_contains_locally                            float64
    description_contains_locate                             float64
    description_contains_location                           float64
    description_contains_lock                               float64
    description_contains_lockbox                            float64
    description_contains_loft                               float64
    description_contains_long                               float64
    description_contains_longer                             float64
    description_contains_look                               float64
    description_contains_looking                            float64
    description_contains_lost                               float64
    description_contains_lot                                float64
    description_contains_loud                               float64
    description_contains_lounge                             float64
    description_contains_love                               float64
    description_contains_lovely                             float64
    description_contains_lover                              float64
    description_contains_lovingly                           float64
    description_contains_lower                              float64
    description_contains_lucky                              float64
    description_contains_luggage                            float64
    description_contains_lunch                              float64
    description_contains_lush                               float64
    description_contains_luxurious                          float64
    description_contains_luxury                             float64
    description_contains_machine                            float64
    description_contains_madison                            float64
    description_contains_magazine                           float64
    description_contains_magical                            float64
    description_contains_magnificent                        float64
    description_contains_mail                               float64
    description_contains_main                               float64
    description_contains_mainly                             float64
    description_contains_maintain                           float64
    description_contains_maintenance                        float64
    description_contains_major                              float64
    description_contains_majority                           float64
    description_contains_make                               float64
    description_contains_maker                              float64
    description_contains_making                             float64
    description_contains_male                               float64
    description_contains_mall                               float64
    description_contains_manage                             float64
    description_contains_management                         float64
    description_contains_manager                            float64
    description_contains_manhattan                          float64
    description_contains_mansion                            float64
    description_contains_marble                             float64
    description_contains_march                              float64
    description_contains_mark                               float64
    description_contains_market                             float64
    description_contains_mass                               float64
    description_contains_massage                            float64
    description_contains_massive                            float64
    description_contains_master                             float64
    description_contains_match                              float64
    description_contains_mate                               float64
    description_contains_matter                             float64
    description_contains_mattress                           float64
    description_contains_mature                             float64
    description_contains_maximum                            float64
    description_contains_maybe                              float64
    description_contains_meadow                             float64
    description_contains_meal                               float64
    description_contains_mean                               float64
    description_contains_meaning                            float64
    description_contains_meat                               float64
    description_contains_meatpacking                        float64
    description_contains_medical                            float64
    description_contains_medium                             float64
    description_contains_meet                               float64
    description_contains_meeting                            float64
    description_contains_member                             float64
    description_contains_memorable                          float64
    description_contains_memorial                           float64
    description_contains_memory                             float64
    description_contains_mention                            float64
    description_contains_mere                               float64
    description_contains_message                            float64
    description_contains_messaging                          float64
    description_contains_messenger                          float64
    description_contains_meter                              float64
    description_contains_metro                              float64
    description_contains_metropolitan                       float64
    description_contains_mexican                            float64
    description_contains_microwave                          float64
    description_contains_middle                             float64
    description_contains_midnight                           float64
    description_contains_mile                               float64
    description_contains_milk                               float64
    description_contains_million                            float64
    description_contains_mind                               float64
    description_contains_mindful                            float64
    description_contains_mini                               float64
    description_contains_minimal                            float64
    description_contains_minimalist                         float64
    description_contains_minimum                            float64
    description_contains_minuet                             float64
    description_contains_minute                             float64
    description_contains_mirror                             float64
    description_contains_miss                               float64
    description_contains_mobile                             float64
    description_contains_modern                             float64
    description_contains_molding                            float64
    description_contains_moment                             float64
    description_contains_monday                             float64
    description_contains_money                              float64
    description_contains_monitor                            float64
    description_contains_month                              float64
    description_contains_monthly                            float64
    description_contains_mood                               float64
    description_contains_morgan                             float64
    description_contains_morning                            float64
    description_contains_morris                             float64
    description_contains_mother                             float64
    description_contains_mount                              float64
    description_contains_movie                              float64
    description_contains_multicultural                      float64
    description_contains_multiple                           float64
    description_contains_mural                              float64
    description_contains_murphy                             float64
    description_contains_murray                             float64
    description_contains_museum                             float64
    description_contains_music                              float64
    description_contains_musical                            float64
    description_contains_musician                           float64
    description_contains_myrtle                             float64
    description_contains_nail                               float64
    description_contains_nassau                             float64
    description_contains_nation                             float64
    description_contains_national                           float64
    description_contains_native                             float64
    description_contains_natural                            float64
    description_contains_nature                             float64
    description_contains_navigate                           float64
    description_contains_navy                               float64
    description_contains_near                               float64
    description_contains_nearby                             float64
    description_contains_nearly                             float64
    description_contains_neat                               float64
    description_contains_necessary                          float64
    description_contains_necessity                          float64
    description_contains_need                               float64
    description_contains_neighbor                           float64
    description_contains_neighborhood                       float64
    description_contains_neighbourhood                      float64
    description_contains_nest                               float64
    description_contains_nestle                             float64
    description_contains_network                            float64
    description_contains_newark                             float64
    description_contains_newly                              float64
    description_contains_nice                               float64
    description_contains_nicely                             float64
    description_contains_nicholas                           float64
    description_contains_night                              float64
    description_contains_nightlife                          float64
    description_contains_noise                              float64
    description_contains_noisy                              float64
    description_contains_nomad                              float64
    description_contains_nook                               float64
    description_contains_normal                             float64
    description_contains_normally                           float64
    description_contains_north                              float64
    description_contains_northern                           float64
    description_contains_note                               float64
    description_contains_notice                             float64
    description_contains_number                             float64
    description_contains_numerous                           float64
    description_contains_oasis                              float64
    description_contains_occasional                         float64
    description_contains_occasionally                       float64
    description_contains_occupancy                          float64
    description_contains_occupant                           float64
    description_contains_occupy                             float64
    description_contains_ocean                              float64
    description_contains_offer                              float64
    description_contains_offering                           float64
    description_contains_office                             float64
    description_contains_okay                               float64
    description_contains_older                              float64
    description_contains_open                               float64
    description_contains_opening                            float64
    description_contains_operate                            float64
    description_contains_opportunity                        float64
    description_contains_opposite                           float64
    description_contains_option                             float64
    description_contains_optional                           float64
    description_contains_orange                             float64
    description_contains_order                              float64
    description_contains_organic                            float64
    description_contains_organize                           float64
    description_contains_orient                             float64
    description_contains_original                           float64
    description_contains_ottoman                            float64
    description_contains_outdoor                            float64
    description_contains_outdoors                           float64
    description_contains_outfit                             float64
    description_contains_outlet                             float64
    description_contains_outpost                            float64
    description_contains_outside                            float64
    description_contains_outstanding                        float64
    description_contains_oven                               float64
    description_contains_overall                            float64
    description_contains_overdue                            float64
    description_contains_overlook                           float64
    description_contains_overnight                          float64
    description_contains_oversized                          float64
    description_contains_owner                              float64
    description_contains_pace                               float64
    description_contains_pack                               float64
    description_contains_package                            float64
    description_contains_packing                            float64
    description_contains_paint                              float64
    description_contains_painting                           float64
    description_contains_panel                              float64
    description_contains_panoramic                          float64
    description_contains_pantry                             float64
    description_contains_paper                              float64
    description_contains_para                               float64
    description_contains_paradise                           float64
    description_contains_parent                             float64
    description_contains_park                               float64
    description_contains_parking                            float64
    description_contains_parkway                            float64
    description_contains_parlor                             float64
    description_contains_parquet                            float64
    description_contains_partial                            float64
    description_contains_partner                            float64
    description_contains_party                              float64
    description_contains_pas                                float64
    description_contains_password                           float64
    description_contains_past                               float64
    description_contains_pastry                             float64
    description_contains_path                               float64
    description_contains_patio                              float64
    description_contains_peace                              float64
    description_contains_peaceful                           float64
    description_contains_peach                              float64
    description_contains_penn                               float64
    description_contains_penthouse                          float64
    description_contains_people                             float64
    description_contains_perfect                            float64
    description_contains_perfectly                          float64
    description_contains_performance                        float64
    description_contains_period                             float64
    description_contains_perk                               float64
    description_contains_permanent                          float64
    description_contains_permit                             float64
    description_contains_person                             float64
    description_contains_personal                           float64
    description_contains_personality                        float64
    description_contains_personally                         float64
    description_contains_pharmacy                           float64
    description_contains_phone                              float64
    description_contains_photo                              float64
    description_contains_photographer                       float64
    description_contains_piano                              float64
    description_contains_pick                               float64
    description_contains_picnic                             float64
    description_contains_picture                            float64
    description_contains_picturesque                        float64
    description_contains_piece                              float64
    description_contains_pier                               float64
    description_contains_pillow                             float64
    description_contains_pizza                              float64
    description_contains_pizzeria                           float64
    description_contains_place                              float64
    description_contains_plan                               float64
    description_contains_planet                             float64
    description_contains_planning                           float64
    description_contains_plant                              float64
    description_contains_plasma                             float64
    description_contains_plate                              float64
    description_contains_platform                           float64
    description_contains_play                               float64
    description_contains_player                             float64
    description_contains_playground                         float64
    description_contains_playing                            float64
    description_contains_playroom                           float64
    description_contains_plaza                              float64
    description_contains_pleasant                           float64
    description_contains_pleasure                           float64
    description_contains_plentiful                          float64
    description_contains_plenty                             float64
    description_contains_plethora                           float64
    description_contains_plug                               float64
    description_contains_plus                               float64
    description_contains_plush                              float64
    description_contains_pocket                             float64
    description_contains_point                              float64
    description_contains_police                             float64
    description_contains_policy                             float64
    description_contains_polish                             float64
    description_contains_pool                               float64
    description_contains_popular                            float64
    description_contains_porch                              float64
    description_contains_port                               float64
    description_contains_portable                           float64
    description_contains_positive                           float64
    description_contains_possible                           float64
    description_contains_possibly                           float64
    description_contains_post                               float64
    description_contains_potential                          float64
    description_contains_pour                               float64
    description_contains_power                              float64
    description_contains_powerful                           float64
    description_contains_prefer                             float64
    description_contains_preference                         float64
    description_contains_premise                            float64
    description_contains_premium                            float64
    description_contains_prepare                            float64
    description_contains_presbyterian                       float64
    description_contains_present                            float64
    description_contains_preserve                           float64
    description_contains_press                              float64
    description_contains_pressure                           float64
    description_contains_prestigious                        float64
    description_contains_pretty                             float64
    description_contains_previous                           float64
    description_contains_prewar                             float64
    description_contains_price                              float64
    description_contains_pricing                            float64
    description_contains_pride                              float64
    description_contains_primarily                          float64
    description_contains_primary                            float64
    description_contains_prime                              float64
    description_contains_printer                            float64
    description_contains_prior                              float64
    description_contains_pristine                           float64
    description_contains_privacy                            float64
    description_contains_private                            float64
    description_contains_privately                          float64
    description_contains_probably                           float64
    description_contains_problem                            float64
    description_contains_process                            float64
    description_contains_product                            float64
    description_contains_professional                       float64
    description_contains_professionally                     float64
    description_contains_profile                            float64
    description_contains_project                            float64
    description_contains_projector                          float64
    description_contains_promenade                          float64
    description_contains_promptly                           float64
    description_contains_proper                             float64
    description_contains_property                           float64
    description_contains_prospect                           float64
    description_contains_provide                            float64
    description_contains_proximity                          float64
    description_contains_public                             float64
    description_contains_pull                               float64
    description_contains_pullout                            float64
    description_contains_purchase                           float64
    description_contains_purpose                            float64
    description_contains_quaint                             float64
    description_contains_quality                            float64
    description_contains_quarter                            float64
    description_contains_queen                              float64
    description_contains_question                           float64
    description_contains_quick                              float64
    description_contains_quickly                            float64
    description_contains_quiet                              float64
    description_contains_quintessential                     float64
    description_contains_quirky                             float64
    description_contains_quite                              float64
    description_contains_rabbit                             float64
    description_contains_race                               float64
    description_contains_rack                               float64
    description_contains_radiator                           float64
    description_contains_radio                              float64
    description_contains_radius                             float64
    description_contains_rail                               float64
    description_contains_railroad                           float64
    description_contains_rain                               float64
    description_contains_range                              float64
    description_contains_rare                               float64
    description_contains_rarely                             float64
    description_contains_rarity                             float64
    description_contains_rate                               float64
    description_contains_rating                             float64
    description_contains_reach                              float64
    description_contains_reachable                          float64
    description_contains_read                               float64
    description_contains_readily                            float64
    description_contains_reading                            float64
    description_contains_ready                              float64
    description_contains_real                               float64
    description_contains_really                             float64
    description_contains_rear                               float64
    description_contains_reason                             float64
    description_contains_reasonable                         float64
    description_contains_receive                            float64
    description_contains_recent                             float64
    description_contains_recently                           float64
    description_contains_recess                             float64
    description_contains_recharge                           float64
    description_contains_recommend                          float64
    description_contains_recommendation                     float64
    description_contains_record                             float64
    description_contains_recreation                         float64
    description_contains_reflect                            float64
    description_contains_refrigerator                       float64
    description_contains_refuge                             float64
    description_contains_refurbish                          float64
    description_contains_regard                             float64
    description_contains_regular                            float64
    description_contains_regularly                          float64
    description_contains_relate                             float64
    description_contains_relatively                         float64
    description_contains_relax                              float64
    description_contains_relaxation                         float64
    description_contains_reliable                           float64
    description_contains_remain                             float64
    description_contains_remember                           float64
    description_contains_remodel                            float64
    description_contains_remote                             float64
    description_contains_remove                             float64
    description_contains_renovate                           float64
    description_contains_renovation                         float64
    description_contains_renowned                           float64
    description_contains_rent                               float64
    description_contains_rental                             float64
    description_contains_renter                             float64
    description_contains_renting                            float64
    description_contains_reply                              float64
    description_contains_request                            float64
    description_contains_require                            float64
    description_contains_reservation                        float64
    description_contains_reserve                            float64
    description_contains_reside                             float64
    description_contains_residence                          float64
    description_contains_resident                           float64
    description_contains_residential                        float64
    description_contains_resort                             float64
    description_contains_respect                            float64
    description_contains_respectful                         float64
    description_contains_respite                            float64
    description_contains_respond                            float64
    description_contains_response                           float64
    description_contains_responsible                        float64
    description_contains_responsive                         float64
    description_contains_rest                               float64
    description_contains_restaurant                         float64
    description_contains_restful                            float64
    description_contains_restore                            float64
    description_contains_restroom                           float64
    description_contains_retail                             float64
    description_contains_retain                             float64
    description_contains_retreat                            float64
    description_contains_return                             float64
    description_contains_review                             float64
    description_contains_rice                               float64
    description_contains_rich                               float64
    description_contains_ride                               float64
    description_contains_ridge                              float64
    description_contains_right                              float64
    description_contains_rise                               float64
    description_contains_rite                               float64
    description_contains_river                              float64
    description_contains_riverbank                          float64
    description_contains_riverside                          float64
    description_contains_road                               float64
    description_contains_rock                               float64
    description_contains_rockefeller                        float64
    description_contains_roll                               float64
    description_contains_romantic                           float64
    description_contains_roof                               float64
    description_contains_rooftop                            float64
    description_contains_room                               float64
    description_contains_roomie                             float64
    description_contains_roommate                           float64
    description_contains_roomy                              float64
    description_contains_roosevelt                          float64
    description_contains_rooster                            float64
    description_contains_root                               float64
    description_contains_round                              float64
    description_contains_route                              float64
    description_contains_rule                               float64
    description_contains_running                            float64
    description_contains_russian                            float64
    description_contains_rustic                             float64
    description_contains_safe                               float64
    description_contains_safety                             float64
    description_contains_saint                              float64
    description_contains_salon                              float64
    description_contains_salt                               float64
    description_contains_sanctuary                          float64
    description_contains_sandwich                           float64
    description_contains_satisfy                            float64
    description_contains_saturday                           float64
    description_contains_sauna                              float64
    description_contains_save                               float64
    description_contains_scene                              float64
    description_contains_scenic                             float64
    description_contains_schedule                           float64
    description_contains_school                             float64
    description_contains_score                              float64
    description_contains_screen                             float64
    description_contains_seaport                            float64
    description_contains_search                             float64
    description_contains_season                             float64
    description_contains_seasonal                           float64
    description_contains_seat                               float64
    description_contains_seating                            float64
    description_contains_seclude                            float64
    description_contains_second                             float64
    description_contains_secret                             float64
    description_contains_section                            float64
    description_contains_sectional                          float64
    description_contains_secure                             float64
    description_contains_security                           float64
    description_contains_seeing                             float64
    description_contains_seek                               float64
    description_contains_seeking                            float64
    description_contains_select                             float64
    description_contains_selection                          float64
    description_contains_self                               float64
    description_contains_semi                               float64
    description_contains_send                               float64
    description_contains_sense                              float64
    description_contains_sensitive                          float64
    description_contains_separate                           float64
    description_contains_september                          float64
    description_contains_serene                             float64
    description_contains_seriously                          float64
    description_contains_serve                              float64
    description_contains_service                            float64
    description_contains_serving                            float64
    description_contains_setting                            float64
    description_contains_settle                             float64
    description_contains_setup                              float64
    description_contains_seven                              float64
    description_contains_shack                              float64
    description_contains_shade                              float64
    description_contains_shake                              float64
    description_contains_shampoo                            float64
    description_contains_shape                              float64
    description_contains_share                              float64
    description_contains_sharing                            float64
    description_contains_sheet                              float64
    description_contains_shelf                              float64
    description_contains_shelve                             float64
    description_contains_shoe                               float64
    description_contains_shoot                              float64
    description_contains_shop                               float64
    description_contains_shopping                           float64
    description_contains_short                              float64
    description_contains_shower                             float64
    description_contains_showtime                           float64
    description_contains_shuttle                            float64
    description_contains_sidewalk                           float64
    description_contains_sight                              float64
    description_contains_sightseeing                        float64
    description_contains_sign                               float64
    description_contains_silverware                         float64
    description_contains_simple                             float64
    description_contains_simplify                           float64
    description_contains_simply                             float64
    description_contains_single                             float64
    description_contains_sink                               float64
    description_contains_sister                             float64
    description_contains_site                               float64
    description_contains_sitting                            float64
    description_contains_situate                            float64
    description_contains_situation                          float64
    description_contains_size                               float64
    description_contains_skating                            float64
    description_contains_skip                               float64
    description_contains_skylight                           float64
    description_contains_skyline                            float64
    description_contains_sleek                              float64
    description_contains_sleep                              float64
    description_contains_sleeper                            float64
    description_contains_sleeping                           float64
    description_contains_slide                              float64
    description_contains_slightly                           float64
    description_contains_slope                              float64
    description_contains_small                              float64
    description_contains_smaller                            float64
    description_contains_smart                              float64
    description_contains_smith                              float64
    description_contains_smoke                              float64
    description_contains_smoker                             float64
    description_contains_smoking                            float64
    description_contains_snack                              float64
    description_contains_soak                               float64
    description_contains_soaking                            float64
    description_contains_soap                               float64
    description_contains_social                             float64
    description_contains_socialize                          float64
    description_contains_socializing                        float64
    description_contains_sofa                               float64
    description_contains_soft                               float64
    description_contains_soho                               float64
    description_contains_solid                              float64
    description_contains_solo                               float64
    description_contains_soon                               float64
    description_contains_sophisticate                       float64
    description_contains_sorry                              float64
    description_contains_sort                               float64
    description_contains_soul                               float64
    description_contains_sound                              float64
    description_contains_south                              float64
    description_contains_southern                           float64
    description_contains_space                              float64
    description_contains_spacious                           float64
    description_contains_spanish                            float64
    description_contains_spare                              float64
    description_contains_sparkling                          float64
    description_contains_speak                              float64
    description_contains_speaker                            float64
    description_contains_special                            float64
    description_contains_specific                           float64
    description_contains_specious                           float64
    description_contains_spectacular                        float64
    description_contains_speed                              float64
    description_contains_spend                              float64
    description_contains_spending                           float64
    description_contains_spice                              float64
    description_contains_spiral                             float64
    description_contains_spirit                             float64
    description_contains_split                              float64
    description_contains_spoon                              float64
    description_contains_sport                              float64
    description_contains_spot                               float64
    description_contains_spotless                           float64
    description_contains_spread                             float64
    description_contains_spring                             float64
    description_contains_square                             float64
    description_contains_stadium                            float64
    description_contains_staff                              float64
    description_contains_stain                              float64
    description_contains_stainless                          float64
    description_contains_stair                              float64
    description_contains_staircase                          float64
    description_contains_stand                              float64
    description_contains_standard                           float64
    description_contains_standing                           float64
    description_contains_star                               float64
    description_contains_start                              float64
    description_contains_starting                           float64
    description_contains_state                              float64
    description_contains_station                            float64
    description_contains_statue                             float64
    description_contains_stay                               float64
    description_contains_steal                              float64
    description_contains_steam                              float64
    description_contains_steamer                            float64
    description_contains_steel                              float64
    description_contains_steinway                           float64
    description_contains_step                               float64
    description_contains_stereo                             float64
    description_contains_stick                              float64
    description_contains_stock                              float64
    description_contains_stone                              float64
    description_contains_stool                              float64
    description_contains_stoop                              float64
    description_contains_stop                               float64
    description_contains_storage                            float64
    description_contains_store                              float64
    description_contains_story                              float64
    description_contains_stove                              float64
    description_contains_straight                           float64
    description_contains_stream                             float64
    description_contains_streaming                          float64
    description_contains_street                             float64
    description_contains_stretch                            float64
    description_contains_strictly                           float64
    description_contains_strip                              float64
    description_contains_stroll                             float64
    description_contains_strong                             float64
    description_contains_student                            float64
    description_contains_studio                             float64
    description_contains_study                              float64
    description_contains_studying                           float64
    description_contains_stuff                              float64
    description_contains_stun                               float64
    description_contains_stuyvesant                         float64
    description_contains_style                              float64
    description_contains_stylish                            float64
    description_contains_sublet                             float64
    description_contains_suburban                           float64
    description_contains_subway                             float64
    description_contains_sugar                              float64
    description_contains_suggest                            float64
    description_contains_suggestion                         float64
    description_contains_suit                               float64
    description_contains_suitable                           float64
    description_contains_suitcase                           float64
    description_contains_suite                              float64
    description_contains_summer                             float64
    description_contains_sunday                             float64
    description_contains_sunlight                           float64
    description_contains_sunlit                             float64
    description_contains_sunny                              float64
    description_contains_sunrise                            float64
    description_contains_sunset                             float64
    description_contains_sunshine                           float64
    description_contains_super                              float64
    description_contains_superb                             float64
    description_contains_superior                           float64
    description_contains_supermarket                        float64
    description_contains_supply                             float64
    description_contains_support                            float64
    description_contains_sure                               float64
    description_contains_surf                               float64
    description_contains_surround                           float64
    description_contains_surveillance                       float64
    description_contains_sushi                              float64
    description_contains_sweeping                           float64
    description_contains_sweet                              float64
    description_contains_swimming                           float64
    description_contains_table                              float64
    description_contains_tableware                          float64
    description_contains_taco                               float64
    description_contains_taking                             float64
    description_contains_talk                               float64
    description_contains_tall                               float64
    description_contains_target                             float64
    description_contains_taste                              float64
    description_contains_tasteful                           float64
    description_contains_tastefully                         float64
    description_contains_tavern                             float64
    description_contains_taxi                               float64
    description_contains_teacher                            float64
    description_contains_team                               float64
    description_contains_tech                               float64
    description_contains_technology                         float64
    description_contains_telephone                          float64
    description_contains_television                         float64
    description_contains_tell                               float64
    description_contains_temperature                        float64
    description_contains_temporary                          float64
    description_contains_tenant                             float64
    description_contains_tend                               float64
    description_contains_tenement                           float64
    description_contains_tennis                             float64
    description_contains_term                               float64
    description_contains_terminal                           float64
    description_contains_terrace                            float64
    description_contains_terrific                           float64
    description_contains_text                               float64
    description_contains_thai                               float64
    description_contains_thank                              float64
    description_contains_thanks                             float64
    description_contains_theater                            float64
    description_contains_theatre                            float64
    description_contains_theme                              float64
    description_contains_thermostat                         float64
    description_contains_thing                              float64
    description_contains_think                              float64
    description_contains_thoroughly                         float64
    description_contains_thought                            float64
    description_contains_thoughtfully                       float64
    description_contains_thread                             float64
    description_contains_thrift                             float64
    description_contains_thrive                             float64
    description_contains_throw                              float64
    description_contains_ticket                             float64
    description_contains_tidy                               float64
    description_contains_tight                              float64
    description_contains_tile                               float64
    description_contains_till                               float64
    description_contains_time                               float64
    description_contains_tiny                               float64
    description_contains_toaster                            float64
    description_contains_today                              float64
    description_contains_toddler                            float64
    description_contains_toilet                             float64
    description_contains_toiletry                           float64
    description_contains_ton                                float64
    description_contains_tool                               float64
    description_contains_toothbrush                         float64
    description_contains_toothpaste                         float64
    description_contains_topper                             float64
    description_contains_total                              float64
    description_contains_totally                            float64
    description_contains_touch                              float64
    description_contains_tour                               float64
    description_contains_tourist                            float64
    description_contains_touristy                           float64
    description_contains_towel                              float64
    description_contains_tower                              float64
    description_contains_town                               float64
    description_contains_track                              float64
    description_contains_trade                              float64
    description_contains_trader                             float64
    description_contains_trading                            float64
    description_contains_traditional                        float64
    description_contains_traffic                            float64
    description_contains_trail                              float64
    description_contains_train                              float64
    description_contains_tranquil                           float64
    description_contains_tranquility                        float64
    description_contains_transfer                           float64
    description_contains_transform                          float64
    description_contains_transit                            float64
    description_contains_transport                          float64
    description_contains_transportation                     float64
    description_contains_trash                              float64
    description_contains_travel                             float64
    description_contains_traveler                           float64
    description_contains_traveling                          float64
    description_contains_traveller                          float64
    description_contains_travelling                         float64
    description_contains_treat                              float64
    description_contains_tree                               float64
    description_contains_trendy                             float64
    description_contains_trip                               float64
    description_contains_triplex                            float64
    description_contains_true                               float64
    description_contains_truly                              float64
    description_contains_trundle                            float64
    description_contains_tuck                               float64
    description_contains_turn                               float64
    description_contains_twice                              float64
    description_contains_twin                               float64
    description_contains_type                               float64
    description_contains_typical                            float64
    description_contains_typically                          float64
    description_contains_ultimate                           float64
    description_contains_ultra                              float64
    description_contains_umbrella                           float64
    description_contains_unbeatable                         float64
    description_contains_underneath                         float64
    description_contains_understand                         float64
    description_contains_unforgettable                      float64
    description_contains_unfortunately                      float64
    description_contains_union                              float64
    description_contains_unique                             float64
    description_contains_uniquely                           float64
    description_contains_unit                               float64
    description_contains_unite                              float64
    description_contains_university                         float64
    description_contains_unlike                             float64
    description_contains_unlimited                          float64
    description_contains_unobstructed                       float64
    description_contains_unwind                             float64
    description_contains_upcoming                           float64
    description_contains_update                             float64
    description_contains_upgrade                            float64
    description_contains_upper                              float64
    description_contains_upscale                            float64
    description_contains_upstairs                           float64
    description_contains_uptown                             float64
    description_contains_urban                              float64
    description_contains_usage                              float64
    description_contains_using                              float64
    description_contains_usually                            float64
    description_contains_utensil                            float64
    description_contains_utica                              float64
    description_contains_utility                            float64
    description_contains_utilize                            float64
    description_contains_vacation                           float64
    description_contains_valet                              float64
    description_contains_value                              float64
    description_contains_vanity                             float64
    description_contains_variety                            float64
    description_contains_various                            float64
    description_contains_vary                               float64
    description_contains_vast                               float64
    description_contains_vegan                              float64
    description_contains_vegetable                          float64
    description_contains_venture                            float64
    description_contains_venue                              float64
    description_contains_verify                             float64
    description_contains_vibe                               float64
    description_contains_vibrant                            float64
    description_contains_vicinity                           float64
    description_contains_victorian                          float64
    description_contains_video                              float64
    description_contains_view                               float64
    description_contains_viewing                            float64
    description_contains_village                            float64
    description_contains_vintage                            float64
    description_contains_vinyl                              float64
    description_contains_virtual                            float64
    description_contains_visit                              float64
    description_contains_visiting                           float64
    description_contains_visitor                            float64
    description_contains_wait                               float64
    description_contains_waiting                            float64
    description_contains_wake                               float64
    description_contains_waking                             float64
    description_contains_walk                               float64
    description_contains_walking                            float64
    description_contains_wall                               float64
    description_contains_want                               float64
    description_contains_wardrobe                           float64
    description_contains_ware                               float64
    description_contains_warehouse                          float64
    description_contains_warm                               float64
    description_contains_warmer                             float64
    description_contains_warmth                             float64
    description_contains_warner                             float64
    description_contains_wash                               float64
    description_contains_washer                             float64
    description_contains_washing                            float64
    description_contains_washington                         float64
    description_contains_watch                              float64
    description_contains_watching                           float64
    description_contains_water                              float64
    description_contains_waterfront                         float64
    description_contains_way                                float64
    description_contains_wear                               float64
    description_contains_weather                            float64
    description_contains_website                            float64
    description_contains_week                               float64
    description_contains_weekday                            float64
    description_contains_weekend                            float64
    description_contains_weekly                             float64
    description_contains_welcome                            float64
    description_contains_west                               float64
    description_contains_western                            float64
    description_contains_westside                           float64
    description_contains_white                              float64
    description_contains_whitney                            float64
    description_contains_wide                               float64
    description_contains_wife                               float64
    description_contains_wifi                               float64
    description_contains_willing                            float64
    description_contains_window                             float64
    description_contains_windsor                            float64
    description_contains_wine                               float64
    description_contains_winter                             float64
    description_contains_wireless                           float64
    description_contains_wish                               float64
    description_contains_woman                              float64
    description_contains_wonderful                          float64
    description_contains_wonderfully                        float64
    description_contains_wont                               float64
    description_contains_wood                               float64
    description_contains_wooden                             float64
    description_contains_work                               float64
    description_contains_working                            float64
    description_contains_workspace                          float64
    description_contains_world                              float64
    description_contains_worry                              float64
    description_contains_worth                              float64
    description_contains_wrap                               float64
    description_contains_write                              float64
    description_contains_writer                             float64
    description_contains_writing                            float64
    description_contains_yankee                             float64
    description_contains_yard                               float64
    description_contains_year                               float64
    description_contains_yellow                             float64
    description_contains_yoga                               float64
    description_contains_york                               float64
    description_contains_young                              float64
    description_contains_yummy                              float64
    description_contains_zero                               float64
    description_contains_zone                               float64
    dtypes: float64(2151)
    memory usage: 748.4 MB


## Appendix


```python
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
```

    [nltk_data] Downloading package wordnet to /home/ec2-user/nltk_data...
    [nltk_data]   Unzipping corpora/wordnet.zip.





    True




```python
import re
import pandas as pd
import numpy as np
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# define some common lingo
custom_stopwords = ['bedroom', 'bathroom', 'apartment']

def remove_hypens(book_text):
    return re.sub(r'(\w+)-(\w+)-?(\w)?', r'\1 \2 \3', book_text)

# tokenize text
def tokenize_text(book_text):
    TOKEN_PATTERN = r'\s+'
    regex_wt = nltk.RegexpTokenizer(pattern=TOKEN_PATTERN, gaps=True)
    word_tokens = regex_wt.tokenize(book_text)
    return word_tokens

def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation))) 
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens]) 
    return filtered_tokens

def convert_to_lowercase(tokens):
    return [token.lower() for token in tokens if token.isalpha()]

def remove_stopwords(tokens, custom_stopwords):
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list += custom_stopwords
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

def get_lemma(tokens):
    lemmas = []
    for word in tokens:
        lemma = wn.morphy(word)
        if lemma is None:
            lemmas.append(word)
        else:
            lemmas.append(lemma)
    return lemmas

def remove_short_tokens(tokens):
    return [token for token in tokens if len(token) > 3]

def keep_only_words_in_wordnet(tokens):
    return [token for token in tokens if wn.synsets(token)]

def apply_lemmatize(tokens, wnl=WordNetLemmatizer()):
    return [wnl.lemmatize(token) for token in tokens]

# I like to think of each row of text as a book
# input to this function is a list of books
def nlp_pipeline(book_texts):
    clean_books = []
    for book in book_texts:
        book = remove_hypens(book)
        book_i = tokenize_text(book)
        book_i = remove_characters_after_tokenization(book_i)
        book_i = convert_to_lowercase(book_i)
        book_i = remove_stopwords(book_i, custom_stopwords)
        book_i = get_lemma(book_i)
        book_i = remove_short_tokens(book_i)
        book_i = keep_only_words_in_wordnet(book_i)
        book_i = apply_lemmatize(book_i)
        clean_books.append(book_i)
    return clean_books
```
