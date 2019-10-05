
## Introduction
The goal of this notebook is show how to train a PCA model on AWS Sagemaker. The training data resides on S3 in `s3://skuchkula-sagemaker-airbnb/` location. Objective is to apply Principal Components Analysis (PCA) on airbnb locations located in NYC. 

### STEP 1: Create an S3 bucket which contains sagemaker name in it.
Having the sagemaker name is not a requirement. However, as per Amazon documentation:

> Note: 
Amazon SageMaker needs permission to access these buckets. You grant permission with an IAM role, which you create in the next step when you create an Amazon SageMaker notebook instance. This IAM role automatically gets permissions to access any bucket that has sagemaker in the name. It gets these permissions through the AmazonSageMakerFullAccess policy, which Amazon SageMaker attaches to the role. If you add a policy to the role that grants the SageMaker service principal S3FullAccess permission, the name of the bucket does not need to contain sagemaker.

### STEP 2: Create an Amazon SageMaker Notebook instance
An Amazon SageMaker notebook instance is a fully managed machine learning (ML) Amazon Elastic Compute Cloud (Amazon EC2) compute instance that runs the Jupyter Notebook App. You use the notebook instance to create and manage Jupyter notebooks that you can use to prepare and process data and to train and deploy machine learning models. 

To create an Amazon SageMaker notebook instance

- Open the Amazon SageMaker console at https://console.aws.amazon.com/sagemaker/.

- Choose Notebook instances, then choose Create notebook instance.

- On the Create notebook instance page, provide the following information (if a field is not mentioned, leave the default values):

- For Notebook instance name, type a name for your notebook instance.

- For Instance type, choose ml.t2.medium. This is the least expensive instance type that notebook instances support, and it suffices for this exercise.

- For IAM role, choose Create a new role, then choose Create role.

- Choose Create notebook instance.

In a few minutes, Amazon SageMaker launches an ML compute instance—in this case, a notebook instance—and attaches an ML storage volume to it. The notebook instance has a preconfigured Jupyter notebook server and a set of Anaconda libraries.

### Step 3: Create notebook instance and start writing code


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
import sagemaker
```

### Step 4:  Get data from S3


```python
# boto3 client to get S3 data
s3_client = boto3.client('s3')
bucket_name='skuchkula-sagemaker-airbnb'
```


```python
# get a list of objects in the bucket
obj_list=s3_client.list_objects(Bucket=bucket_name)

# print object(s)in S3 bucket
files=[]
for contents in obj_list['Contents']:
    files.append(contents['Key'])
    
print(files)
```

    ['detailed_listings.csv', 'summary_listings.csv']



```python
# there is one file --> one key
detailed_listings=files[0]
summary_listings=files[1]
```


```python
# check the file formats
file_name = detailed_listings

# get an S3 object by passing in the bucket and file name
data_object = s3_client.get_object(Bucket=bucket_name, Key=file_name)

# what info does the object contain?
display(data_object)
```


    {'ResponseMetadata': {'RequestId': 'C6DAE855FAFB68CD',
      'HostId': 'AshhLC/JbPE5I3Gokgj0/kzcz8T89oog0bYTlfMdLNGmVprf3c67vmPyjV9YvXeLq/e/aHIr/r8=',
      'HTTPStatusCode': 200,
      'HTTPHeaders': {'x-amz-id-2': 'AshhLC/JbPE5I3Gokgj0/kzcz8T89oog0bYTlfMdLNGmVprf3c67vmPyjV9YvXeLq/e/aHIr/r8=',
       'x-amz-request-id': 'C6DAE855FAFB68CD',
       'date': 'Sat, 05 Oct 2019 00:30:54 GMT',
       'last-modified': 'Thu, 03 Oct 2019 21:24:49 GMT',
       'etag': '"01067810107b6eb6cfc6bf52cf02de2c-22"',
       'accept-ranges': 'bytes',
       'content-type': 'text/csv',
       'content-length': '184372589',
       'server': 'AmazonS3'},
      'RetryAttempts': 0},
     'AcceptRanges': 'bytes',
     'LastModified': datetime.datetime(2019, 10, 3, 21, 24, 49, tzinfo=tzutc()),
     'ContentLength': 184372589,
     'ETag': '"01067810107b6eb6cfc6bf52cf02de2c-22"',
     'ContentType': 'text/csv',
     'Metadata': {},
     'Body': <botocore.response.StreamingBody at 0x7f5634aa90f0>}



```python
# information is in the "Body" of the object
data_body = data_object["Body"].read()
print('Data type: ', type(data_body))
```

    Data type:  <class 'bytes'>



```python
# read in bytes data
data_stream = io.BytesIO(data_body)

# create a dataframe
counties_df = pd.read_csv(data_stream, header=0, delimiter=",") 
counties_df.head()
```

    /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (61,62,94,95) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)





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
      <th>listing_url</th>
      <th>scrape_id</th>
      <th>last_scraped</th>
      <th>name</th>
      <th>summary</th>
      <th>space</th>
      <th>description</th>
      <th>experiences_offered</th>
      <th>neighborhood_overview</th>
      <th>...</th>
      <th>instant_bookable</th>
      <th>is_business_travel_ready</th>
      <th>cancellation_policy</th>
      <th>require_guest_profile_picture</th>
      <th>require_guest_phone_verification</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>reviews_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>https://www.airbnb.com/rooms/2595</td>
      <td>20190806030549</td>
      <td>2019-08-07</td>
      <td>Skylit Midtown Castle</td>
      <td>Find your romantic getaway to this beautiful, ...</td>
      <td>- Spacious (500+ft²), immaculate and nicely fu...</td>
      <td>Find your romantic getaway to this beautiful, ...</td>
      <td>none</td>
      <td>Centrally located in the heart of Manhattan ju...</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3647</td>
      <td>https://www.airbnb.com/rooms/3647</td>
      <td>20190806030549</td>
      <td>2019-08-06</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>NaN</td>
      <td>WELCOME TO OUR INTERNATIONAL URBAN COMMUNITY T...</td>
      <td>WELCOME TO OUR INTERNATIONAL URBAN COMMUNITY T...</td>
      <td>none</td>
      <td>NaN</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3831</td>
      <td>https://www.airbnb.com/rooms/3831</td>
      <td>20190806030549</td>
      <td>2019-08-06</td>
      <td>Cozy Entire Floor of Brownstone</td>
      <td>Urban retreat: enjoy 500 s.f. floor in 1899 br...</td>
      <td>Greetings!      We own a double-duplex brownst...</td>
      <td>Urban retreat: enjoy 500 s.f. floor in 1899 br...</td>
      <td>none</td>
      <td>Just the right mix of urban center and local n...</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>moderate</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5022</td>
      <td>https://www.airbnb.com/rooms/5022</td>
      <td>20190806030549</td>
      <td>2019-08-06</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>NaN</td>
      <td>Loft apartment with high ceiling and wood floo...</td>
      <td>Loft apartment with high ceiling and wood floo...</td>
      <td>none</td>
      <td>NaN</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5099</td>
      <td>https://www.airbnb.com/rooms/5099</td>
      <td>20190806030549</td>
      <td>2019-08-06</td>
      <td>Large Cozy 1 BR Apartment In Midtown East</td>
      <td>My large 1 bedroom apartment is true New York ...</td>
      <td>I have a large 1 bedroom apartment centrally l...</td>
      <td>My large 1 bedroom apartment is true New York ...</td>
      <td>none</td>
      <td>My neighborhood in Midtown East is called Murr...</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.60</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 106 columns</p>
</div>




```python
def get_data_frame(bucket_name, file_name):
    # get an S3 object by passing in the bucket and file name
    data_object = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    
    # information is in the "Body" of the object
    data_body = data_object["Body"].read()
    
    # read in bytes data
    data_stream = io.BytesIO(data_body)
    
    # create a dataframe
    df = pd.read_csv(data_stream, header=0, delimiter=",", low_memory=False)
    
    return df
```


```python
df_summary_listings = get_data_frame(bucket_name, summary_listings)
df_detailed_listings = get_data_frame(bucket_name, detailed_listings)
```


```python
df_summary_listings.head()
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
      <th>host_id</th>
      <th>host_name</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>Skylit Midtown Castle</td>
      <td>2845</td>
      <td>Jennifer</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>Entire home/apt</td>
      <td>225</td>
      <td>1</td>
      <td>46</td>
      <td>2019-07-14</td>
      <td>0.39</td>
      <td>2</td>
      <td>288</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3647</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>4632</td>
      <td>Elisabeth</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>Private room</td>
      <td>150</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>365</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3831</td>
      <td>Cozy Entire Floor of Brownstone</td>
      <td>4869</td>
      <td>LisaRoxanne</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>Entire home/apt</td>
      <td>89</td>
      <td>1</td>
      <td>274</td>
      <td>2019-07-26</td>
      <td>4.64</td>
      <td>1</td>
      <td>212</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5022</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>7192</td>
      <td>Laura</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>Entire home/apt</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>2018-11-19</td>
      <td>0.10</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5099</td>
      <td>Large Cozy 1 BR Apartment In Midtown East</td>
      <td>7322</td>
      <td>Chris</td>
      <td>Manhattan</td>
      <td>Murray Hill</td>
      <td>40.74767</td>
      <td>-73.97500</td>
      <td>Entire home/apt</td>
      <td>200</td>
      <td>3</td>
      <td>75</td>
      <td>2019-07-21</td>
      <td>0.60</td>
      <td>1</td>
      <td>127</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_detailed_listings.head()
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
      <th>listing_url</th>
      <th>scrape_id</th>
      <th>last_scraped</th>
      <th>name</th>
      <th>summary</th>
      <th>space</th>
      <th>description</th>
      <th>experiences_offered</th>
      <th>neighborhood_overview</th>
      <th>...</th>
      <th>instant_bookable</th>
      <th>is_business_travel_ready</th>
      <th>cancellation_policy</th>
      <th>require_guest_profile_picture</th>
      <th>require_guest_phone_verification</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>reviews_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>https://www.airbnb.com/rooms/2595</td>
      <td>20190806030549</td>
      <td>2019-08-07</td>
      <td>Skylit Midtown Castle</td>
      <td>Find your romantic getaway to this beautiful, ...</td>
      <td>- Spacious (500+ft²), immaculate and nicely fu...</td>
      <td>Find your romantic getaway to this beautiful, ...</td>
      <td>none</td>
      <td>Centrally located in the heart of Manhattan ju...</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3647</td>
      <td>https://www.airbnb.com/rooms/3647</td>
      <td>20190806030549</td>
      <td>2019-08-06</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>NaN</td>
      <td>WELCOME TO OUR INTERNATIONAL URBAN COMMUNITY T...</td>
      <td>WELCOME TO OUR INTERNATIONAL URBAN COMMUNITY T...</td>
      <td>none</td>
      <td>NaN</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3831</td>
      <td>https://www.airbnb.com/rooms/3831</td>
      <td>20190806030549</td>
      <td>2019-08-06</td>
      <td>Cozy Entire Floor of Brownstone</td>
      <td>Urban retreat: enjoy 500 s.f. floor in 1899 br...</td>
      <td>Greetings!      We own a double-duplex brownst...</td>
      <td>Urban retreat: enjoy 500 s.f. floor in 1899 br...</td>
      <td>none</td>
      <td>Just the right mix of urban center and local n...</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>moderate</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5022</td>
      <td>https://www.airbnb.com/rooms/5022</td>
      <td>20190806030549</td>
      <td>2019-08-06</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>NaN</td>
      <td>Loft apartment with high ceiling and wood floo...</td>
      <td>Loft apartment with high ceiling and wood floo...</td>
      <td>none</td>
      <td>NaN</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5099</td>
      <td>https://www.airbnb.com/rooms/5099</td>
      <td>20190806030549</td>
      <td>2019-08-06</td>
      <td>Large Cozy 1 BR Apartment In Midtown East</td>
      <td>My large 1 bedroom apartment is true New York ...</td>
      <td>I have a large 1 bedroom apartment centrally l...</td>
      <td>My large 1 bedroom apartment is true New York ...</td>
      <td>none</td>
      <td>My neighborhood in Midtown East is called Murr...</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.60</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 106 columns</p>
</div>



The Detailed listings contains about 106 different features for each of the Airbnb locations. We need to do some EDA to check if we can trim down some features.


```python
print("Summary Listings row, cols: ", df_summary_listings.shape)
print("Detailed Listings row, cols: ", df_detailed_listings.shape)
```

    Summary Listings row, cols:  (48864, 16)
    Detailed Listings row, cols:  (48864, 106)


The summary listings contains a small subset of features contained in the detailed listings. Summary listings is good for visualization purposes, but for machine learning models, it is better to have more features from which the model can learn.


```python
df_summary_listings.columns
```




    Index(['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
           'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
           'minimum_nights', 'number_of_reviews', 'last_review',
           'reviews_per_month', 'calculated_host_listings_count',
           'availability_365'],
          dtype='object')




```python
# check how the data types are distributed.
# pandas does its best to interpret the datatype while reading in
# however it is our duty to check if the datatype makes sense
df_detailed_listings.dtypes.value_counts()
```




    object     63
    float64    22
    int64      21
    dtype: int64



### Data Cleaning


```python
df_detailed_listings.select_dtypes(include=['object']).columns
```




    Index(['listing_url', 'last_scraped', 'name', 'summary', 'space',
           'description', 'experiences_offered', 'neighborhood_overview', 'notes',
           'transit', 'access', 'interaction', 'house_rules', 'picture_url',
           'host_url', 'host_name', 'host_since', 'host_location', 'host_about',
           'host_response_time', 'host_response_rate', 'host_is_superhost',
           'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
           'host_verifications', 'host_has_profile_pic', 'host_identity_verified',
           'street', 'neighbourhood', 'neighbourhood_cleansed',
           'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
           'smart_location', 'country_code', 'country', 'is_location_exact',
           'property_type', 'room_type', 'bed_type', 'amenities', 'price',
           'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee',
           'extra_people', 'calendar_updated', 'has_availability',
           'calendar_last_scraped', 'first_review', 'last_review',
           'requires_license', 'license', 'jurisdiction_names', 'instant_bookable',
           'is_business_travel_ready', 'cancellation_policy',
           'require_guest_profile_picture', 'require_guest_phone_verification'],
          dtype='object')




```python
print(pd.get_option("display.max_columns"))
pd.set_option("display.max_columns", 100)
print(pd.get_option("display.max_columns"))
```

    20
    100



```python
drop_object_cols = ['listing_url',
             'last_scraped',
             #'name',
             'picture_url',
             'host_url',
             'host_name',
             'host_since',
             'host_location',
             'host_about',
             'host_thumbnail_url',
             'host_picture_url',
             'host_neighbourhood',
             'street',
             #'neighbourhood',
             #'neighbourhood_cleansed',
             #'neighbourhood_group_cleansed',
             'city',
             'state',
             'zipcode',
             'market',
             'smart_location',
             'country_code',
             'country',
             'calendar_updated',
             'calendar_last_scraped',
             'first_review',
             'last_review' 
            ]
```


```python
df_detailed_listings.select_dtypes(include=['object']).head()
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
      <th>listing_url</th>
      <th>last_scraped</th>
      <th>name</th>
      <th>summary</th>
      <th>space</th>
      <th>description</th>
      <th>experiences_offered</th>
      <th>neighborhood_overview</th>
      <th>notes</th>
      <th>transit</th>
      <th>access</th>
      <th>interaction</th>
      <th>house_rules</th>
      <th>picture_url</th>
      <th>host_url</th>
      <th>host_name</th>
      <th>host_since</th>
      <th>host_location</th>
      <th>host_about</th>
      <th>host_response_time</th>
      <th>host_response_rate</th>
      <th>host_is_superhost</th>
      <th>host_thumbnail_url</th>
      <th>host_picture_url</th>
      <th>host_neighbourhood</th>
      <th>host_verifications</th>
      <th>host_has_profile_pic</th>
      <th>host_identity_verified</th>
      <th>street</th>
      <th>neighbourhood</th>
      <th>neighbourhood_cleansed</th>
      <th>neighbourhood_group_cleansed</th>
      <th>city</th>
      <th>state</th>
      <th>zipcode</th>
      <th>market</th>
      <th>smart_location</th>
      <th>country_code</th>
      <th>country</th>
      <th>is_location_exact</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>bed_type</th>
      <th>amenities</th>
      <th>price</th>
      <th>weekly_price</th>
      <th>monthly_price</th>
      <th>security_deposit</th>
      <th>cleaning_fee</th>
      <th>extra_people</th>
      <th>calendar_updated</th>
      <th>has_availability</th>
      <th>calendar_last_scraped</th>
      <th>first_review</th>
      <th>last_review</th>
      <th>requires_license</th>
      <th>license</th>
      <th>jurisdiction_names</th>
      <th>instant_bookable</th>
      <th>is_business_travel_ready</th>
      <th>cancellation_policy</th>
      <th>require_guest_profile_picture</th>
      <th>require_guest_phone_verification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.airbnb.com/rooms/2595</td>
      <td>2019-08-07</td>
      <td>Skylit Midtown Castle</td>
      <td>Find your romantic getaway to this beautiful, ...</td>
      <td>- Spacious (500+ft²), immaculate and nicely fu...</td>
      <td>Find your romantic getaway to this beautiful, ...</td>
      <td>none</td>
      <td>Centrally located in the heart of Manhattan ju...</td>
      <td>NaN</td>
      <td>Apartment is located on 37th Street between 5t...</td>
      <td>Guests have full access to the kitchen, bathro...</td>
      <td>I am a Sound Therapy Practitioner and Kundalin...</td>
      <td>Make yourself at home, respect the space and t...</td>
      <td>https://a0.muscache.com/im/pictures/f0813a11-4...</td>
      <td>https://www.airbnb.com/users/show/2845</td>
      <td>Jennifer</td>
      <td>2008-09-09</td>
      <td>New York, New York, United States</td>
      <td>A New Yorker since 2000! My passion is creatin...</td>
      <td>within a few hours</td>
      <td>90%</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/users/2845/profile_...</td>
      <td>https://a0.muscache.com/im/users/2845/profile_...</td>
      <td>Midtown</td>
      <td>['email', 'phone', 'reviews', 'kba', 'work_ema...</td>
      <td>t</td>
      <td>t</td>
      <td>New York, NY, United States</td>
      <td>Midtown</td>
      <td>Midtown</td>
      <td>Manhattan</td>
      <td>New York</td>
      <td>NY</td>
      <td>10018</td>
      <td>New York</td>
      <td>New York, NY</td>
      <td>US</td>
      <td>United States</td>
      <td>f</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>Real Bed</td>
      <td>{TV,Wifi,"Air conditioning",Kitchen,"Paid park...</td>
      <td>$225.00</td>
      <td>$1,995.00</td>
      <td>NaN</td>
      <td>$350.00</td>
      <td>$100.00</td>
      <td>$0.00</td>
      <td>a week ago</td>
      <td>t</td>
      <td>2019-08-07</td>
      <td>2009-11-21</td>
      <td>2019-07-14</td>
      <td>f</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.airbnb.com/rooms/3647</td>
      <td>2019-08-06</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>NaN</td>
      <td>WELCOME TO OUR INTERNATIONAL URBAN COMMUNITY T...</td>
      <td>WELCOME TO OUR INTERNATIONAL URBAN COMMUNITY T...</td>
      <td>none</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Upon arrival please have a legibile copy of yo...</td>
      <td>https://a0.muscache.com/im/pictures/838341/9b3...</td>
      <td>https://www.airbnb.com/users/show/4632</td>
      <td>Elisabeth</td>
      <td>2008-11-25</td>
      <td>New York, New York, United States</td>
      <td>Make Up Artist National/ (Website hidden by Ai...</td>
      <td>within a day</td>
      <td>100%</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/users/4632/profile_...</td>
      <td>https://a0.muscache.com/im/users/4632/profile_...</td>
      <td>Harlem</td>
      <td>['email', 'phone', 'google', 'reviews', 'jumio...</td>
      <td>t</td>
      <td>t</td>
      <td>New York, NY, United States</td>
      <td>Harlem</td>
      <td>Harlem</td>
      <td>Manhattan</td>
      <td>New York</td>
      <td>NY</td>
      <td>10027</td>
      <td>New York</td>
      <td>New York, NY</td>
      <td>US</td>
      <td>United States</td>
      <td>t</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>Pull-out Sofa</td>
      <td>{"Cable TV",Internet,Wifi,"Air conditioning",K...</td>
      <td>$150.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>$200.00</td>
      <td>$75.00</td>
      <td>$20.00</td>
      <td>35 months ago</td>
      <td>t</td>
      <td>2019-08-06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.airbnb.com/rooms/3831</td>
      <td>2019-08-06</td>
      <td>Cozy Entire Floor of Brownstone</td>
      <td>Urban retreat: enjoy 500 s.f. floor in 1899 br...</td>
      <td>Greetings!      We own a double-duplex brownst...</td>
      <td>Urban retreat: enjoy 500 s.f. floor in 1899 br...</td>
      <td>none</td>
      <td>Just the right mix of urban center and local n...</td>
      <td>NaN</td>
      <td>B52 bus for a 10-minute ride to downtown Brook...</td>
      <td>You will have exclusive use of and access to: ...</td>
      <td>We'll be around, but since you have the top fl...</td>
      <td>Smoking - outside please; pets allowed but ple...</td>
      <td>https://a0.muscache.com/im/pictures/e49999c2-9...</td>
      <td>https://www.airbnb.com/users/show/4869</td>
      <td>LisaRoxanne</td>
      <td>2008-12-07</td>
      <td>New York, New York, United States</td>
      <td>Laid-back bi-coastal actor/professor/attorney.</td>
      <td>within an hour</td>
      <td>90%</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/users/4869/profile_...</td>
      <td>https://a0.muscache.com/im/users/4869/profile_...</td>
      <td>Clinton Hill</td>
      <td>['email', 'phone', 'reviews', 'kba']</td>
      <td>t</td>
      <td>t</td>
      <td>Brooklyn, NY, United States</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>Brooklyn</td>
      <td>Brooklyn</td>
      <td>NY</td>
      <td>11238</td>
      <td>New York</td>
      <td>Brooklyn, NY</td>
      <td>US</td>
      <td>United States</td>
      <td>t</td>
      <td>Guest suite</td>
      <td>Entire home/apt</td>
      <td>Real Bed</td>
      <td>{TV,"Cable TV",Internet,Wifi,"Air conditioning...</td>
      <td>$89.00</td>
      <td>$575.00</td>
      <td>$2,100.00</td>
      <td>$500.00</td>
      <td>NaN</td>
      <td>$0.00</td>
      <td>today</td>
      <td>t</td>
      <td>2019-08-06</td>
      <td>2014-09-30</td>
      <td>2019-07-26</td>
      <td>f</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>moderate</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.airbnb.com/rooms/5022</td>
      <td>2019-08-06</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>NaN</td>
      <td>Loft apartment with high ceiling and wood floo...</td>
      <td>Loft apartment with high ceiling and wood floo...</td>
      <td>none</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Please be considerate when staying in the apar...</td>
      <td>https://a0.muscache.com/im/pictures/feb453bd-f...</td>
      <td>https://www.airbnb.com/users/show/7192</td>
      <td>Laura</td>
      <td>2009-01-29</td>
      <td>Miami, Florida, United States</td>
      <td>I have been a NYer for almost 10 years. I came...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/users/7192/profile_...</td>
      <td>https://a0.muscache.com/im/users/7192/profile_...</td>
      <td>East Harlem</td>
      <td>['email', 'phone', 'facebook', 'reviews', 'kba']</td>
      <td>t</td>
      <td>t</td>
      <td>New York, NY, United States</td>
      <td>East Harlem</td>
      <td>East Harlem</td>
      <td>Manhattan</td>
      <td>New York</td>
      <td>NY</td>
      <td>10029</td>
      <td>New York</td>
      <td>New York, NY</td>
      <td>US</td>
      <td>United States</td>
      <td>t</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>Real Bed</td>
      <td>{Internet,Wifi,"Air conditioning",Kitchen,Elev...</td>
      <td>$80.00</td>
      <td>$600.00</td>
      <td>$1,600.00</td>
      <td>$100.00</td>
      <td>$80.00</td>
      <td>$20.00</td>
      <td>4 months ago</td>
      <td>t</td>
      <td>2019-08-06</td>
      <td>2012-03-20</td>
      <td>2018-11-19</td>
      <td>f</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.airbnb.com/rooms/5099</td>
      <td>2019-08-06</td>
      <td>Large Cozy 1 BR Apartment In Midtown East</td>
      <td>My large 1 bedroom apartment is true New York ...</td>
      <td>I have a large 1 bedroom apartment centrally l...</td>
      <td>My large 1 bedroom apartment is true New York ...</td>
      <td>none</td>
      <td>My neighborhood in Midtown East is called Murr...</td>
      <td>Read My Full Listing For All Information. New ...</td>
      <td>From the apartment is a 10 minute walk to Gran...</td>
      <td>I will meet you upon arrival.</td>
      <td>I usually check in with guests via text or ema...</td>
      <td>• Check-in time is 2PM. • Check-out time is 12...</td>
      <td>https://a0.muscache.com/im/pictures/be2fdcf6-e...</td>
      <td>https://www.airbnb.com/users/show/7322</td>
      <td>Chris</td>
      <td>2009-02-02</td>
      <td>New York, New York, United States</td>
      <td>I'm an artist, writer, traveler, and a native ...</td>
      <td>within a few hours</td>
      <td>90%</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/pictures/user/26745...</td>
      <td>https://a0.muscache.com/im/pictures/user/26745...</td>
      <td>Flatiron District</td>
      <td>['email', 'phone', 'reviews', 'jumio', 'govern...</td>
      <td>t</td>
      <td>f</td>
      <td>New York, NY, United States</td>
      <td>Midtown East</td>
      <td>Murray Hill</td>
      <td>Manhattan</td>
      <td>New York</td>
      <td>NY</td>
      <td>10016</td>
      <td>New York</td>
      <td>New York, NY</td>
      <td>US</td>
      <td>United States</td>
      <td>f</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>Real Bed</td>
      <td>{TV,"Cable TV",Internet,Wifi,Kitchen,"Buzzer/w...</td>
      <td>$200.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>$300.00</td>
      <td>$125.00</td>
      <td>$100.00</td>
      <td>4 days ago</td>
      <td>t</td>
      <td>2019-08-06</td>
      <td>2009-04-20</td>
      <td>2019-07-21</td>
      <td>f</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>t</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_detailed_listings.select_dtypes(include=['float64']).columns
```




    Index(['thumbnail_url', 'medium_url', 'xl_picture_url', 'host_acceptance_rate',
           'host_listings_count', 'host_total_listings_count', 'latitude',
           'longitude', 'bathrooms', 'bedrooms', 'beds', 'square_feet',
           'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',
           'review_scores_rating', 'review_scores_accuracy',
           'review_scores_cleanliness', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location',
           'review_scores_value', 'reviews_per_month'],
          dtype='object')




```python
drop_float_cols = ['thumbnail_url',
                   'medium_url',
                   'xl_picture_url',
                   #'latitude',
                   #'longitude'
                  ]
```


```python
df_detailed_listings.select_dtypes(include=['float64']).head()
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
      <th>thumbnail_url</th>
      <th>medium_url</th>
      <th>xl_picture_url</th>
      <th>host_acceptance_rate</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>square_feet</th>
      <th>minimum_nights_avg_ntm</th>
      <th>maximum_nights_avg_ntm</th>
      <th>review_scores_rating</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>reviews_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1125.0</td>
      <td>95.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>500.0</td>
      <td>1.0</td>
      <td>730.0</td>
      <td>90.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>4.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>120.0</td>
      <td>93.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>40.74767</td>
      <td>-73.97500</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>21.0</td>
      <td>89.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>0.60</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_detailed_listings.select_dtypes(include=['int64']).columns
```




    Index(['id', 'scrape_id', 'host_id', 'accommodates', 'guests_included',
           'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
           'maximum_minimum_nights', 'minimum_maximum_nights',
           'maximum_maximum_nights', 'availability_30', 'availability_60',
           'availability_90', 'availability_365', 'number_of_reviews',
           'number_of_reviews_ltm', 'calculated_host_listings_count',
           'calculated_host_listings_count_entire_homes',
           'calculated_host_listings_count_private_rooms',
           'calculated_host_listings_count_shared_rooms'],
          dtype='object')




```python
drop_int_cols = ['scrape_id',
                 'host_id',
                 'minimum_minimum_nights',
                 'maximum_minimum_nights',
                 'minimum_maximum_nights',
                 'maximum_maximum_nights'
                ]
```


```python
df_detailed_listings.select_dtypes(include=['int64']).head()
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
      <th>scrape_id</th>
      <th>host_id</th>
      <th>accommodates</th>
      <th>guests_included</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>minimum_minimum_nights</th>
      <th>maximum_minimum_nights</th>
      <th>minimum_maximum_nights</th>
      <th>maximum_maximum_nights</th>
      <th>availability_30</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>availability_365</th>
      <th>number_of_reviews</th>
      <th>number_of_reviews_ltm</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>20190806030549</td>
      <td>2845</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1125</td>
      <td>1</td>
      <td>1</td>
      <td>1125</td>
      <td>1125</td>
      <td>13</td>
      <td>17</td>
      <td>31</td>
      <td>288</td>
      <td>46</td>
      <td>12</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3647</td>
      <td>20190806030549</td>
      <td>4632</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>7</td>
      <td>30</td>
      <td>60</td>
      <td>90</td>
      <td>365</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3831</td>
      <td>20190806030549</td>
      <td>4869</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>730</td>
      <td>1</td>
      <td>1</td>
      <td>730</td>
      <td>730</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>212</td>
      <td>274</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5022</td>
      <td>20190806030549</td>
      <td>7192</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>120</td>
      <td>10</td>
      <td>10</td>
      <td>120</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5099</td>
      <td>20190806030549</td>
      <td>7322</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>21</td>
      <td>3</td>
      <td>3</td>
      <td>21</td>
      <td>21</td>
      <td>24</td>
      <td>33</td>
      <td>63</td>
      <td>127</td>
      <td>75</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
drop_cols = drop_object_cols + drop_float_cols + drop_int_cols + ['experiences_offered']
len(drop_cols)
```




    33




```python
drop_cols
```




    ['listing_url',
     'last_scraped',
     'picture_url',
     'host_url',
     'host_name',
     'host_since',
     'host_location',
     'host_about',
     'host_thumbnail_url',
     'host_picture_url',
     'host_neighbourhood',
     'street',
     'city',
     'state',
     'zipcode',
     'market',
     'smart_location',
     'country_code',
     'country',
     'calendar_updated',
     'calendar_last_scraped',
     'first_review',
     'last_review',
     'thumbnail_url',
     'medium_url',
     'xl_picture_url',
     'scrape_id',
     'host_id',
     'minimum_minimum_nights',
     'maximum_minimum_nights',
     'minimum_maximum_nights',
     'maximum_maximum_nights',
     'experiences_offered']




```python
df_dl_clean_df = df_detailed_listings.drop(columns=drop_cols)
```


```python
df_dl_clean_df.shape
```




    (48864, 73)



## Missing Value analysis
Calculate missing value statistics


```python
print(pd.get_option("display.max_rows"))
pd.set_option("display.max_rows", 100)
print(pd.get_option("display.max_rows"))
```

    60
    100



```python
num_missing = df_dl_clean_df.isnull().sum().to_frame()
num_missing.columns = ['num_missing']
num_missing['pct_missing'] = np.round(100 * (num_missing['num_missing'] / df_dl_clean_df.shape[0]))
num_missing.sort_values(by='num_missing', ascending=False)
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
      <th>num_missing</th>
      <th>pct_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>host_acceptance_rate</th>
      <td>48864</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>jurisdiction_names</th>
      <td>48853</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>license</th>
      <td>48842</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>square_feet</th>
      <td>48469</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>monthly_price</th>
      <td>43715</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>weekly_price</th>
      <td>42964</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>notes</th>
      <td>28692</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>access</th>
      <td>21916</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>interaction</th>
      <td>19947</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>house_rules</th>
      <td>18912</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>neighborhood_overview</th>
      <td>17297</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>security_deposit</th>
      <td>17290</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>transit</th>
      <td>16975</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>16582</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>host_response_time</th>
      <td>16582</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>space</th>
      <td>13985</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>review_scores_location</th>
      <td>11163</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>review_scores_value</th>
      <td>11161</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>review_scores_checkin</th>
      <td>11158</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>review_scores_accuracy</th>
      <td>11142</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>review_scores_communication</th>
      <td>11136</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>review_scores_cleanliness</th>
      <td>11126</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>11104</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>cleaning_fee</th>
      <td>10584</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>10131</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>summary</th>
      <td>2075</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>description</th>
      <td>843</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>56</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>42</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>27</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>host_identity_verified</th>
      <td>18</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>host_has_profile_pic</th>
      <td>18</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>host_total_listings_count</th>
      <td>18</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>host_listings_count</th>
      <td>18</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>host_is_superhost</th>
      <td>18</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>name</th>
      <td>16</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>neighbourhood</th>
      <td>11</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>cancellation_policy</th>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>is_location_exact</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>host_verifications</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>requires_license</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>is_business_travel_ready</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>instant_bookable</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>require_guest_profile_picture</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>require_guest_phone_verification</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count_entire_homes</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count_private_rooms</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count_shared_rooms</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>number_of_reviews_ltm</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>availability_90</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>neighbourhood_cleansed</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>room_type</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>bed_type</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>amenities</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>price</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>neighbourhood_group_cleansed</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>guests_included</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>property_type</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>extra_people</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>minimum_nights</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>maximum_nights</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>minimum_nights_avg_ntm</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>maximum_nights_avg_ntm</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>has_availability</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>availability_30</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>availability_60</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>id</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Analyze amenities


```python
df_dl_clean_df['amenities'] =  df_dl_clean_df['amenities'].apply(lambda x: x[1:-1])
```


```python
amenities = df_dl_clean_df['amenities']
```


```python
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
len(amenities_idx)
```




    131




```python
len(corpus)
```




    48864




```python
# Get the number of items and tokens 
M = len(amenities)
N = len(amenities_idx)

# Initialize a matrix of zeros
A = np.zeros((M, N))
```


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




    (48864, 131)




```python
from sklearn.manifold import TSNE
```


```python
# Dimension reduction with t-SNE
model = TSNE(n_components=2, learning_rate=200, random_state=42)
tsne_features = model.fit_transform(A)
```


```python
tsne_df = pd.DataFrame({'TSNE1': tsne_features[:,0], 
              'TSNE2': tsne_features[:,1]
             })
```


```python
subset_cols = ['id', 'price', 'amenities', 
               'neighbourhood_cleansed', 
               'neighbourhood_group_cleansed',
               'latitude', 'longitude'
              ]
```


```python
df_dl_clean_df = df_dl_clean_df[subset_cols]
```


```python
df_dl_clean_df = pd.concat([df_dl_clean_df, tsne_df], axis='columns')
```


```python
df_dl_clean_df.head()
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
      <th>price</th>
      <th>amenities</th>
      <th>neighbourhood_cleansed</th>
      <th>neighbourhood_group_cleansed</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>TSNE1</th>
      <th>TSNE2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>$225.00</td>
      <td>TV,Wifi,"Air conditioning",Kitchen,"Paid parki...</td>
      <td>Midtown</td>
      <td>Manhattan</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>42.803360</td>
      <td>-10.623417</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3647</td>
      <td>$150.00</td>
      <td>"Cable TV",Internet,Wifi,"Air conditioning",Ki...</td>
      <td>Harlem</td>
      <td>Manhattan</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>-34.314285</td>
      <td>13.559643</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3831</td>
      <td>$89.00</td>
      <td>TV,"Cable TV",Internet,Wifi,"Air conditioning"...</td>
      <td>Clinton Hill</td>
      <td>Brooklyn</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>3.406300</td>
      <td>-22.669363</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5022</td>
      <td>$80.00</td>
      <td>Internet,Wifi,"Air conditioning",Kitchen,Eleva...</td>
      <td>East Harlem</td>
      <td>Manhattan</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>-11.913917</td>
      <td>-8.425117</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5099</td>
      <td>$200.00</td>
      <td>TV,"Cable TV",Internet,Wifi,Kitchen,"Buzzer/wi...</td>
      <td>Murray Hill</td>
      <td>Manhattan</td>
      <td>40.74767</td>
      <td>-73.97500</td>
      <td>-12.099396</td>
      <td>-1.838848</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_dl_clean_df.he
```

### Visualizing t-sne results


```python
from bokeh.io import show, output_notebook, push_notebook, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
output_notebook()

```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="1001">Loading BokehJS ...</span>
    </div>





```python
# Make a source and a scatter plot  
source = ColumnDataSource(df_dl_clean_df[:100])
```


```python
plot = figure(x_axis_label = 'T-SNE 1', 
              y_axis_label = 'T-SNE 2', 
              width = 500, height = 400)
plot.circle(x = 'TSNE1', 
    y = 'TSNE2', 
    source = source, 
    size = 10, color = '#FF7373', alpha = .8)
```




<div style="display: table;"><div style="display: table-row;"><div style="display: table-cell;"><b title="bokeh.models.renderers.GlyphRenderer">GlyphRenderer</b>(</div><div style="display: table-cell;">id&nbsp;=&nbsp;'1408', <span id="1411" style="cursor: pointer;">&hellip;)</span></div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">data_source&nbsp;=&nbsp;ColumnDataSource(id='1370', ...),</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">glyph&nbsp;=&nbsp;Circle(id='1406', ...),</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">hover_glyph&nbsp;=&nbsp;None,</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_event_callbacks&nbsp;=&nbsp;{},</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_property_callbacks&nbsp;=&nbsp;{},</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">level&nbsp;=&nbsp;'glyph',</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">muted&nbsp;=&nbsp;False,</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">muted_glyph&nbsp;=&nbsp;None,</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">name&nbsp;=&nbsp;None,</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">nonselection_glyph&nbsp;=&nbsp;Circle(id='1407', ...),</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">selection_glyph&nbsp;=&nbsp;None,</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">subscribed_events&nbsp;=&nbsp;[],</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">tags&nbsp;=&nbsp;[],</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">view&nbsp;=&nbsp;CDSView(id='1409', ...),</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">visible&nbsp;=&nbsp;True,</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_range_name&nbsp;=&nbsp;'default',</div></div><div class="1410" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_range_name&nbsp;=&nbsp;'default')</div></div></div>
<script>
(function() {
  var expanded = false;
  var ellipsis = document.getElementById("1411");
  ellipsis.addEventListener("click", function() {
    var rows = document.getElementsByClassName("1410");
    for (var i = 0; i < rows.length; i++) {
      var el = rows[i];
      el.style.display = expanded ? "none" : "table-row";
    }
    ellipsis.innerHTML = expanded ? "&hellip;)" : "&lsaquo;&lsaquo;&lsaquo;";
    expanded = !expanded;
  });
})();
</script>





```python
# Create a HoverTool object
hover = HoverTool(tooltips = [('id', '@id'),
                              ('price', '$@price')
                             ])
plot.add_tools(hover)
```


```python
output_file('amenities.html')
show(plot)
```

<iframe src="amenities.html"
    sandbox="allow-same-origin allow-scripts"
    width="100%"
    height="600"
    scrolling="yes"
    seamless="seamless"
    frameborder="0">
</iframe>











## To do


```python
# more columns that can be removed
# experiences_offered, contains all none's so no use.
more_cols = ['experiences_offered']
```
