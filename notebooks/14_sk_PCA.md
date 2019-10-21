

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


```python
sagemaker.__version__
```




    '1.34.0'




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
     'feature/airbnb_final.csv',
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
#file = 'feature/airbnb_final.csv'
file = 'feature_eng/min_max_scaled_final_df.csv'
df_airbnb = get_data_frame(bucket_name, file)
```


```python
from sagemaker import get_execution_role

session = sagemaker.Session() # store the current SageMaker session

# get IAM role
role = get_execution_role()
print(role)
```

    arn:aws:iam::506140549518:role/service-role/AmazonSageMaker-ExecutionRole-20190827T125122



```python
# get default bucket
bucket_name = session.default_bucket()
print(bucket_name)
print()
```

    sagemaker-us-east-1-506140549518
    



```python
# define location to store model artifacts
prefix = 'pca'

output_path='s3://{}/{}/'.format(bucket_name, prefix)

print('Training artifacts will be uploaded to: {}'.format(output_path))
```

    Training artifacts will be uploaded to: s3://sagemaker-us-east-1-506140549518/pca/



```python
# define a PCA model
from sagemaker import PCA

# this is current features - 1
# you'll select only a portion of these to use, later
N_COMPONENTS=50

pca_SM = PCA(role=role,
             train_instance_count=1,
             train_instance_type='ml.c4.xlarge',
             output_path=output_path, # specified, above
             num_components=N_COMPONENTS, 
             sagemaker_session=session)
```


```python
# convert df to np array
train_data_np = df_airbnb.values.astype('float32')

# convert to RecordSet format
formatted_train_data = pca_SM.record_set(train_data_np)
```


```python
%%time

# train the PCA mode on the formatted data
pca_SM.fit(formatted_train_data)
```

    2019-10-10 15:40:44 Starting - Starting the training job...
    2019-10-10 15:40:46 Starting - Launching requested ML instances......
    2019-10-10 15:41:47 Starting - Preparing the instances for training...
    2019-10-10 15:42:45 Downloading - Downloading input data...
    2019-10-10 15:43:11 Training - Downloading the training image...
    2019-10-10 15:43:30 Training - Training image download completed. Training in progress.
    [31mDocker entrypoint called with argument(s): train[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/resources/default-conf.json: {u'_num_gpus': u'auto', u'_log_level': u'info', u'subtract_mean': u'true', u'force_dense': u'true', u'epochs': 1, u'algorithm_mode': u'regular', u'extra_components': u'-1', u'_kvstore': u'dist_sync', u'_num_kv_servers': u'auto'}[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'feature_dim': u'2201', u'mini_batch_size': u'500', u'num_components': u'50'}[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] Final configuration: {u'num_components': u'50', u'_num_gpus': u'auto', u'_log_level': u'info', u'subtract_mean': u'true', u'force_dense': u'true', u'epochs': 1, u'algorithm_mode': u'regular', u'feature_dim': u'2201', u'extra_components': u'-1', u'_kvstore': u'dist_sync', u'_num_kv_servers': u'auto', u'mini_batch_size': u'500'}[0m
    [31m[10/10/2019 15:43:33 WARNING 139778775783232] Loggers have already been setup.[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] Launching parameter server for role scheduler[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/ba5eebc5-ee5c-460d-bbf7-3d6d5aee844f', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'AWS_REGION': 'us-east-1', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'pca-2019-10-10-15-40-44-219', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-78-79.ec2.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/4a0a522e-6c3f-42a8-927c-3175035591be', 'PWD': '/', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-1:506140549518:training-job/pca-2019-10-10-15-40-44-219', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] envs={'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/ba5eebc5-ee5c-460d-bbf7-3d6d5aee844f', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_NUM_WORKER': '1', 'DMLC_PS_ROOT_PORT': '9000', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.0.78.79', 'AWS_REGION': 'us-east-1', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'pca-2019-10-10-15-40-44-219', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-78-79.ec2.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/4a0a522e-6c3f-42a8-927c-3175035591be', 'DMLC_ROLE': 'scheduler', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-1:506140549518:training-job/pca-2019-10-10-15-40-44-219', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] Launching parameter server for role server[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/ba5eebc5-ee5c-460d-bbf7-3d6d5aee844f', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'AWS_REGION': 'us-east-1', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'pca-2019-10-10-15-40-44-219', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-78-79.ec2.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/4a0a522e-6c3f-42a8-927c-3175035591be', 'PWD': '/', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-1:506140549518:training-job/pca-2019-10-10-15-40-44-219', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] envs={'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/ba5eebc5-ee5c-460d-bbf7-3d6d5aee844f', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_NUM_WORKER': '1', 'DMLC_PS_ROOT_PORT': '9000', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.0.78.79', 'AWS_REGION': 'us-east-1', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'pca-2019-10-10-15-40-44-219', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-78-79.ec2.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/4a0a522e-6c3f-42a8-927c-3175035591be', 'DMLC_ROLE': 'server', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-1:506140549518:training-job/pca-2019-10-10-15-40-44-219', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] Environment: {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/ba5eebc5-ee5c-460d-bbf7-3d6d5aee844f', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_PS_ROOT_PORT': '9000', 'DMLC_NUM_WORKER': '1', 'SAGEMAKER_HTTP_PORT': '8080', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.0.78.79', 'AWS_REGION': 'us-east-1', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'pca-2019-10-10-15-40-44-219', 'HOME': '/root', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-78-79.ec2.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/4a0a522e-6c3f-42a8-927c-3175035591be', 'DMLC_ROLE': 'worker', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-1:506140549518:training-job/pca-2019-10-10-15-40-44-219', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [31mProcess 61 is a shell:scheduler.[0m
    [31mProcess 70 is a shell:server.[0m
    [31mProcess 1 is a worker.[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] Using default worker.[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] Loaded iterator creator application/x-recordio-protobuf for content type ('application/x-recordio-protobuf', '1.0')[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] Loaded iterator creator application/x-labeled-vector-protobuf for content type ('application/x-labeled-vector-protobuf', '1.0')[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] Loaded iterator creator protobuf for content type ('protobuf', '1.0')[0m
    [31m[10/10/2019 15:43:33 INFO 139778775783232] Create Store: dist_sync[0m
    [31m[10/10/2019 15:43:34 INFO 139778775783232] nvidia-smi took: 0.0251910686493 secs to identify 0 gpus[0m
    [31m[10/10/2019 15:43:34 INFO 139778775783232] Number of GPUs being used: 0[0m
    [31m[10/10/2019 15:43:34 INFO 139778775783232] The default executor is <PCAExecutor on cpu(0)>.[0m
    [31m[10/10/2019 15:43:34 INFO 139778775783232] 2201 feature(s) found in 'data'.[0m
    [31m[10/10/2019 15:43:34 INFO 139778775783232] <PCAExecutor on cpu(0)> is assigned to batch slice from 0 to 499.[0m
    [31m#metrics {"Metrics": {"initialize.time": {"count": 1, "max": 748.8198280334473, "sum": 748.8198280334473, "min": 748.8198280334473}}, "EndTime": 1570722214.024792, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1570722213.260874}
    [0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Batches Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Records Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Batches Seen": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Records Seen": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Max Records Seen Between Resets": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Reset Count": {"count": 1, "max": 0, "sum": 0.0, "min": 0}}, "EndTime": 1570722214.025017, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1570722214.024979}
    [0m
    [31m[2019-10-10 15:43:34.033] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 0, "duration": 771, "num_examples": 1, "num_bytes": 4416000}[0m
    [31m[2019-10-10 15:43:38.249] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 1, "duration": 4196, "num_examples": 92, "num_bytes": 402783360}[0m
    [31m#metrics {"Metrics": {"epochs": {"count": 1, "max": 1, "sum": 1.0, "min": 1}, "update.time": {"count": 1, "max": 4215.699911117554, "sum": 4215.699911117554, "min": 4215.699911117554}}, "EndTime": 1570722218.249515, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1570722214.0249}
    [0m
    [31m[10/10/2019 15:43:38 INFO 139778775783232] #progress_metric: host=algo-1, completed 100 % of epochs[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 92, "sum": 92.0, "min": 92}, "Number of Batches Since Last Reset": {"count": 1, "max": 92, "sum": 92.0, "min": 92}, "Number of Records Since Last Reset": {"count": 1, "max": 45605, "sum": 45605.0, "min": 45605}, "Total Batches Seen": {"count": 1, "max": 92, "sum": 92.0, "min": 92}, "Total Records Seen": {"count": 1, "max": 45605, "sum": 45605.0, "min": 45605}, "Max Records Seen Between Resets": {"count": 1, "max": 45605, "sum": 45605.0, "min": 45605}, "Reset Count": {"count": 1, "max": 1, "sum": 1.0, "min": 1}}, "EndTime": 1570722218.250209, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "PCA", "epoch": 0}, "StartTime": 1570722214.033756}
    [0m
    [31m[10/10/2019 15:43:38 INFO 139778775783232] #throughput_metric: host=algo-1, train throughput=10814.9061736 records/second[0m
    [31m#metrics {"Metrics": {"finalize.time": {"count": 1, "max": 2927.04701423645, "sum": 2927.04701423645, "min": 2927.04701423645}}, "EndTime": 1570722221.178063, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1570722218.249734}
    [0m
    [31m[10/10/2019 15:43:41 INFO 139778775783232] Test data is not provided.[0m
    [31m#metrics {"Metrics": {"totaltime": {"count": 1, "max": 8062.474966049194, "sum": 8062.474966049194, "min": 8062.474966049194}, "setuptime": {"count": 1, "max": 38.812875747680664, "sum": 38.812875747680664, "min": 38.812875747680664}}, "EndTime": 1570722221.197231, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1570722221.178139}
    [0m
    
    2019-10-10 15:43:52 Uploading - Uploading generated training model
    2019-10-10 15:43:52 Completed - Training job completed
    Billable seconds: 67
    CPU times: user 442 ms, sys: 31.5 ms, total: 473 ms
    Wall time: 3min 42s



```python
training_job_name='pca-2019-10-10-15-40-44-219'

# where the model is saved, by default
model_key = os.path.join(prefix, training_job_name, 'output/model.tar.gz')
print(model_key)

# download and unzip model
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')

# unzipping as model_algo-1
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')
```

    pca/pca-2019-10-10-15-40-44-219/output/model.tar.gz





    2304




```python
import mxnet as mx

# loading the unzipped artifacts
pca_model_params = mx.ndarray.load('model_algo-1')

# what are the params
print(pca_model_params)
```

    {'s': 
    [ 57.05088   57.874645  58.433697  59.175858  59.865147  61.96779
      63.226593  64.11992   66.35956   67.24171   67.67469   68.12071
      69.39432   70.41575   71.05591   71.61029   72.413445  72.68548
      74.65268   77.155876  79.46062   80.077995  82.66995   85.92465
      86.539955  87.50056   88.780014  89.40145   90.21015   94.273415
      96.85173   97.366425 100.695    103.57861  105.91536  107.30245
     107.82814  109.70534  112.907974 116.73672  118.375206 121.84918
     129.90123  139.94371  151.06177  163.72157  169.06854  177.17932
     211.70601  311.55603 ]
    <NDArray 50 @cpu(0)>, 'v': 
    [[-5.7720104e-03 -5.3913011e-03  9.0847909e-03 ...  3.0375753e-02
      -1.0498090e-02 -1.1980495e-02]
     [-2.6859946e-03  5.4610777e-04  8.8122051e-04 ...  3.9079497e-03
       3.9688114e-04 -1.6078530e-03]
     [ 8.4759603e-04 -1.2613757e-03  2.8918835e-03 ...  9.3708942e-03
      -3.0037272e-03 -5.2033747e-03]
     ...
     [-4.3243897e-04 -1.7060355e-04 -2.5075878e-04 ... -1.8658712e-04
       1.1779055e-05 -1.6026984e-06]
     [-6.7385123e-04  2.9283887e-04  3.0488582e-05 ...  3.8099787e-04
      -1.5027729e-04 -2.9955751e-05]
     [-1.7711901e-06  3.9155810e-05 -2.6563127e-04 ... -2.2606198e-05
      -7.4653940e-06 -3.0313136e-05]]
    <NDArray 2201x50 @cpu(0)>, 'mean': 
    [[0.07277273 0.07373131 0.0841825  ... 0.00040985 0.00064066 0.00043103]]
    <NDArray 1x2201 @cpu(0)>}



```python
# get selected params
s=pd.DataFrame(pca_model_params['s'].asnumpy())
v=pd.DataFrame(pca_model_params['v'].asnumpy())
```


```python
# looking at top 20 components
n_principal_components = 20

start_idx = N_COMPONENTS - n_principal_components  # 50-n

# print a selection of s
print(s.iloc[start_idx:, :])
```

                 0
    30   96.851730
    31   97.366425
    32  100.695000
    33  103.578613
    34  105.915359
    35  107.302452
    36  107.828140
    37  109.705338
    38  112.907974
    39  116.736717
    40  118.375206
    41  121.849182
    42  129.901230
    43  139.943710
    44  151.061768
    45  163.721573
    46  169.068542
    47  177.179321
    48  211.706009
    49  311.556030



```python
# Calculate the explained variance for the top n principal components
# you may assume you have access to the global var N_COMPONENTS
def explained_variance(s, n_top_components):
    '''Calculates the approx. data variance that n_top_components captures.
       :param s: A dataframe of singular values for top components; 
           the top value is in the last row.
       :param n_top_components: An integer, the number of top components to use.
       :return: The expected data variance covered by the n_top_components.'''
    
    start_idx = N_COMPONENTS - n_top_components  ## 33-3 = 30, for example
    # calculate approx variance
    exp_variance = np.square(s.iloc[start_idx:,:]).sum()/np.square(s).sum()
    
    return exp_variance[0]
```


```python
# test cell
n_top_components = 30 # select a value for the number of top components

# calculate the explained variance
exp_variance = explained_variance(s, n_top_components)
print('Explained variance: ', exp_variance)
```

    Explained variance:  0.8496999



```python
# features
features_list = df_airbnb.columns.values
print('Features: \n', features_list)
```

    Features: 
     ['accommodates' 'bathrooms' 'bedrooms' ... 'description_contains_yummy'
     'description_contains_zero' 'description_contains_zone']



```python
import seaborn as sns

def display_component(v, features_list, component_num, n_weights=10):
    
    # get index of component (last row - component_num)
    row_idx = N_COMPONENTS-component_num

    # get the list of weights from a row in v, dataframe
    v_1_row = v.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe, using list comporehension
    comps = pd.DataFrame(list(zip(v_1, features_list)), 
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(n_weights)

    # display using seaborn
    ax=plt.subplots(figsize=(10,6))
    ax=sns.barplot(data=sorted_weight_data, 
                   x="weights", 
                   y="features", 
                   palette="Blues_d")
    ax.set_title("PCA Component Makeup, Component #" + str(component_num))
    plt.show()
```


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


## Deploy PCA


```python
%%time
# this takes a little while, around 7mins
pca_predictor = pca_SM.deploy(initial_instance_count=1, 
                              instance_type='ml.c4.xlarge')
```

    --------------------------------------------------------------------------------------------------!CPU times: user 534 ms, sys: 27.2 ms, total: 561 ms
    Wall time: 8min 15s



```python
type(train_data_np)
```




    numpy.ndarray




```python
train_data_np = train_data_np[:1000,:]
```


```python
# pass np train data to the PCA model
train_pca = pca_predictor.predict(train_data_np)
```


```python
train_pca
```




    [label {
       key: "projection"
       value {
         float32_tensor {
           values: -0.21395865082740784
           values: 0.3977692425251007
           values: 0.7837661504745483
           values: -0.9496099948883057
           values: -1.658820390701294
           values: 0.33218762278556824
           values: 0.249675452709198
           values: -0.32349181175231934
           values: 0.4614548087120056
           values: 0.5768586993217468
           values: -0.1243261992931366
           values: 0.26902806758880615
           values: 0.2574237287044525
           values: 0.3664625287055969
           values: 0.2585570812225342
           values: 0.6496644020080566
           values: -0.39550015330314636
           values: 0.1076347827911377
           values: 0.5901657342910767
           values: 0.5803667306900024
           values: 0.772720992565155
           values: -0.02226695418357849
           values: -0.2864333987236023
           values: 0.011405829340219498
           values: 0.04967791587114334
           values: 0.182930588722229
           values: 0.0781865119934082
           values: 0.2640266716480255
           values: 0.33040809631347656
           values: 0.2742791175842285
           values: -0.36675137281417847
           values: 0.21018344163894653
           values: -0.09859494864940643
           values: 0.5178123116493225
           values: -0.30106881260871887
           values: -1.1945672035217285
           values: 0.6428004503250122
           values: -0.5585151314735413
           values: 0.10400271415710449
           values: -0.016680851578712463
           values: 1.3331857919692993
           values: -0.10858157277107239
           values: -0.0013564229011535645
           values: 0.04471306502819061
           values: 0.6541873812675476
           values: 0.12538379430770874
           values: -0.5534112453460693
           values: -0.6990922689437866
           values: -1.4555258750915527
           values: -2.658553123474121
         }
       }
     }, label {
       key: "projection"
       value {
         float32_tensor {
           values: -0.13851720094680786
           values: 0.7106360197067261
           values: 0.2878350615501404
           values: -0.7536268830299377
           values: -1.3909412622451782
           values: -0.15032155811786652
           values: -0.1984279602766037
           values: -0.2614770531654358
           values: 0.21960783004760742
           values: -0.2503553330898285
           values: -0.30338364839553833
           values: 0.2088761180639267
           values: 0.24588486552238464
           values: -0.39024803042411804
           values: -0.32250040769577026
           values: -0.22463619709014893
           values: -0.213418111205101
           values: 0.10443514585494995
           values: -0.03314867615699768
           values: -0.2534470558166504
           values: 0.15767759084701538
           values: -0.267532616853714
           values: 0.49453139305114746
           values: -0.030308060348033905
           values: 0.028240207582712173
           values: -0.2023022472858429
           values: -0.21827279031276703
           values: 0.04996345192193985
           values: 0.25285422801971436
           values: 0.4854140281677246
           values: 0.4116573631763458
           values: -0.24031013250350952
           values: -1.1106432676315308
           values: -0.37595656514167786
           values: -0.38762396574020386
           values: 0.818548858165741
           values: -0.18803207576274872
           values: 0.26144206523895264
           values: -0.43905508518218994
           values: 0.09178748726844788
           values: 0.15190742909908295
           values: -0.9968162775039673
           values: -1.132571816444397
           values: 1.1838425397872925
           values: 0.1397506296634674
           values: -0.39901405572891235
           values: 0.8938336372375488
           values: -1.097098708152771
           values: -0.7153205275535583
           values: 1.644171953201294
         }
       }
     }, label {
       key: "projection"
       value {
         float32_tensor {
           values: -0.2064109444618225
           values: 0.8512427806854248
           values: -0.046416282653808594
           values: -0.7389469146728516
           values: -1.1430038213729858
           values: 0.16617056727409363
           values: 0.19558162987232208
           values: -0.008042484521865845
           values: 0.0640716552734375
           values: 0.19374945759773254
           values: 0.20728787779808044
           values: -0.009931221604347229
           values: 0.17424187064170837
           values: -0.37693899869918823
           values: -0.34815752506256104
           values: -0.07355278730392456
           values: -0.2439824789762497
           values: 0.11462992429733276
           values: -0.37285512685775757
           values: -0.23959194123744965
           values: -0.3089516758918762
           values: -1.0409849882125854
           values: -0.03016635775566101
           values: -0.158334881067276
           values: -0.18962129950523376
           values: -0.5417731404304504
           values: 0.5900906324386597
           values: -0.7460067272186279
           values: -0.7637325525283813
           values: 0.7670818567276001
           values: -0.2516971230506897
           values: -0.9787700176239014
           values: 0.2811269462108612
           values: -0.3954581320285797
           values: 0.29448050260543823
           values: -0.839026927947998
           values: 0.48325201869010925
           values: -0.298027902841568
           values: 0.5877614617347717
           values: -0.08970455825328827
           values: -0.20068460702896118
           values: 0.1334562599658966
           values: -0.8190857172012329
           values: 0.39956891536712646
           values: -0.39278021454811096
           values: -0.113797128200531
           values: -0.3561122417449951
           values: 0.32088935375213623
           values: -1.7180061340332031
           values: 1.0121126174926758
         }
       }
     }, label {
       key: "projection"
       value {
         float32_tensor {
           values: -0.20509755611419678
           values: 0.8615318536758423
           values: 0.0834275558590889
           values: -0.6357015371322632
           values: -1.4638893604278564
           values: -0.19650201499462128
           values: 0.13995406031608582
           values: -0.06671357154846191
           values: -0.06162291765213013
           values: 0.41699135303497314
           values: 0.5131653547286987
           values: -0.19667057693004608
           values: 0.0874955952167511
           values: -0.2453630268573761
           values: 0.5001711249351501
           values: -0.043177202343940735
           values: -0.35928887128829956
           values: -0.5198426246643066
           values: -0.6334847211837769
           values: 0.09694081544876099
           values: 0.4976940155029297
           values: -0.3448166847229004
           values: 0.46302783489227295
           values: 0.1340734213590622
           values: 0.08215348422527313
           values: -0.4435919523239136
           values: -0.313030481338501
           values: 0.6207235455513
           values: -0.2620559334754944
           values: -0.4471712112426758
           values: 0.20006299018859863
           values: -0.8956047296524048
           values: -0.7054430246353149
           values: -0.52272629737854
           values: -0.2041417956352234
           values: 0.023345649242401123
           values: -0.34916257858276367
           values: -0.09813927114009857
           values: -0.8931775093078613
           values: 1.7370473146438599
           values: 0.7802249193191528
           values: 0.07157206535339355
           values: -0.6246670484542847
           values: 1.1588881015777588
           values: 0.9060518145561218
           values: 0.9071215987205505
           values: 0.35388028621673584
           values: 0.26386523246765137
           values: -0.5209187865257263
           values: -0.06740367412567139
         }
       }
     }, label {
       key: "projection"
       value {
         float32_tensor {
           values: 0.2501308023929596
           values: -0.7153280973434448
           values: -0.3311329483985901
           values: 0.022556066513061523
           values: -0.4054851531982422
           values: 0.19486813247203827
           values: 0.08833807706832886
           values: -0.2258608341217041
           values: -0.09147170186042786
           values: -0.213420107960701
           values: -0.10260772705078125
           values: -0.08259693533182144
           values: 0.2940676510334015
           values: -0.022778701037168503
           values: -0.25677746534347534
           values: -0.31123465299606323
           values: -0.03363800048828125
           values: 0.26643621921539307
           values: 0.03740313649177551
           values: -0.01506778597831726
           values: 0.07985860109329224
           values: 0.0628783106803894
           values: -0.08831170201301575
           values: -0.4336599111557007
           values: -0.3484685719013214
           values: -0.33926838636398315
           values: 0.4900835156440735
           values: 0.38715922832489014
           values: -0.04190504550933838
           values: -0.5256760120391846
           values: 0.07550954818725586
           values: 0.4151379466056824
           values: 0.35024499893188477
           values: 0.1988077610731125
           values: -0.10519754886627197
           values: -1.1099836826324463
           values: 0.43931201100349426
           values: 0.30570727586746216
           values: 0.6841707229614258
           values: 0.5769422054290771
           values: 0.18060439825057983
           values: -0.010625362396240234
           values: -0.6786844730377197
           values: 0.8275101780891418
           values: 0.516450047492981
           values: -1.2622392177581787
           values: 0.6590332984924316
           values: -0.520532488822937
           values: 1.2962031364440918
           values: 1.666905403137207
         }
       }
     }, label {
       key: "projection"
       value {
         float32_tensor {
           values: 0.14163830876350403
           values: -0.2802896797657013
           values: 0.15324299037456512
           values: -0.5633378028869629
           values: 0.5992924571037292
           values: 0.6380820870399475
           values: 0.07077392190694809
           values: -1.0696269273757935
           values: 0.0036900490522384644
           values: 0.15354694426059723
           values: -0.06172342598438263
           values: 0.13671241700649261
           values: 0.14789247512817383
           values: 0.07177020609378815
           values: 0.36216622591018677
           values: -0.37686705589294434
           values: 0.025177687406539917
           values: -0.13659322261810303
           values: 0.8344812989234924
           values: 0.4049750864505768
           values: 0.25906699895858765
           values: 0.49874642491340637
           values: -0.12310685962438583
           values: -0.7394569516181946
           values: 0.31479769945144653
           values: -1.0155630111694336
           values: 0.35460472106933594
           values: -0.1511269211769104
           values: -0.4476776719093323
           values: -0.4751843810081482
           values: -0.8676798939704895
           values: -0.025009647011756897
           values: 0.17393255233764648
           values: 0.6282758712768555
           values: -0.5544185638427734
           values: 0.059601008892059326
           values: -0.15958066284656525
           values: -0.7866604328155518
           values: -0.15836966037750244
           values: 0.1578144133090973
           values: -0.2071094512939453
           values: 0.6523106098175049
           values: -0.3140662908554077
           values: 0.744392991065979
           values: 0.18536821007728577
           values: 0.2014126181602478
           values: -0.5161429643630981
           values: -1.0810562372207642
           values: 0.046402692794799805
           values: -0.22471225261688232
         }
       }
     }, label {
       key: "projection"
       value {
         float32_tensor {
           values: 0.2746610939502716
           values: -0.27568379044532776
           values: 0.04715646058320999
           values: -1.2746964693069458
           values: -1.5352091789245605
           values: 0.20815427601337433
           values: 0.010859020054340363
           values: -0.7050411105155945
           values: 0.4280134439468384
           values: 0.15874135494232178
           values: 0.23133635520935059
           values: -0.13916577398777008
           values: 0.2590964138507843
           values: -0.11804334819316864
           values: -0.706478476524353
           values: -0.6639591455459595
           values: -0.29119595885276794
           values: -0.1849147081375122
           values: -0.4742773771286011
           values: 0.48630115389823914
           values: -0.4297211766242981
           values: -0.7556835412979126
           values: -0.7944867014884949
           values: 0.26790574193000793
           values: -0.4737450182437897
           values: -0.2902553081512451
           values: 0.48863470554351807
           values: -0.292998731136322
           values: -0.5220605134963989
           values: 0.7102686166763306
           values: 0.35730868577957153
           values: -1.548667311668396
           values: -0.3718993067741394
           values: -0.1263469159603119
           values: 0.14627662301063538
           values: -0.20152133703231812
           values: -0.2737290561199188
           values: -0.1884356439113617
           values: -0.6491206884384155
           values: 0.8458350896835327
           values: -0.3092692494392395
           values: 0.14799603819847107
           values: 0.24830332398414612
           values: 0.2162850946187973
           values: -0.6676642894744873
           values: 0.3370961546897888
           values: 1.1703094244003296
           values: -1.0749139785766602
           values: -0.9858936667442322
           values: 0.9986637830734253
         }
       }
     }, label {
       key: "projection"
       value {
         float32_tensor {
           values: 0.4854779839515686
           values: 0.11105558276176453
           values: -0.018837466835975647
           values: 0.30717289447784424
           values: 0.04492729902267456
           values: -0.3001968264579773
           values: -0.028481919318437576
           values: -0.1362604796886444
           values: -0.09201416373252869
           values: 0.3795551061630249
           values: -0.10602876543998718
           values: -0.13658300042152405
           values: 0.25565996766090393
           values: -0.3511468470096588
           values: 0.18002764880657196
           values: -0.37637394666671753
           values: -0.20005279779434204
           values: 0.28363847732543945
           values: 0.5097999572753906
           values: 0.01988011598587036
           values: 0.10520046949386597
           values: -0.6156152486801147
           values: -0.43476712703704834
           values: 0.1253090351819992
           values: -0.2890267074108124
           values: 0.4138968586921692
           values: -0.4311639070510864
           values: 0.3289690911769867
           values: 0.20543991029262543
           values: 0.46740782260894775
           values: -0.30953019857406616
           values: 0.06499174237251282
           values: -0.5821793675422668
           values: -0.23320774734020233
           values: 0.28483641147613525
           values: -0.2323845475912094
           values: 0.32469817996025085
           values: 0.15998238325119019
           values: -0.04265856742858887
           values: -0.07772776484489441
           values: -0.04046604037284851
           values: -1.4394522905349731
           values: -0.5657315850257874
           values: 0.6928197145462036
           values: 0.743298351764679
           values: -0.42322415113449097
           values: -0.695752739906311
           values: 0.06131279468536377
           values: -1.495945930480957
           values: 1.2981746196746826
         }
       }
     }, label {
       key: "projection"
       value {
         float32_tensor {
           values: -0.42339250445365906
           values: -0.02289065718650818
           values: 0.188869908452034
           values: -0.13190627098083496
           values: 0.29869574308395386
           values: 0.11022693663835526
           values: -0.34373369812965393
           values: -0.19917301833629608
           values: -0.26699596643447876
           values: -0.271071195602417
           values: 0.10552862286567688
           values: -0.12034082412719727
           values: 0.027505546808242798
           values: -0.5026354789733887
           values: 0.24481621384620667
           values: 0.019176818430423737
           values: 0.098385751247406
           values: -0.1461726427078247
           values: -0.823872983455658
           values: 0.1707848608493805
           values: 0.06009012460708618
           values: -0.562066912651062
           values: -0.17999941110610962
           values: 0.2318914830684662
           values: -0.25322338938713074
           values: 0.5999882817268372
           values: -0.013188749551773071
           values: -0.5034663081169128
           values: 0.20484326779842377
           values: 0.19268137216567993
           values: -0.21077686548233032
           values: -0.001534000039100647
           values: 0.5160103440284729
           values: -0.08106818795204163
           values: 1.3308932781219482
           values: 0.03199244290590286
           values: -0.15218111872673035
           values: 0.6111699342727661
           values: -0.15292489528656006
           values: 0.2456464022397995
           values: -0.45539504289627075
           values: 0.4350896179676056
           values: -0.4488588571548462
           values: 0.22141756117343903
           values: 0.8619904518127441
           values: 0.03939706087112427
           values: -0.5759202241897583
           values: -0.5214102268218994
           values: -1.7575619220733643
           values: -1.6690689325332642
         }
       }
     }, label {
       key: "projection"
       value {
         float32_tensor {
           values: 0.18495000898838043
           values: 0.11214172840118408
           values: -0.08923976123332977
           values: 0.05986452102661133
           values: 0.15011537075042725
           values: -0.21941164135932922
           values: -0.09698201715946198
           values: 0.5193964242935181
           values: 0.09591835737228394
           values: -0.06708450615406036
           values: 0.5321345925331116
           values: -0.3593146502971649
           values: 0.43345198035240173
           values: 0.04125887155532837
           values: -0.05602315068244934
           values: -0.2747052311897278
           values: -0.5123181939125061
           values: 0.21006733179092407
           values: 0.27895358204841614
           values: 0.3731425702571869
           values: -0.3541407585144043
           values: -1.0782705545425415
           values: -0.4792827367782593
           values: 0.24636399745941162
           values: -0.2543420195579529
           values: 0.46660739183425903
           values: -0.6420791149139404
           values: 0.1873590350151062
           values: -0.39170125126838684
           values: 1.0770469903945923
           values: 0.40922436118125916
           values: -0.340203195810318
           values: -0.1297895461320877
           values: 0.30375364422798157
           values: -0.8523931503295898
           values: -0.4236856698989868
           values: -0.45189112424850464
           values: 0.202458918094635
           values: -0.09895932674407959
           values: -0.03728306293487549
           values: 0.02796006202697754
           values: 0.2603796422481537
           values: -0.2870347201824188
           values: 0.058608606457710266
           values: 0.7438739538192749
           values: 0.5767285227775574
           values: 0.8348431587219238
           values: 0.5194814205169678
           values: -1.3384506702423096
           values: 0.523772120475769
         }
       }
     }]




```python
# check out the first item in the produced training features
data_idx = 0
print(train_pca[data_idx])
```


```python
# delete predictor endpoint
session.delete_endpoint(pca_predictor.endpoint)
```


```python
!pip install -U sagemaker==1.34.0
```

    Collecting sagemaker==1.34.0
    [?25l  Downloading https://files.pythonhosted.org/packages/fc/bf/3733873b0f870344aaa3757e583fe349e09f2ab125aabcd5a039542f8aa4/sagemaker-1.34.0.tar.gz (202kB)
    [K    100% |████████████████████████████████| 204kB 26.8MB/s ta 0:00:01
    [?25hRequirement not upgraded as not directly required: boto3>=1.9.169 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from sagemaker==1.34.0) (1.9.242)
    Requirement not upgraded as not directly required: numpy>=1.9.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from sagemaker==1.34.0) (1.14.5)
    Requirement not upgraded as not directly required: protobuf>=3.1 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from sagemaker==1.34.0) (3.5.2)
    Requirement not upgraded as not directly required: scipy>=0.19.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from sagemaker==1.34.0) (1.2.1)
    Requirement not upgraded as not directly required: urllib3<1.25,>=1.21 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from sagemaker==1.34.0) (1.23)
    Requirement not upgraded as not directly required: protobuf3-to-dict>=0.1.5 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from sagemaker==1.34.0) (0.1.5)
    Requirement not upgraded as not directly required: docker-compose>=1.23.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from sagemaker==1.34.0) (1.24.1)
    Requirement not upgraded as not directly required: requests<2.21,>=2.20.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from sagemaker==1.34.0) (2.20.0)
    Requirement not upgraded as not directly required: botocore<1.13.0,>=1.12.242 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from boto3>=1.9.169->sagemaker==1.34.0) (1.12.242)
    Requirement not upgraded as not directly required: jmespath<1.0.0,>=0.7.1 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from boto3>=1.9.169->sagemaker==1.34.0) (0.9.4)
    Requirement not upgraded as not directly required: s3transfer<0.3.0,>=0.2.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from boto3>=1.9.169->sagemaker==1.34.0) (0.2.1)
    Requirement not upgraded as not directly required: six>=1.9 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from protobuf>=3.1->sagemaker==1.34.0) (1.11.0)
    Requirement not upgraded as not directly required: setuptools in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from protobuf>=3.1->sagemaker==1.34.0) (39.1.0)
    Requirement not upgraded as not directly required: PyYAML<4.3,>=3.10 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from docker-compose>=1.23.0->sagemaker==1.34.0) (3.12)
    Requirement not upgraded as not directly required: dockerpty<0.5,>=0.4.1 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from docker-compose>=1.23.0->sagemaker==1.34.0) (0.4.1)
    Requirement not upgraded as not directly required: docopt<0.7,>=0.6.1 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from docker-compose>=1.23.0->sagemaker==1.34.0) (0.6.2)
    Requirement not upgraded as not directly required: jsonschema<3,>=2.5.1 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from docker-compose>=1.23.0->sagemaker==1.34.0) (2.6.0)
    Requirement not upgraded as not directly required: docker[ssh]<4.0,>=3.7.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from docker-compose>=1.23.0->sagemaker==1.34.0) (3.7.3)
    Requirement not upgraded as not directly required: cached-property<2,>=1.2.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from docker-compose>=1.23.0->sagemaker==1.34.0) (1.5.1)
    Requirement not upgraded as not directly required: websocket-client<1.0,>=0.32.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from docker-compose>=1.23.0->sagemaker==1.34.0) (0.56.0)
    Requirement not upgraded as not directly required: texttable<0.10,>=0.9.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from docker-compose>=1.23.0->sagemaker==1.34.0) (0.9.1)
    Requirement not upgraded as not directly required: idna<2.8,>=2.5 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from requests<2.21,>=2.20.0->sagemaker==1.34.0) (2.6)
    Requirement not upgraded as not directly required: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from requests<2.21,>=2.20.0->sagemaker==1.34.0) (2019.6.16)
    Requirement not upgraded as not directly required: chardet<3.1.0,>=3.0.2 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from requests<2.21,>=2.20.0->sagemaker==1.34.0) (3.0.4)
    Requirement not upgraded as not directly required: docutils<0.16,>=0.10 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from botocore<1.13.0,>=1.12.242->boto3>=1.9.169->sagemaker==1.34.0) (0.14)
    Requirement not upgraded as not directly required: python-dateutil<3.0.0,>=2.1; python_version >= "2.7" in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from botocore<1.13.0,>=1.12.242->boto3>=1.9.169->sagemaker==1.34.0) (2.7.3)
    Requirement not upgraded as not directly required: docker-pycreds>=0.4.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from docker[ssh]<4.0,>=3.7.0->docker-compose>=1.23.0->sagemaker==1.34.0) (0.4.0)
    Requirement not upgraded as not directly required: paramiko>=2.4.2; extra == "ssh" in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from docker[ssh]<4.0,>=3.7.0->docker-compose>=1.23.0->sagemaker==1.34.0) (2.6.0)
    Requirement not upgraded as not directly required: bcrypt>=3.1.3 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from paramiko>=2.4.2; extra == "ssh"->docker[ssh]<4.0,>=3.7.0->docker-compose>=1.23.0->sagemaker==1.34.0) (3.1.7)
    Requirement not upgraded as not directly required: pynacl>=1.0.1 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from paramiko>=2.4.2; extra == "ssh"->docker[ssh]<4.0,>=3.7.0->docker-compose>=1.23.0->sagemaker==1.34.0) (1.3.0)
    Requirement not upgraded as not directly required: cryptography>=2.5 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from paramiko>=2.4.2; extra == "ssh"->docker[ssh]<4.0,>=3.7.0->docker-compose>=1.23.0->sagemaker==1.34.0) (2.7)
    Requirement not upgraded as not directly required: cffi>=1.1 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from bcrypt>=3.1.3->paramiko>=2.4.2; extra == "ssh"->docker[ssh]<4.0,>=3.7.0->docker-compose>=1.23.0->sagemaker==1.34.0) (1.11.5)
    Requirement not upgraded as not directly required: asn1crypto>=0.21.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from cryptography>=2.5->paramiko>=2.4.2; extra == "ssh"->docker[ssh]<4.0,>=3.7.0->docker-compose>=1.23.0->sagemaker==1.34.0) (0.24.0)
    Requirement not upgraded as not directly required: pycparser in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from cffi>=1.1->bcrypt>=3.1.3->paramiko>=2.4.2; extra == "ssh"->docker[ssh]<4.0,>=3.7.0->docker-compose>=1.23.0->sagemaker==1.34.0) (2.18)
    Building wheels for collected packages: sagemaker
      Running setup.py bdist_wheel for sagemaker ... [?25ldone
    [?25h  Stored in directory: /home/ec2-user/.cache/pip/wheels/c2/bc/63/787701efab41789384bbd3c48b265d61d3990502efdc8c98fa
    Successfully built sagemaker
    Installing collected packages: sagemaker
      Found existing installation: sagemaker 1.42.6
        Uninstalling sagemaker-1.42.6:
          Successfully uninstalled sagemaker-1.42.6
    Successfully installed sagemaker-1.34.0
    [33mYou are using pip version 10.0.1, however version 19.2.3 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m

