#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tommy Yong
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("Detail_listings.csv")

features = list(data)
colnames = np.transpose(features)

# (1) Generate Summary Statistics
print("-----------------------")
print("Data Dimensions:  ", data.shape)

sumry = data.describe().transpose()
print("-----------------------")
print("Summary Statistics:\n",sumry,'\n')

#check datatyoe of each column
print("-----------------------")
print("Data Type of columns:  ", data.dtypes)

#check datatyoe of each column
print("-----------------------")
print("Top 10 Rows:", data.head(10))

#check if the dataset has missing values
print("-----------------------")
print("Number of empty cells" , data.isna().sum())

#one hot encoding
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)

#one hot encoding for 'host response time'
arr = ['host_response_time','host_is_superhost','host_has_profile_pic',
       'host_identity_verified','is_location_exact','property_type','room_type','bed_type','instant_bookable','cancellation_policy',
       'require_guest_profile_picture','require_guest_phone_verification']
response = data
for i in arr:
 encode_and_bind(response,i)
 response = encode_and_bind(response,i)
print("Data Type of columns:  ", response.dtypes)
        
#drop the colums with datatype as Object
df = response.select_dtypes(exclude=['object'])
to_drop = ['id',
           'scrape_id',
           'host_id']
df = df.drop(to_drop,axis=1)
print("-----------------------")
print("Data Dimensions:  ", df.shape)
print("Data Type of columns:  ", df.dtypes)

#check datatyoe of each column
print("-----------------------")
print("Top 20 Rows:", df.head(20))

#check if the dataset has missing values
print("-----------------------")
print("Number of empty cells" , df.isna().sum())

#drop the colums which are less filled
to_drop = ['neighbourhood_group_cleansed',
           'square_feet','has_availability']
df = df.drop(to_drop,axis=1)
print("-----------------------")
print("Data Dimensions:  ", df.shape)
print("Data Type of columns:  ", df.dtypes)

#check if the dataset has missing values
print("-----------------------")
print("Number of empty cells" , df.isna().sum())

#drop the rows with empty cells
df = df.dropna(axis=0,how='any')

#check if the dataset has missing values
print("-----------------------")
print("Number of empty cells" , df.isna().sum())
print("-----------------------")
print("Data Dimensions:  ", df.shape)
print("Data Type of columns:  ", df.dtypes)

# (2) Histograms Visualisation
print("Frequency Distributions:")
df.hist(grid=True, figsize=(10,6), color='blue')
plt.tight_layout()
plt.show()
