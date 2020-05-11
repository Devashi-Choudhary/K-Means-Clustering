#!/usr/bin/env python
# coding: utf-8

# # Importing the required Libraries

# In[25]:


import numpy as np
import cv2
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# # Loading the Image

# In[26]:


img = cv2.imread('C:/Users/Devashi Jain/Desktop/IIIT-D/Computer Vision/Assignment-3/Assignment-3/Q1-images/variableObjects.jpeg')
Z = img.reshape((-1,3))
print(Z.shape)
Z = np.float32(Z)
df=pd.DataFrame(Z)


# # K_Means Clustering Algorithm

# In[27]:


def K_Means(df,k):
    centroids = {
        i: [np.random.randint(0, 255), np.random.randint(0, 255),np.random.randint(0,255)]
        for i in range(k)
    }
    print(centroids)
    def Assignment(df, centroids):
        for i in centroids.keys():
            df['distance_from_{}'.format(i)] = (np.sqrt((df[0] - centroids[i][0]) ** 2+ (df[1] - centroids[i][1]) ** 2+(df[2] - centroids[i][2]) ** 2))
        centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
        df['label'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
        df['label'] = df['label'].map(lambda x: int(x.lstrip('distance_from_')))
        return df
    df=Assignment(df,centroids)
    def Update_Centroid(k):
        for i in centroids.keys():
            centroids[i][0] = np.mean(df[df['label'] == i][0])
            centroids[i][1] = np.mean(df[df['label'] == i][1])
            centroids[i][2] = np.mean(df[df['label'] == i][2])
        return k
    while True:
        closest_centroids = df['label'].copy(deep=True)
        centroids = Update_Centroid(centroids)
        df = assignment(df, centroids)
        if closest_centroids.equals(df['label']):
            break
    return centroids,df['label'],df


# In[28]:


Centroids,Label,df=K_Means(df,4)


# # Output of Image in 3D color Space

# In[30]:


list1=list(Label)
list1=np.array(list1)
centers=list(Centroids.values())
centers=np.array(centers)
centers=np.uint8(centers)
print(centers)
res = centers[list1.flatten()]
res2 = res.reshape((img.shape))
q=cv2.resize(res2,(int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.imshow('Images Using K-Means Clustering in 3D color Space',q)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 3D scatter plot

# In[19]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
color=['red','blue','green','orange','yellow','magenta','orange','pink']
plt.title("RGB Color Space for 2or4objects")
for i in Centroids.keys():
    plt.scatter(*Centroids[i],c=color[i])
    df1=df[df['closest']==i]
    ax.scatter(df1[0], df1[1], df1[2],c=color[i], marker='o')
    #ax.scatter(*Centroids[i],marker='*',c=color[i], s=1000)
    ax.set_xlabel('Blue Color Space')
    ax.set_ylabel('Green Color Space')
    ax.set_zlabel('Red Color Space')
plt.show()


# In[ ]:




