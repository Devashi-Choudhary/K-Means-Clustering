#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import pandas as pd
img = cv2.imread('C:/Users/Devashi Jain/Desktop/IIIT-D/Computer Vision/Assignment-3/Assignment-3/Q1-images/variableObjects.jpeg')
print(img.shape)
rows,cols,y=img.shape
coordinate=[]
for i in range((rows)):
    for j in range((cols)):
        coordinate.append((i,j))
Z = img.reshape((-1,3))
print(Z.shape)
Z = np.float32(Z)


# In[2]:


df_coordinate=pd.DataFrame(coordinate)
df_coordinate.columns=[3,4]
df_color=pd.DataFrame(Z)
df=pd.concat([df_coordinate,df_color],axis=1)


# In[3]:


df


# In[ ]:


def K_Means(df,k):
    k = 4
    centroids = {
        i: [np.random.randint(0, 255), np.random.randint(0, 255),np.random.randint(0,255),np.random.randint(0,rows-1),np.random.randint(0,cols-1)]
        for i in range(k)
    }
    def Assignment(df, centroids):
        for i in centroids.keys():
            df['distance_from_{}'.format(i)] = (np.sqrt(
                (df[0] - centroids[i][0]) ** 2+ 
                (df[1] - centroids[i][1]) ** 2+
                (df[2] - centroids[i][2]) ** 2+
                (df[3] - centroids[i][3]) ** 2+ 
                (df[4] - centroids[i][4]) ** 2
                                                       )
                                                )
        centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
        df['label'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
        df['label'] = df['label'].map(lambda x: int(x.lstrip('distance_from_')))
        #df['color'] = df['closest'].map(lambda x: colmap[x])
        return df
    df=Assignment(df,centroids)
    def Update_Centroid(k):
        for i in centroids.keys():
            centroids[i][0] = np.mean(df[df['label'] == i][0])
            centroids[i][1] = np.mean(df[df['label'] == i][1])
            centroids[i][2] = np.mean(df[df['label'] == i][2])
            centroids[i][3] = np.mean(df[df['label'] == i][3])
            centroids[i][4] = np.mean(df[df['label'] == i][4])
        return k
    while True:
        closest_centroids = df['label'].copy(deep=True)
        centroids = Update_Centroid(centroids)
        df = assignment(df, centroids)
        if closest_centroids.equals(df['label']):
            break
    return centroids,df['label'],df


# In[ ]:


centroids,Label,df=K_Means(df,4)


# In[8]:


list1=list(df['label'])
list1=np.array(list1)
centers=[]
for i in range(len(centroids)):
    centers.append((centroids[i][0],centroids[i][1],centroids[i][2]))

centers=np.array(centers)
centers=np.uint8(centers)
print(centers)
res = centers[list1.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('Images Using K-Means Clustering in 5D Space',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


q=cv2.resize(res2,(int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.imshow('Images Using K-Means Clustering in 5D Space',q)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
color=['red','blue','green','orange','yellow','magenta','orange','pink']
for i in centroids.keys():
    #plt.scatter(*centroids[i])
    df1=df[df['closest']==i]
    ax.scatter(df1[0], df1[1], df1[2],c=color[i], marker='o')
    #ax.scatter(*centroids[i],marker='*',c='#050505', s=1000)
plt.show()


# In[ ]:


q=cv2.resize(res2,(int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.imshow('Images Using K-Means Clustering in 5D Space',q)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




