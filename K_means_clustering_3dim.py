import numpy as np
import cv2
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse

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
        df = Assignment(df, centroids)
        if closest_centroids.equals(df['label']):
            break
    return centroids,df['label'],df

def segmentation(imagepath, k):
    img = cv2.imread(imagepath)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    df = pd.DataFrame(Z)
    Centroids, Label, df = K_Means(df, k)
    list1 = list(Label)
    list1 = np.array(list1)
    centers = list(Centroids.values())
    centers = np.array(centers)
    centers = np.uint8(centers)
    res = centers[list1.flatten()]
    res2 = res.reshape((img.shape))
    q = cv2.resize(res2,(int(img.shape[1]/2),int(img.shape[0]/2)))
    cv2.imshow('Images Using K-Means Clustering in 3D color Space',q)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagepath", required = True, help = "path to input image")
ap.add_argument("-k", "--num_cluster", type = int, default = 4, help = "number of clusters for k means clustering")
args = ap.parse_args()

if __name__ == "__main__":
    imagepath = args.imagepath
    k = args.num_cluster
    segmentation(imagepath, k)

