# Segmentation using K-means clustering 
The goal of the project is to implement K-means clustering for image segmentation in :
1. 3-dimensional color space.
2. 5-dimensional space, where first 3 dimensions are color space and extra two dimensions correspond to X and Y dimension of the image. 
 
# How to execute code :
1. You will first have to download the repository and then extract the contents into a folder.
2. Make sure you have the correct version of Python installed on your machine. This code runs on Python 3.6 above.
3. You can open the folder and run following on command prompt.
 > `python K_means_clustering_3dim --imagepath data/2apples.jpg --num_cluster 2`, where default value of num_cluster is 4.
 
 Same command can be used for 5-dimensional space like :
 > `python K_means_clustering_5dim --imagepath data/2apples.jpg --num_cluster 2`, where default value of num_cluster is 4.
 
 # Results
 ![output](https://github.com/Devashi-Choudhary/K_Means_Clustering/blob/master/Results/output.png)
 
From above image, we can infer that while we are considering the colour space(intensity values) of object only that is it’s  pixel values in RGB channel so clustering result is based on that. Now, when we are in 5-dimensional space, we are considering an image as a heap of pixels that is  each pixel is characterized by its coordinates (spatial information) and colour intensities (R, G, B).So, we are representing  pixels in vector form that is in 5D space .The  clustering approach applied to the heap of pixels represented as vectors in 5D space There is difference in both 3D space and 5D space is that 3D colour space only consider pixel values while in 5D  one part is relating to object tracking in images(intensities) and the other part is related  to spatial/spectral part of segmentation process(position of objects).

# References

This project is done as a part of college assignment.
