# local_feature_mapping_python #
In this repository, we come up with written-from-scratch algorithms to perform the following computer vision tasks specifically in python:
* Feature detectors - Identify the interest points
* Feature descriptors - Extract feature vector descriptor surrounding each interest point
* Feature matching - Determine correspondence between descriptors in 2 views

Why do we need to learn feature matching and what is the application of it?
To provide you with a clear answer to this question, let’s come up with an example. take a look at the following images: This is the Episcopal Palace building in Spain. You can see the same building from different angles and distances. 

![Selection_220](https://user-images.githubusercontent.com/47978272/147425883-2bca24a9-9b3c-4ec5-9c42-032aa8a1f641.png)


We as a human can simply identify that there is one unique building in 2 images from 2 different angles and distances. But for a computer, this similarity identification is very difficult and sometimes impossible to find the answer. We are learning Feature Matching to enable a computer to find similar features in different images. Ultimately, the computer will able to provide the following applications:
* Automate object tracking
* Point matching for computing disparity
* Stereo calibration
* Motion-based segmentation
* Recognition
* 3D object reconstruction
* Robot navigation
* image indexing
In order to write a specific algorithm for feature detection, the followings are required:

![Selection_221](https://user-images.githubusercontent.com/47978272/147425956-37fcc187-a290-46aa-a3cd-e035eaa09e14.png)



## Harris Corner Detector ##
The feature detector or interest point detector algorithm is responsible for finding the most concrete and reliable features in an image that can be easily located in the second image. For instance, the following image pair is illustrated in the following figure:

![Selection_222](https://user-images.githubusercontent.com/47978272/147426034-c65eac51-0c0b-4a59-bd7c-f0a6d441be3c.png)

A good interest point detector should be able to:
* Detect most of the true interest points
* Deal with noise
* Time-efficient


## Mathematical Idea ##
In this section, we will be using the Harris Corner detector algorithm to find interest points in an image. In this algorithm, a small window will be shifted all over the image to find corner points in the image.

![Selection_223](https://user-images.githubusercontent.com/47978272/147426112-42ec12d1-ef45-49c7-a738-88d611709ef4.png)

Now the question is how to find corner points in an image. To answer this question, we need to dig into the mathematical idea behind Harris Corner Detector:

![Selection_224](https://user-images.githubusercontent.com/47978272/147426173-7cc11a46-cca7-4b00-80eb-5ffa50f97635.png)

Now, we need to find the eigenvalues and eigenvectors of the derivative matrix M:

![Selection_225](https://user-images.githubusercontent.com/47978272/147426256-db66ca1e-65fd-4ed6-877e-9a339358e455.png)

using the eigenvalues for each pixel, we can decide which pixel is the corner point. This concept is illustrated in the following figure:

![Selection_226](https://user-images.githubusercontent.com/47978272/147426305-a31b0c1c-8742-42d3-81f1-ed02709f26a8.png)

Having the eigenvalues calculated, we can simply calculate the cornerness of every corner point using the Forstner-Harris equation. with α = 0.06. Unlike eigenvalue analysis, this quantity does not require the use of square roots and yet is still rotationally invariant and also downweights edge-like features.


