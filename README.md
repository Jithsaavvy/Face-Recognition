# Eigen-Faces and OpenCV
In this task, I developed code for my own facial recognition library using Eigen faces and OpenCV (i.e.) by using API or libraries and without any available APIs or libraries. Eigenvectors have many applications which are not limited to obtaining surface normals from a set of point clouds.

**Programming Language Used: Python**

An eigenface really is nothing else than an eigenvector, in this case reshaped for plotting. Eigenfaces can be used in facial recognition, allowing a robot to distinguish between different persons, but can also be applied to other use cases, such as voice or gesture recognition.

The task consists of the following subtasks:
    1. Implement the eigenface algorithm. In particular, create a Python class that exposes (at least) two methods:
    2. A method for calculating eigenfaces given two parameters, namely (i) a set of images and (ii) subject ids that uniquely identify the subjects in the images.
    3. A method that takes one parameter - a list of query faces - and, for each face in the input list, finds the subject id of the most similar face. This method should thus return a list of subject ids.

A dataset for training your recognition algorithm is given in the data/training folder. The images in the data/test folder should be used for testing the algorithm.

Note: Principal Component Analysis (PCA) is used here

Following to it, the face recognition task is also performed using inbuild library - OpenCV. 

