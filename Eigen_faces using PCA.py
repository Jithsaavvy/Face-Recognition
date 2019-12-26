import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs
import IPython
import imageio
from typing import Tuple, Sequence

class FaceRecognition(object):
    def __init__(self):
        self.subject_ids = None
        self.faces = None
        self.eigenface_weights = None
        self.mean_image = None

    def eigenfaces(self, image_filenames: Sequence[str],
                   subject_ids: Sequence[int]) -> None:
        
        '''Finds a set of eigenfaces based on the input images.
        The eigenfaces are saved in self.faces and self.eigenface_weights.

        Keyword arguments:
        image_filenames -- A list of image filenames
        subject_ids -- A list of IDs that uniquely identify the subjects
                       in the images

        '''
        self.subject_ids=subject_ids
        images=imageio.imread(image_filenames[0])
        m,n=shape(images)
        sh=m*n
        image_vector=np.empty(shape=(sh,110))
        for i in range(len(image_filenames)):
            images=imageio.imread(image_filenames[i])        
            image_flatten=np.ndarray.flatten(images)
            image_vector[:,i]=image_flatten
            
        imglen=len(self.subject_ids)
        aver = np.zeros((45045,1))
        
        #Using PCA method
        for i in range(0,110):
            aver[:,0] = aver[:,0] + image_vector[:,i]
        aver[:,0] = aver[:,0]/110
        nor = np.zeros((45045,110))
        for i in range(0,110):
            nor[:,i] = image_vector[:,i] - aver[:,0]            
        self.faces = nor
        
        #self.mean_image=(np.sum(image_vector,axis=1)/imglen)[np.newaxis].T
        #print(self.mean_image.shape)
        #self.faces=np.subtract(image_vector,self.mean_image)
        #print(self.faces.shape)
        #cov=np.cov(self.faces.T)
        cov=self.faces.T.dot(self.faces)
        eigval,eigvec=np.linalg.eigh(cov)                
        self.eigenface_weights=self.faces.dot(eigvec)
        #Eigen faces for total image set
        self.faces=self.eigenface_weights.T.dot(image_vector)
                        

    def recognize_faces(self, image_filenames: Sequence[str]) -> Sequence[int]:
        '''Finds the eigenfaces that have the highest similarity
        to the input images and returns a list with their indices.

        Keyword arguments:
        image_filenames -- A list of image filenames

        Returns:
        recognised_ids -- A list of ids that correspond to the classifier
                          predictions for the input images

        '''        
        #Test image set
        images=imageio.imread(image_filenames[0])
        m,n=shape(images)
        sh=m*n
        image_vector1=np.empty(shape=(sh,55))
        for i in range(len(image_filenames)):
            image=imageio.imread(image_filenames[i])        
            image_flatten1=np.ndarray.flatten(image)
            image_vector1[:,i]=image_flatten1
            
        leng=len(self.subject_ids)
        means=(np.sum(image_vector1,axis=1)/leng)
        facess=np.subtract(image_vector1,means)
        #Eigen faces for test image set
        eigenface2= self.eigenface_weights.T.dot(image_vector1)
        mylist=[]
        addn=np.empty(shape=(1,110))
        
        for i in range(0,eigenface2.shape[1]):
            for j in range(0, self.faces.shape[1]):
                addn[:,j]=np.sum(abs(self.faces[:,j]-eigenface2[:,i]))
                minelement=np.argmin(addn)                
            mylist.append(self.subject_ids[minelement])
        return mylist
                

#Testing
import os
import glob

#loading training images
training_image_filenames = sorted(glob.iglob('data/training/*.pgm'))

#loading test images
test_image_filenames = sorted(glob.iglob('data/test/*.pgm'))

#creating a lambda function for extracting filenames;
#the filename of each image is the subject id
subject_number = lambda filename: int(os.path.basename(filename)[7:9])

#extracting the filename using the lambda function
train_subject_ids = list(map(subject_number, training_image_filenames))
test_subject_ids = list(map(subject_number, test_image_filenames))

print('Test subject ids:', np.array(test_subject_ids))

face_recognition = FaceRecognition()
face_recognition.eigenfaces(training_image_filenames, train_subject_ids)
recognized_ids = face_recognition.recognize_faces(test_image_filenames)
print('Predicted subject ids:', recognized_ids)

different_results = np.array(test_subject_ids) - np.array(recognized_ids)
positives = (different_results == 0).sum()
accuracy = positives / (len(test_subject_ids) * 1.)
print('Number of correct predictions =', positives)
print('Prediction accuracy =', accuracy)