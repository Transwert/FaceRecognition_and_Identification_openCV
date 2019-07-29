### FaceRecognition_and_Identification_openCV
### openCV_timeLapse

## Introduction : 
this project includes facial recognition as well as identification of the person in live video,
using openCV and python

## Brief :
A facial recognition system is a technology capable of identifying or verifying
a person from a digital image or a video frame from a video source.  { source : [wiki](https://en.wikipedia.org/wiki/Facial_recognition_system) }.
So with pre-defined dataset containing images of person ( which to be identified ) or new dataset 
( which can be formed from live stream from webcam ). Using LBPH facial recogniser in openCV 
( in total, their are 3 inbuilt facial recognisers :  Eigenface recogniser, Fisherface recogniser and LBPH face recogniser ),
which being trained on the dataset, and then is used to identify person.

## Observations :
this classifier does show a good result, as it is faster than normal yet slow neural network model in terms of training, 
but it lacks in terms of accuracy as compared to those slow neural network models. so it depends on user, to trade off between
accuracy or training and deploymemt speed. 
