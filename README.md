# GDSC_ML_Project

## GOAL of our Project

We chose this project because we wanted people to engage in video calls in a more fun way. Our intent was that video calls using filters with friends or professional colleagues could be made less intimidating and more enjoyable. 

## Introduction

We used various machine-learning tools and libraries to create a face filter that detects a face using the computer's webcam and overlays a custom animated filter. We used data from the face_recognition library to train our model, and through this process, our model could detect as many human faces as present in the webcam video accurately. Using the trained model, we were able to successfully overlay our custom filter and make it move along as people move. 

## Initial Steps

To start this project, we first researched different data sets that might be useful for this project to train our machine learning model. We also looked for different libraries and tools that might be useful in this project, like the cv2 module and the Haar Cascade classifiers.

Using this information, we started implementing a filter on static images. We wanted to see if our model would correctly identify faces and correctly overlay images on top of it. We encountered some errors, for example, we quickly learned that our filter would have to be transparent and we also had some issues with using the cv2 module. But, we were able to successfully overlay filters on static pictures, even pictures that made multiple people. 

We then tried implementing the filters on downloaded videos. We wanted to see if our model would work with movement as well. For this part of the project, we again used the same tools, the cv2 module, and the Haar Cascade classifiers. We again encountered some errors, for example, the filter did not move with the person and stayed static. We also weren’t able to get the filter to overlay multiple people in the video. In the end, we were able to successfully implement this feature. 

## Methods

After researching and learning how to apply filters on static images and downloaded videos, we moved towards using the webcam so people could overlay their faces with filters. Like we did during our initial steps, we used open cv (Open Source Computer Vision Library) and face_recognition library to get our webcam to open and detect our faces. 

We first started out using the Haar Cascade classifiers to detect our eyes and faces, but it wasn’t getting much trained and accurately measuring all faces when more than 2 people were joining. Therefore, we decided to use the face_recognition library that trains our model to better detect all faces. This allowed us to detect all faces that are present in the webcam and was also able detect faces that are on my phone when I hold them up to the webcam. 

Before applying our custom-built filter, we tried to first detect our faces using rectangles and make sure that they are following our faces. We encountered some difficulties because opening the webcam was a bit different than detecting faces statically. Through various research, we found face recognition library that we could use and was able to successfully detect all of our faces. 

We then tried to implement custom and downloaded filters. We tried to reference to our codes used from our initial steps (recognizing faces from video), but as we used different libraries and tools, we encountered some errors that were easily fixed through researching and some errors that took time for us to consider. One major error that we encountered was that the filter was not getting applied to the correct position of our faces and was not following us as we moved. In the end, we were able to fix the dimensions of the filter and switch variables and were able to successfully implement the feature. 

## Future Applications

In the future, we would like to extend our filter’s capabilities and have it work on multiple features of the face, like the eyes, nose, and mouth. Currently, it only detects the face shape and sits on top of it. 

We also want to create multiple filters that are better suited to different situations, like a filter for professional calls. We would like to implement a filter where users can click a button, and the screen stops until the user quits this action. We think this filter can be useful when the user needs to quickly adjust something, but at the same time does not want to turn their camera off. 

We also want to create an interface where users can access all the filters at one spot, and choose which filter to apply. In the future, we are considering to create an application or a website. 
