# face-mask-detection
This is my first approach to face mask detection problem using Open CV face recogniser and trained on the about 7000 images convolutional neural network

Content:

1. face_capture.ipynb
    Python notebook, opens stream video from your webcam (first webcam if more than 1 are detected) and gives prediction if you are wearing a mask. You might want to run this file     to test the model. To succesfully run this notebook you need to have "model_masks_trial2.h5" file in the same folder where you have this notebook and good illumination. The       face recognizer classifier is not optimised for different illumination conditions and black masks (I am happy to hear idea why haarcascade face recongiser performs bad in my       case with black masks).
2. model_masks_trial2.h5
    Trained model for mask classification. Necessary to have to run face_capture.ipynb.
3. mask_classifier.py
    Python file, there classification model is trained to detect face mask on a person, includes also image preprocessing
4. inputs.npy, labels.npy
    Images as multidimensonal numpy array and corresponding labels (0 - no mask, 1 - mask). These were used to train the model from mask_classifier.py.
5. Mask detection_demo.mp4
    Recording of my laptop screen where I test my mask detector
    
Additional comments you will find also in each file.
Raw images you can download from here:
https://1drv.ms/u/s!AtdSTPoAYdUFhrVItqBbJ1jWNXvtRg?e=Ifsyz1
