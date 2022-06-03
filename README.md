# Hand-gesture-recognition
This project uses a combination of OpenCv and Cnn model. OpenCv is used to capture the current frame from your webcam and Cnn model is used to classify the image in the current frame.

The dataset used has 7 hand gestures; Fist, Five, None, Okay, Peace, Rad, Straight, Thumbs



# How to run
1. Firstly, run the Hand-gesture-Cnn file to generate the model, or directly skip to Step 3. 
2. Use the generated model in the Hand-gesture-OpenCv.py
3. Run the Hand-gesture-OpenCv file
4. A window will popup which will calculate the background, wait until background is loaded. Now, you can place your hand in the green box or ROI (Region of Interest). Finally, the name of your hand gesture will be displayed on the video window. 



# How to get hand gesture:
For detecting the hand segment, contours are used to outline the hand gesture.  Make sure the background is clear and free of edges and corners when the computer calculates the background; otherwise, it will take them into account. If the camera is moved after the background is calculated, the background will be distorted. While taking the hand gesture, ensure the hand is less than 100 cm away from the webcam, the gesture is angled towards the camera and a proper lighting is used; to detect the gesture accurately. 



# Data-set
The pretrained dataset (hand_gestures.h5) is used from the referenced project (Dabhi, S., 2020). The pretrained dataset is then used to recognize the gestures made in real time on the webcam. 



By running the Hand-gesture-Cnn file, a training set consisting of 7999 images belongs to 8 classes and a test set consisting of 4000 images belongs to 8 classes. The accuracy of the training set was 98.69%. 



Epoch 1/1
7999/7999 [==============================] - 551s 69ms/step - loss: 0.0401 - accuracy: 0.9869 - val_loss: 2.0493 - val_accuracy: 0.9762

