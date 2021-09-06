import os
import numpy as np
import pandas as pd
import cv2
import random
import pickle
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

build_model_first = False

seed_constant = 23
np.random.seed(seed_constant)
random.seed(seed_constant)

image_height, image_width = 32, 32 
max_images_per_class = 600

dataset_directory = "Dataset/Activity_Dataset"
classes_list = ["Control", "Sitting", "Sleeping", "Stairs", "Standing", "Walking"]
# the number of frame we will take in order to make prediction
n_frame = 10

''' 
This if statement which execute the  folling if build_madel_first is  true helps us to not waste
time during run time because model take long to build thats why we buld_model_first and save it in
file called finalized_model.sav and never need to build it forever so for the first time we set
build_model_first to True and we never do that again.
'''
if build_model_first:

  def read_frames(class_name,class_dir_path):
    # Empty List declared to store class frames
    frames_list = []
    # read all files in class
    frames = os.listdir(class_dir_path)
    # loop through all images in a class
    for frame in frames:
      # Construct the complete image path
      img_dir_path = os.path.join(dataset_directory, class_name, frame)
      # Reading the image File Using the imread
      image = cv2.imread(img_dir_path, 0)
      # print(image.shape)
      # Resize the Frame to fixed Dimensions
      resized_frame = cv2.resize(image, (image_height, image_width))
      # print(resized_frame.shape)
      # Normalize the resized frame by dividing it with 255 so that each pixel value
      # then lies between 0 and 1
      normalized_frame = resized_frame / 255
      # Appending the normalized frame into the frames list
      frames_list.append(normalized_frame)
    # returning the frames list
    return frames_list


  def create_dataset():

    # Declaring Empty Lists to store the features and labels values.
    features = []
    labels = []

    for class_index, class_name in enumerate(classes_list):
      print(f'Extracting Data of Class: {class_name}')
      # Getting the class folder
      dir_path = os.path.join(dataset_directory, class_name)
      # Calling the read_frames method for every class
      frames = read_frames(class_name=class_name, class_dir_path=dir_path)
      # Adding randomly selected frames to the features list
      features.extend(random.sample(frames, max_images_per_class))
      # Adding Fixed number of labels to the labels list
      labels.extend([class_index] * max_images_per_class)
    # Converting the features and labels lists to numpy arrays
    features = np.asarray(features)
    features_shape = features.shape
    features = features.reshape(features_shape[0], features_shape[1]*features_shape[2])
    labels = np.array(labels)
    return features, labels


  features, labels = create_dataset()

  def get_model_object():
    mlp = MLPClassifier(
    max_iter=500,
    random_state = seed_constant
    )
    return mlp

  mlp = get_model_object()
  mlp.fit(features, labels)

  scores = []
  def get_model_accuracies():
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(features):
      X_train, X_test = features[train_index], features[test_index]
      y_train, y_test = labels[train_index], labels[test_index]
      model = get_model_object()
      model.fit(X_train, y_train)
      scores.append(model.score(X_test, y_test))
  get_model_accuracies()
  # my observered model accuracy was 45.9 %
  print("model accuracy is : " + str(np.mean(scores) * 100))
  # save the model for future use
  filename = 'constructed_mlp_model.sav'
  pickle.dump(mlp, open(filename, 'wb'))


filename = 'constructed_mlp_model.sav'
mlp = pickle.load(open(filename, 'rb'))

def predict_on_live_video():
  # define a video capture object
  cap = cv2.VideoCapture(0)
    
  if (cap.isOpened()== False):
    print("Error opening video stream")
  
  prediction_list = []
  current_prediction = "Recognizing activity"
  green = 0
  red = 255

  while(cap.isOpened()):

      # Capture the video frame
      # by frame
      ret, frame = cap.read()
    
      if ret == True:
        # convert frame to grayscale color
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(gray, (image_height, image_width))
        # Normalize the resized frame by dividing it with 255 so that each pixel
        # value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        normalized_frame_shape = normalized_frame.shape
        normalized_frame = normalized_frame.reshape(normalized_frame_shape[0] * normalized_frame_shape[1])
        prediction = mlp.predict_proba([normalized_frame])
        #print(prediction)
        # append prediction to a list
        prediction_list.append(prediction)
        #print(len(prediction_list))

         # Assuring that the PREDICTION_LIST is completely filled before starting the
         # averaging process
        if len(prediction_list) >= n_frame:
          # Converting Predicted Labels Probabilities into Numpy array
          predicted_labels_prob_np = np.array(prediction_list)
          # Calculating Average of Predicted Labels Probabilities Column Wise
          predicted_labels_prob_averaged =predicted_labels_prob_np.mean(axis=0)
          # check if the high plobabilty if is greater than 74 to make ure that we
          # return correct prediction most of the time
          maximum_label = np.max(predicted_labels_prob_averaged) * 100
          print(maximum_label)
          if maximum_label >= 75:
            # Converting the predicted probabilities into labels by returning the 
            # index of the maximum value.
            predicted_label = np.argmax(predicted_labels_prob_averaged)
            predicted_class_name = classes_list[predicted_label]
            current_prediction = predicted_class_name
            green = 255
            red = 0
          else:
            current_prediction = "Recognizing activity"
            green = 0
            red = 255 
          prediction_list.clear()
        cv2.putText(
          frame, current_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
          1, (0, green, red), 2
        )
        cv2.imshow('frame', frame)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      else:
        break
    
  # After the loop release the cap object
  cap.release()
  # Destroy all the windows
  cv2.destroyAllWindows()

predict_on_live_video()
