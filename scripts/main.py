import cv2
import face_detection
import facial_landmark
import numpy as np
import tensorflow as tf
import math
import os


if __name__ == "__main__":
    dir = os.getcwd()
    dir = dir + "/../models/"
    #load model for face detection
    model = dir + "opencv_face_detector_uint8.pb"
    config = dir + "opencv_face_detector.pbtxt"
    face_detection_model = cv2.dnn.readNetFromTensorflow(model, config)    
    
    #Load model for landmark detection
    saved_model=dir + 'pose_model'
    landmark_model = tf.saved_model.load(saved_model)
    
    #Estimated 3D points of face key points
    face_3D_model = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    cap = cv2.VideoCapture(0)
    _, img = cap.read()

    # Camera parameters
    f_length = img.shape[1]
    optical_centre = (img.shape[1]/2, img.shape[0]/2)
    K_matrix = np.array(    [[f_length, 0, optical_centre[0]],
                            [0, f_length, optical_centre[1]],
                            [0, 0, 1]])



    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame . Exiting ...")
            break
        img=frame.copy()

        #Returns the ROI based on the scale(between 0 to 1)
        roi,border,roi_dim=face_detection.calculate_ROI(img,0.8)
        
        #Returns the co-ordinates of all the faces detected in the ROI
        faces=face_detection.detect_face(roi,face_detection_model)
        face_detection.draw_boundbox(faces,roi)

        #Returns the 68 landmarks for every face detected
        landmarks=facial_landmark.face_landmark(faces,roi,landmark_model)
        
       
        for face_landmarks in landmarks:
            key_mark = np.array([   face_landmarks[30],     # Nose tip
                                    face_landmarks[8],      # Chin
                                    face_landmarks[36],     # Left eye left corner
                                    face_landmarks[45],     # Right eye right corne
                                    face_landmarks[48],     # Left Mouth corner
                                    face_landmarks[54]],    # Right mouth corner
                                    dtype="double")    
                                    
            distortion = np.zeros((4,1)) # Assuming no lens distortion

            #Calculates Rotation and translation matrix
            _, R, T = cv2.solvePnP(face_3D_model, key_mark, K_matrix, distortion, flags=cv2.SOLVEPNP_UPNP)
            nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1500.0)]), R, T, K_matrix, distortion)
        
            nose1 = (int(key_mark[0][0]), int(key_mark[0][1]))
            nose2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(roi, nose1, nose2, (255,0,0), 2)
            
            #Estimates the head pose for side ways movement
            a1, a2 = facial_landmark.head_pose(img, R, T, K_matrix)
            cv2.line(roi, tuple(a1), tuple(a2), (255, 255, 255), 2)
            
            #Overlap the ROI back on to the original image
            img[border[1]:border[1]+roi_dim[1],border[0]:border[0]+roi_dim[0]]=roi
            
            try:
                theta = int(math.degrees(math.atan((nose2[1]-nose1[1])/(nose2[0]-nose1[0]))))
                phi = int(math.degrees(math.atan(-1/((a2[1]-a1[1])/(a2[0]-a1[0])))))
            except:
                theta = 0
                phi = 0    
            
            #Angle thresholds to detect the head pose
            Theta_threshold=20
            Phi_Threshold=45
            
            if theta >= Theta_threshold:
                cv2.putText(img, 'Head down', (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL , 2, (0, 0, 0), 3)
            elif theta <= -Theta_threshold:
                cv2.putText(img, 'Head up', (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL , 2, (0, 0, 0), 3)    
            if phi >= Phi_Threshold:
                cv2.putText(img, 'Head right', (350, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 3)
            elif phi <= -Phi_Threshold:
                cv2.putText(img, 'Head left', (350, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 3)        
   
        cv2.imshow("WEBCAM FEED",img)
        if cv2.waitKey(1) == ord('q'):
            break  
