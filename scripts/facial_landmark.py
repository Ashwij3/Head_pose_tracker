import cv2
import tensorflow as tf
import numpy as np



def face_landmark(faces,image,model):
    """
    Find the facial landmarks of all the detected face coordinates in an image 

    Parameters
    ----------
    face : list
        list of Face coordinates (x, y, w, h) of all faces in which the landmarks are to be found    
    image : np.uint8
        The image in which landmarks are to be found
    model : Tensorflow model
        Loaded facial landmark model

    Returns
    -------
    faces_landmark : list of numpy array
        facial landmark points of all the faces

    """
    faces_landmark=[]
    for face in faces:
        face_img=image[int(face[1]):int(face[3]),int(face[0]):int(face[2])]
        face_img = cv2.resize(face_img, (128, 128))
        #face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        #roi_128 = cv2.resize(face_img, (128, 128))
        predictions = model.signatures["predict"](tf.constant([face_img], dtype=tf.uint8))
        marks = np.array(predictions['output']).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))
        marks[:, 0] *= (face[2]-face[0])
        marks[:, 1] *= (face[3]-face[1])
        marks[:, 0] += face[0]
        marks[:, 1] += face[1]
        marks = marks.astype(np.uint)
        faces_landmark.append(marks)    
        for mark in marks:
            cv2.circle(image, (mark[0], mark[1]), 2, (255,255,0), -1, cv2.LINE_AA)
    return faces_landmark



def head_pose(image,R,T,camera_matrix):
    """
    Get the points to estimate head pose sideways    

    Parameters
    ----------
    image : np.unit8
        Original Image.
    R : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    T : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix

    Returns
    -------
    (x, y) : tuple
        Coordinates of line to estimate head pose

    """  
    rear_size = 0
    rear_depth = 0
    front_size = image.shape[1]
    front_depth = front_size*2
    point_3d=[]
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
    dist_coeffs = np.zeros((4,1))
    (point_2d, _) = cv2.projectPoints(point_3d,R,T,camera_matrix,dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    y = (point_2d[1] + point_2d[2])//2
    x = point_2d[0]
    return (x,y)