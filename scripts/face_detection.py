import cv2 
import numpy as np
import math



def calculate_ROI(img,scale):
    """
    Returns the ROI based from the image based on scale

    Parameters
    ----------
    img : np.uint8
        The original image which could be a frame from the camera
    scale : value between 0 to 1
        Percentage of the original image being the ROI
        1 will return the original image back


    Returns
    -------
    new_img : np.uint8
        Calculated ROI
    border : list
        Coordinates of the ROI box
    roi_size : list
        Dimensions of ROI        
    """
    img_dim=img.shape
    roi_height=int(scale*img_dim[1])
    roi_width=int(scale*img_dim[0])
    
    h_border=int((img_dim[1]-roi_height)/2)
    w_border=int((img_dim[0]-roi_width)/2)
    cv2.rectangle(img,(h_border,w_border),(h_border+roi_height,w_border+roi_width),(255,0,0),2)
    
    new_img=img[w_border:w_border+roi_width,h_border:h_border+roi_height]
    border=[h_border,w_border]
    roi_size=[roi_height,roi_width]
    
    return new_img,border,roi_size
    

def detect_face(image,model):
    """
    Find the faces in the image

    Parameters
    ----------
    img : np.uint8
       The image in which faces are to be found
    model : Tensorflow model
    
    Returns
    -------
    faces: list
        Face coordinates (x, y, w, h)     
    """
    roi_area=image.shape[0]*image.shape[1] 
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    model.setInput(blob)
    pred_faces = model.forward()
    faces = []
    for i in range(pred_faces.shape[2]):
        likelihood = pred_faces[0, 0, i, 2]
        if likelihood > 0.7:
            x = int(pred_faces[0, 0, i, 3]*image.shape[1])
            y = int(pred_faces[0, 0, i, 4]*image.shape[0])
            w = int(pred_faces[0, 0, i, 5]*image.shape[1])
            h = int(pred_faces[0, 0, i, 6] *image.shape[0])
            area=(w-x)*(h-y)
            # discard the faces which is less than 10% area of the ROI
            if((area/roi_area)>=0.1):    
                faces.append([x,y,w,h])
    return faces  


def draw_boundbox(faces,image):
    """
    Draw the bounding box around the detected faces

    Parameters
    ----------
    faces : list
        Face coordinates (x, y, x1, y1)     
    
    image : np.uint8
        The image in which face bounding box to be drawn
    """    
    roi_centre=[int(image.shape[1]/2),int(image.shape[0]/2)]
    dist=[]
    for (x,y, w, h) in faces:
        dist.append(math.dist(roi_centre,[int((x+w)/2),int((y+h)/2)]))
    if(len(dist)!=0):
        min_index = np.argmin(dist)
        for i in range(len(dist)):
            if(i == min_index):
                cv2.rectangle(image,(faces[i][0],faces[i][1]),(faces[i][2],faces[i][3]),(0,255,0),2)
            else:
                cv2.rectangle(image,(faces[i][0],faces[i][1]),(faces[i][2],faces[i][3]),(0,0,255),2)






    
