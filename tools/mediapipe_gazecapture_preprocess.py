from mediapipe_eye_detector import MPFacialLandmarkDetector
import cv2
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import json

error_list=[]

def process_image(landmark_detector, image_path, face_folder, left_eye_folder, right_eye_folder, mask_folder, original_grid_folder, crop_list, error_list):
    try:
        img = cv2.imread(image_path)

        h, w = img.shape[:2]

        landmark_mask = np.zeros((h, w), dtype=np.uint8)

        landmarks = landmark_detector.get_landmarks(img)

        if landmarks is not None:
            for landmark in landmarks:
                left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157]
                right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381]

                left_eye_points = [landmark[i] for i in left_eye_indices]
                right_eye_points = [landmark[i] for i in right_eye_indices]

                #face = landmark_detector.get_square_roi(img, landmark, scale=1.2)
                face, original_grid, face_bbox = landmark_detector.get_square_roi_and_grid(img, landmark, scale=1.2)
                left_eye, left_eye_bbox = landmark_detector.get_square_roi(img, left_eye_points, scale=1.5)
                right_eye, right_eye_bbox = landmark_detector.get_square_roi(img, right_eye_points, scale=1.5)

                # save the landmark grid
                for i in landmark:
                    adjusted_coordinate = (min(i[1], h-1), min(i[0], w-1))
                    landmark_mask[adjusted_coordinate] = 1

            # Save the processed images
            
            if face is not None and left_eye is not None and right_eye is not None and landmark_mask is not None and original_grid is not None:
                cv2.imwrite(os.path.join(face_folder, f"{Path(image_path).stem}.jpg"), face)
                cv2.imwrite(os.path.join(left_eye_folder, f"{Path(image_path).stem}.jpg"), left_eye)
                cv2.imwrite(os.path.join(right_eye_folder, f"{Path(image_path).stem}.jpg"), right_eye)
                cv2.imwrite(os.path.join(mask_folder, f"{Path(image_path).stem}.jpg"), landmark_mask)
                cv2.imwrite(os.path.join(original_grid_folder, f"{Path(image_path).stem}.jpg"), original_grid)
                crop_list.append([f"{Path(image_path).stem}.jpg", face_bbox, left_eye_bbox, right_eye_bbox])
            else:
                error_list.append(f"{Path(image_path).stem}.jpg")

    except Exception as e:
        print('image:',f"{Path(image_path).stem}.jpg",'Error occur:', e) 
        name_str = f"{Path(image_path).stem}.jpg"
        error_list.append(name_str)
        

def process_dataset(root_path):
    landmark_detector = MPFacialLandmarkDetector()
    img_dir = os.path.join(root_path, 'images')
    img_list = os.listdir(img_dir)
    face_folder = os.path.join(root_path, "Face")
    left_eye_folder = os.path.join(root_path, "LeftEye")
    right_eye_folder = os.path.join(root_path, "RightEye")
    mask_folder = os.path.join(root_path, "LandMarkMask")
    original_grid_folder = os.path.join(root_path, "Original_Grid")
    if not os.path.exists(face_folder):
        os.makedirs(face_folder)
    if not os.path.exists(left_eye_folder):
         os.makedirs(left_eye_folder)
    if not os.path.exists(right_eye_folder):
        os.makedirs(right_eye_folder)
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)  
    if not os.path.exists(original_grid_folder):
        os.makedirs(original_grid_folder)      
    crop_list = []
    error_list = []
    for img in img_list:
        img_path = os.path.join(img_dir, img)
        process_image(landmark_detector, img_path, face_folder, left_eye_folder, right_eye_folder, mask_folder, original_grid_folder, crop_list, error_list)
    with open("eye_face_bbox.json", 'w') as bb:
        json.dump(crop_list, bb)
    with open('error.json', 'w') as er:
        json.dump(error_list, er)
                            


if __name__ == '__main__':

    root_path = r'C:\Users\Xuli\Desktop\MPIIGazeCode'
    process_dataset(root_path)
    
