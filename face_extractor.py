import os
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from pathlib import PurePath, Path
from moviepy.editor import VideoFileClip
from matplotlib import pyplot as plt
from umeyama import umeyama
from mtcnn import MTCNN

class VideoInfo:
    def __init__(self):
        self.frame = 0

class FaceExtractor:
    def __init__(self, weights_file: str = None, min_face_size: int = 20, steps_threshold: list = None,
                 scale_factor: float = 0.709):
        self.mtcnn = MTCNN(min_face_size=min_face_size, steps_threshold=steps_threshold,
                 scale_factor=scale_factor)
    
    @staticmethod
    def get_src_landmarks(x0, x1, y0, y1, pnts):
        """
        x0, x1, y0, y1: (smoothed) bbox coord.
        pnts: landmarks predicted by MTCNN
        """    
        src_landmarks = [(int(pnts[i][1]-x0), 
                          int(pnts[i][0]-y0)) for i in range(5)]
        return src_landmarks
    
    @staticmethod
    def get_tar_landmarks(img):
        """    
        img: detected face image
        """         
        ratio_landmarks = [
            (0.31339227236234224, 0.3259269274198092),
            (0.31075140146108776, 0.7228453709528997),
            (0.5523683107816256, 0.5187296867370605),
            (0.7752419985257663, 0.37262483743520886),
            (0.7759613623985877, 0.6772957581740159)
            ]   

        img_size = img.shape
        tar_landmarks = [(int(xy[0]*img_size[0]), 
                          int(xy[1]*img_size[1])) for xy in ratio_landmarks]
        return tar_landmarks
    
    @staticmethod
    def landmarks_match_mtcnn(src_im, src_landmarks, tar_landmarks): 
        """
        umeyama(src, dst, estimate_scale)
        landmarks coord. for umeyama should be (width, height) or (y, x)
        """
        src_size = src_im.shape
        src_tmp = [(int(xy[1]), int(xy[0])) for xy in src_landmarks]
        tar_tmp = [(int(xy[1]), int(xy[0])) for xy in tar_landmarks]
        M = umeyama(np.array(src_tmp), np.array(tar_tmp), True)[0:2]
        result = cv2.warpAffine(src_im, M, (src_size[1], src_size[0]), borderMode=cv2.BORDER_REPLICATE) 
        return result
    
    @staticmethod
    def process_mtcnn_bbox(bbox, im_shape):
        """
        output bbox coordinate of MTCNN is (y0, x0, y1, x1)
        Here we process the bbox coord. to a square bbox with ordering (x0, y1, x1, y0)
        """
        y0, x0, y1, x1 = bbox[0:4]
        w, h = int(y1 - y0), int(x1 - x0)
        length = (w + h)/2
        center = (int((x1+x0)/2),int((y1+y0)/2))
        new_x0 = np.max([0, (center[0]-length//2)])#.astype(np.int32)
        new_x1 = np.min([im_shape[0], (center[0]+length//2)])#.astype(np.int32)
        new_y0 = np.max([0, (center[1]-length//2)])#.astype(np.int32)
        new_y1 = np.min([im_shape[1], (center[1]+length//2)])#.astype(np.int32)
        bbox[0:4] = new_x0, new_y1, new_x1, new_y0
        return bbox

    def process_image(self, input_img, info, save_interval, save_path, name_prefix): 
        min_conf=0.9

        info.frame += 1
        frame = info.frame 
        
        if frame % save_interval == 0:
            detections = self.mtcnn.detect_faces(input_img)
            for idx, det in enumerate(detections):
                box = det['box']
                conf = det['confidence']
                if conf >= min_conf:
                    size = box[3]
                    x0, y1, x1, y0 = FaceExtractor.process_mtcnn_bbox([box[0], box[1], box[0] + size, box[1] + size], input_img.shape)
                    det_face_im = input_img[int(x0):int(x1),int(y0):int(y1),:]

                    # get src/tar landmarks
                    src_landmarks = FaceExtractor.get_src_landmarks(x0, x1, y0, y1, list(det['keypoints'].values()))
                    tar_landmarks = FaceExtractor.get_tar_landmarks(det_face_im)

                    # align detected face
                    aligned_det_face_im = FaceExtractor.landmarks_match_mtcnn(
                        det_face_im, src_landmarks, tar_landmarks)
                    
                    Path(os.path.join(f"{save_path}", "rgb")).mkdir(parents=True, exist_ok=True)
                    fname = f"./{save_path}/rgb/frame{frame}_{name_prefix}_face{str(idx)}.jpg"
                    plt.imsave(fname, aligned_det_face_im, format="jpg")

                    bm = np.zeros_like(aligned_det_face_im)
                    h, w = bm.shape[:2]
                    bm[int(src_landmarks[0][0]-h/15):int(src_landmarks[0][0]+h/15),
                       int(src_landmarks[0][1]-w/8):int(src_landmarks[0][1]+w/8),:] = 255
                    bm[int(src_landmarks[1][0]-h/15):int(src_landmarks[1][0]+h/15),
                       int(src_landmarks[1][1]-w/8):int(src_landmarks[1][1]+w/8),:] = 255
                    bm = FaceExtractor.landmarks_match_mtcnn(bm, src_landmarks, tar_landmarks)
                    
                    Path(os.path.join(f"{save_path}", "binary_mask")).mkdir(parents=True, exist_ok=True)
                    fname = f"./{save_path}/binary_mask/frame{frame}_{name_prefix}_face{str(idx)}.jpg"
                    plt.imsave(fname, bm, format="jpg")

        return np.zeros((3,3,3))

    def preprocess_video(self, fn_input_video, save_interval, save_path, name_prefix=""):
        info = VideoInfo()
        output = 'dummy.mp4'
        clip1 = VideoFileClip(fn_input_video)
        clip = clip1.fl_image(lambda img: self.process_image(img, info, save_interval, save_path, name_prefix))
        clip.write_videofile(output, audio=False, verbose=False)
        clip1.reader.close()