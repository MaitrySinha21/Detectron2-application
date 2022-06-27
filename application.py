from detectron import *
import cv2
"""
read any bgr image with cv2 otherwise convert rgb-to-bgr
"""
# imgPath = 'images/city.jpg'
imgPath = 'scene_data/test/malignant_256.jpg'
img = cv2.imread(imgPath)


detector = Detector(model_type='CUSTOM')
detector.seg_img(img)
# detector.seg_video(video='video.mp4')
detector.object_detection(img)
# detector.video_detection(path='video.mp4')
# detector.key_point_detection(img)
# detector.key_point_video(path='video.mp4')

