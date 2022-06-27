from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
import os, pickle
from custom import *
import cv2
import warnings
warnings.filterwarnings("ignore")

classFile = 'coco.names'
"""
parameters to set for different custom models, uncomment bellow 
"""
# classFile = 'custom.names'
cfg_save_path = 'scene_data/IS_cfg.pickle'


def readClasses(classFile):
    with open(classFile, 'r') as f:
        classList = f.read().splitlines()
    return classList


class Detector:
    def __init__(self, model_type='OD'):
        self.cfg = get_cfg()
        self.model_type = model_type
        if model_type == 'OD':       # Object Detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == 'IS':     # Instance Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == 'KPS':    # Keypoint Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif model_type == 'PS':     # Panoptic Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
        elif model_type == 'CUSTOM':  # Custom model
            self.cfg_save_path = cfg_save_path
            with open(self.cfg_save_path, 'rb') as f:
                self.cfg = pickle.load(f)
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.predictor = DefaultPredictor(self.cfg)

    def segmentation(self, img):
        image = img[:, :, ::-1].copy()  # converted to rgb
        prediction = self.predictor(image)
        viz = Visualizer(image, metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                         instance_mode=ColorMode.SEGMENTATION)
        output = viz.draw_instance_predictions(prediction['instances'].to('cpu'))
        return output.get_image()[:, :, ::-1]  # converted to bgr

    def seg_img(self, img):
        img = self.segmentation(img)
        cv2.imshow('Results', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def seg_video(self, video=0):
        cap = cv2.VideoCapture(video)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (1000, 600), interpolation=cv2.INTER_NEAREST).copy()
                frame = self.segmentation(frame)
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) == 13:
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def detection(self, img):
        image = img[:, :, ::-1].copy()   # converted to rgb
        prediction = self.predictor(image)
        class_idxs = prediction["instances"].pred_classes.tolist()
        cls_names = readClasses(classFile)
        boxes = prediction["instances"].pred_boxes
        color = (255, 255, 255)
        for box, idx in zip(boxes, class_idxs):
            box = box.tolist()
            cls = cls_names[idx]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img, (x1, y1-15), (x1+len(cls)*7, y1-1), color=(138, 73, 17), thickness=-1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=1)
            cv2.putText(img, text=str(cls), org=(x1, y1-5),\
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.35,
                        color=(255, 255, 255), thickness=1)

        return img

    def key_points(self, img):
        image = img[:, :, ::-1].copy()  # converted to rgb
        outputs = self.predictor(image)
        """
        extracting all 17 key points
        name = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_soulder', 'right_soulder',
                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                'right_knee', 'left_ankle', 'right_ankle']
        """
        col = (255, 255, 255)
        col1 = (0, 200, 0)
        black = (0, 0, 0)
        for det_keypoints in outputs["instances"].pred_keypoints:
            person_keypoint = {
                "nose": {'x': int(det_keypoints.cpu().numpy()[0][0]),
                         'y': int(det_keypoints.cpu().numpy()[0][1]),
                         'conf': det_keypoints.cpu().numpy()[0][2]},
                "left_eye": {'x': int(det_keypoints.cpu().numpy()[1][0]),
                             'y': int(det_keypoints.cpu().numpy()[1][1]),
                             'conf': det_keypoints.cpu().numpy()[1][2]},
                "right_eye": {'x': int(det_keypoints.cpu().numpy()[2][0]),
                              'y': int(det_keypoints.cpu().numpy()[2][1]),
                              'conf': det_keypoints.cpu().numpy()[2][2]},
                "left_ear": {'x': int(det_keypoints.cpu().numpy()[3][0]),
                             'y': int(det_keypoints.cpu().numpy()[3][1]),
                             'conf': det_keypoints.cpu().numpy()[3][2]},
                "right_ear": {'x': int(det_keypoints.cpu().numpy()[4][0]),
                              'y': int(det_keypoints.cpu().numpy()[4][1]),
                              'conf': det_keypoints.cpu().numpy()[4][2]},
                "left_shoulder": {'x': int(det_keypoints.cpu().numpy()[5][0]),
                                  'y': int(det_keypoints.cpu().numpy()[5][1]),
                                  'conf': det_keypoints.cpu().numpy()[5][2]},
                "right_shoulder": {'x': int(det_keypoints.cpu().numpy()[6][0]),
                                   'y': int(det_keypoints.cpu().numpy()[6][1]),
                                   'conf': det_keypoints.cpu().numpy()[6][2]},
                "left_elbow": {'x': int(det_keypoints.cpu().numpy()[7][0]),
                               'y': int(det_keypoints.cpu().numpy()[7][1]),
                               'conf': det_keypoints.cpu().numpy()[7][2]},
                "right_elbow": {'x': int(det_keypoints.cpu().numpy()[8][0]),
                                'y': int(det_keypoints.cpu().numpy()[8][1]),
                                'conf': det_keypoints.cpu().numpy()[8][2]},
                "left_wrist": {'x': int(det_keypoints.cpu().numpy()[9][0]),
                               'y': int(det_keypoints.cpu().numpy()[9][1]),
                               'conf': det_keypoints.cpu().numpy()[9][2]},
                "right_wrist": {'x': int(det_keypoints.cpu().numpy()[10][0]),
                                'y': int(det_keypoints.cpu().numpy()[10][1]),
                                'conf': det_keypoints.cpu().numpy()[10][2]},
                "left_hip": {'x': int(det_keypoints.cpu().numpy()[11][0]),
                             'y': int(det_keypoints.cpu().numpy()[11][1]),
                             'conf': det_keypoints.cpu().numpy()[11][2]},
                "right_hip": {'x': int(det_keypoints.cpu().numpy()[12][0]),
                              'y': int(det_keypoints.cpu().numpy()[12][1]),
                              'conf': det_keypoints.cpu().numpy()[12][2]},
                "left_knee": {'x': int(det_keypoints.cpu().numpy()[13][0]),
                              'y': int(det_keypoints.cpu().numpy()[13][1]),
                              'conf': det_keypoints.cpu().numpy()[13][2]},
                "right_knee": {'x': int(det_keypoints.cpu().numpy()[14][0]),
                               'y': int(det_keypoints.cpu().numpy()[14][1]),
                               'conf': det_keypoints.cpu().numpy()[14][2]},
                "left_ankle": {'x': int(det_keypoints.cpu().numpy()[15][0]),
                               'y': int(det_keypoints.cpu().numpy()[15][1]),
                               'conf': det_keypoints.cpu().numpy()[15][2]},
                "right_ankle": {'x': int(det_keypoints.cpu().numpy()[16][0]),
                                'y': int(det_keypoints.cpu().numpy()[16][1]),
                                'conf': det_keypoints.cpu().numpy()[16][2]}
            }

            noses = person_keypoint['nose']
            l_eyes = person_keypoint['left_eye']
            r_eyes = person_keypoint['right_eye']
            l_ear = person_keypoint['left_ear']
            r_ear = person_keypoint['right_ear']
            l_shoulder = person_keypoint['left_shoulder']
            r_shoulder = person_keypoint['right_shoulder']
            l_elbow = person_keypoint['left_elbow']
            r_elbow = person_keypoint['right_elbow']
            l_wrist = person_keypoint['left_wrist']
            r_wrist = person_keypoint['right_wrist']
            l_hip = person_keypoint['left_hip']
            r_hip = person_keypoint['right_hip']
            l_knee = person_keypoint['left_knee']
            r_knee = person_keypoint['right_knee']
            l_ankle = person_keypoint['left_ankle']
            r_ankle = person_keypoint['right_ankle']
            # extract 17 key points
            x1, y1 = noses['x'], noses['y']
            x2, y2 = l_eyes['x'], l_eyes['y']
            x3, y3 = r_eyes['x'], r_eyes['y']
            x4, y4 = l_ear['x'], l_ear['y']
            x5, y5 = r_ear['x'], r_ear['y']
            x6, y6 = l_shoulder['x'], l_shoulder['y']
            x7, y7 = r_shoulder['x'], r_shoulder['y']
            x8, y8 = l_elbow['x'], l_elbow['y']
            x9, y9 = r_elbow['x'], r_elbow['y']
            x10, y10 = l_wrist['x'], l_wrist['y']
            x11, y11 = r_wrist['x'], r_wrist['y']
            x12, y12 = l_hip['x'], l_hip['y']
            x13, y13 = r_hip['x'], r_hip['y']
            x14, y14 = l_knee['x'], l_knee['y']
            x15, y15 = r_knee['x'], r_knee['y']
            x16, y16 = l_ankle['x'], l_ankle['y']
            x17, y17 = r_ankle['x'], r_ankle['y']

            # print lines
            cv2.line(img, (x1, y1), (x2, y2), color=col1, thickness=2)
            cv2.line(img, (x1, y1), (x3, y3), color=col1, thickness=2)
            cv2.line(img, (x4, y4), (x2, y2), color=col1, thickness=2)
            cv2.line(img, (x5, y5), (x3, y3), color=col1, thickness=2)
            pnt1 = (x6+(x7-x6)//2, y6)
            cv2.line(img, (x1, y1), pnt1, color=(0, 255, 255), thickness=3)
            pnt2 = (x12+(x13-x12)//2, y12)
            cv2.line(img, pnt1, pnt2, color=(0, 255, 255), thickness=3)

            cv2.line(img, (x6, y6), (x7, y7), color=(24, 61, 242), thickness=3)

            cv2.line(img, (x6, y6), (x8, y8), color=col1, thickness=3)
            cv2.line(img, (x7, y7), (x9, y9), color=col1, thickness=3)
            cv2.line(img, (x8, y8), (x10, y10), color=col1, thickness=3)
            cv2.line(img, (x9, y9), (x11, y11), color=col1, thickness=3)

            cv2.line(img, (x12, y12), (x13, y13), color=(24, 61, 242), thickness=3)

            cv2.line(img, (x12, y12), (x14, y14), color=col1, thickness=3)
            cv2.line(img, (x13, y13), (x15, y15), color=col1, thickness=3)
            cv2.line(img, (x14, y14), (x16, y16), color=col1, thickness=3)
            cv2.line(img, (x15, y15), (x17, y17), color=col1, thickness=3)
            # print circles
            cv2.circle(img, (x1, y1), radius=3, thickness=-1, color=col)
            cv2.circle(img, (x2, y2), radius=3, thickness=1, color=black)
            cv2.circle(img, (x3, y3), radius=3, thickness=1, color=black)
            cv2.circle(img, (x4, y4), radius=2, thickness=-1, color=black)
            cv2.circle(img, (x5, y5), radius=2, thickness=-1, color=black)
            cv2.circle(img, (x6, y6), radius=4, thickness=1, color=col)
            cv2.circle(img, (x7, y7), radius=4, thickness=1, color=col)
            cv2.circle(img, (x8, y8), radius=4, thickness=1, color=col)
            cv2.circle(img, (x9, y9), radius=4, thickness=1, color=col)
            cv2.circle(img, (x10, y10), radius=4, thickness=1, color=col)
            cv2.circle(img, (x11, y11), radius=4, thickness=1, color=col)
            cv2.circle(img, (x12, y12), radius=4, thickness=1, color=col)
            cv2.circle(img, (x13, y13), radius=4, thickness=1, color=col)
            cv2.circle(img, (x14, y14), radius=4, thickness=1, color=col)
            cv2.circle(img, (x15, y15), radius=4, thickness=1, color=col)
            cv2.circle(img, (x16, y16), radius=4, thickness=1, color=col)
            cv2.circle(img, (x17, y17), radius=4, thickness=1, color=col)

        return img

    def object_detection(self, img):
        img = self.detection(img)
        cv2.imshow('Results', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def video_detection(self, path=0):
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (1000, 600), interpolation=cv2.INTER_NEAREST).copy()
                frame = self.detection(frame)
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) == 13:
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def key_point_detection(self, img):
        img = self.key_points(img)
        cv2.imshow('Results', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def key_point_video(self, path=0):
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (1000, 600), interpolation=cv2.INTER_NEAREST).copy()
                frame = self.key_points(frame)
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) == 13:
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
