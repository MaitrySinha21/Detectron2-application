from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
import os, pickle
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

classFile = 'custom.names'
cfg_save_path = 'data/IS_cfg.pickle'


def readClasses(classFile):
    with open(classFile, 'r') as f:
        classList = f.read().splitlines()
    return classList


class Detector:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg_save_path = cfg_save_path
        with open(self.cfg_save_path, 'rb') as f:
            self.cfg = pickle.load(f)
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.predictor = DefaultPredictor(self.cfg)

    def segmentation(self, img):
        image = img[:, :, ::-1].copy()  # converted to rgb
        prediction = self.predictor(image)
        class_idxs = prediction["instances"].pred_classes.tolist()
        cls_names = readClasses(classFile)
        viz = Visualizer(image, metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                         instance_mode=ColorMode.SEGMENTATION)
        output = viz.draw_instance_predictions(prediction['instances'].to('cpu'))
        im = np.array(output.get_image()[:, :, ::-1])  # converted to bgr
        im = cv2.resize(im, (320, 320), interpolation=cv2.INTER_NEAREST)
        for idx in class_idxs:
            clss = cls_names[idx]
            cv2.rectangle(im, (0, 23), (30 + len(clss) * 9, 0), color=(138, 73, 17), thickness=-1)
            cv2.putText(im, text=clss, org=(5, 15),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.6,
                        color=(255, 255, 255), thickness=1)
        cv2.imwrite('static/segment.png', im)

    def detection(self, img):
        image = img.copy()  # ##[:, :, ::-1].copy()   # converted to rgb
        prediction = self.predictor(image)
        class_idxs = prediction["instances"].pred_classes.tolist()
        cls_names = readClasses(classFile)
        boxes = prediction["instances"].pred_boxes
        color = (255, 25, 25)
        for box, idx in zip(boxes, class_idxs):
            box = box.tolist()
            cls = cls_names[idx]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
            img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(img, (0, 23), (30 + len(cls) * 9, 0), color=(138, 73, 17), thickness=-1)
            cv2.putText(img, text=str(cls), org=(5, 16),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.6, color=(255, 255, 255), thickness=1)
        cv2.imwrite('static/detect.png', img)
        cv2.imwrite('static/input.png', image)
