from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import random, cv2
import matplotlib.pyplot as plt


def plot_sample(dataset_name, n=1):
    custom_dataset = DatasetCatalog.get(dataset_name)
    custom_metadata = MetadataCatalog.get(dataset_name)
    for s in random.sample(custom_dataset, n):
        img = cv2.imread(s['file_name'])
        v = Visualizer(img[:, :, ::-1], metadata=custom_metadata, scale=0.5)  # bgr to rgb
        v = v.draw_dataset_dict(s)
        plt.imshow(v.get_image())
        plt.show()


def get_train_cfg(config_file_path, check_point_url, train_dataset_name,
                  test_dataset_name, num_classes, steps, device, output):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(check_point_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = steps
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output
    return cfg
