from custom import *
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
import os, pickle, warnings
warnings.filterwarnings("ignore")

setup_logger()

# ### initialise any pre-trained model from model zoo #####
config_file_path = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
check_point_url = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
device = 'cuda'  # or 'cpu'

# ################ set training parameters #############
output = './output/instance_segmentation'
num_classes = 2
train_dataset_name = 'scene_cancer_train'
test_dataset_name = 'scene_cancer_test'
cfg_save_path = 'scene_data/IS_cfg.pickle'
steps = 2000
# #######################################################
train_images_path = 'scene_data/train'
train_annot_path = 'scene_data/train.json'
test_images_path = 'scene_data/test'
test_annot_path = 'scene_data/test.json'
# ############ register dataset ##########################
register_coco_instances(name=train_dataset_name, metadata={},
                        json_file=train_annot_path, image_root=train_images_path)
register_coco_instances(name=test_dataset_name, metadata={},
                        json_file=test_annot_path, image_root=test_images_path)

# check registered or not, uncomment plot_sample...
# plot_sample(dataset_name=train_dataset_name, n=2)
# ########################## configure ##########################


def main():
    cfg = get_train_cfg(config_file_path=config_file_path, check_point_url=check_point_url,
                        train_dataset_name=train_dataset_name, test_dataset_name=test_dataset_name,
                        num_classes=num_classes, steps=steps, device=device, output=output)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)  # saving cfg
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  # making output directory if not exists
    trainer = DefaultTrainer(cfg)  # loading default trainer
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    main()
