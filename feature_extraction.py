import sys; sys.path.append('./scene_graph_benchmark')
from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir

import os
import glob
import cv2
import torch
from PIL import Image
import numpy as np
import tqdm
import json
import h5py
import torch.nn as nn
from tqdm import tqdm_notebook
import os

import json
import numpy as np
# import fasttext.util
import tqdm
import argparse

def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    cv2_img = cv2.resize(cv2_img, (1024, 1024))
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

def extract_features(img_paths, transforms, model):
    model.eval()
    img_inputs = []
    for img_path in img_paths:
        try:
            image = cv2.imread(img_path)
            img_input = cv2Img_to_Image(image)
        except:
            image = Image.open(img_path)
            img_input = np.array(image)
            if img_input.shape[-1] < 3:
                img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
            img_input = cv2Img_to_Image(img_input)

        img_input, _ = transforms(img_input, target=None)
        img_input = img_input.to(cfg.MODEL.DEVICE)
        raw_height, raw_width = img_input.shape[-2:]
        img_inputs.append(img_input.unsqueeze(0))

    img_inputs = torch.cat(img_inputs, dim=0)
    
    with torch.no_grad():
        predictions = model(img_inputs.type(torch.FloatTensor))

    # prediction = prediction.resize((raw_width, raw_height))
    batch_box_features = []
    for prediction in predictions:
        prediction = prediction.to('cpu')
        det_dict = {key : prediction.get_field(key) for key in prediction.fields()}
        box_features = det_dict['box_features']
        batch_box_features.append(box_features)
    
    return batch_box_features

def feature_extraction(frame_dir, cfg_file, save_features_path):
    files = os.listdir(frame_dir)
    folder_path = frame_dir
    img_paths = [folder_path + file for file in files]
    full_img_paths = []
    # filter_videos = open('./filter_videos.txt', 'r').read().split('\n')
    for img_path in img_paths:
        # if os.path.basename(img_path) in filter_videos:
        img_names = glob.glob(img_path + '/*')
        img_names.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
        for i in range(0, len(img_names), 15):
            full_img_paths.append(img_names[i])

    #Setting configuration
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)

    #Configuring VinVl
    cfg.merge_from_file(cfg_file)
    argument_list = [
                    'MODEL.WEIGHT', './vinvl_vg_x152c4.pth',
                    'MODEL.ROI_HEADS.NMS_FILTER', 1,
                    'MODEL.ROI_HEADS.SCORE_THRESH', 0.2, 
                    'TEST.IGNORE_BOX_REGRESSION', False,
                    'MODEL.ATTRIBUTE_ON', True,
                    'MODEL.DEVICE', 'cuda:0',
                    'TEST.OUTPUT_FEATURE', True,
    ]

    cfg.merge_from_list(argument_list)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR

    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)

    transforms = build_transforms(cfg, is_train=False)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    if not os.path.exists(save_features_path):
        os.mkdir(save_features_path)

    batch_size = 16
    batch_full_img_paths = [full_img_paths[i:i + batch_size] for i in range(0, len(full_img_paths), batch_size)]
    
    for batch in tqdm.tqdm(batch_full_img_paths):
        batch_box_features = extract_features(batch, transforms, model)
        for box_features, img_path in zip(batch_box_features, batch):
            np.save(save_features_path + '/' + os.path.basename(img_path).split('.')[0] + '.npy',
                    box_features.cpu().detach().numpy())
        
def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='VinVL feature extraction Args')

    parser.add_argument('--frame_dir',
                      help='test video frame path',
                      default='./thompson_data/thompson_test_vqa_frames/',
                      type=str, required=True)

    parser.add_argument('--cfg_file',
                      help='path to cfg file of VinVL model',
                      default='./scene_graph_benchmark/sgg_configs/vgattr/vinvl_x152c4.yaml', type=str)
    
    parser.add_argument('--save_features_path',
                      help='path to save npy features',
                      default='./thompson_data/vinvl_thompson_test_features/', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    feature_extraction(args.frame_dir, args.cfg_file, args.save_features_path)
