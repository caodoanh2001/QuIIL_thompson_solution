import cv2
import glob
import os
from tqdm import tqdm
import json
import argparse

def process(video_path, save_path, test_question_json_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dict_video_frames = {}
    test_questions = json.load(open(test_question_json_path))
    for item in test_questions:
        dict_video_frames[item.split('.')[0]] = len(test_questions[item]['frames'].keys())

    for video_path in tqdm(glob.glob(video_path + '/*')):
        print(video_path)
        vidcap = cv2.VideoCapture(video_path)
        frame_num = dict_video_frames[os.path.basename(video_path).split('.')[0]]
        write_path = save_path + '/' + os.path.basename(video_path).split('.')[0]
        if not os.path.exists(write_path):
            os.mkdir(write_path)
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = vidcap.read()
        count = 0
        for i in tqdm(range(length)):
            cv2.imwrite(write_path + "_%d.jpg" % count, image) 
            success, image = vidcap.read()
            count += 1

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='MCAN Args')

    parser.add_argument('--video_path',
                      help='test video path',
                      default='./thompson_data/Test/Videos/',
                      type=str, required=True)

    parser.add_argument('--save_path',
                      help='path to save frames',
                      default='./thompson_data/thompson_test_vqa_frames/', type=str)
    
    parser.add_argument('--test_question_json_path',
                      help='path to json file storing test questions',
                      default='./thompson_data/Test/Annotations/test_questions.json', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    process(args.video_path, args.save_path, args.test_question_json_path)