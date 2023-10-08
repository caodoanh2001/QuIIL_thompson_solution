import sys
sys.path.append("./mcan-vqa")
from core.model.net import Net2
from cfgs.base_cfgs_test import Cfgs
from core.data.load_data import DataSet
from core.data.data_utils import proc_img_feat, proc_ques, tokenize

import json, torch
import numpy as np
import yaml
from tqdm import tqdm
import glob
import os
import argparse

def inference(path_test_video_frames, path_features, ckpt_path, submission_template_path, out_path):
    __C = Cfgs()
    cfg_file = "./mcan-vqa/cfgs/{}_model.yml".format("large")
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.full_load(f)

    args_dict = {**yaml_dict}
    __C.add_args(args_dict)
    __C.proc()

    # dataset = DataSet(__C)
    token_size = 39
    ans_size = 16
    # pretrained_emb = dataset.pretrained_emb
    pretrained_emb = np.load("pretrained_emb.npy")
    ix_to_ans = {'0': 'there is no bleeding', '1': 'no catheter is used', '2': 'throat', '3': 'no limb is injured', '4': 'lower limb', '5': 'right arm', '6': "can't identify", '7': 'right leg', '8': 'no', '9': 'left arm', '10': 'yes', '11': 'left leg', '12': 'upper limb', '13': 'abdomen', '14': 'thorax', '15': 'none'}

    net = Net2(
        __C ,
        pretrained_emb,
        token_size,
        ans_size
    )
    net.cuda()
    net.eval()

    stat_ques_list = \
                json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions'] + \
                json.load(open(__C.QUESTION_PATH['val'], 'r'))['questions'] + \
                json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
                json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']

    token_to_ix, pretrained_emb = tokenize(stat_ques_list, __C.USE_GLOVE)

    chkponts = torch.load(ckpt_path)["state_dict"]
    net.load_state_dict({k.replace('module.',''):chkponts[k] for k in chkponts.keys()})

    # build answer_id
    dict_ques_ans = dict()
    all_questions = [item for item in open('./mcan-vqa/all_questions.txt').read().split('\n')]
    all_answers = [item.split(',') for item in open('./mcan-vqa/all_answers.txt').read().split('\n')]

    for ques, ans in zip(all_questions, all_answers):
        dict_ques_ans[ques] = {answer.strip():i+1 for i, answer in enumerate(ans)}

    frame_step = 15
    submission_template = json.load(open(submission_template_path))
    valid_videos = [item.split('.')[0] for item in submission_template.keys()]

    f = open(out_path, 'w')

    for i_vid, video in enumerate(valid_videos):
        print("Predicting video", video, str(i_vid+1) + '/' + str(len(valid_videos)))
        questions = submission_template[video + '.json']["frames"]
        frame_indices = list(range(int(list(questions.keys())[0]), int(list(questions.keys())[-1])+1))
        len_cut_frames = len(glob.glob(path_test_video_frames + '/' + video + '/*'))
        # For videos cut sucessfully all frames
        sub_frame_indices = [frame_indices[i:i + frame_step] for i in range(0, len(frame_indices), frame_step)]
        only_one_feat = False
        if len_cut_frames < frame_step:
            only_one_feat = True
            name_feature = path_features + video + '_0' + '.npy'
        
        # List all available featres
        list_names = glob.glob(path_features + video + '_*')
        list_names.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

        for s_frame_indices in tqdm(sub_frame_indices):
            if not only_one_feat: # If not using one feature for whole video
                if len(s_frame_indices) < frame_step:
                    name_feature = path_features + video + '_' + str(s_frame_indices[0]) + '.npy'
                else:    
                    name_feature = path_features + video + '_' + str(s_frame_indices[-1]+1) + '.npy'          
            
                if not os.path.isfile(name_feature): # If frame not found, use the last feature
                    name_feature = list_names[-1]

            features = proc_img_feat(np.load(name_feature), __C.IMG_FEAT_PAD_SIZE)
            img_feat_iter = torch.from_numpy(features).cuda().unsqueeze(0)
            ques_ix_iters = []
            
            # For submission
            text_questions = []
            dup_frame_indices = []
            
            for frame_idx in s_frame_indices:
                set_question = questions[str(frame_idx)]
                for question in set_question.keys():
                    ques_ix_iter = proc_ques({'question': question.strip()}, token_to_ix, __C.MAX_TOKEN)
                    ques_ix_iter = torch.from_numpy(ques_ix_iter).cuda()
                    ques_ix_iters.append(ques_ix_iter.unsqueeze(0))
                    text_questions.append(question)
                    dup_frame_indices.append(frame_idx)

            ques_ix_iters = torch.cat(ques_ix_iters, dim=0)
            img_feat_iter = img_feat_iter.repeat(ques_ix_iters.shape[0], 1, 1)
            pred = net(img_feat_iter, ques_ix_iters)
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)
            answers = [ix_to_ans[str(i)] for i in pred_argmax]

            for f_id, ques, ans, in zip(dup_frame_indices, text_questions, answers):
                try:
                    try:
                        line = [video + '.json', str(f_id), ques, str(dict_ques_ans[ques][ans]), ans]
                    except:
                        line = [video + '.json', str(f_id), ques, str(dict_ques_ans[ques]['none']), ans]
                except:
                    print("Detect out-of-idex answer")
                    line = [video + '.json', str(f_id), ques, str(dict_ques_ans[ques][list(dict_ques_ans[ques].keys())[-1]]), list(dict_ques_ans[ques].keys())[-1]]
                f.write(','.join(line)+'\n')
    f.close()

        
def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Inference Args')

    parser.add_argument('--path_test_video_frames',
                      help='test video frame path',
                      default='./thompson_data/thompson_test_vqa_frames/',
                      type=str)

    parser.add_argument('--path_features',
                      help='path to save npy features',
                      default='./thompson_data/vinvl_thompson_test_features/', type=str)
    
    parser.add_argument('--ckpt_path',
                      help='checkpoint path',
                      default='./best_cpkt_vqa.pkl', type=str)
    
    parser.add_argument('--submission_template_path',
                      help='submission template path',
                      default='./thompson_data/Test/Annotations/test_questions.json', type=str)

    parser.add_argument('--out_path',
                      help='path to save output submission file',
                      default='./submission.csv', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    inference(args.path_test_video_frames, args.path_features, args.ckpt_path, args.submission_template_path, args.out_path)