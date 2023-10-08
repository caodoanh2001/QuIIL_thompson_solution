
Implementation of the report:

"QuilL at T3 challenge: Towards Automation in Life-Saving Intervention Procedures from First-Person View"

for visual question answering task.

*Please follow the provided documentation and instructions to use the code effectively for visual question answering tasks. If you have any questions or need assistance, feel free to refer to the documentation or reach out to us.*

## Step 1. Download dataset & video-to-frame processing

Please download the dataset provided by T3 challenge here: https://thompson-challenge.grand-challenge.org/

There would be two files: `Train.zip` and `Test.zip`. Unzip them and obtain the following structure:

```
Train/Test
|__ Annotations
|____ *.json
|__ Videos
|____ *.mp4
```

Then, run the following script to cut video to frames:

```
python cut_video_to_frames.py \
--video_path TESTING_VIDEO_PATH \
--save_path PATH_TO_SAVE_FRAMES \
--test_question_json_path PATH_TO_TEST_JSON_FILES
```

Then, frames are saved into `PATH_TO_SAVE_FRAMES`.

```
PATH_TO_SAVE_FEATURES
|__ *_[idx frame].jpg
```

## Step 2. Feature extraction
As mentioned in the paper, we used VinVL pre-trained model for feature extraction. Features of each frame will be N object feature vectors, storing in the `*.npy` files. Please run the following script for feature extracion:

```
CUDA_VISIBLE_DEVICES=[gpu id] python feature_extraction.py \
--frame_dir PATH_TO_SAVE_FRAMES \
--cfg_file PATH_TO_VINVL_CFG \
--save_features_path PATH_TO_SAVE_FEATURES
```

`*.npy` files storing feature vectors of each frame will be stored at `PATH_TO_SAVE_FEATURES`.

```
PATH_TO_SAVE_FEATURES
|__ *.npy
```

## Step 3. Inference

Run the below script for generating output answers:

```
CUDA_VISIBLE_DEVICES=[gpu id] python inference_submit.py \
--path_test_video_frames PATH_TO_SAVE_FRAMES \
--path_features PATH_TO_SAVE_FEATURES \
--ckpt_path PATH_TO_CHECKPOINT \
--submission_template_path PATH_TO_JSON_TEMPLATE_FILE \
--out_path PATH_TO_SAVE_OUTPUT_ANSWERS
```

* `PATH_TO_SAVE_FRAMES`: This argument represents the directory path where frames from step 1 will be stored.
* `PATH_TO_SAVE_FEATURES`: Specify the directory path where the extracted features from step 2 are saved.
* `PATH_TO_CHECKPOINT`: This argument should contain the checkpoint path. You can download the required checkpoint file from [here](link-to-download-checkpoint).
* `submission_template_path`: This refers to the `*.json` file that stores questions per frame for testing videos. You can download the template from [here](link-to-download-template).
* `PATH_TO_SAVE_OUTPUT_ANSWERS`: This argument indicates the directory path where the generated `*.csv` file, containing answers per frame for testing videos, will be saved.

The structure of generated `*.csv` file:
```
file_name,frame_num,question,answer_index,answer
CT1.json,0,What limb is injured?,5,no limb is injured
CT1.json,0,Is the patient intubated?,3,can't identify
CT1.json,0,Where is the catheter inserted?,3,no catheter is used
CT1.json,0,Is there bleeding?,2,no
CT1.json,0,Has the bleeding stopped?,3,there is no bleeding
CT1.json,0,Is the patient moving?,3,can't identify
CT1.json,0,Is the patient breathing?,3,can't identify
CT1.json,0,Is there a tourniquet?,2,no
CT1.json,0,Is there an incision?,1,yes
CT1.json,0,Is there a chest tube?,2,no
```
