# How To Optimze Quantization

*By Ryan, Jenny and Vinson*

**CTRL+SHIFT+V for pretty view in VSCode**

## Overview

This folder (testing) includes a workflow for tuning our settings for quantization. First by collecting a bunch of unquantized images of Huey. Next, we manually lable the orientation of this folder of pictures. 'test_detect_corners' functionon that takes quantizations as parameters (threshold and weiIn execution, itTghts).  runs quant then corner detection on each

image. For each image we calculate a "score" based on the difference of the true (labled) angle and the angle produced by corner detection. Also, an arbitrary' score, NO_ORIENTATI'ON_SCORE is applied when no orientation is given from corner detection. The average score is returned by this function. 

Finally, a differenc script uses Optuna to treat this function like a black box and tries a bunch of params and spits out the combination that results in the lowest score. (Low is good like golf). Below is a step by step guide on how to run this workflow.

## 1. Data collection
Note: for this part you need to be in the terminal for the main directory because we're using models from main
- Alter 'collect_huey_imgs.py' settings so that VIDEO_NAME is the name of the video (in main_files) you want to tune for
- You may want to alter COLLECTION_FREQ to set the number of frames the code will wait between saving each images (higher = save less often = less images)
- Run python .\testing\collect_huey_imgs.py
- Your dataset will be in testing/test_data (Recomended: change the name)

## 2. Angle labeling
- You may now cd into testing
- In angle_data_label.py, rename DATA_SET_NAME to the name of your dataset folder in test_data
- Select your data labeling style (more details on that below) and begin labeling angles
- Skip (press S) any images not of huey or where it is impossible to get an orientation
- The information will be stored in a csv in your image dataset
- **If you run it again you will override your data**

### Data Labeling style
There are two style of data labeling: 2 point selection orientation, 1 point selection with assigned center orientation

Labels: 
- Data path: "DATA_SET_NAME" (current state: pictures)
- Bool: 'MANUAL_CENTER'; True = 2 point selection, False = 1 point selection 
    - True = 2 points: user selected points of 'state["tail"]' and 'state["head"]'
    - False = 1 point: first point is predetermined in 'State[center]', second point is user selected 'state["head"]'
- Points (handled in code): 
    - 'state[center]' = the center of the full image 
    - 'state["tail"]' = base of the vector 
    - 'state["head"]' = head of vector; final point dictating orientation 

## 3. Tuning
- In corner_testing_main.py set your DATA_SET_NAME and set COLOR_SELECT_IMG to an image in your data set that looks good for color selection
- In optimization.py set NO_ORIENTATION_SCORE. This number is the score we assign if corner detection does not pick up an orientation. Put simply, a lower number means your settings are more likely to give you detection with no orientation (though, when you get an orientation it may be more accurate).
- Run optimization.py and it will spit out settings for your dataset. You can add these to quant_settings.json and select them in main.