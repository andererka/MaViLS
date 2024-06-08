# MaViLS algorithm
This repository presents an algorithm to match audioscripts to PDF slides based on OCR, image features and audioscript content or even the combination of the three. It introduces a jump penality that punishes jumps between non-consecutive slides. For further details, we refer to our paper 'MaViLS, a Benchmark Dataset for Video-to-Slide Alignment, Assessing Baseline Accuracy with a Multimodal Alignment Algorithm Leveraging Speech, OCR, and Visual Features' (link to paper is following) and the short overview and description of the code's structure below.

### Overview

The algorithm aligns lecture videos with corresponding slides with a multimodal algorithm that uses audio, OCR and image features all together. The approach uses dynamic programming to include a penalizing term for slide transitions, therefore rewarding linear and non-jumpy slide transitions. Running the mavils algorithm, the user can choose between different options:

* single feature algorithm (that only leverages either OCR or audio or image features)
* merged feature algorithm (were the user can select between the method of merging the feature matrices. Please refer to the paper for an explanation on this. One can select between max, mean and weighted sum)


### Structure of the code

The example dataset used is stored in the folder 'data'. As the lecture videos used are to large to upload to Github, you can get them on Kaggle under the following link: kagglelinktofollow. A video folder with the according content therefore needs to be added to the data folder.
The dataset includes the audioscripts of the lectures (preprocessed with faster-whisper: https://github.com/SYSTRAN/faster-whisper), ground truth files in excel format that include the human labeled slide alignement for the video lectures, and the lecture themselves in pdf format.

The evaluation folder includes the python scripts for calculating the F1, the recall and the precision scores, as well as the label consistency between different (human) labelers. Lastly, the python script 'analyse_videos.py' includes code to calculate the 'jumpiness' and the ratio between no slide and slide view of the video recordings according to the ground truth. 

The helpers folder includes relevant functions how to preprocess the audioscript and how to create the feature matrices and run dynamic programming for finding the optimal slide sequence according to a jump penalty. 

The mavils folder includes the script 'matching_algorithm.py' that needs to be run in order to align pdf slides to the audioscript. A more thorough description is found below.

The results folder includes results from all the example lectures included in the data folder with different jump penalities and different merge methods to combine the feature matrices like it is described in the paper. 

An empty folder named 'frame_images' should exist that is collecting the video frames during the time the matching algorithm is run. 

## How to run matching algorithm

The matching algorithm can be found in the mavils folder. In order to run it, first install the neceasry packages (ideally in a new conda environment). This can be done with:

``pip install requirements.txt``

Additionally, it is probably required for you to install tesseract on your computer and add the path to the tesseract.exe file to the script or create a 'local_settings.py' file like we've done it and store the path there. Then it can be easily imported with:

``from local_settings import path_to_tesseract``

You should be all set now. In order to run the matching_algorithm.py script, direct to the mavils folder and run:

``python matching_algorithm.py``

In order to change the jump penalty, you can add the option --jump_penalty. Default is 0.1.

``python matching_algorithm.py --jump_penalty 0.2``

In order to change the merge method, you can add the option --merge_method. Default is 'max'

``python matching_algorithm.py --merge_method mean``

The following other options exist:

* --sentence_model that sets the sentence transformer model. Default is ''sentence-transformers/distiluse-base-multilingual-cased' but could be changed
* autoimage_name that sets the image processor and model. Default is 'MBZUAI/swiftformer-xs'
* --sift that is either false or true, depending whether one want to run the SIFT algorithm for comparison. This is much time-consuming if set to true.
* --audio_script. Path to the audioscript
* --file_path. Path to the lecture PDF
* --video_path. Path to the video recording of the according lecture
* --file_name. Path to the result file that is created through the processing of the script.




