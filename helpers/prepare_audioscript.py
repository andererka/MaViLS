import re
import sys
import pandas as pd
sys.path.append('../')

def generate_output_dict_by_word(file):
    """generates dictionary with time of video recording as key and audio text said by the lecturer at this time as value.

    Args:
        file (str): file name/path to audioscript which was generated in such a format: [00:00:13.260 --> 00:00:14.200] class
                    The audioscript stores a word said in every row together with the exact time of the video recording (start and stop).
                    For generating these audioscripts please refer to e.g. to this repo: https://github.com/SYSTRAN/faster-whisper
        output_file (str, optional): output file path/name. Defaults to '../data/unlabeled_ground_truth/output_file.xlsx'.

    Returns:
        dict: dictionary with time of video recording as key and audio text as value by WORD
    """
    with open(file, encoding='utf-8') as f:
        input_text = f.read()

    # Regular expression pattern to match timestamps
    pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\s-->\s(\d{2}:\d{2}:\d{2}\.\d{3})\] (.+?)\n'

    # Find all matches in the input text
    matches = re.findall(pattern, input_text)

    # Initialize dictionary to store converted timestamps and corresponding text
    output_dict = {}

    # Convert timestamps to seconds and populate the dictionary
    for match in matches:
        start_time = match[0]
        end_time = match[1]
        text = match[2]
        end_seconds = int(start_time[:2]) * 3600  + int(end_time[3:5]) * 60 + float(end_time[6:])
        
        output_dict[end_seconds] = text  

    return output_dict


def generate_output_dict_by_sentence(file, output_file='../data/unlabeled_ground_truth/output_file.xlsx'):
    """generates dictionary with time of video recording as key and audio text said by the lecturer at this time as value.

    Args:
        file (str): file name/path to audioscript which was generated in such a format: [00:00:13.260 --> 00:00:14.200] class
                    The audioscript stores a word said in every row together with the exact time of the video recording (start and stop).
                    For generating these audioscripts please refer to e.g. to this repo: https://github.com/SYSTRAN/faster-whisper
        output_file (str, optional): output file path/name. Defaults to '../data/unlabeled_ground_truth/output_file.xlsx'.

    Returns:
        dict: dictionary with time of video recording as key and audio text as value by SENTENCE
    """
    with open(file, encoding='utf-8') as f:
        input_text = f.read()

    # Regular expression pattern to match timestamps
    pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\s-->\s(\d{2}:\d{2}:\d{2}\.\d{3})\] (.+?)\n'

    # Find all matches in the input text
    matches = re.findall(pattern, input_text)

    # Initialize dictionary to store converted timestamps and corresponding text
    output_dict = {}

    # Convert timestamps to seconds and populate the dictionary
    text_string = ""
    for match in matches:
        start_time = match[0]
        end_time = match[1]
        text = match[2]
        text_string += " " + text

        # conditional was necessary here because of slightly different formats of the audioscript files
        if ('cryptocurrency' in file) or  ('short_range' in file) or ('deeplearning' in file) or ('solar' in file) or  ('psychology' in file):  # ground truth files for these were only separated by '.' not by '?' or '!'
            condition = ('.' in text)
        else:
            condition = ('.' in text) or ('?' in text) or ('!' in text)
        if condition: 
        
            start_seconds = int(start_time[:2]) * 3600 + int(start_time[3:5]) * 60 + float(start_time[6:])
            end_seconds = int(start_time[:2]) * 3600  + int(end_time[3:5]) * 60 + float(end_time[6:])

            middle = (end_seconds + start_seconds) / 2
            
            output_dict[middle] = text_string

            text_string = ""

    start_seconds = int(start_time[:2]) * 3600 + int(start_time[3:5]) * 60 + float(start_time[6:])
    end_seconds = int(start_time[:2]) * 3600  + int(end_time[3:5]) * 60 + float(end_time[6:])

    middle = (end_seconds + start_seconds) / 2
    
    output_dict[middle] = text_string

    # create pandas dataframe
    df = pd.DataFrame(list(output_dict.items()), columns=['Key', 'Value'])
    # write to excel format
    df.to_excel(output_file, index=False, engine='openpyxl')

    return output_dict




