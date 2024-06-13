import sys
sys.path.append('../')
from helpers.prepare_audioscript import generate_output_dict_by_sentence
from sentence_transformers import SentenceTransformer, util
import fitz
import numpy as np
import pandas as pd
import cv2
from helpers.prepare_audioscript import generate_output_dict_by_sentence
from PIL import Image
from datetime import datetime

from transformers import AutoImageProcessor, SwiftFormerModel
from helpers.utils import calculate_dp_with_jumps, compute_similarity_matrix, create_video_frames
from helpers.utils import extract_features_from_images
from helpers.utils import calculate_sift_normalized_similarity
from helpers.utils import gradient_descent_with_adam
import argparse
#local_settings is a python file that is ignored by github: please create this locally in order to set the path to tesseract exe like:
# path_to_tesseract = ..\..\Library\bin\tesseract.exe':
from local_settings import path_to_tesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = path_to_tesseract
from tqdm import tqdm

def main(video_path, file_name, file_path, audio_script, autoimage_name='MBZUAI/swiftformer-xs',
         sentence_model_name='sentence-transformers/distiluse-base-multilingual-cased', jump_penalty=0.1, merge_method='max', sift=False):

    # Load the sentence transformer model
    sentence_model = SentenceTransformer(sentence_model_name)

    image_processor = AutoImageProcessor.from_pretrained(autoimage_name)
    image_model = SwiftFormerModel.from_pretrained(autoimage_name)

    jump_penality_str = str(jump_penalty)
    jump_penality_string = jump_penality_str.replace('.', 'comma')

    with open('time.txt', 'a') as file:
        file.write("Time for alignment algorithms.\n")

    time_start = datetime.now()

    # Step 1: Extract frames from video
    output_dict = generate_output_dict_by_sentence(audio_script)

    # we take a frame according to the sentences of the audioscript. Possible is also to choose a higher resolution
    interval_list = list(output_dict.keys())

    frames = create_video_frames(video_path, interval_list)

    print('len frames', len(frames))
    # Step 2: Convert PDF to images
    pdf_file = fitz.open(file_path)
    # Iterate over PDF pages
    text_pdf = []
    pdf_images = []
    pdf_images_cv2 = []
    pil_images = []


    for page_index in tqdm(range(len(pdf_file)), desc='PDF pages are extracted'):
        # Get the page itself
        page = pdf_file[page_index]

        # Extract text from pdf:
        text = page.get_text()
        text_pdf.append(text)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        pdf_images.append(img)
        if pix.n == 4:
            img = img[:, :, :3]  # Drop the alpha channel
        if pix.n == 3:  # assuming the image is RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pdf_images_cv2.append(img)
        # Convert to PIL image
        pil_image = Image.fromarray(img)
        pil_image.save('../pdf_images/{}_image.png'.format(page_index))
        pil_images.append(pil_image)

    # Vectorize sentences
    sentences1 = list(output_dict.values())

    pdf_file.close()

    # Resize frames to match PDF image dimensions
    frame_height, frame_width, _ = pdf_images[0].shape
    resized_frames = [cv2.resize(frame, (frame_width, frame_height)) for frame in tqdm(frames, desc='video frames are resized')]

    # get text features:
    audio_features = sentence_model.encode(sentences1, convert_to_tensor=True)

    image_texts = [pytesseract.image_to_string(image, lang='eng+ell+equ+deu') for image in tqdm(pil_images, desc='text is extracted from slide images')]
    text_features = sentence_model.encode(image_texts, convert_to_tensor=True)

    similarity_matrix_audio = compute_similarity_matrix(audio_features, text_features)

    ### optimal path regarding audio features is calculated:
    optimal_path_audio, _ = calculate_dp_with_jumps(similarity_matrix_audio, jump_penalty)

    time_audio = datetime.now()

    print('Time for audio algorithm', time_audio - time_start)
    with open('time.txt', 'a') as file:
        file.write("Time for audio algorithm: {}\n".format(str(time_audio - time_start)))

    # images are processed by selected image processor
    pdf_images_processed = [image_processor(image, return_tensors="pt") for image in tqdm(pdf_images_cv2, desc='pdf images are processed')]
    resized_frames_processed = [image_processor(image, return_tensors="pt") for image in tqdm(resized_frames, desc='video frames are processed')]

    # calculate image features
    features_pdf = np.array(extract_features_from_images(pdf_images_processed, image_model))
    features_frames = np.array(extract_features_from_images(resized_frames_processed, image_model))

    similarity_matrix_image = compute_similarity_matrix(features_frames, features_pdf)

    ### optimal path regarding image features is calulcated:
    optimal_path_image, _ = calculate_dp_with_jumps(similarity_matrix_image, jump_penalty)

    time_image = datetime.now()

    print('Time for image algorithm: ', time_image - time_start)

    with open('time.txt', 'a') as file:
        file.write("Time for image algorithm: {}\n".format(str(time_image - time_start)))

    frame_texts = [pytesseract.image_to_string(frame, lang='eng+ell+equ+deu') for frame in tqdm(frames, desc='text is extracted from video frames')]
    frame_features = sentence_model.encode(frame_texts, convert_to_tensor=True)

    similarity_matrix_ocr = compute_similarity_matrix(frame_features, text_features)

    ### optimal path regarding ocr features is calculated:
    optimal_path_ocr, _ = calculate_dp_with_jumps(similarity_matrix_ocr, jump_penalty)

    time_ocr = datetime.now()

    print('Time for ocr algorithm: ', time_ocr - time_start - (time_image-time_audio))

    with open('time.txt', 'a') as file:
        file.write("Time for ocr algorithm: {}\n".format(str(time_ocr - time_start - (time_image-time_audio))))

    print('similarity_matrix_audio.shape', similarity_matrix_audio.shape)
    print('similarity_matrix_ocr.shape', similarity_matrix_ocr.shape)
    print('similarity_matrix_image.shape', similarity_matrix_image.shape)


    result_dict_ocr = {}
    for chunk_index in optimal_path_ocr:
        result_dict_ocr[interval_list[chunk_index[0]]] =  chunk_index[1] + 1

    df_ocr = pd.DataFrame(list(result_dict_ocr.items()), columns=['Key', 'Value'])
    # writing results regarding ocr to excel sheet
    df_ocr.to_excel('{}_ocr_{}.xlsx'.format(file_name, jump_penality_string), index=False, engine='openpyxl')


    result_dict_audio = {}
    for chunk_index in optimal_path_audio:
        result_dict_audio[interval_list[chunk_index[0]]] =  chunk_index[1] + 1

    df_audio = pd.DataFrame(list(result_dict_audio.items()), columns=['Key', 'Value'])
    # writing results regarding audio to excel sheet
    df_audio.to_excel('{}_audiomatching_{}.xlsx'.format(file_name, jump_penality_string), index=False, engine='openpyxl')

    result_dict_image= {}
    for chunk_index in optimal_path_image:
        result_dict_image[interval_list[chunk_index[0]]] =  chunk_index[1] + 1

    df_image = pd.DataFrame(list(result_dict_image.items()), columns=['Key', 'Value'])
    # writing results regarding image features to excel sheet
    df_image.to_excel('{}_image_matching_{}.xlsx'.format(file_name, jump_penality_string), index=False, engine='openpyxl')


    if merge_method == 'mean':
        similarity_matrix_merged = np.mean((similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image), axis=0)
        optimal_path, _ = calculate_dp_with_jumps(similarity_matrix_merged, jump_penalty)

        result_dict = {}
        for chunk_index in optimal_path:
            result_dict[interval_list[chunk_index[0]]] =  chunk_index[1] + 1

        df = pd.DataFrame(list(result_dict.items()), columns=['Key', 'Value'])
        # writing results regarding mean merge to excel sheet
        df.to_excel('{}_mean_matching_all_{}.xlsx'.format(file_name, jump_penality_string), index=False, engine='openpyxl')

    elif merge_method == 'max':
        similarity_matrix_merged = np.max((similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image), axis=0)

        optimal_path, _ = calculate_dp_with_jumps(similarity_matrix_merged, jump_penalty)

        result_dict = {}
        for chunk_index in optimal_path:
            result_dict[interval_list[chunk_index[0]]] =  chunk_index[1] + 1

        df = pd.DataFrame(list(result_dict.items()), columns=['Key', 'Value'])
        # writing results regarding max merge to excel sheet
        df.to_excel('{}_max_matching_all_{}.xlsx'.format(file_name, jump_penality_string), index=False, engine='openpyxl')

    elif merge_method=='all':
        similarity_matrix_merged = np.mean((similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image), axis=0)
        optimal_path, _ = calculate_dp_with_jumps(similarity_matrix_merged, jump_penalty)

        result_dict = {}
        for chunk_index in optimal_path:
            result_dict[interval_list[chunk_index[0]]] =  chunk_index[1] + 1
        time_mean = datetime.now()
        print('Time for matching all with mean: ', time_mean - time_start)
        with open('time.txt', 'a') as file:
            file.write("Time for matching all with mean algorithm: {}\n".format(str(time_mean - time_start)))

        df = pd.DataFrame(list(result_dict.items()), columns=['Key', 'Value'])
        # writing results regarding mean merge to excel sheet
        df.to_excel('{}_mean_matching_all_{}.xlsx'.format(file_name, jump_penality_string), index=False, engine='openpyxl')

        similarity_matrix_merged = np.max((similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image), axis=0)
        optimal_path, _ = calculate_dp_with_jumps(similarity_matrix_merged, jump_penalty)

        result_dict = {}
        for chunk_index in optimal_path:
            result_dict[interval_list[chunk_index[0]]] =  chunk_index[1] + 1

        time_max = datetime.now()

        print('Time for matching all with max: ', time_max - time_start - (time_mean-time_ocr))

        with open('time.txt', 'a') as file:
            file.write("Time for matching all with max: {}\n".format(str(time_max - time_start - (time_mean-time_ocr))))

        df = pd.DataFrame(list(result_dict.items()), columns=['Key', 'Value'])
        # writing results regarding max merge to excel sheet
        df.to_excel('{}_max_matching_all_{}.xlsx'.format(file_name, jump_penality_string), index=False, engine='openpyxl')

        ## weighted sum through gradient descent:
        matrices = [similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image]

        optimal_weights = gradient_descent_with_adam(matrices, jump_penalty)
        print("Optimal Weights:", optimal_weights)

        similarity_matrix_merged = optimal_weights[0] * similarity_matrix_ocr + optimal_weights[1] * similarity_matrix_audio + optimal_weights[2] * similarity_matrix_image
        optimal_path, _ = calculate_dp_with_jumps(similarity_matrix_merged, jump_penalty)

        result_dict = {}
        for chunk_index in optimal_path:
            result_dict[interval_list[chunk_index[0]]] =  chunk_index[1] + 1

        time_weights = datetime.now()
        print('Time for matching all with weight: ', time_weights - time_start - (time_weights-time_mean))
        with open('time.txt', 'a') as file:
            file.write("Time for matching all with weight: {}\n".format(str(time_weights - time_start - (time_max-time_ocr))))

        df = pd.DataFrame(list(result_dict.items()), columns=['Key', 'Value'])
        # writing results regarding weighted sum merge to excel sheet
        df.to_excel('{}_weighted_sum_matching_all_{}.xlsx'.format(file_name, jump_penality_string), index=False, engine='openpyxl')

    elif merge_method == 'weighted_sum':
        ## weighted sum through gradient descent:
        matrices = [similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image]

        optimal_weights = gradient_descent_with_adam(matrices, jump_penalty, num_iterations=50)
        print("Optimal Weights:", optimal_weights)

        similarity_matrix_merged = optimal_weights[0] * similarity_matrix_ocr + optimal_weights[1] * similarity_matrix_audio + optimal_weights[2] * similarity_matrix_image
        optimal_path, _ = calculate_dp_with_jumps(similarity_matrix_merged, jump_penalty)

        result_dict = {}
        for chunk_index in optimal_path:
            result_dict[interval_list[chunk_index[0]]] =  chunk_index[1] + 1

        df = pd.DataFrame(list(result_dict.items()), columns=['Key', 'Value'])
        # writing results regarding weighted sum merge to excel sheet
        df.to_excel('{}_weighted_sum_matching_all_{}_50iterations_with_adam.xlsx'.format(file_name, jump_penality_string), index=False, engine='openpyxl')
    
    if sift:
        time_sift1 = datetime.now()
        ### calculate SIFT:
        similarity_matrix_sift = calculate_sift_normalized_similarity(resized_frames, pdf_images_cv2)

        optimal_path, _ = calculate_dp_with_jumps(similarity_matrix_sift, jump_penalty)


        result_dict = {}
        for chunk_index in optimal_path:
            result_dict[interval_list[chunk_index[0]]] =  chunk_index[1] + 1

        time_sift2 = datetime.now()

        print('calc time SIFT: ',  time_sift2 - time_sift1)

        with open('time.txt', 'a') as file:
            file.write("Time for SIFT: {}\n".format(str(time_sift2 - time_sift1)))


        df = pd.DataFrame(list(result_dict.items()), columns=['Key', 'Value'])
        # writing results regarding SIFT to excel sheet
        df.to_excel('{}_SIFT.xlsx'.format(file_name), index=False, engine='openpyxl')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="This script matches videoframes to lecture slides")


    parser.add_argument('--sentence_model', nargs='?', default='sentence-transformers/distiluse-base-multilingual-cased', type=str, required=False,
                        help='sentence transformer model')
    parser.add_argument('--jump_penalty', nargs='?', default=0.1, type=str, required=False,
                        help='penality for large jumps')
    parser.add_argument('--autoimage_name', nargs='?', default="MBZUAI/swiftformer-xs", type=str, required=False,
                        help='model for visual features')
    parser.add_argument('--merge_method', nargs='?', default='max', type=str, required=False,
                        help='merge method for different features; either mean or max')
    parser.add_argument('--sift', nargs='?', default=False, type=bool, required=False,
                        help='running SIFT argument or not. Default is False')
    parser.add_argument('--audio_script', default='../data/audioscripts/numerics_hennig.srt', type=str, required=False,
                        help='path to audioscript')
    parser.add_argument('--file_path', default='../data/lectures/numerics.pdf', type=str, required=False,
                        help='path to file')
    parser.add_argument('--video_path', default='../data/video/numerics_high_res.mp4', type=str, required=False,
                        help='path to lecture video')
    parser.add_argument('--file_name', nargs='?', default='../results/default', type=str, required=False,
                        help='name of result file')

    args = parser.parse_args()

    main(video_path = args.video_path, file_name=args.file_name,
         file_path=args.file_path, audio_script=args.audio_script, 
         sift=args.sift, merge_method=args.merge_method, autoimage_name=args.autoimage_name, 
         jump_penalty=args.jump_penaty, sentence_model=args.sentence_model)

