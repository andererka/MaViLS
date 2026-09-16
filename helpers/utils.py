import sys
sys.path.append('../')
import os
import numpy as np
import cv2
from scipy.spatial.distance import cosine
import torch
from tqdm import tqdm
import fitz
from sklearn.model_selection import ParameterGrid
import torch

def calculate_dp_with_jumps(similarity_matrix, jump_penalty, linearity_penalty = 0.0000001):
    """calculates the decision matrix D according to a similarity matrix and a jump penality

    Args:
        similarity_matrix (numpy matrix): matrix containing similarity scores for each pair of video frame and lecture slide
        jump_penalty (float): jump penality for punsihing large jumps back and forth within a lecture

    Returns:
        list, matrix: list of indices to follow for an optimized sequence of slide numbers, decision matrix D
    """
    rows, cols = similarity_matrix.shape
    print('rows and cols', rows, cols)
    # dp matrix is initalized with 0s
    dp = np.zeros((rows + 1, cols + 1))
    # path[i][j] stores the (1-based) slide index k that was chosen as predecessor of cell (i, j)
    path = np.zeros((rows + 1, cols + 1), dtype=np.int64)

    # jump penality for every pair of slide j (rows) and candidate slide k (columns), scaled by size of jump.
    # penality is higher if jump is backward compared to forward, and 0 if k == j
    slide_indices = np.arange(1, cols + 1)
    jump_sizes = np.abs(slide_indices[None, :] - slide_indices[:, None])
    jump_penalty_scaled = jump_penalty * jump_sizes.astype(float)
    jump_penalty_scaled[slide_indices[None, :] < slide_indices[:, None]] *= 2

    # go through every row; all slide pairs (j, k) of a row are evaluated at once.
    # Same arithmetic (and order of operations) as the former triple loop, so results are bit-identical.
    for i in tqdm(range(1, rows + 1), desc='go through similarity matrix to calculate dp matrix'):   # frame chunks
        expected_frame_index = 1 + (rows/(cols-1)) * i
        linearity_penalty_scaled = linearity_penalty * np.abs(slide_indices - expected_frame_index)
        current_values = (similarity_matrix[i - 1] - linearity_penalty_scaled)[None, :] - jump_penalty_scaled + dp[i - 1][1:][None, :]
        # argmax takes the first maximum, like the strict '>' comparison of the former loop
        max_indices = np.argmax(current_values, axis=1)
        dp[i][1:] = current_values[np.arange(cols), max_indices]
        path[i][1:] = max_indices + 1

    # trace back paths to find the one with the highest correlation:
    optimal_end = int(np.argmax(dp[rows][1:])) + 1

    optimal_path = []
    i, j = rows, optimal_end
    while i > 0:
        previous_slide = int(path[i][j])
        optimal_path.append((i - 1, previous_slide - 1))
        i, j = i - 1, previous_slide

    return list(reversed(optimal_path)), dp

def compute_similarity_matrix(embeddings1, embeddings2):
    """Computes matrix with cosine similarity values between every embedding entry

    Args:
        embeddings1 (_embedding_): _feature embeddings, e.g. sentence transformer embedding or image vectorizer_
        embeddings2 (_embedding_): _feature embeddings, e.g. sentence transformer embedding or image vectorizer_

    Returns:
        _numpy matrix_: _similarity matrix with cosine similarity values_
    """
    # Initialize an empty matrix with the appropriate size
    similarity_matrix = np.zeros((len(embeddings1), len(embeddings2)))

    # Iterate over each pair of embeddings and calculate cosine similarity
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            # Cosine similarity is 1 - cosine distance
            similarity_matrix[i, j] = 1 - cosine(emb1, emb2)

    return similarity_matrix

def create_video_frames(video_path, interval_list, save_frames=False):
    """Create video frames according to the time distances stored in interval_list

    Args:
        video_path (_str_): _path to video_
        interval_list (_type_): _defines intervals of frames. Can be individualized according to audioscript chunks_
        save_frames (bool, optional): whether to also write every frame as an uncompressed PNG to '../frame_images' for debugging. Defaults to False.
    """
    # Process videos and store frames
    frames = []  # Store frames
    timestamps = []  # Store timestamps of captured frames for debugging or verification
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    if save_frames:
        os.makedirs('../frame_images', exist_ok=True)

    for interval in interval_list:
        # Set video position to the current interval time
        cap.set(cv2.CAP_PROP_POS_MSEC, interval * 1000)
        
        ret, frame = cap.read()  # Read the frame at the current interval

        if not ret:
            print(f"Failed to capture frame at {interval} seconds.")
            if not frames:
                # there is no earlier frame to fall back to
                raise ValueError(f"Could not read the first frame (at {interval} seconds) of video: {video_path}")
            # in case capture fails, append frames with last frame in order to get equal length for arrays later on
            frames.append(frames[-1])
            continue

        frames.append(frame)  # Append the successfully captured frame to the frames list
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
        timestamps.append(current_time)  # Append the timestamp for debugging
        
        # For debugging: Save frames as images (optional)
        if save_frames:
            cv2.imwrite('../frame_images/{}.png'.format(str(len(frames))), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    cap.release()  # Release the video capture object
    
    return frames

def preprocess_cv2_image(cv2_img, target_size):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    # Resize image to match model's expected sizing
    img_resized = cv2.resize(img_rgb, target_size)
    return img_resized

def extract_features_from_images(cv2_images, model):
    """extracts features of images according to a (transformer) model

    Args:
        cv2_images (_type_): _description_
        model (_type_): should be model that processes images like swiftformer

    Returns:
        list: list of features
    """
    feature_list = []

    for img in cv2_images:
        # Predict and extract features
        with torch.no_grad():
            outputs = model(**img)
        last_hidden_states = outputs.last_hidden_state
        feature_list.append(last_hidden_states.view(-1).numpy().flatten())

    return feature_list

def calculate_sift_normalized_similarity(images1, images2):
    """calculates similarity measure according to SIFT keypoints

    Args:
        images1 (_type_): images of video frames
        images2 (_type_): images of slides

    Returns:
        numpy matrix: similarty matrix
    """
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    descriptors1 = []
    descriptors2 = []

    # Compute keypoints and descriptors for the first list of images
    for img in images1:
        kp, des = sift.detectAndCompute(img, None)
        descriptors1.append(des)

    # Compute keypoints and descriptors for the second list of images
    for img in images2:
        kp, des = sift.detectAndCompute(img, None)
        descriptors2.append(des)

    similarity_matrix = np.zeros((len(images1), len(images2)), dtype=float)

    for i, des1 in tqdm(enumerate(descriptors1), desc='going through each frame image'):
        for j, des2 in enumerate(descriptors2):
            if des1 is not None and des2 is not None and len(des1) > 0:
                matches = bf.match(des1, des2)
                similarity_matrix[i, j] = len(matches)
            else:
                print('no keypoints found')
                similarity_matrix[i, j] = 0 

    return similarity_matrix


def convert_pdf_to_images(pdf_path, output_folder='../pdf_images', zoom_x=2.0, zoom_y=2.0):
    """converts all slides of a pdf to images and returns the pixmap images and the paths to the stored images

    Args:
        pdf_path (str): path to PDF
        output_folder (str, optional): _description_. Defaults to '../pdf_images'.
        zoom_x (float, optional): increase resolution by factor. Defaults to 2.0.
        zoom_y (float, optional): increase resolution by factor. Defaults to 2.0.

    Returns:
        list of paths, list of pixmap images
    """
    doc = fitz.open(pdf_path)
    image_paths = []
    images = []
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        
        # Create a matrix for transformation. Default resolution is 72 dpi.
        # By multiplying with a factor (e.g., 2.0), we increase the resolution to 144 dpi.
        mat = fitz.Matrix(zoom_x, zoom_y)
        
        # Get the pixmap with the transformation matrix
        image = page.get_pixmap(matrix=mat)
        
        image_path = f"{output_folder}/page_{page_num + 1}.png"
        image.save(image_path)
        image_paths.append(image_path)
        images.append(image)
    return image_paths, images



def combine_similarity_matrices(weights, matrices):
    """combine matrices according to weights as weigted sum

    Args:
        weights (list): list of floats
        matrices (list): list of matrices

    Returns:
        numpy matrix: combined matrix; weighted sum of matrices
    """
    combined_matrix = np.zeros_like(matrices[0])
    for weight, matrix in zip(weights, matrices):
        combined_matrix += weight * matrix
    return combined_matrix

def objective_function(weights, matrices, jump_penalty):
    """defines objective function for dynamic programming decision matrix

    Args:
        weights (list of floats): weights for weighted sum of matrices
        matrices (list): list of matrices
        jump_penalty (float): penality for jumps of non-consecutive slides

    Returns:
        numpy float: scoe of objective function
    """
    combined_matrix = combine_similarity_matrices(weights, matrices)
    _, dp = calculate_dp_with_jumps(combined_matrix, jump_penalty)
    # Assuming the objective is to maximize the final value in the DP table
    final_score = np.max(dp[-1])
    return -final_score 



def gradient_descent_with_adam(matrices, jump_penalty, learning_rate=0.001, num_iterations=50, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """calculate (numerical) gradient descent in order to find optimized weighted sum

    Args:
        matrices (_list): list of matrices
        jump_penalty (float): penality for jumps of non-consecutive slides
        learning_rate (float, optional): Defaults to 0.001.
        num_iterations (int, optional): number of iterations of gradient descent. Defaults to 50.
        beta1 (float, optional): constant to update first moment estimate. Defaults to 0.9.
        beta2 (float, optional): constant to update first moment estimate. Defaults to 0.999.
        epsilon (_type_, optional): Defaults to 1e-8.

    Returns:
        list: list of optimized weights
    """
    initial_weights = [1/3, 1/3, 1/3]
    weights = np.array(initial_weights)
    m = np.zeros_like(weights)  # First moment vector
    v = np.zeros_like(weights)  # Second moment vector
    t = 0  # Timestep

    for _ in range(num_iterations):
        grad = np.zeros_like(weights)
        for i in range(len(weights)):
            weights_plus = np.copy(weights)
            weights_minus = np.copy(weights)
            weights_plus[i] += epsilon
            weights_minus[i] -= epsilon
            grad[i] = (objective_function(weights_plus, matrices, jump_penalty) -
                       objective_function(weights_minus, matrices, jump_penalty)) / (2 * epsilon)
        
        t += 1  # Increment timestep
        m = beta1 * m + (1 - beta1) * grad  # Update biased first moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)  # Update biased second raw moment estimate
        m_hat = m / (1 - beta1 ** t)  # Compute bias-corrected first moment estimate
        v_hat = v / (1 - beta2 ** t)  # Compute bias-corrected second raw moment estimate
        
        # Update weights
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Ensure weights are normalized and non-negative
        weights = np.maximum(weights, 0)
        weights /= np.sum(weights)

    return weights



    