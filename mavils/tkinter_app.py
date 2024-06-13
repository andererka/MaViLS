import tkinter as tk
from tkinter import filedialog, messagebox
from PyPDF2 import PdfFileReader
from moviepy.editor import VideoFileClip
import os
import matching_algorithm
from faster_whisper import WhisperModel
import pandas as pd

model_size = "tiny"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def read_excel_files(file_paths):
    dataframes = [pd.read_excel(file_paths[0])]
    col_names = ['Audio']
    for i, file in enumerate(file_paths[1:]):
        df = pd.read_excel(file)
        dataframes.append(df['Value'].rename(col_names[i]))
    return dataframes

def seconds_to_time_format(s):
    # Convert seconds to hours, minutes, seconds, and milliseconds
    hours = s // 3600
    s %= 3600
    minutes = s // 60
    s %= 60
    seconds = s // 1
    milliseconds = round((s % 1) * 1000)

    # Return the formatted string
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"

def process_files(pdf_path, video_path):
    # Your Python code to process the PDF and video
    print(f"Processing PDF: {pdf_path}")
    print(f"Processing Video: {video_path}")
    status_label.config(text="Transcribing video, please wait...")
    status_label.update_idletasks()
    segments, _ = model.transcribe(video_path, beam_size=5, language="en", condition_on_previous_text=False, word_timestamps=True)
    with open('audioscript.txt', 'w') as f:
        for segment in segments:
            for word in segment.words:
                print("[{} --> {}] {}".format(seconds_to_time_format(word.start), seconds_to_time_format(word.end), word.word))
                f.write("[{} --> {}] {}\n".format(seconds_to_time_format(word.start), seconds_to_time_format(word.end), word.word))
    status_label.config(text="")
    try:
        # Update status to indicate processing
        status_label.config(text="Matching slides to audiotext, please wait...")
        status_label.update_idletasks()
        matching_algorithm.main(file_path=pdf_path, video_path=video_path, audio_script='audioscript.txt', file_name='app_results/result_file', )  # Call the main function from processing_script
        status_label.config(text="")
        messagebox.showinfo("Success", "Files successfully uploaded and processed")
        show_download_button('app_results/result_file_max_matching_all_0comma1.xlsx', button_text='Download result for max matching', row=4)
        show_download_button('app_results/result_file_audiomatching_0comma1.xlsx', button_text='Download result for audio matching', row=5)
        show_download_button('app_results/result_file_image_matching_0comma1.xlsx', button_text='Download result for image matching', row=6)
        show_download_button('app_results/result_file_ocr_0comma1.xlsx', button_text='Download result for OCR matching', row=7)
        dframes = read_excel_files([os.path.abspath('../data/unlabeled_ground_truth/output_file.xlsx'),os.path.abspath('app_results/result_file_max_matching_all_0comma1.xlsx')])
        dframes.to_excel('app_results/merged_file.xlsx', index=False) 
        show_download_button('app_results/merged_file.xlsx', button_text='Download result for OCR matching', row=8)   
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def download_file(output_path):
    os.startfile(os.path.abspath(output_path)) # Opens the file with the default application

def show_download_button(output_path, button_text="Download Output", row=4):
    download_button = tk.Button(main_frame, text=button_text, command=lambda: download_file(output_path))
    download_button.grid(row=row, column=0, columnspan=2, pady=10)

def upload_files():
    if pdf_path and video_path:
        process_files(pdf_path, video_path)
        messagebox.showinfo("Success", "Files successfully uploaded and processed")
    else:
        messagebox.showwarning("Warning", "Please select both a PDF and a video file")

def upload_file(file_type):
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")] if file_type == "pdf" else [("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        if file_type == "pdf":
            global pdf_path
            pdf_path = file_path
            pdf_button.config(text=os.path.basename(file_path))
        elif file_type == "video":
            global video_path
            video_path = file_path
            video_button.config(text=os.path.basename(file_path))
# Setting up the main window
root = tk.Tk()
root.title("MaViLS Audio-Slide Alignment")

# Accessibility: Adding labels and proper structure
main_frame = tk.Frame(root)
main_frame.pack(pady=20, padx=20)

# Instructions Label
instructions_label = tk.Label(main_frame, text="Please upload a PDF and a video file for processing:")
instructions_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

# PDF Upload Button
pdf_button_label = tk.Label(main_frame, text="PDF File:")
pdf_button_label.grid(row=1, column=0, sticky='e')
pdf_button = tk.Button(main_frame, text="Browse...", command=lambda: upload_file("pdf"))
pdf_button.grid(row=1, column=1, sticky='w')

# Video Upload Button
video_button_label = tk.Label(main_frame, text="Video File:")
video_button_label.grid(row=2, column=0, sticky='e')
video_button = tk.Button(main_frame, text="Browse...", command=lambda: upload_file("video"))
video_button.grid(row=2, column=1, sticky='w')

# Process Button
process_button = tk.Button(main_frame, text="Process Files", command=upload_files)
process_button.grid(row=3, column=0, columnspan=2, pady=10)

# Status Label
status_label = tk.Label(main_frame, text="", foreground="blue")
status_label.grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()