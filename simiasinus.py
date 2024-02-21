import os
import tkinter as tk
from tkinter import filedialog
from faster_whisper import WhisperModel

def converter(path_to_file_in_any_format) :
    parts = path_to_file_in_any_format.rsplit('.', 1)
    
    if  parts[1] == 'wav':
        print("Format audio is already '.wav'")
        result_text.insert(tk.END, "Format audio is already '.wav'")
        return path_to_file_in_any_format
    else:
        print("Converting audio file...")
        result_text.insert(tk.END, "Converting audio file...")
        new_path = parts[0]+'.wav'
        os.system(f'ffmpeg -i {path_to_file_in_any_format} {new_path} -loglevel "error" ')
        return new_path

def filter(path_to_noisy_speech_file) :
    #TODO

    #just returning the same file for the moment
    path_to_denoised_speech_file = path_to_noisy_speech_file

    return path_to_denoised_speech_file

def STT(path_to_clean_speech_file) :

    model_size = "large-v3"
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(path_to_clean_speech_file, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    return segments

def diarizer(segments, path_to_denoised_speech_file) :

    #TODO
    
    #just returning original segments for the moment
    diarized_segments = segments

    return diarized_segments

def organizer(diarized_segments) :

    #TODO

    #just returning original segments for the moment
    organized_text = ""
    
    return organized_text

def nSTT(path_to_noisy_speech_file) :

    #convert the file in wav
    path_to_noisy_speech_file_in_wav_format = converter(path_to_noisy_speech_file)

    #filter out noise from the signal
    path_to_denoised_speech_file = filter(path_to_noisy_speech_file_in_wav_format)

    #provide clean speech to wishper (running on noisy for the moment)
    segments = STT(path_to_denoised_speech_file)

    #input the segments into the diarizer
    diarized_segments = diarizer(segments, path_to_denoised_speech_file)

    #organize the output
    output_text = organizer(diarized_segments)

    return output_text


# Function to handle file selection
def select_audio_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        audio_file_label.config(text=f"Selected Audio File: {file_path}")
        # Enable the difficulty dropdown and start button after selecting the file
        difficulty_dropdown.config(state=tk.NORMAL)
        start_button.config(state=tk.NORMAL)
        return file_path


def start_program():
    file_path = audio_file_label.cget("text").split(": ")[1]
    speed = difficulty_var.get()
    path_to_noisy_speech_file = converter(file_path)
    filter(path_to_noisy_speech_file)
    
    
    #result_text.insert(tk.END, result + '\n')

def main():        

    global audio_file_label, difficulty_dropdown, start_button, difficulty_var, result_text
    root = tk.Tk()
    root.title("Simiasinus")

    select_audio_button = tk.Button(root, text="Select Audio File", command=select_audio_file)
    select_audio_button.pack(pady=10)

    audio_file_label = tk.Label(root, text="")
    audio_file_label.pack()

    speed_label = tk.Label(root, text="Select the speed")
    speed_label.pack(pady=5)

    difficulty_var = tk.StringVar(root)
    difficulty_var.set("Fast")
    difficulty_options = ["Fast", "Medium", "Slow"]
    difficulty_dropdown = tk.OptionMenu(root, difficulty_var, *difficulty_options)
    difficulty_dropdown.pack(pady=5)
    difficulty_dropdown.config(state=tk.DISABLED)

    start_button = tk.Button(root, text="Start Program", command=start_program)
    start_button.pack(pady=5)
    start_button.config(state=tk.DISABLED)

    result_text = tk.Text(root, height=10, width=50)
    result_text.pack(pady=10)

    root.mainloop()



    #input_file = './audio_sample/Noisy_Input.mp3'
    #converter(input_file)
    #nSTT("/home/thomas/projects/tide_hackathon/data/deep_audi_denoiser/Denoised.mp3")


if __name__ == "__main__":
    main()