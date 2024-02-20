from faster_whisper import WhisperModel

def converter(path_to_file_in_any_format) :
    #TODO

    #just returning the same file for the moment 
    path_to_file_in_wav_format = path_to_file_in_any_format

    return path_to_file_in_wav_format

def filter(path_to_noisy_speech_file) :
    #TODO

    #just returning the same file for the moment
    path_to_denoised_speech_file = path_to_noisy_speech_file

    return path_to_denoised_speech_file

def STT(path_to_noisy_speech_file) :

    model_size = "large-v3"
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(path_to_noisy_speech_file, beam_size=5)

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


def main() :

    nSTT("/home/thomas/projects/tide_hackathon/data/deep_audi_denoiser/Denoised.mp3")


if __name__ == "__main__":
    main()