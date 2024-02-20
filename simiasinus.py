from faster_whisper import WhisperModel

def converter(path_to_file_in_any_format) :
    path_to_file_in_wav_format = ""

    #TODO

    return path_to_file_in_wav_format

def filter(path_to_noisy_speech_file) :
    path_to_denoised_speech_file = ""

    #TODO

    return path_to_denoised_speech_file

def STT(path_to_noisy_speech_file) :

    model_size = "large-v3"
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(path_to_noisy_speech_file, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))



def nSTT(path_to_noisy_speech_file) :

    #convert the file in wav
    path_to_noisy_speech_file_in_wav_format = converter(path_to_noisy_speech_file)

    #filter out noise from the signal
    path_to_denoised_speech_file = filter(path_to_noisy_speech_file_in_wav_format)

    #provide clean speech to wishper (running on noisy for the moment)
    STT(path_to_noisy_speech_file)


def main() :

    nSTT("/home/thomas/projects/tide_hackathon/data/deep_audi_denoiser/Noisy Input.mp3")


if __name__ == "__main__":
    main()
