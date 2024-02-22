import librosa as lb
import soundfile as sf
#install samplerate resampy

target = 16000
#name = 'Noisy_Input'
name = 'diar.wav'
path = '/home/drago/Coding/TideHackaton/segan/audio_sample/' + name


y, sr = lb.load(path)
y_resample = lb.resample(y, orig_sr=sr, target_sr=target, res_type='zero_order_hold')

# Save the resampled audio
output_path = path.rsplit(".")[0] + '_resampled.wav'
sf.write(output_path, y_resample, target)



