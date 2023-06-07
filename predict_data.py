import os

import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import statistics

import pandas as pd 
import librosa 
import librosa.display

import tensorflow as tf
from tensorflow import keras

from pydub.silence import split_on_silence
from pydub import AudioSegment, effects 
import scipy
from scipy.io.wavfile import read, write

from IPython.display import Audio
import noisereduce as nr
from noisereduce.generate_noise import band_limited_noise

def clean_data(filename, output_folder):
  #read data
  rate, y = read(filename)
  # reduce noise
  audio = nr.reduce_noise(y = y, sr=rate, n_std_thresh_stationary=1.5,stationary=True)
  # make the audio in pydub audio segment format
  aud = AudioSegment(audio.tobytes(),frame_rate = rate,
                      sample_width = audio.dtype.itemsize,channels = 1)
  # use split on sience method to split the audio based on the silence, 
  # here we can pass the min_silence_len as silent length threshold in ms and intensity thershold
  audio_chunks = split_on_silence(
      aud,
      min_silence_len = 700,
      silence_thresh = -60,
      keep_silence = 400,)
  try:
    #audio chunks are combined here
    audio_processed = sum(audio_chunks)
    audio_processed = np.array(audio_processed.get_array_of_samples())
  except:
    print(filename)
  else:
    #save audio file to wav
    scipy.io.wavfile.write(output_folder, rate, audio_processed)

def predict(filename):
    #load metadata
    TEST = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)),"test_data_2.csv"))
    LABELS = sorted(TEST.primary_label.unique())

    # Global vars
    SPEC_SHAPE = (48, 128) # height x width
    FMIN = 500

    # load model 
    model = keras.models.load_model(os.path.join(os.path.abspath(os.path.dirname(__file__)),'model_per_dense_50_clean.h5'))

    # load path
    input_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "file", filename)

    # #### convert to wav (if not wav)
    # file_extension = filename.rsplit('.', 1)[1]
    
    # if file_extension != "wav":
    #     import pydub
    #     pydub.AudioSegment.ffmpeg = "d:\ffmpeg"
    #     dest_path = input_path.rsplit('.', 1)[0] + '.wav'
    #     print(dest_path)
    #     if file_extension == "ogg" :
    #         song = AudioSegment.from_ogg(input_path)
    #         song.export(dest_path, format="wav")
    #     elif file_extension == "mp3" :
    #         print(input_path)
    #         song = AudioSegment.from_mp3(input_path)
    #         song.export(dest_path, format="wav")
    #     else :
    #        print("######### ERROR ######")
    #        return
    #     input_path = dest_path

    ### cleaning data #####
    test_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'file', 'data_clean.wav')
    clean_data(input_path, test_path)

    ### extract data & predict data ####
    # Open it with librosa
    wave_data, wave_rate = librosa.load(test_path)
    D = librosa.stft(wave_data)
    D_harmonic, D_percussive = librosa.decompose.hpss(D)

    # Pre-compute a global reference power from the input spectrum
    rp = np.max(np.abs(D))

    #wave data
    y_harmonic = librosa.istft(D_harmonic, length=len(wave_data))
    y_percussive = librosa.istft(D_percussive, length=len(wave_data))

    # Store results so that we can analyze them later
    data = {'row_id': [], 'prediction': [], 'score': []}

    # Split signal into 5-second chunks
    # Just like we did before (well, this could actually be a seperate function)
    sig_splits = []
    sample_length = 5*wave_rate
    for i in range(0, len(wave_data), sample_length):
        split = y_percussive[i:i + sample_length]

        # End of signal?
        if len(split) < sample_length:
            break

        sig_splits.append(split)
        
    # Get the spectrograms and run inference on each of them
    # This should be the exact same process as we used to
    # generate training samples!
    # Extract mel spectrograms for each audio chunk
    s_cnt = 0
    seconds = 0
    for chunk in sig_splits:
        # Keep track of the end time of each chunk
        seconds += 5
        hop_length = int(sample_length/ (SPEC_SHAPE[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=chunk, 
                                                sr=wave_rate, 
                                                n_fft=1024, 
                                                hop_length=hop_length, 
                                                n_mels=SPEC_SHAPE[0], 
                                                fmin=FMIN)

        mel_spec = librosa.power_to_db(mel_spec, ref=rp) 
        
        # Normalize
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()
        
        # Add channel axis to 2D array
        mel_spec = np.expand_dims(mel_spec, -1)

        # Add new dimension for batch size
        mel_spec = np.expand_dims(mel_spec, 0)
        
        # Predict
        p = model.predict(mel_spec)[0]
        
        # Get highest scoring species
        idx = p.argmax()
        species = LABELS[idx]
        score = p[idx]
        
        # Prepare submission entry
        data['row_id'].append(test_path.split(os.sep)[-1].rsplit('_', 1)[0] + 
                            '_' + str(seconds))    
        
        data['prediction'].append(species)
        s_cnt += 1
            
        # Add the confidence score as well
        data['score'].append(score)
            
    print('SOUNSCAPE ANALYSIS DONE. FOUND {} BIRDS.'.format(s_cnt))
    # Make a new data frame
    results = pd.DataFrame(data, columns = ['row_id', 'prediction', 'score'])
    # print(results)
    prediction = statistics.mode(np.array(results[['prediction']].values).flatten())
    return prediction

# print(predict("dataset_example_file.wav"))