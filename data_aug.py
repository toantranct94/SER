from collections import namedtuple
import random
import torch
import torchaudio
from torchaudio import transforms
from nb_SparseImageWarp import sparse_image_warp
from config import *
import numpy as np
import scipy.misc
from scipy.io import wavfile
import os
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import librosa
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class MelLoader(Dataset):
    """
        1) computes mel-spectrograms from audio files.
    """
    def __init__(self):
        self.AudioData = namedtuple('AudioData', ['sig', 'sr'])
        self.to_db_scale = False
        pass

    def normalize(self, data):
        if data.ndim == 3:
            data = np.squeeze(data, axis=0)
        if type(data) is not np.ndarray:
            data = data.detach().numpy()

        data = scipy.misc.imresize(data, (224, 224), interp='bicubic')
        return data

    def get_mel(self, filename, duration):
        # print(filename)

        # mfccs = transforms.MFCC(sample_rate=int(RATE),n_mfcc=13)
        if filename[-3:] == 'npz':
            features = np.load(filename)
            spectrogram = self.normalize(features['inputs'])
        else:
            audio = self.AudioData(*torchaudio.load(filename))
            data = audio.sig.detach().numpy()
            N = int(duration*audio.sr)
            new_sig = random_segment(audio.sig, N)
            
            # spectro = tfm_spectro(ad=audio, sig=new_sig, hop=256, n_mels=128, to_db_scale=self.to_db_scale, f_max=8000, f_min=-100.0)
            spectro = self.log_specgram(new_sig.detach().numpy().flatten(), int(RATE)).astype(np.float32)
            # spectro =  self.extract_hpss_melspec(filename)
            spectrogram = self.normalize(spectro)
        
        out = np.zeros((3, 224, 224), dtype = np.float32)
        out[0, :, :] = spectrogram
        out[1, :, :] = spectrogram 
        out[2, :, :] = spectrogram 

        # img = Image.fromarray(out.T, 'RGB')
        # img.save('out.png')

        return out

    def tensor_to_img(self, spectrogram):
        plt.figure(figsize=(14,1)) # arbitrary, looks good on my screen.
        plt.imshow(spectrogram.detach().numpy()[0])
        plt.show()
        # display(spectrogram.shape)

    def log_specgram(self, audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))

        freqs, _, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window='hann', # 'text' , 'hamming'
                                            nperseg=nperseg,
                                            noverlap = noverlap,
                                            detrend=False)
        return np.log(spec.T.astype(np.float32) + eps)

    def extract_hpss_melspec(self, filename):
        y, sr = librosa.load(filename, sr=16000)

        # Harmonic-percussive source separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        S_h = librosa.feature.melspectrogram(y_harmonic, sr=sr, n_mels=128)
        S_p = librosa.feature.melspectrogram(y_percussive, sr=sr, n_mels=128)

        log_S_h = librosa.power_to_db(S_h, ref=np.max)
        log_S_p = librosa.power_to_db(S_p, ref=np.max)
        
        # log_S_h = scipy.misc.imresize(log_S_h, (224, 224), interp='bicubic')
        # log_S_p = scipy.misc.imresize(log_S_p, (224, 224), interp='bicubic')

        return log_S_h

    def aug(self, duration = 2):

        data = pd.read_csv(TRAINING_GT)
        df_train = pd.DataFrame(data, columns= ['File', 'Label'])

        labels = df_train.values[:,1]
        files = df_train.values[:,0]
        files = [x for x in files]

        count_labels = Counter(labels)

        sfdata = list(zip(files, labels))

        def write_audio(filename,transform_type,data,label):
            filename = filename[:-4]
            output_dir = AGU_DATA_PATH + transform_type
            os.makedirs(output_dir, exist_ok=True)
            file_path = output_dir + filename
            data = np.squeeze(data, axis=0)
            data = data.detach().numpy()
            np.savez(file_path, inputs=data, label=label)
            return transform_type + filename + '.npz', str(label)

        # df_aug = pd.DataFrame(columns= ['File', 'Label'])
        data_aug = []

        for index, filenames in enumerate(sfdata):
            filename = filenames[0]
            label = filenames[1]
            print("Preparing: ", filename)
            audio = self.AudioData(*torchaudio.load(BASE_TRAIN + filename))
            N = int(duration*audio.sr)
            new_sig = random_segment(audio.sig, N)
            # spectro = tfm_spectro(ad=audio, sig=new_sig, ws=512, hop=256, n_mels=128, to_db_scale=self.to_db_scale, f_max=8000, f_min=-80.0)
            spectro = self.log_specgram(new_sig.detach().numpy().flatten(), int(RATE)).astype(np.float32)
            # spectro = self.extract_hpss_melspec(BASE_TRAIN + filename)
            spectro = spectro.reshape(1, spectro.shape[0], spectro.shape[1])
            spectro = torch.from_numpy(spectro)

            if index % 3 == 0:
                time_warp = self.time_warp(spectro, 2)
                info = write_audio(filename=filename,transform_type=TRANSFORM_TIMEWARP,data=time_warp,label=label)
                # df_aug.loc[index + 1] = [info]
                data_aug.append(info)

            if index % 3 == 1:
                freq_mask = self.freq_mask(spectro, num_masks=2, replace_with_zero=True)
                # self.tensor_to_img(freq_mask)
                info = write_audio(filename=filename,transform_type=TRANSFORM_FREQ_MASK,data=freq_mask,label=label)
                # df_aug.loc[index + 1] = [info]
                data_aug.append(info)
            
            if index % 3 == 2:
                time_mask = self.time_mask(spectro, num_masks=2, replace_with_zero=True)
                info = write_audio(filename=filename,transform_type=TRANSFORM_TIME_MASK,data=time_mask,label=label)
                # df_aug.loc[index + 1] = [info]
                data_aug.append(info)

            combine = self.combine(spectro)

            info = write_audio(filename=filename,transform_type=TRANSFORM_COMBINE,data=combine,label=label)

            data_aug.append(info)

        df = pd.DataFrame(data_aug)
        df.to_csv(AGU_DATA_PATH + 'aug.csv', index=None, header=False, sep = ',')

    def time_warp(self, spec, W=5):
        num_rows = spec.shape[1]
        spec_len = spec.shape[2]
        device = spec.device
        
        y = num_rows//2
        horizontal_line_at_ctr = spec[0][y]
        assert len(horizontal_line_at_ctr) == spec_len
        
        point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
        assert isinstance(point_to_warp, torch.Tensor)

        # Uniform distribution from (0,W) with chance to be up to W negative
        dist_to_warp = random.randrange(-W, W)
        src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device), 
                            torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
        warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
        return warped_spectro.squeeze(3)

    def freq_mask(self, spec, F=30, num_masks=1, replace_with_zero=False):
        cloned = spec.clone()
        num_mel_channels = cloned.shape[1]
        
        for i in range(0, num_masks):        
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f) 
            if (replace_with_zero): cloned[0][f_zero:mask_end] = 0
            else: cloned[0][f_zero:mask_end] = cloned.mean()
        
        return cloned

    def time_mask(self, spec, T=40, num_masks=1, replace_with_zero=False):
        cloned = spec.clone()
        len_spectro = cloned.shape[2]
        
        for i in range(0, num_masks):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if (replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
            else: cloned[0][:,t_zero:mask_end] = cloned.mean()
        return cloned

    def combine(self, spectro):
        combined = self.time_mask(self.freq_mask(self.time_warp(spectro), num_masks=1), num_masks=1)
        return combined

def tfm_spectro(ad=None, sig=None,  sr=16000, to_db_scale=False, n_fft=1024, 
                ws=None, hop=None, f_min=0.0, f_max=-80, pad=0, n_mels=128):
    # We must reshape signal for torchaudio to generate the spectrogram.
    mel = transforms.MelSpectrogram(sample_rate=ad.sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop, 
                                    f_min=f_min, f_max=f_max, pad=pad)(sig.reshape(1, -1))

    mel = mel.permute(0,2,1) # swap dimension, mostly to look sane to a human.
    if to_db_scale:
        mel = transforms.AmplitudeToDB(stype='magnitude', top_db=f_max)(mel)
    return mel

def random_segment(audio_signal, N):

    audio_signal = audio_signal.reshape(-1,1)
    length = audio_signal.shape[0]
    if N < length:
        start = random.randint(0, length - N)
        audio_signal = audio_signal[start:start + N]
    else: 
        audio_signal = torch.flatten(audio_signal)
        tmp = np.zeros((N,))
        start = random.randint(0, N - length)
        tmp[start: start + length] = audio_signal 
        audio_signal = tmp
        audio_signal = audio_signal.reshape(-1, 1)
        audio_signal = torch.from_numpy(audio_signal).float()
        # test_sound = np.pad(test_sound, (N - test_sound.shape[0])//2, mode = 'constant')
    return audio_signal

class MyDatasetSTFT(Dataset):
    def __init__(self, filenames, labels, transform=None, duration=2, test=False):
        assert len(filenames) == len(labels), "Number of files != number of labels"
        self.fns = filenames
        self.lbs = labels
        self.transform = transform
        self.duration = duration  # audio duration in second
        self.test = test
        self.root_test = BASE_PUBLIC_TEST
        self.extractor = MelLoader()

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        if self.test:
            fname = self.fns[idx]
            # fname = self.fns[idx].split("/")[-1]
            # fname = self.root_test + fname
        else:
            fname = self.fns[idx]
            if not os.path.isfile(fname):
                fname = self.fns[idx].split("/")[-1]
                fname = self.root_test + fname
            
        feats = self.extractor.get_mel(fname, self.duration)

        # print(feats)
        return feats, self.lbs[idx], self.fns[idx]

def build_dataloaders(args):
    fns = []
    lbs = []
    # train

    data = pd.read_csv(TRAINING_GT)
    df_train = pd.DataFrame(data, columns= ['File', 'Label'])

    labels = df_train.values[:,1]
    files = df_train.values[:,0]

    files = [BASE_TRAIN + x for x in files]

    # files = ['../SER2/data/spectr1/' + x[:-4] + '.npz' for x in files]

    df_aug = pd.read_csv(AUG_GT)
    labels_aug = df_aug.values[:,1]
    files_aug = df_aug.values[:,0]
    files_aug = [AGU_DATA_PATH + x for x in files_aug]
    
    labels = labels.tolist()
    labels_aug = labels_aug.tolist()

    sfdata_aug = list(zip(files_aug, labels_aug))
    sfdata = list(zip(files, labels))
    sfdata = shuffle(sfdata)
    sfdata_aug = shuffle(sfdata_aug)

    fns = [x[0] for x in sfdata]
    lbs = [x[1] for x in sfdata]

    fns_aug = [x[0] for x in sfdata_aug]
    lbs_aug = [x[1] for x in sfdata_aug]

    train_fns, val_fns, train_lbs, val_lbs = train_test_split(fns, lbs, test_size = 0.2, random_state = args.random_state)

    if USE_DATA_AUG:

        train_fns = train_fns + fns_aug
        train_lbs = train_lbs + lbs_aug

    num_classes = len(set(train_lbs))
    print('Total training files: {}'.format(len(train_fns)))
    print('Total validation files: {}'.format(len(val_fns)))
    print('Total classes: {}'.format(num_classes))

    dsets = dict()
    dsets['train'] =  MyDatasetSTFT(train_fns, train_lbs, duration = args.duration)
    dsets['val'] =  MyDatasetSTFT(val_fns, val_lbs, duration = args.duration)
    
    dset_loaders = dict() 
    dset_loaders['train'] = DataLoader(dsets['train'],
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = NUM_WORKERS)

    dset_loaders['val'] = DataLoader(dsets['val'],
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = NUM_WORKERS)

    return dset_loaders, (train_fns, val_fns, train_lbs, val_lbs)


def build_data_aug():
    ex = MelLoader()
    ex.aug()

if __name__ == "__main__":
    ex = MelLoader()
    ex.aug()

    # b = ex.get_mel(BASE_TRAIN + 'PAEP-000265.wav', 2)
    # ex.load_images('../SER2/data/spectr/PAEP-000001_h3.png')
    # extract_features(BASE_TRAIN + 'PAEP-000265.wav')
    # b = ex.get_mel(BASE_TRAIN + 'PAEP-000018.wav', 2)
    # b = ex.get_mel(AGU_DATA_PATH + 'CB/PAEP-000001.npz', 2)
    pass

