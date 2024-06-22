#Ya malekal haghol mobin

#Author: Maryam Afshari -Iranian

#Date: 13.6.2024 June/ 24.3.1403 Khordad Panjshanbe

import glob
import numpy as np
import os
import random
import soundfile as sf
import torch
from scipy import signal

class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
        
        # Load data & labels
        self.data_list = []
        self.speaker_labels = []
        self.phoneme_labels = []  # New phoneme labels

        # Create a dictionary to map phoneme labels to integers
        phoneme_dict = {}
        phoneme_counter = 0
        ignored_lines = 0  # Counter for ignored lines

        lines = open(train_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        dictkeys = list(set([x.split()[1] for x in lines]))  # Changed to index 1 for speaker-id
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        
        for index, line in enumerate(lines):
            try:
                speaker_label = dictkeys[line.split()[1]]  # Changed to index 1 for speaker-id
                phoneme_label = line.split()[2]  # Treat phoneme ID as a string

                # Ignore lines with 'FT' in phoneme_label
                if phoneme_label == 'FT':
                    ignored_lines += 1
                    continue

                # Convert phoneme label to integer
                if phoneme_label not in phoneme_dict:
                    phoneme_dict[phoneme_label] = phoneme_counter
                    phoneme_counter += 1
                phoneme_label = phoneme_dict[phoneme_label]

                file_name = os.path.join(train_path, line.split()[0])  # Changed to index 0 for train-file-id
                file_name += ".wav"
                
                self.speaker_labels.append(speaker_label)
                self.phoneme_labels.append(phoneme_label)
                self.data_list.append(file_name)
            except ValueError:
                print(f"Skipping line with non-integer phoneme ID: {line}")

        print(f"Ignored {ignored_lines} lines with 'FT' as phoneme_label")

    def __getitem__(self, index):
        file_name = self.data_list[index]
        audio, sr = sf.read(file_name)
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_frame = np.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = np.stack([audio], axis=0)
        
        # Data Augmentation
        augtype = 0  # I have changed it at 18-2-1403 Ordibehesht - May
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        
        return torch.FloatTensor(audio[0]), self.speaker_labels[index], self.phoneme_labels[index]

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = sf.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = sf.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = np.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
