import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import os
import sys
import tqdm
import time
import pickle

from tools2B import *
from loss2B import AAMsoftmax
from model2B import ECAPA_TDNN

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, n_class_phoneme, alphaB, betaB, m, s, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        ## ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()

        ## Classifier
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()
        
        # Phoneme loss
        self.phoneme_loss = AAMsoftmax(n_class=n_class_phoneme, m=m, s=s).cuda()

        # Combine Loss weights
        self.alpha = alphaB
        self.beta = betaB

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        print("hello, this in train network ... ECAPAModel.py")
        self.train()
        self.scheduler.step(epoch - 1)
        total_samples = 0
        speaker_top1, phoneme_top1, total_loss = 0, 0, 0
        total_speaker_loss, total_phoneme_loss = 0, 0
        lr = self.optim.param_groups[0]['lr']
        print("Loader Length = ", len(loader))
        for num, (data, speaker_labels, phoneme_labels) in enumerate(loader, start=1):
            self.zero_grad()
            speaker_labels = torch.LongTensor(speaker_labels).cuda()
            phoneme_labels = torch.LongTensor(phoneme_labels).cuda()

            # Forward pass
            speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
            
            # Speaker loss
            speaker_loss, speaker_acc = self.speaker_loss.forward(speaker_embedding, speaker_labels)
            
            # Phoneme loss
            phoneme_loss, phoneme_acc = self.phoneme_loss.forward(speaker_embedding, phoneme_labels)

            # Combined loss
            combined_loss = self.alpha * speaker_loss + self.beta * phoneme_loss
            
            # Backward pass and optimization
            combined_loss.backward()
            self.optim.step()

            batch_size = len(speaker_labels)
            total_samples += batch_size
            speaker_top1 += speaker_acc * batch_size
            phoneme_top1 += phoneme_acc * batch_size
            total_loss += combined_loss.detach().cpu().numpy() * batch_size
            total_speaker_loss += speaker_loss.detach().cpu().numpy() * batch_size
            total_phoneme_loss += phoneme_loss.detach().cpu().numpy() * batch_size

            combined_acc = ((speaker_top1 / total_samples) + (phoneme_top1 / total_samples)) / 2

            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / len(loader))) + \
                             " Loss: %.5f, Speaker Loss: %.5f, Phoneme Loss: %.5f, Speaker ACC: %2.2f%%, Phoneme ACC: %2.2f%%, Combined ACC: %2.2f%% \r" % 
                             (total_loss / total_samples, total_speaker_loss / total_samples, total_phoneme_loss / total_samples, speaker_top1 / total_samples, phoneme_top1 / total_samples, combined_acc * 100))
            sys.stderr.flush()
        sys.stdout.write("\n")

        return total_loss / total_samples, lr, combined_acc

    def eval_network(self, eval_list, eval_path):
        print("hello, this in eval network ... ECAPAModel2.py")
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        with torch.no_grad():
            for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
                file_name = os.path.join(eval_path, file)
                file_name += ".wav"
                audio, _ = sf.read(file_name)
                # Full utterance
                data_1 = torch.FloatTensor(np.stack([audio], axis=0)).cuda()

                # Splitted utterance matrix
                max_audio = 300 * 160 + 240
                if audio.shape[0] <= max_audio:
                    shortage = max_audio - audio.shape[0]
                    audio = np.pad(audio, (0, shortage), 'wrap')
                feats = []
                startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)
                for asf in startframe:
                    feats.append(audio[int(asf):int(asf) + max_audio])
                feats = np.stack(feats, axis=0).astype(np.float)
                data_2 = torch.FloatTensor(feats).cuda()
                # Speaker embeddings
                embedding_1 = self.speaker_encoder.forward(data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.speaker_encoder.forward(data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
                embeddings[file] = [embedding_1, embedding_2]

        scores, labels = []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        # Compute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF, scores, labels

    def enroll_network(self, enroll_list, enroll_path, path_save_model):
        print("hello, this in enroll network ... ECAPAModel2.py")
        self.eval()
        enrollments = {}
        lines = open(enroll_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        for line in lines:
            parts = line.split()
            model_id = parts[0]
            #phrase_id = parts[1]
            phrase_id = int(parts[1])  # Convert phrase_id to integer
            enroll_files = parts[3:]  # Enrollment file IDs
            embeddings = []

            for file in enroll_files:
                file_name = os.path.join(enroll_path, file) + ".wav"
                try:
                    audio, _ = sf.read(file_name)
                except Exception as e:
                    print(f"Error reading audio file {file_name}: {e}")
                    continue
                #data = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
                try:
                    data = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
                except Exception as e:
                    print(f"Error converting audio to tensor for file {file_name}: {e}")
                    continue

                with torch.no_grad():
                    #embedding = self.speaker_encoder.forward(data, aug=False)
                    #embedding = F.normalize(embedding, p=2, dim=1)
                    try:
                        embedding = self.speaker_encoder.forward(data, aug=False)
                        embedding = F.normalize(embedding, p=2, dim=1)
                    except Exception as e:
                        print(f"Error generating embedding for file {file_name}: {e}")
                        continue
                embeddings.append(embedding)
            
            if len(embeddings) == 0:
                print(f"No valid embeddings found for model_id {model_id}, phrase_id {phrase_id}")
                continue

            avg_embedding = torch.mean(torch.stack(embeddings), dim=0)  # make avg of 3 enrollment file embeddings
            if model_id not in enrollments:
                enrollments[model_id] = {}
            enrollments[model_id][phrase_id] = avg_embedding

        print("After reading lines in enroll networks ...")
        os.makedirs(path_save_model, exist_ok=True)

        # Save enrollments using the provided path
        with open(os.path.join(path_save_model, "enrollments.pkl"), "wb") as f:
            pickle.dump(enrollments, f)
        print("After writing lines in pkl enroll networks ...")

    def test_network(self, test_list, test_path, path_save_model):
        print("hello, this in test network ... ECAPAModel2B.py")
        self.eval()
        enrollments_path = os.path.join(path_save_model, "enrollments.pkl")
        print(f"Loading enrollments from {enrollments_path}")
        with open(enrollments_path, "rb") as f:
            enrollments = pickle.load(f)

        scores, labels = []
        lines = open(test_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        
        for line in lines:
            parts = line.split()
            model_id = parts[0]
            test_file = parts[1]
            trial_type = parts[2]
            
            file_name = os.path.join(test_path, test_file)
            file_name += ".wav"
            audio, _ = sf.read(file_name)
            data = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
            
            with torch.no_grad():
                # Generate speaker embedding
                speaker_emb = self.speaker_encoder.forward(data, aug=False)
                speaker_emb = F.normalize(speaker_emb, p=2, dim=1)

                # Check if speaker_emb is None
                if speaker_emb is None:
                    print(f"Error: speaker_emb is None for {file_name}")
                    continue

                # Generate phrase logits
                phrase_logits = self.phoneme_loss(speaker_emb)

                # Check if phrase_logits is None
                if phrase_logits is None:
                    print(f"Error: phrase_logits is None for {file_name}")
                    continue

                _, predicted_phrase_label = torch.max(phrase_logits, 1)
                predicted_phrase_label = predicted_phrase_label.item()
                
                # Initialize predicted scores
                predicted_speaker_score = 0
                predicted_phrase_score = 0
                
                if model_id in enrollments:
                    enrollment_data = enrollments[model_id]
                    
                    if predicted_phrase_label in enrollment_data:
                        enrollment_embedding = enrollment_data[predicted_phrase_label]
                        predicted_speaker_score = torch.mean(torch.matmul(speaker_emb, enrollment_embedding.T)).detach().cpu().numpy()
                        # Simulating the phrase score based on the predicted logits
                        predicted_phrase_score = torch.mean((predicted_phrase_label == torch.argmax(phrase_logits, dim=1)).float()).item()
                    else:
                        predicted_speaker_score = 0  # Handle missing phrase labels appropriately
                        predicted_phrase_score = 0
                
                # Set actual scores based on trial type
                if trial_type == 'TC':
                    speaker_score = 1
                    phrase_score = 1
                    label = 1
                elif trial_type == 'TW':
                    speaker_score = 1
                    phrase_score = 0
                    label = 0
                elif trial_type == 'IC':
                    speaker_score = 0
                    phrase_score = 1
                    label = 0
                elif trial_type == 'IW':
                    speaker_score = 0
                    phrase_score = 0
                    label = 0
                else:
                    raise ValueError(f"Unknown trial type: {trial_type}")
            
            # Final score: if both predicted_speaker_score and predicted_phrase_score are correct, then score is 1, otherwise 0
            if (predicted_speaker_score == speaker_score) and (predicted_phrase_score == phrase_score):
                score = 1
            else:
                score = 0

            scores.append(score)
            labels.append(label)

        # Calculate EER and minDCF for speaker verification
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF, scores, labels

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
