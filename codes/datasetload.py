import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.autograd import Variable

class inputDataset(Dataset):
    """
    The Class will act as the container for our dataset. It will take your dataframe, the root path, and also the transform function for transforming the dataset.
    """
    def __init__(self, data_frame, audio_root):
        self.data_frame = data_frame
        self.audio_root = audio_root

    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        audio_file = self.audio_root + self.data_frame.iloc[idx, 0]
        input_signal, sr = torchaudio.load(audio_file, normalize = True)
        channel_dim = input_signal.shape[0]
        if channel_dim > 1:
            input_signal = torch.mean(input_signal, 0, keepdim=True)
        if sr!=16000:
            resampler = torchaudio.transforms.Resample(orig_freq = sr, new_freq = 16000)
            input_signal = resampler(input_signal)
        input_signal_length = torch.tensor([input_signal.shape[-1]], dtype=torch.long)
         
        transcript = self.data_frame.iloc[idx,2] # changed when df is created from json nemo style manifest files
        transcript = str(transcript)
       
        return input_signal, input_signal_length, transcript
    

def collate_batch_input(batch):
    
    input_signal_temp_list, input_signal_length_list, references = [], [], []

    for (_input_signal, _input_signal_length, _transcript) in batch:
        input_signal_temp_list.append(_input_signal.squeeze())
        input_signal_length_list.append(_input_signal_length)
        references.append(_transcript)
        
    input_signal_list = nn.utils.rnn.pad_sequence(input_signal_temp_list, batch_first=True)    
    return input_signal_list, input_signal_length_list, references