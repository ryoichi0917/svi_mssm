import torch
from torch.utils.data import Dataset
import numpy as np

class SIR_Toy_Dataset(Dataset):
    def __init__(self, data_arr, alldata = False):
        self.I_obs = data_arr["I_obs"]
        self.alldata = alldata
        if self.alldata:
            self.cluster = data_arr["cluster"]
            self.S = data_arr["S"]
            self.I = data_arr["I"]
            self.R = data_arr["R"]

    def __len__(self):
        return len(self.I_obs)

    def __getitem__(self, idx):
        #LSTMの入力にするため、特徴量の次元を追加する。
        I_obs = torch.tensor(self.I_obs[idx], dtype=torch.float32).unsqueeze(1)
        
        if self.alldata:
            cluster = torch.tensor(self.cluster[idx], dtype=torch.int32)
            S = torch.tensor(self.S[idx], dtype=torch.float32)
            I = torch.tensor(self.I[idx], dtype=torch.float32)
            R = torch.tensor(self.R[idx], dtype=torch.float32)
            return {"cluster":cluster, 
                    "S" : S, "I" : I, 
                    "R" : R, "I_obs" : I_obs}

        else:
            return {"I_obs" : I_obs}