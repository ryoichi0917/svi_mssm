import torch
from torch.utils.data import Dataset
import numpy as np

class SL_Toy_Dataset(Dataset):
    def __init__(self, data_arr, alldata = False):
        self.y_obs = data_arr["y_obs"]#観測データ(モデルへの入力)
        self.alldata = alldata
        if self.alldata:
            self.cluster = data_arr["cluster"]
            self.x = data_arr["x"]#潜在変数
            self.y = data_arr["y"]#潜在変数
            self.x_obs = data_arr["x_obs"]

    def __len__(self):
        return len(self.y_obs)

    def __getitem__(self, idx):
        #LSTMの入力にするため、特徴量の次元を追加する。
        y_obs = torch.tensor(self.y_obs[idx], dtype=torch.float32).unsqueeze(1)
        
        if self.alldata:
            cluster = torch.tensor(self.cluster[idx], dtype=torch.int32)
            x = torch.tensor(self.x[idx], dtype=torch.float32)
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            x_obs = torch.tensor(self.x_obs[idx], dtype=torch.float32)
            return {"cluster":cluster, 
                    "x" : x, "y" : y, 
                    "x_obs" : x_obs, "y_obs" : y_obs}

        else:
            return {"y_obs" : y_obs}


class SL_Sample_Dataset(Dataset):
    #可視化用のsampleデータセットを作成する関数
    def __init__(self, data_arr):
        #各クラスタに属する最初のデータを抜き出す
        tgt_data_list = [0,300, 600]
        
        self.y_obs = data_arr["y_obs"][tgt_data_list]
        self.cluster = data_arr["cluster"][tgt_data_list]
        self.x = data_arr["x"][tgt_data_list]
        self.y = data_arr["y"][tgt_data_list]
        self.x_obs = data_arr["x_obs"][tgt_data_list]

    def __len__(self):
        return len(self.y_obs)

    def __getitem__(self, idx):
        #LSTMの入力にするため、特徴量の次元を追加する。
        y_obs = torch.tensor(self.y_obs[idx], dtype=torch.float32).unsqueeze(1)
        cluster = torch.tensor(self.cluster[idx], dtype=torch.int32)
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        x_obs = torch.tensor(self.x_obs[idx], dtype=torch.float32)
        return {"cluster":cluster, 
                "x" : x, "y" : y, 
                "x_obs" : x_obs, "y_obs" : y_obs}
