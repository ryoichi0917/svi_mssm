import torch
import pytorch_lightning as pl
from SGVB_SSM_TS_Clustering_Model_ResFlow import SGVB_SSM_TS_Clustering_Model_With_NF
from torch import optim


class Base_SGVB_SSM_TS_Clustering_Model_With_NF(pl.LightningModule):
    def __init__(self, 
                 base_param_dict, 
                 lstm_param_dict, 
                 resflow_param_dict, 
                 model_param_dict,
                 joint_prob_param_dict,
                 optimizer,
                 annealing_params=None,
                 GPU=False):

        super().__init__()
        self.model = SGVB_SSM_TS_Clustering_Model_With_NF(
                 base_param_dict = base_param_dict,
                 lstm_param_dict = lstm_param_dict,
                 resflow_param_dict = resflow_param_dict,
                 model_param_dict = model_param_dict,
                 joint_prob_param_dict = joint_prob_param_dict,
                 GPU=GPU)
        self.mode = base_param_dict["mode"]
        self.optim_spec = optimizer
        self.annealing_params = annealing_params
        self.GPU = GPU


    def get_alpha(self, now_epoch):
        #現在のアニーリングの係数を返す関数
        if now_epoch < self.annealing_params["start"]:
            alpha=self.annealing_params["alpha"]

        elif (now_epoch >= self.annealing_params["start"]) and (now_epoch < self.annealing_params["end"]):
            coef = (self.annealing_params["end"]-now_epoch)/(self.annealing_params["end"]-self.annealing_params["start"])
            alpha=self.annealing_params["alpha"]*coef
        else:
            alpha=0

        self.log("alpha", alpha, on_epoch=True)

        return alpha


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        now_alpha = self.get_alpha(self.trainer.current_epoch)
        if self.mode == "sl":
            x = batch["y_obs"]
        elif self.mode == "sir":
            x = batch["I_obs"]
        loss = self.model.calc_loss(x, alpha=now_alpha)
        return loss

    def configure_optimizers(self):
        if self.optim_spec["optimizer"] == "SGD_sc":
            optimizer = optim.SGD(self.parameters(),lr=self.optim_spec["lr"])
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, 
                                        base_lr = self.optim_spec["lr"], 
                                        max_lr = self.optim_spec["max_lr"],
                                        step_size_up = self.optim_spec["step_size"],
                                        mode = 'triangular')
            return {"optimizer": optimizer,
                    "lr_scheduler": {
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1,
                        }
                   }
        elif self.optim_spec["optimizer"] == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.optim_spec["lr"])
            return {
                "optimizer": optimizer,
            }


class WD_FT_SGVB_SSM_TS_Clustering_Model_With_NF(pl.LightningModule):
    #pytorch lightningで学習するためのrapper関数
    def __init__(self,
                 pretraind_model,
                 optimizer, 
                 GPU,
                 fine_tune=False):
        """
        【SGVB_SSM_TS_Clustering_Model以外の引数】
        l2_regularization : l2正則化のlambda
        sample_batch : 学習結果の可視化に利用するサンプルデータ
                        評価のdataloaderからサンプルしたbatchを一つ渡す
                        全てのクラスターのデータが含まれている必要がある
                        LV_Toy_Dataset関数で、alldata=Trueとする必要がある
        """

        super().__init__()
        self.model = pretraind_model
        self.mode = pretraind_model.mode
        self.optim_spec = optimizer
        self.GPU = GPU
        self.fine_tune = fine_tune

        self.validation_step_outputs = []


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.mode == "sl":
            x = batch["y_obs"]
        elif self.mode == "sir":
            x = batch["I_obs"]
        loss = self.model.calc_loss(x)
        return loss


    def get_First_term_params(self):
        #P側の訓練対象のパラメータを抽出する関数
        First_term_params = []
        for name, p in self.model.named_parameters():
            if ("joint_probability_multi_cluster" in name):
                if (".C" in name) or (".mu" in name) or (".alpha_order" in name) or (".init_SIR_seed" in name):
                    First_term_params.append(p)
        return First_term_params
    
    def get_Q_params(self):
        #Flowに含まれる訓練対象のパラメータを抽出する関数
        q_params = []
        
        for name, p in self.model.named_parameters():
            #Flowのパラメータを訓練可能なパラメータとして指定する。
            if ("resflow_multi_cluster" in name):
                q_params.append(p)
        return q_params


    def configure_optimizers(self):
        if self.optim_spec["optimizer"] == "Adam":
            if self.fine_tune == False:
                optimizer = optim.Adam(self.parameters(), lr=self.optim_spec["lr"])
                return {
                    "optimizer": optimizer,
                }
            elif self.fine_tune == True:
                First_term_params = self.get_First_term_params()
                Q_params = self.get_Q_params()
                optimizer = optim.Adam(params=[
                     {"params":First_term_params, "lr":self.optim_spec["lr"]},
                     {"params":Q_params, "lr":self.optim_spec["lr"]}
                    ])
            return optimizer

