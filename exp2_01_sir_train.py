from pathlib import Path
import sys
relative_path = Path('model')
absolute_path = relative_path.resolve()
sys.path.append(str(absolute_path))


import torch
import pytorch_lightning as pl
from LT_SGVB_SSM_TS_Clustering_Model_With_NF import Base_SGVB_SSM_TS_Clustering_Model_With_NF, WD_FT_SGVB_SSM_TS_Clustering_Model_With_NF
from SIR_Toy_Dataset import SIR_Toy_Dataset
from torch import optim
from torch.utils.data import DataLoader
import numpy as np



base_param_dict_sir = {"input_dim":1,
                   "latent_dim":3,
                   "num_clusters":2,
                   "mode":"sir"}

lstm_param_dict = {"embed_hidden_dim":32,
                 "num_lstm_layers_base":2,
                 "num_lstm_layers_other":1,}

resflow_param_dict_sir = {"num_flow_module":4,
                      "kernel_size":2,
                      "dims" : [3, 32, 32, 32, 32, 3],
                      "bias" : True,
                      "coeff" : 0.9,
                      "n_iterations":None,
                      "tolerance":0.001,
                      "reduce_memory": False}


sir_model_param_dict={"c0":{"beta" : None,
                            "gamma" : None},
                      "c1":{"beta" : None,
                            "gamma" : None}}

sir_joint_prob_param_dict={"k_order" : None,
                           "lambda_order" : None,
                           "alpha_order" : None,
                           "init_SIR" : None}




base_optim = {"optimizer":"SGD_sc", 
               "lr":1e-3, 
               "max_lr":1e-1, 
               "steps_per_lr_cycle":20}

wd_ft_optim ={"optimizer":"Adam", 
              "lr":0.001}
batch_size = 30

num_base_epoch = 1500
num_wd_epoch = 200
num_ft_epoch = 30

annealing_params={"start":100, "end":200, "alpha":1200}

GPU = (lambda : True if torch.cuda.is_available() else False)()









data_arr = np.load("data/sir_data.npz")#npzデータのload()
dataset = SIR_Toy_Dataset(data_arr, alldata=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
base_optim["step_size"]=(base_optim["steps_per_lr_cycle"]//2)*np.ceil(len(dataset)/batch_size)#スケジューラの周期を設定


lt_model_base = Base_SGVB_SSM_TS_Clustering_Model_With_NF(base_param_dict=base_param_dict_sir, 
                                         lstm_param_dict=lstm_param_dict, 
                                         resflow_param_dict=resflow_param_dict_sir, 
                                         model_param_dict=sir_model_param_dict,
                                         joint_prob_param_dict=sir_joint_prob_param_dict,
                                         optimizer=base_optim,
                                         annealing_params=annealing_params,
                                         GPU=GPU)

trainer = pl.Trainer(max_epochs=num_base_epoch, gradient_clip_val=1)
trainer.fit(lt_model_base, dataloader)




lt_model_wd= WD_FT_SGVB_SSM_TS_Clustering_Model_With_NF(
                 pretraind_model=lt_model_base.model,
                 optimizer=wd_ft_optim, 
                 GPU = GPU,
                 fine_tune=False)


# In[10]:


trainer = pl.Trainer(max_epochs=num_wd_epoch, 
                    gradient_clip_val=1)
trainer.fit(lt_model_wd, dataloader)


# In[11]:


lt_model_ft= WD_FT_SGVB_SSM_TS_Clustering_Model_With_NF(
                 pretraind_model=lt_model_wd.model,
                 optimizer=wd_ft_optim, 
                 GPU = GPU,
                 fine_tune=True)


# In[12]:


trainer = pl.Trainer(max_epochs=num_ft_epoch, 
                    gradient_clip_val=1)
trainer.fit(lt_model_ft, dataloader)

