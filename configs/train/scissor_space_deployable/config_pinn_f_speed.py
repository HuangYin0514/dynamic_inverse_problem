import os
import torch
import numpy as np

####################################
# config.py

####################################
# For general settings
taskname = "task_scissor_space_deployable_pinn_f"
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.double  # torch.float32
outputs_dir = "./outputs/"

####################################
# Dynamic
obj = 24
dim = 3
dof = obj * dim

unitnum = 5
l = (1.6, 1.59)

M_A = 0.03 * 0.03
M_rho = 3000

F_a = 2**0.5 / 2
F_f = 600  # 驱动力
# F_f = 599.9924  # 驱动力
F_r = 5  # 伸展臂载荷
# ---------------------
# dae
dynamic_class = "DynamicScissorSpaceDeployableDAE"

####################################
# data
data_name = "DynamicData_UniformSpeed"
dataset_path = os.path.join(outputs_dir, "data", dynamic_class)
physic_num = 1000
interval = 1

####################################
# net
net_name = "PINN_F"
load_net_path = ""

BackboneNet_input_dim = 1
BackboneNet_hidden_dim = 200
BackboneNet_output_dim = 1
BackboneNet_layers_num = 5

loss_physic_weight = 1

####################################
# For training settings
learning_rate = 1e-4
optimizer = "adam_LBFGS"
# scheduler = "no_scheduler"
scheduler = "StepLR"
iterations = 3000
optimize_next_iterations = 2900
print_every = 500
