import Conv_model
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from tqdm import tqdm
import logging 
from utils import funcs
import sys
from itertools import repeat

network_name = "Schoups"

# Task params
precision_hard = np.pi/60
precision_easy = np.pi/18
scales = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
v1_weight_scales = scales * len(scales)
phase_weight_scale = 0.01
v4_weight_scales = [x for item in scales for x in repeat(item, len(scales))]
# v1_weight_scale = 0.5
# v4_weight_scale = 2
learning_rate = 0.01
iterations = 10000
inp_size = 33
v1_size = 11
v1_orientation_number = 32
v4_size = 11
v4_stride = 1
v4_orientation_number=16
phis_sfs = 4
training_size = 20
phis = True
sfs = False
random_sf = False

# v1_gammas = [i/10 for i in range(5, 16)] + [0.5] * 10
# v4_orientation_stds = [0.7] * 11 + [i/10 for i in range(1, 11)]
v1_gamma = 0.5
v4_orientation_std = 0.7

# v1_gamma = v1_gammas[int(sys.argv[1]) - 1]
# v4_orientation_std = v4_orientation_stds[int(sys.argv[1]) - 1]
       
v1_weight_scale = v1_weight_scales[int(sys.argv[1]) - 1]
v4_weight_scale = v4_weight_scales[int(sys.argv[1]) - 1]


folder_savepath = 'trained_models/conv_high_fixed_normalized_pooled/weight_scale_' + str(v1_weight_scale).replace('.', '') + '_001_' + str(v4_weight_scale).replace('.', '') #path to existing directory for saving trained models
savepath = folder_savepath + '/' + network_name + '_32_orientations_model_' 
# savepath = 'trained_models/automated_stds/' + str(v1_gamma).replace('.', '') + "_"+ str(v4_orientation_std).replace('.', '') + '/Schoups_32_orientations_model_' #path to existing directory for saving trained models


np.random.seed(0)
torch.manual_seed(0)


net = Conv_model.convnet(
    input_size = inp_size, v1_size = v1_size, v1_orientation_number = v1_orientation_number,
    v4_size = v4_size, v4_stride = v4_stride, v4_orientation_number= v4_orientation_number,
    phis_sfs = phis_sfs, training_size = training_size, phis = phis, sfs = sfs,
    alpha = learning_rate, v1_rescale = v1_weight_scale, phase_rescale = phase_weight_scale, v4_rescale = v4_weight_scale,
    v1_gamma = v1_gamma, v4_orientation_std = v4_orientation_std)
net.load_state_dict(torch.load(savepath[:-1] + ".pt"))
net.angle1 = torch.load(savepath + "angle1.pt")
net.angle2 = torch.load(savepath + "angle2.pt")
net.train_x_location = torch.load(savepath + "x_location.pt")
net.train_y_location = torch.load(savepath + "y_location.pt")
net.trained_phis = torch.load(savepath + "phases.pt")
net.trained_sfs = torch.load(savepath + "sfs.pt")
net.v1_angles = torch.load(savepath + "v1_angles.pt")
net.v1_tuning_curve()
net.v4_tuning_curve()

net.otc_curve()
net.v1_tuning_params()
net.v4_tuning_params() 

torch.save(net.v1_max_diff, savepath + "v1_max_diff.pt")
torch.save(net.v4_max_diff, savepath + "v4_max_diff.pt")
# torch.save(net.v1_max_diff_angle, savepath + "v1_max_diff_angle.pt")
# torch.save(net.v4_max_diff_angle, savepath + "v4_max_diff_angle.pt")
# torch.save(net.v1_mean_before_bandwidth, savepath + "v1_bandwidth.pt")
# torch.save(net.v4_mean_before_bandwidth, savepath + "v4_bandwidth.pt")


plt.figure(figsize = [35, 35])
difference = False
plt.subplot(3, 3, 1)
net.plot_v1_tuning_curve(orientation = 3, phi_sf = 0, orientations = True, differences = difference);
plt.axvline(net.angle1 * 180/np.pi, 0, 1, linestyle = 'dashed', color = 'black');
plt.axvline(net.angle2 * 180/np.pi, 0, 1, linestyle = 'dashed', color = 'black');
plt.title("V1 tuning curves after training")

plt.subplot(3, 3, 2)
net.plot_v4_tuning_curve(differences = difference)
plt.axvline(net.angle1 * 180/np.pi, 0, 1, linestyle = 'dashed', color = 'black');
plt.axvline(net.angle2 * 180/np.pi, 0, 1, linestyle = 'dashed', color = 'black');
plt.title("V4 tuning curves after training")
plt.savefig(savepath + "tuning_curves.png")

plt.figure(figsize = [10, 10])
net.plot_otc_curve()
plt.title(network_name + " model")
plt.savefig(savepath + "OTC.png")

# SAVE
torch.save(net.v1_binary_heatmap,savepath + "v1_binary_heatmap.pt")
torch.save(net.v4_binary_heatmap,savepath + "v4_binary_heatmap.pt")

