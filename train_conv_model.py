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


logger = funcs.initLogger('train_log','train_log.log')

# network_name = "Schoups model"

# Task params
precision_hard = np.pi/60
precision_easy = np.pi/18
# scales = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25]
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
v4_stride = 6
v4_orientation_number = 16
phis = 4
training_size = 20
random_sf = False

# v1_gammas = [i/10 for i in range(5, 16)] + [0.5] * 10
# v4_orientation_stds = [0.7] * 11 + [i/10 for i in range(1, 11)]
v1_gamma = 0.5
v4_orientation_std = 0.7

# v1_gamma = v1_gammas[int(sys.argv[1]) - 1]
# v4_orientation_std = v4_orientation_stds[int(sys.argv[1]) - 1]
v1_weight_scale = v1_weight_scales[int(sys.argv[1]) - 1]
v4_weight_scale = v4_weight_scales[int(sys.argv[1]) - 1]

folder_savepath = 'trained_models/conv_high_fixed_normalized/weight_scale_' + str(v1_weight_scale).replace('.', '') + '_001_' + str(v4_weight_scale).replace('.', '') #path to existing directory for saving trained models
savepath = folder_savepath + '/' 


if not funcs.path_exists(folder_savepath):
    raise ValueError('Save directory does not exist! Create directory or amend savepath')
logger.info('Model save path validated - saving to {}'.format(folder_savepath))



paramdict= {'precision_hard' : precision_hard, 'precision_easy':precision_easy, 'v1_weight_scalar': v1_weight_scale, 'phase_weight_scalar': phase_weight_scale, 'v4_weight_scalar': v4_weight_scale, 'learning_rate':learning_rate, 'train_iters':iterations,
           'input_size':inp_size,'v1_size':v1_size,'v1_orientation_number':v1_orientation_number,'v4_size':v4_size,'v4_stride':v4_stride,
            'v4_orientation_number':v4_orientation_number,'phis':phis,'training_size':training_size, 'random_sf':random_sf,
            'v1_gamma': v1_gamma, 'v4_orientation_std': v4_orientation_std}
np.save(folder_savepath + '/params',paramdict)

#TRAIN NETWORK 
np.random.seed(0)
torch.manual_seed(0)

done = False
while done == False:
    net = Conv_model.convnet(
        input_size = inp_size, v1_size = v1_size, v1_orientation_number = v1_orientation_number,
        v4_size = v4_size, v4_stride = v4_stride, v4_orientation_number= v4_orientation_number,
        phis = phis, training_size = training_size, alpha = learning_rate, v1_rescale = v1_weight_scale, 
        phase_rescale = phase_weight_scale, v4_rescale = v4_weight_scale, v1_gamma = v1_gamma, 
        v4_orientation_std = v4_orientation_std)

    net.transfer_inputting(-precision_hard, precision_hard, v1_size, v1_size, random_sf = random_sf)
    net.desired_outputting()
    optimizer = optim.SGD(net.parameters(), lr = net.alpha)
    logger.info('Network initialized')
    net.train(iterations, optimizer)
    
    # If all losses are below 1, then quit training loop
    if not False in [i < 1 for i in net.losses]:
        done = True
    else:
        learning_rate /= 10
    
    logger.info('Network training complete - {} iterations'.format(iterations))

# SAVE
torch.save(net.state_dict(),savepath[:-1] + ".pt")
torch.save(net.losses,savepath + "loss.pt")
torch.save(net.training_scores,savepath + "performance.pt")
torch.save(net.v1_weight_changes,savepath + "v1_weight_change.pt")
torch.save(net.v4_weight_changes,savepath + "v4_weight_change.pt")
torch.save(net.decision_weight_changes,savepath + "decision_weight_change.pt")
torch.save(net.angle1,savepath + "angle1.pt")
torch.save(net.angle2,savepath + "angle2.pt")
torch.save(net.train_x_location,savepath + "x_location.pt")
torch.save(net.train_y_location,savepath + "y_location.pt")
torch.save(net.generalize_perform,savepath + "generalization_performance.pt")
torch.save(net.generalize_error,savepath + "generalization_error.pt")
torch.save(net.trained_phis,savepath + "phases.pt")
torch.save(net.trained_sfs,savepath + "sfs.pt")
torch.save(net.v1_angles,savepath + "v1_angles.pt")
# torch.save(net.v1_otc_max_diffs,savepath + "v1_otc_max_diffs.pt")
# torch.save(net.v4_otc_max_diffs,savepath + "v4_otc_max_diffs.pt")


net.otc_curve(v1_position_1 = int((net.v1_dimensions - 1) / 2), v1_position_2 = int((net.v1_dimensions - 1) / 2), v4_position_1 = int((net.v4_dimensions - 1) / 2), v4_position_2 = int((net.v4_dimensions - 1) / 2))
net.v1_tuning_params()
net.v4_tuning_params()

        
torch.save(net.v1_max_diff, savepath + "v1_max_diff.pt")
torch.save(net.v4_max_diff, savepath + "v4_max_diff.pt")
torch.save(net.v1_binary_heatmap,savepath + "v1_binary_heatmap.pt")
torch.save(net.v4_binary_heatmap,savepath + "v4_binary_heatmap.pt")
torch.save(net.v1_amplitude_difference, savepath + "v1_amplitude.pt")
torch.save(net.v4_amplitude_difference, savepath + "v4_amplitude.pt")
# torch.save(net.v1_spatial_heatmap,savepath + "v1_spatial_heatmap.pt")
# torch.save(net.v4_spatial_heatmap,savepath + "v4_spatial_heatmap.pt")
# torch.save(net.v1_max_diff_angle, savepath + "v1_max_diff_angle.pt")
# torch.save(net.v4_max_diff_angle, savepath + "v4_max_diff_angle.pt")
# torch.save(net.v1_mean_before_bandwidth, savepath + "v1_bandwidth.pt")
# torch.save(net.v4_mean_before_bandwidth, savepath + "v4_bandwidth.pt")
logger.info('Model saved to {}'.format(savepath))

# Figures

plt.figure(figsize = [35, 35])
difference = False
plt.subplot(3, 3, 1)
net.plot_v1_tuning_curve(orientation = 3, sf = 0, phi = 0, orientations = True, differences = difference);
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
plt.savefig(savepath + "OTC.png")

# plt.figure(figsize = [35, 35])
# plt.subplot(3, 3, 1)
# net.plot_otc_curve()
# plt.title(network_name + " model OTC")
            
# plt.subplot(3, 3, 2)
# net.plot_otc_curve_diff(absolute_diff = False)
# plt.title(network_name + " model OTC diff")
# plt.savefig(savepath + "OTC.png")
            


                                                                                                                   