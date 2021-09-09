import LCN_model
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

logger = funcs.initLogger('train_log','train_log.log')

savepath = 'trained_models/automated/weight_scale_01_001_5/' #path to existing directory for saving trained models
if not funcs.path_exists(savepath):
    raise ValueError('Save directory does not exist! Create directory or amend savepath')
logger.info('Model save path validated - saving to {}'.format(savepath))
    
# Task params
precision_hard = np.pi/60
precision_easy = np.pi/18
v1_weight_scale = 0.1
phase_weight_scale = 0.01
v4_weight_scale = 5
learning_rate = 0.01
iterations = 10000
inp_size = 33
v1_size = 11
v1_orientation_number = 32
v4_size = 11
v4_stride = 6
v4_orientation_number=16
phis_sfs = 4
training_size = 20
phis = True
sfs = False
random_sf = False

paramdict= {'precision_hard' : precision_hard, 'precision_easy':precision_easy, 'v1_weight_scalar': v1_weight_scale, 'phase_weight_scalar': phase_weight_scale, 'v4_weight_scalar': v4_weight_scale, 'learning_rate':learning_rate, 'train_iters':iterations,
           'input_size':inp_size,'v1_size':v1_size,'v1_orientation_number':v1_orientation_number,'v4_size':v4_size,'v4_stride':v4_stride,
            'v4_orientation_number':v4_orientation_number,'phis_sfs':phis_sfs,'training_size':training_size,'phis':phis,'sfs':sfs, 'random_sf':random_sf}
np.save(savepath + '/params',paramdict)

#TRAIN NETWORK 1 [hard//schoups]
np.random.seed(0)
torch.manual_seed(0)

schoups_net = LCN_model.LCN(
    input_size = inp_size, v1_size = v1_size, v1_orientation_number = v1_orientation_number,
    v4_size = v4_size, v4_stride = v4_stride, v4_orientation_number= v4_orientation_number,
    phis_sfs = phis_sfs, training_size = training_size, phis = phis, sfs = sfs,
    alpha = learning_rate, v1_rescale = v1_weight_scale, phase_rescale = phase_weight_scale, v4_rescale = v4_weight_scale)

schoups_net.transfer_inputting(-precision_hard, precision_hard, v1_size, v1_size, random_sf = random_sf)
schoups_net.desired_outputting()
optimizer = optim.SGD(schoups_net.parameters(), lr = schoups_net.alpha)
logger.info('Schoups net initialized')
schoups_net.train(iterations, optimizer)
logger.info('Schoups net training complete - {} iterations'.format(iterations))

# SAVE
torch.save(schoups_net.state_dict(),savepath + "Schoups_32_orientations_model.pt")
torch.save(schoups_net.losses,savepath + "Schoups_32_orientations_model_loss.pt")
torch.save(schoups_net.training_scores,savepath + "Schoups_32_orientations_model_performance.pt")
torch.save(schoups_net.v1_weight_changes,savepath + "Schoups_32_orientations_model_v1_weight_change.pt")
torch.save(schoups_net.v4_weight_changes,savepath + "Schoups_32_orientations_model_v4_weight_change.pt")
torch.save(schoups_net.decision_weight_changes,savepath + "Schoups_32_orientations_model_decision_weight_change.pt")
torch.save(schoups_net.angle1,savepath + "Schoups_32_orientations_model_angle1.pt")
torch.save(schoups_net.angle2,savepath + "Schoups_32_orientations_model_angle2.pt")
torch.save(schoups_net.train_x_location,savepath + "Schoups_32_orientations_model_x_location.pt")
torch.save(schoups_net.train_y_location,savepath + "Schoups_32_orientations_model_y_location.pt")
torch.save(schoups_net.generalize_perform,savepath + "Schoups_32_orientations_model_generalization_performance.pt")
torch.save(schoups_net.generalize_error,savepath + "Schoups_32_orientations_model_generalization_error.pt")
torch.save(schoups_net.trained_phis,savepath + "Schoups_32_orientations_model_phases.pt")
torch.save(schoups_net.trained_sfs,savepath + "Schoups_32_orientations_model_sfs.pt")
torch.save(schoups_net.v1_angles,savepath + "Schoups_32_orientations_model_v1_angles.pt")
logger.info('Schoups model saved to {}'.format(savepath))

# #TRAIN NETWORK 2
# np.random.seed(0)
# torch.manual_seed(0)

# ghose_net = LCN_model.LCN(
#     input_size = 33, v1_size = 11, v1_orientation_number = 32, v4_size = 11, v4_stride = 6, 
#     v4_orientation_number = 16, phis_sfs = 4, training_size = 20, phis = True, sfs = False, alpha = learning_rate, rescale = weight_scale)
# ghose_net.transfer_inputting(-precision_easy, precision_easy, 11, 11, random_sf = False)
# ghose_net.desired_outputting()
# optimizer = optim.SGD(ghose_net.parameters(), lr = ghose_net.alpha)
# logger.info('Ghose net initialized')
# ghose_net.train(5000, optimizer)
# logger.info('Ghose net training complete')

# # SAVE
# torch.save(ghose_net.state_dict(),savepath +  "Ghose_32_orientations_model.pt")
# torch.save(ghose_net.losses,savepath +  "Ghose_32_orientations_model_loss.pt")
# torch.save(ghose_net.training_scores,savepath +  "Ghose_32_orientations_model_performance.pt")
# torch.save(ghose_net.v1_weight_changes,savepath +  "Ghose_32_orientations_model_v1_weight_change.pt")
# torch.save(ghose_net.v4_weight_changes,savepath +  "Ghose_32_orientations_model_v4_weight_change.pt")
# torch.save(ghose_net.decision_weight_changes,savepath +  "Ghose_32_orientations_model_decision_weight_change.pt")
# torch.save(ghose_net.angle1,savepath +  "Ghose_32_orientations_model_angle1.pt")
# torch.save(ghose_net.angle2,savepath +  "Ghose_32_orientations_model_angle2.pt")
# torch.save(ghose_net.train_x_location,savepath + "Ghose_32_orientations_model_x_location.pt")
# torch.save(ghose_net.train_y_location,savepath + "Ghose_32_orientations_model_y_location.pt")
# torch.save(ghose_net.generalize_perform,savepath + "Ghose_32_orientations_model_generalization_performance.pt")
# torch.save(ghose_net.generalize_error,savepath + "Ghose_32_orientations_model_generalization_error.pt")
# torch.save(ghose_net.trained_phis,savepath + "Ghose_32_orientations_model_phases.pt")
# torch.save(ghose_net.trained_sfs,savepath + "Ghose_32_orientations_model_sfs.pt")
# torch.save(ghose_net.v1_angles,savepath + "Ghose_32_orientations_model_v1_angles.pt")
# logger.info('Ghose model saved to {}'.format(savepath))

# # Task params
# precision_hard = np.pi/60
# precision_easy = np.pi/18
# v1_weight_scales = [0.1, 0.1, 0.5, 0.5, 0.5, 0.5]
# phase_weight_scale = 0.01
# v4_weight_scales = [5, 10, 1, 2, 5, 10]
# learning_rate = 0.01
# iterations = 10000
# inp_size = 33
# v1_size = 11
# v1_orientation_number = 32
# v4_size = 11
# v4_stride = 6
# v4_orientation_number=16
# phis_sfs = 4
# training_size = 20
# phis = True
# sfs = False
# random_sf = False

# # v1_grid = torch.zeros(len(v1_weight_scales), len(v4_weight_scales))
# # v4_grid = torch.zeros(len(v1_weight_scales), len(v4_weight_scales))

# for i in range(len(v1_weight_scales)):
#     v1_weight_scale = v1_weight_scales[i]
#     v4_weight_scale = v4_weight_scales[i]
        
#     savepath = 'trained_models/automated/weight_scale_' + str(v1_weight_scale).replace('.', '') + '_001_' + str(v4_weight_scale).replace('.', '') + '/' #path to existing directory for saving trained models

#     if not funcs.path_exists(savepath):
#         raise ValueError('Save directory does not exist! Create directory or amend savepath')
#     logger.info('Model save path validated - saving to {}'.format(savepath))



#     paramdict= {'precision_hard' : precision_hard, 'precision_easy':precision_easy, 'v1_weight_scalar': v1_weight_scale, 'phase_weight_scalar': phase_weight_scale, 'v4_weight_scalar': v4_weight_scale, 'learning_rate':learning_rate, 'train_iters':iterations,
#                'input_size':inp_size,'v1_size':v1_size,'v1_orientation_number':v1_orientation_number,'v4_size':v4_size,'v4_stride':v4_stride,
#                 'v4_orientation_number':v4_orientation_number,'phis_sfs':phis_sfs,'training_size':training_size,'phis':phis,'sfs':sfs, 'random_sf':random_sf}
#     np.save(savepath + '/params',paramdict)

#     #TRAIN NETWORK 1 [hard//schoups]
#     np.random.seed(0)
#     torch.manual_seed(0)

#     schoups_net = LCN_model.LCN(
#         input_size = inp_size, v1_size = v1_size, v1_orientation_number = v1_orientation_number,
#         v4_size = v4_size, v4_stride = v4_stride, v4_orientation_number= v4_orientation_number,
#         phis_sfs = phis_sfs, training_size = training_size, phis = phis, sfs = sfs,
#         alpha = learning_rate, v1_rescale = v1_weight_scale, phase_rescale = phase_weight_scale, v4_rescale = v4_weight_scale)

#     schoups_net.transfer_inputting(-precision_hard, precision_hard, v1_size, v1_size, random_sf = random_sf)
#     schoups_net.desired_outputting()
#     optimizer = optim.SGD(schoups_net.parameters(), lr = schoups_net.alpha)
#     logger.info('Schoups net initialized')
#     schoups_net.train(iterations, optimizer)
#     logger.info('Schoups net training complete - {} iterations'.format(iterations))

#     # SAVE
#     torch.save(schoups_net.state_dict(),savepath + "Schoups_32_orientations_model.pt")
#     torch.save(schoups_net.losses,savepath + "Schoups_32_orientations_model_loss.pt")
#     torch.save(schoups_net.training_scores,savepath + "Schoups_32_orientations_model_performance.pt")
#     torch.save(schoups_net.v1_weight_changes,savepath + "Schoups_32_orientations_model_v1_weight_change.pt")
#     torch.save(schoups_net.v4_weight_changes,savepath + "Schoups_32_orientations_model_v4_weight_change.pt")
#     torch.save(schoups_net.decision_weight_changes,savepath + "Schoups_32_orientations_model_decision_weight_change.pt")
#     torch.save(schoups_net.angle1,savepath + "Schoups_32_orientations_model_angle1.pt")
#     torch.save(schoups_net.angle2,savepath + "Schoups_32_orientations_model_angle2.pt")
#     torch.save(schoups_net.train_x_location,savepath + "Schoups_32_orientations_model_x_location.pt")
#     torch.save(schoups_net.train_y_location,savepath + "Schoups_32_orientations_model_y_location.pt")
#     torch.save(schoups_net.generalize_perform,savepath + "Schoups_32_orientations_model_generalization_performance.pt")
#     torch.save(schoups_net.generalize_error,savepath + "Schoups_32_orientations_model_generalization_error.pt")
#     torch.save(schoups_net.trained_phis,savepath + "Schoups_32_orientations_model_phases.pt")
#     torch.save(schoups_net.trained_sfs,savepath + "Schoups_32_orientations_model_sfs.pt")
#     torch.save(schoups_net.v1_angles,savepath + "Schoups_32_orientations_model_v1_angles.pt")
#     logger.info('Schoups model saved to {}'.format(savepath))

#     schoups_net.otc_curve(v1_position = 11, v4_position = 1)

#     v1_grid[i][j] = schoups_net.v1_max_diff
#     v4_grid[i][j] = schoups_net.v4_max_diff
        
# torch.save(schoups_net.v1_grid, "trained_models/automated/v1_grid.pt")
# torch.save(schoups_net.v4_grid, "trained_models/automated/v4_grid.pt")
# logger.info('V1 grid and V4 grid saved')
                                                                                                                   