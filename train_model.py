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
from utils import infoLogger

savepath = 'trained_models/weight_scale.05/' #for saving trained model

logger = infoLogger.initLogger('train_log','train_log.log')

# Task params
precision_hard = np.pi/180
precision_easy = np.pi/18
weight_scale = 0.05
learning_rate = 0.01

#TRAIN NETWORK 1
np.random.seed(0)
torch.manual_seed(0)


schoups_net = LCN_model.LCN(
    input_size = 33, v1_size = 11, v1_orientation_number = 32, v4_size = 11, v4_stride = 6, 
    v4_orientation_number = 16, phis_sfs = 4, training_size = 20, phis = True, sfs = False, alpha = learning_rate, rescale = weight_scale)
schoups_net.transfer_inputting(-precision_hard, precision_hard, 11, 11, random_sf = False)
schoups_net.desired_outputting()
optimizer = optim.SGD(schoups_net.parameters(), lr = schoups_net.alpha)
logger.info('Schoups net initialized')
schoups_net.train(5000, optimizer)
logger.info('Schoups net training complete')

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
logger.info('Schoups model saved')

#TRAIN NETWORK 2
np.random.seed(0)
torch.manual_seed(0)

ghose_net = LCN_model.LCN(
    input_size = 33, v1_size = 11, v1_orientation_number = 32, v4_size = 11, v4_stride = 6, 
    v4_orientation_number = 16, phis_sfs = 4, training_size = 20, phis = True, sfs = False, alpha = learning_rate, rescale = weight_scale)
ghose_net.transfer_inputting(-precision_easy, precision_easy, 11, 11, random_sf = False)
ghose_net.desired_outputting()
optimizer = optim.SGD(ghose_net.parameters(), lr = ghose_net.alpha)
logger.info('Ghose net initialized')
ghose_net.train(5000, optimizer)
logger.info('Ghose net training complete')

# SAVE
torch.save(ghose_net.state_dict(),savepath +  "Ghose_32_orientations_model.pt")
torch.save(ghose_net.losses,savepath +  "Ghose_32_orientations_model_loss.pt")
torch.save(ghose_net.training_scores,savepath +  "Ghose_32_orientations_model_performance.pt")
torch.save(ghose_net.v1_weight_changes,savepath +  "Ghose_32_orientations_model_v1_weight_change.pt")
torch.save(ghose_net.v4_weight_changes,savepath +  "Ghose_32_orientations_model_v4_weight_change.pt")
torch.save(ghose_net.decision_weight_changes,savepath +  "Ghose_32_orientations_model_decision_weight_change.pt")
torch.save(ghose_net.angle1,savepath +  "Ghose_32_orientations_model_angle1.pt")
torch.save(ghose_net.angle2,savepath +  "Ghose_32_orientations_model_angle2.pt")
torch.save(ghose_net.train_x_location,savepath + "Ghose_32_orientations_model_x_location.pt")
torch.save(ghose_net.train_y_location,savepath + "Ghose_32_orientations_model_y_location.pt")
torch.save(ghose_net.generalize_perform,savepath + "Ghose_32_orientations_model_generalization_performance.pt")
torch.save(ghose_net.generalize_error,savepath + "Ghose_32_orientations_model_generalization_error.pt")
torch.save(ghose_net.trained_phis,savepath + "Ghose_32_orientations_model_phases.pt")
torch.save(ghose_net.trained_sfs,savepath + "Ghose_32_orientations_model_sfs.pt")
torch.save(ghose_net.v1_angles,savepath + "Ghose_32_orientations_model_v1_angles.pt")
logger.info('Ghose model saved')