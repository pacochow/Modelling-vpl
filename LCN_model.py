#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from tqdm import tqdm
import logging
import matplotlib.pylab as pl


class LCN(nn.Module):
    
    def __init__(self, input_size, v1_size, v1_orientation_number, v4_size, v4_stride, v4_orientation_number, phis, 
                 training_size, alpha = 0.01, v1_rescale = 1, phase_rescale = 1, v4_rescale = 1, 
                 v1_gamma = 0.5, v4_orientation_std = 0.7):

        """
        Initialize network parameters.
        """
        
        super(LCN, self).__init__()
        
        # Ensure V1 activation map size is a positive integer
        if (input_size - v1_size + 1) % 1 != 0 or (input_size - v1_size + 1) < 0:
            return("input_size needs to be larger than v1_size")
        
        # Ensure V4 activation map size is a positive integer
        if (((input_size - v1_size + 1 - v4_size)/v4_stride) % 1) != 0 or (((input_size - v1_size + 1 - v4_size)/v4_stride) % 1) < 0:
            return("V4 size and stride needs to fit within V1 activation maps of size (input_size - v1_size + 1)")
        
        # Ensure training_size is an even number
        if training_size % 2 != 0:
            return("training_size argument needs to be a multiple of 2")
        
        self.input_size = input_size # Size of input gabors (pixels)
        self.v1_orientation_number = v1_orientation_number # Number of V1 gabor filters at different orientations
        self.v1_size = v1_size # Size of V1 gabor filters (pixels)
        self.phis = phis # Number of V1 gabor filters at different phases
        self.sfs = 2 # Number of V1 gabor filters at different sfs
        self.v1_rescale = v1_rescale # scalar multiplier for scaling of V1 weights, defaults to 1 (no scaling)
        self.phase_rescale = phase_rescale # scalar multiplier for scaling of phase pooling, defaults to 1 (no scaling)
        self.v4_rescale = v4_rescale # scalar multiplier for scaling of V4 weights, defaults to 1 (no scaling)
        self.v1_gamma = v1_gamma
        self.v4_orientation_std = v4_orientation_std
        
        
        self.training_size = training_size # Batch size
        self.alpha = alpha # Learning rate
        self.v1_dimensions = self.input_size - self.v1_size + 1 # Dimension of activation map after V1 simple cell filtering
        
        self.v4_size = v4_size # Size of 2D V4 gaussian filter (pixels)
        self.v4_stride = v4_stride # V4 filter stride
        self.v4_orientation_number = v4_orientation_number # Number of V4 filters selective for different orientations
        self.v4_dimensions = int(((self.v1_dimensions - self.v4_size)/self.v4_stride) + 1) # Dimension of activation map after V4 filtering
        
        self.ambiguous = [-np.pi/2, 0, np.pi/2, np.pi] # List of ambiguous gabor angles that needs to be removed
        
        # Setting device to GPU or CPU but GPU currently not needed
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu") 
        
        # Initialising V1 weights – learnable parameter 
        self.simple_weight = torch.nn.Parameter(self.init_weights()) 
        
        # Initialising V4 weights – learnable parameter
        self.v4_weight = torch.nn.Parameter(self.init_gaussian_weights().view(
            self.v4_orientation_number, self.v1_orientation_number, self.v4_dimensions, self.v4_dimensions, self.v4_size ** 2)) 
        
        # Initialising decision layer
        self.decision = nn.Linear(v4_orientation_number * self.v4_dimensions * self.v4_dimensions, 2, bias = False) 
        self.decision.weight = torch.nn.Parameter(
            torch.zeros((2, self.v4_orientation_number * self.v4_dimensions * self.v4_dimensions))) 
        
        # Saving initial weights
        self.before_v1weight = self.simple_weight.clone()#.to(self.device)
        self.before_v4weight = self.v4_weight.clone()#.to(self.device)
        self.before_decision_weight = self.decision.weight.clone()#.to(self.device)
        
        # Initialise logger
#         logger = logging.getLogger('train_log')
#         logger.setLevel(logging.INFO)
#         f_handler = logging.FileHandler('train_log.log')
#         f_handler.setLevel(logging.INFO)
#         f_format = logging.Formatter('%(asctime)s - %(message)s')
#         f_handler.setFormatter(f_format)
#         f_handler.setLevel(logging.INFO)
#         logger.addHandler(f_handler)
    
    # Network training functions
    
    def init_weights(self):
        
        """
        Initialize V1 simple cell weights with gabor filters at different orientations and different phases/spatial 
        frequencies, depending on what is chosen when initializing network. Returns torch tensor weights.
        """        
        # Create range of orientations between -pi/2 and pi/2 for each V1 gabor that are equally spaced and symmetrical around 0
        self.v1_angles = np.linspace(-np.pi/2 + np.pi/(2 * self.v1_orientation_number), np.pi/2 - np.pi/(2 * self.v1_orientation_number), self.v1_orientation_number)
        # Create range of phases between 0 and pi for each V1 gabor
        self.phis_range = np.linspace(0, np.pi, self.phis) 
        self.sfs_range = [5, 9]
        weights = []

         # For each orientation and phase, create gabor filter with those parameters
        for theta in self.v1_angles:
            for sf in self.sfs_range:
                for phi in self.phis_range:
                    for k in range(self.v1_dimensions ** 2):
                        kernel = self.generate_gabor(self.v1_size, theta, phi, sf, gamma = self.v1_gamma)
                        kernel = kernel * self.v1_rescale

                        # Add random noise from normal distribution 
#                         noise = torch.normal(0, 0.03, (self.v1_size, self.v1_size)) 
#                         kernel = kernel + noise 
                        weights.append(kernel)

        # Return torch tensor weights for each V1 gabor filter
        weight = torch.stack(weights).view(
            1, self.v1_orientation_number*self.sfs*self.phis, 1, self.v1_dimensions, self.v1_dimensions, self.v1_size ** 2) 
        return weight 

    def init_gaussian_weights(self):
        
        """
        Initialize V4 weights with gaussian filters. Returns torch tensor weights.
        """
        # Create range of orientations between -pi/2 and pi/2 for each V4 filter
        self.v4_angles = np.linspace(-np.pi/2 + np.pi/(2 * self.v4_orientation_number), np.pi/2 - np.pi/(2 * self.v4_orientation_number), self.v4_orientation_number)
        
        x = np.linspace(-np.pi/2, np.pi/2, self.v1_orientation_number)
        
        # For each V4 orientation, create 3D V4 gaussian filter that has dimensions v1_orientation_number x v4_dimensions x v4_dimensions
        v4 = torch.empty(self.v4_orientation_number, self.v1_orientation_number, self.v4_dimensions, self.v4_dimensions, self.v4_size, self.v4_size) 
        
        for pool_orientation in range(self.v4_orientation_number):
            for orientation in range(self.v1_orientation_number):
                for i in range(self.v4_dimensions):
                    for j in range(self.v4_dimensions):
                        # Find index in v1_angles using desired v4_angle to initialise preferred orientation of V4 orientation tuning curve
                        index = self.find_nearest(torch.tensor(x), self.v4_angles[pool_orientation])
                        
                        # Generate 3D gaussian filter and roll the orientation gaussian to change the peaks
                        kernel = self.generate_3d_gaussian(mean = self.v1_angles[int(round(self.v1_orientation_number/2, 0))], spatial_std = 0.5, orientation_std = self.v4_orientation_std, roll = (int(round(self.v1_orientation_number/2, 0)) - index))[orientation] 
                        kernel = kernel * self.v4_rescale
                        
                        # Add random noise from normal distribution scaled by mean of each gaussian filter so noise does not cover up gaussian
#                         noise = torch.normal(0, 0.015, (self.v4_size, self.v4_size)) * kernel.mean() 
#                         kernel = kernel + noise
                        v4[pool_orientation][orientation][i][j] = kernel
        return v4
    
    
    def binary_loss(self, scores, desired_output):
        
        """
        Loss function. Takes in prediction and desired output as arguments and returns cross entropy loss. 
        """
        
        # Cross entropy loss because decision layer output is binary
        loss = nn.CrossEntropyLoss() 
        return loss(scores, desired_output)
    
    def inputting(self, angle1, angle2, random_sf):
        
        """
        Creates gabor filters to train network with. Takes in two angles that give the range of orientation of gabors
        and a Boolean random_sf to determine if spatial frequency of gabors is randomized. Returns a torch tensor 
        consisting of all the training gabors. 
        """
        
        if angle1 in self.ambiguous or angle2 in self.ambiguous:
            return("Angles cannot be ambiguously clockwise or counterclockwise relative to 0°")
        
        self.labels = []
        self.inputs = []
        
        self.trained_phis = np.linspace(0, np.pi, self.phis) # Use phases regularly spaced across 0 to pi
        self.trained_sfs = []
    
        self.angle1 = angle1
        self.angle2 = angle2
        
        if random_sf == True:
            sfs = [5, 9]
        else:
            sfs = [5]
        
        # For each orientation, create input gabor stimulus at that orientation with test_size/2 random phases
        for angle in [angle1, angle2]:
            for sf in sfs:
                for i in range(int(self.training_size/(2*len(sfs)))):
                    theta = angle
                    phi = self.trained_phis[i % len(self.trained_phis)]
                    kernel = self.generate_gabor(self.input_size, theta, phi, sf) 
                    self.trained_sfs.append(self.lamda)
                    self.inputs.append(kernel)

                    # Label stimulus as 0 if it is clockwise to reference orientation of 0 and 1 if counterclockwise
                    if 0 < theta < np.pi/2: 
                        label = torch.tensor([0])
                    else:
                        label = torch.tensor([1])
                    self.labels.append(label)
            
        # Stack all input gabor stimuli into one tensor
        self.input = torch.stack(self.inputs).view(self.training_size, 1, self.input_size, self.input_size)#.to(self.device) 
        return self.input
    
    def transfer_inputting(self, angle1, angle2, x_location, y_location, random_sf):
        
        """
        Creates gabor filters at particular location to train network with. Takes in two angles that give the range of 
        orientation of gabors and a Boolean random_sf to determine if spatial frequency of gabors is randomized. Returns a torch 
        tensor consisting of all the training gabors. 
        """
        
        # Ensure angles are not ambiguous
        if angle1 in self.ambiguous or angle2 in self.ambiguous:
            return("Angles cannot be ambiguously clockwise or counterclockwise relative to 0°")
        
        # Ensure locations are within v1_dimensions
        if not 0 <= x_location <= self.v1_dimensions - 1 or not 0 <= y_location <= self.v1_dimensions - 1:
            return("x_location and y_location needs to be between 0 and " + str(self.v1_dimensions - 1))
        
        self.labels = []
        self.inputs = []
    
        self.angle1 = angle1
        self.angle2 = angle2
        
        self.train_x_location = x_location
        self.train_y_location = y_location
        
        self.trained_phis = np.linspace(0, np.pi, self.phis) # Use phases regularly spaced across 0 to pi
        self.trained_sfs = []

        if random_sf == True:
            sfs = [5, 9]
        else:
            sfs = [5]
        
        # For each orientation, create input gabor stimulus at that orientation at particular location
        for angle in [angle1, angle2]:
            for sf in sfs:
                for i in range(int(self.training_size/(2*len(sfs)))):
                    theta = angle
                    phi = self.trained_phis[i % len(self.trained_phis)]
                    kernel = self.generate_location_gabor(theta, phi, sf, x_location, y_location)
                    self.trained_sfs.append(self.lamda)
                    self.inputs.append(kernel)

                    # Label stimulus as 0 if it is clockwise to reference orientation of 0 and 1 if counterclockwise
                    if 0 < theta < np.pi/2: 
                        label = torch.tensor([0])
                    else:
                        label = torch.tensor([1])
                    self.labels.append(label)
            
        # Stack all input gabor stimuli into one tensor
        self.input = torch.stack(self.inputs).view(self.training_size, 1, self.input_size, self.input_size)#.to(self.device) 
        return self.input
    
    def desired_outputting(self):
        
        """
        Returns a torch tensor consisting of the correct labels of the training set. 
        """
        
        self.desired_output = torch.tensor(self.labels)#.to(self.device)
        return self.desired_output
    
    def forward(self, x):
        
        """
        Forward function. Takes in one training gabor filter as argument and creates a prediction.
        """
        
        # V1 simple cell filter
        
        # Unfold input gabor stimuli with stride 1 to perform locally connected weight multiplication – returns shape 1 x 1 x v1_dimensions x v1_dimensions x v1_size x v1_size
        x = x.unfold(2, self.v1_size, 1).unfold(3, self.v1_size, 1) 
        x = x.reshape(1, 1, self.v1_dimensions, self.v1_dimensions, self.v1_size * self.v1_size)
        
        # Locally connected multiplication, then summing over space to space to create activation maps with dimensions 1 x v1_orientation_number * sfs * phis x v1_dimensions x v1_dimensions
        out = (x.unsqueeze(1) * self.simple_weight).sum([2, -1]) 
        
        # Apply RELU 
        out = F.relu(out)
        
        # V1 complex cell pooling over phase/sf
        phase_pools = []
        
        for i in range(0, self.v1_orientation_number*self.sfs*self.phis, self.phis):
            # Pick all activation maps with same orientation but different phase and square them
            pool = out[0][i:i+self.phis] ** 2
            
            # Sum all of these activation maps 
            pool = (torch.sqrt(torch.sum(pool, dim = 0)) * self.phase_rescale / self.phis).view(1, self.v1_dimensions, self.v1_dimensions) 
            phase_pools.append(pool)
        out = torch.stack(phase_pools).view(self.v1_orientation_number * self.sfs, self.v1_dimensions, self.v1_dimensions)

        sf_pools = []
        for i in range(0, self.v1_orientation_number*self.sfs, self.sfs):
            # Pick all activation maps with same orientation but different sfs
            pool = out[i:i+self.sfs] 
            
            # Mean sum all of these activation maps  
            pool = (torch.sum(pool, dim = 0) / self.sfs).view(1, self.v1_dimensions, self.v1_dimensions) 
            sf_pools.append(pool)
        sf_pools = torch.stack(sf_pools).view(self.v1_orientation_number, self.v1_dimensions, self.v1_dimensions)

        # V4 pooling
        v4_pools = []
        
        # Unfold activation maps with stride v4_stride to perform V4 locally connected weight multiplication – returns shape v1_size x v4_dimensions x v4_dimensions x v4_size x v4_size
        out = sf_pools.unfold(1, self.v4_size, self.v4_stride).unfold(2, self.v4_size, self.v4_stride) 
        out = out.reshape(self.v1_orientation_number, self.v4_dimensions, self.v4_dimensions, self.v4_size * self.v4_size)
        
        # For each V4 orientation, perform locally connected multiplication, then sum over space and orientation to create tensor of activation maps with dimensions v4_orientation_number, v4_dimensions, v4_dimensions
        for j in range(self.v4_orientation_number):
            pooled = (out*self.v4_weight[j]).sum([-1, 0])
            v4_pools.append(pooled)
        v4_pool = torch.stack(v4_pools).view(self.v4_orientation_number, self.v4_dimensions, self.v4_dimensions)
       
        # Apply RELU
        v4_pool = F.relu(v4_pool)

        # Feed flattened activation maps into decision layer
        out = v4_pool.view(1, self.v4_orientation_number * self.v4_dimensions * self.v4_dimensions)
        out = self.decision(out.float()) 
        out = F.softmax(out, dim = 1)

        return out
    
    def train(self, iterations, optimizer):
        
        """
        Training loop. Takes in the number of iteractions and optimizer as arguments and trains the network over the 
        number of iteractions. Calculates mean loss over batch size and optimizes over this loss. 
        """
        logger = logging.getLogger('train_log')

        self.v1_weight_changes = []
        self.v4_weight_changes = []
        self.decision_weight_changes = []

        self.losses = []
        self.training_scores = []

        self.generalize_perform = []
        self.generalize_error = []

        self.v1_otc_max_diffs = []
        self.v4_otc_max_diffs = []

        self.v1_spatial_heatmap = torch.zeros(int(iterations/1000 + 1), self.v1_dimensions, self.v1_dimensions)
        self.v4_spatial_heatmap = torch.zeros(int(iterations/1000 + 1), self.v4_dimensions, self.v4_dimensions)
        
        # Iteration loop
        for i in tqdm(range(iterations)):

            if i % 50 == 0:
                logger.info('Iterations completed: {:.0f}'.format(i))

#             # End training if performance reaches over 85% 
#             if len(self.training_scores) != 0 and self.training_scores[-1] >= 85:
#                 logger.info('Training terminated at {:.0f} iterations, {:.0f}% training performance reached'.format(i, self.training_scores[-1]))
#                 break


            optimizer.zero_grad() # Reset gradients each iteration
            self.training_score = 0 
            loss2 = torch.empty(self.training_size)
            # Batch loop
            for j in range(self.training_size):
                # Generate predictions for each input
                self.scores = self.forward(self.input[j][0].view(1, 1, self.input_size, self.input_size))

                # Calculate loss
                loss1 = self.binary_loss(self.scores, self.desired_output[j].view(1)) 

                # Increase score if predicted label matches actual label
                if torch.argmax(self.scores) == self.desired_output[j]: 
                    self.training_score += 1

                # Keep track of loss
                loss2[j] = loss1 

            # Calculate average loss for each batch
            loss = torch.sum(loss2)/self.training_size 
            self.losses.append(float(loss))
            
            # If loss goes over 1, break training loop 
            if float(loss) > 1:
                break

            # Calculate performance for each batch
            self.training_score = self.training_score/self.training_size * 100 
            self.training_scores.append(self.training_score)

            # Keep track of weight changes in each layer
            self.v1_weight_changes.append(self.v1_weight_change(self.before_v1weight, self.simple_weight)) 
            self.v4_weight_changes.append(self.v4_weight_change(self.before_v4weight, self.v4_weight)) 
            self.decision_weight_changes.append(self.decision_weight_change(self.before_decision_weight, self.decision.weight))

            # Backpropagate error and update weights
            loss.backward() 
            optimizer.step() 

            self.generalization(self.angle2, self.training_size)
            self.generalize_perform.append(self.generalization_score)
            self.generalize_error.append(self.general_mean_error)

        # Generate tuning curves
        self.v1_tuning_curve()
        self.v4_tuning_curve()
            
#             if i % 500 == 0 or i == iterations - 1:
#                 self.v1_tuning_curve()
#                 self.v4_tuning_curve()
#                 self.otc_curve(v1_position_1 = 11, v1_position_2 = 11, v4_position_1 = 1, v4_position_2 = 1)
#                 self.v1_otc_max_diffs.append(self.v1_max_diff)
#                 self.v4_otc_max_diffs.append(self.v4_max_diff)
                    
                    
#             if i % 1000 == 0 or i == iterations - 1:
#                 self.v1_tuning_curve()
#                 self.v4_tuning_curve()
#                 if i == iterations - 1:
#                     for a in range(self.v1_dimensions):
#                         for b in range(self.v1_dimensions):
#                             self.otc_curve(v1_position_1 = a, v1_position_2 = b, v4_position_1 = 1, v4_position_2 = 1)
#                             self.v1_spatial_heatmap[int((i+1)/1000)][a][b] = self.v1_max_diff
#                     for c in range(self.v4_dimensions):
#                         for d in range(self.v4_dimensions):
#                             self.otc_curve(v1_position_1 = 11, v1_position_2 = 11, v4_position_1 = c, v4_position_2 = d)
#                             self.v4_spatial_heatmap[int((i+1)/1000)][c][d] = self.v4_max_diff
#                 else:       
#                     for a in range(self.v1_dimensions):
#                         for b in range(self.v1_dimensions):
#                             self.otc_curve(v1_position_1 = a, v1_position_2 = b, v4_position_1 = 1, v4_position_2 = 1)
#                             self.v1_spatial_heatmap[int(i/1000)][a][b] = self.v1_max_diff
#                     for c in range(self.v4_dimensions):
#                         for d in range(self.v4_dimensions):
#                             self.otc_curve(v1_position_1 = 11, v1_position_2 = 11, v4_position_1 = c, v4_position_2 = d)
#                             self.v4_spatial_heatmap[int(i/1000)][c][d] = self.v4_max_diff

        
    def double_train(self, iterations, optimizer, angle1, angle2, x_location, y_location):
        
        """
        Training loop for sequential curriculum. Takes in the number of iteractions, optimizer, the first angle to be
        trained on (will be trained on -angle and angle), the second angle to be trained on after training on first 
        angle, the test angle and the test size. For each iteraction, the network is tested on the test_size number of 
        gabors between -test_angle and test_angle to see how well it generalizes. 
        """
        
       
        
        self.v1_weight_changes = []
        self.v4_weight_changes = []
        self.decision_weight_changes = []
        
        self.losses = []
        self.training_scores = []
        
        self.generalize_perform = []
        self.generalize_error = []
        
        self.angle1 = -angle2
        self.angle2 = angle2
        
        # Training over 2 angles sequentially
        for angle in [angle1, angle2]:
            input = self.transfer_inputting(-angle, angle, x_location, y_location, random_sf = False)
            desired_output = self.desired_outputting()
            
            # Iteration loop
            for i in tqdm(range(iterations)):
                if i % 50 == 0:
                    logger.info('Iterations completed: {:.0f}'.format(i))
                optimizer.zero_grad() # Reset gradients each iteration
                self.training_score = 0
                loss2 = torch.empty(self.training_size)

                # Batch loop
                for j in range(self.training_size):
                    
                    # Generate predictions for each input
                    self.scores = self.forward(input[j][0].view(1, 1, self.input_size, self.input_size))
                    
                    # Calculate loss
                    loss1 = self.binary_loss(self.scores, desired_output[j].view([1]))
                    
                    # Increase score if predicted label matches actual label
                    if torch.argmax(self.scores) == desired_output[j]:
                        self.training_score += 1
                    
                    # Keep track of loss
                    loss2[j] = loss1

                # Calculate average loss for each batch
                loss = torch.sum(loss2)/self.training_size
                self.losses.append(float(loss))

                # Calculate performance for each batch
                self.training_score = self.training_score/self.training_size * 100
                self.training_scores.append(self.training_score)


                # Keep track of weight changes in each layer
                self.v1_weight_changes.append(self.v1_weight_change(self.before_v1weight, self.simple_weight))
                self.v4_weight_changes.append(self.v4_weight_change(self.before_v4weight, self.v4_weight))
                self.decision_weight_changes.append(self.decision_weight_change(self.before_decision_weight, self.decision.weight))

                # Backpropagate error and update weights
                loss.backward()
                optimizer.step()
                
                self.generalization(angle, self.training_size)
                self.generalize_perform.append(self.generalization_score)
                self.generalize_error.append(self.general_mean_error)
    
        # Generate tuning curves   
        self.v1_tuning_curve()
        self.v4_tuning_curve()
        
    def generalization(self, angle, phase_number):
        
        """
        Function used to test model against test_size number of gabors between -angle and angle to measure 
        generalization error and performance.
        """
        
        # Create list of phases between 0 and pi
        phases = np.linspace(0, np.pi, int(phase_number/2))
        self.generalization_score = 0
        general_error = []
        
        # Create gabors at each orientation and phase and present to network
        for angle in [-angle, angle]:
            for phi in phases:
                # Create gabor
                gabor = self.generate_location_gabor(angle, phi, 5, int((self.v1_dimensions - 1)/2), int((self.v1_dimensions - 1)/2)).clone().detach()

                # Get correct labels for test gabors
                if 0 < angle < np.pi/2:
                    label = torch.tensor([0])
                else:
                    label = torch.tensor([1])

                # Present to network and measure performance and error
                with torch.no_grad():
                    a = self.forward(gabor)
                    if torch.argmax(a) == label:
                        self.generalization_score += 1
                    general_error.append(float(self.binary_loss(a, label)))
        
        # Calculate generalization performance and error
        self.generalization_score = self.generalization_score/(phase_number) * 100
        self.general_mean_error = np.mean(general_error)
    
    # Network analytical functions
    
    def plot_training_error(self, color):
        
        """
        Plots the training error for each iteraction. Calculated by mean loss between prediction and desired output
        for each batch of training data. Takes in color as argument. 
        """
        
        plt.plot([loss for loss in self.losses], color = color)
        plt.xlabel("Time (epochs)")
        plt.ylabel("Error")

    def plot_training_performance(self, color):
        
        """
        Plots the training performance for each iteraction. If the prediction has a higher value for the accurate 
        rotation (clockwise vs anticlockwise), the model is assumed to have produced the correct output and so the
        score increases by 1. Percentage accuracy is then calculated after. Takes in color as argument. 
        """
        
        plt.plot(self.training_scores, color = color)
        plt.xlabel("Time (epochs)")
        plt.ylabel("Performance (%)")

    def plot_generalization_performance(self, color):
        
        """
        Plots generalization performance. Takes in color as argument.
        """
        
        plt.plot(self.generalize_perform, color = color, linestyle = "dotted")
        plt.xlabel("Time (epochs)")
        plt.ylabel("Performance (%)")

    def plot_generalization_error(self, color):
        
        """
        Plots generalization error. Takes in color as argument. 
        """
        
        plt.plot(self.generalize_error, color = color, linestyle = "dotted")
        plt.xlabel("Time (epochs)")
        plt.ylabel("Error")
   
    def plot_angle_performance(self, angle_number, color):
        
        """
        Plots the angle/performance graph for the model. Takes in the number of test angles and plot color as 
        arguments. Creates the number of angles between 0 and pi/2 and tests model with test angles between -angle 
        and angle. 
        """
        
        # Use trained phases and sfs. If there are no trained phases and sfs because model is just initialized, then use 0 as phase and 5 as sf
        try:
            phases = self.trained_phis
            sfs = self.trained_sfs
        except AttributeError: 
            phases = list((0,) * self.training_size)
            sfs = list((5, ) * self.training_size)

        
        # Create list of angles between 0 and pi/2
        angles = self.remove_ambiguous_stimuli(0, np.pi/2, angle_number, even_space = False)
        self.angle_scores = []
        
        # For each angle, calculate generalization performance on 50 gabors with orientations between -angle and angle
        for i in tqdm(range(angle_number)):
            score = 0
            
            for j in range(self.training_size):
                if j < 10:
                    angle = -angles[i]

                else: 
                    angle = angles[i]
                
                kernel = self.generate_location_gabor(angle, phases[j], sfs[j], int((self.v1_dimensions - 1)/2), int((self.v1_dimensions - 1)/2))
                
                # Label stimulus as 0 if it is clockwise to reference orientation of 0 and 1 if counterclockwise
                if 0 < angle < np.pi/2: 
                    label = torch.tensor([0])
                else:
                    label = torch.tensor([1])

                with torch.no_grad():
                    pred = self.forward(kernel)
                    if torch.argmax(pred) == label:
                        score += 1
                
            self.angle_scores.append((score/self.training_size) * 100) 

        
        # Plot performance on each angle
        plt.plot((angles * 180)/np.pi, self.angle_scores, color = color)
        plt.xlabel("Separation angle (Degrees)")
        plt.ylabel("Performance (%)")

    def v1_weight_change(self, before, after):
        
        """
        Calculates frobenius norm of difference between weights in V1 simple cells and initial weights for each iteraction 
        of training. Takes in initial weights and current weights as arguments and returns scalar value.
        """
        
        # For each V1 simple cell gabor selective for different orientation, phase and position, calculate difference before and after training
        diff = after - before
        net_diff = []
        
        # Calculate frobenius norm of each gabor difference and return mean magnitude of change
        for i in diff.view(self.v1_orientation_number*self.sfs*self.phis, 1, self.v1_dimensions, self.v1_dimensions, self.v1_size ** 2):
            for j in i.view(self.v1_dimensions, self.v1_dimensions, self.v1_size ** 2):
                for k in j:
                    net_diff.append(torch.linalg.norm(k.view(self.v1_size, self.v1_size), ord = 'fro').item())
        return np.mean(net_diff)
    
    def v4_weight_change(self, before, after):
        
        """
        Calculates frobenius norm of difference between weights in V4 cells and initial weights for each iteraction 
        of training. Takes in initial weights and current weights as arguments and returns scalar value.
        """
        
        # For each V4 gaussian filter selective for different orientations and position, calculate difference before and after training
        diff = after - before
        net_diff = []
        
        # Calculate frobenius norm of each filter difference and return mean magnitude of change
        for v4_orientation in diff.view(self.v4_orientation_number, self.v1_orientation_number, self.v4_dimensions, self.v4_dimensions, self.v4_size ** 2):
            for simple in v4_orientation.view(self.v1_orientation_number, self.v4_dimensions, self.v4_dimensions, self.v4_size ** 2):
                for j in simple.view(self.v4_dimensions, self.v4_dimensions, self.v4_size ** 2):
                    for k in j:
                        net_diff.append(torch.linalg.norm(k.view(self.v4_size, self.v4_size), ord = 'fro').item())
        return np.mean(net_diff)
    
    def decision_weight_change(self, before, after):
        
        """
        Calculates frobenius norm of difference between weights in decision layer and initial weights for each iteraction
        of training. Takes in initial weights and current weights as arguments and returns scalar value. 
        """
        
        # Calculate frobenius norm of decision layer weight changes before and after training
        diff = after - before
        net_diff = (torch.linalg.norm(
            diff.view(2, self.v4_orientation_number * self.v4_dimensions * self.v4_dimensions), ord = 'fro').item())
        return net_diff / (self.v4_orientation_number * self.v4_dimensions * self.v4_dimensions)
    
    def plot_weight_change(self, color, v1 = False, v4 = False, decision = False):
        
        """
        Plot weight changes. Takes in plot color as argument. Setting v1 = True plots V1 simple cell weight change and
        setting decision = True plots deicision layer weight changes. 
        """
        
        if v1 == True:
            plt.plot(self.v1_weight_changes, color = color)
            plt.title("Weight changes in V1 during training");
        if v4 == True:
            plt.plot(self.v4_weight_changes, color = color)
            plt.title("Weight changes in V4 during training");
        if decision == True:
            plt.plot(self.decision_weight_changes, color = color)
            plt.title("Weight changes in decision layer during training");
        plt.xlabel("Time (epochs)")
        plt.ylabel("Weight change")
    
    def v1_tuning_curve(self):
        
        """
        Creates tuning curves for each gabor filter in V1 simple cell layer by testing each filter with a number of
        orientations at a fixed phase and spatial frequency. Returns a torch tensor consisting of all tuning curves 
        organized with dimensions orientation, spatial frequency, horizontal position, vertical position and the 
        tuning curve data plot.
        """
        
        self.tuning_curve_sample = 100
        
        # Initialise tensor for all tuning curves after training, organized into orientations, phase/sfs, horizontal position, vertical position, tuning curve data
        self.results = torch.empty(self.v1_orientation_number, self.sfs, self.phis, self.v1_dimensions, self.v1_dimensions, self.tuning_curve_sample)
        
        # Initialise tensor for all tuning curves before training
        self.initial_tuning_curves = torch.empty(
            self.v1_orientation_number, self.sfs, self.phis, self.v1_dimensions, self.v1_dimensions, self.tuning_curve_sample)
        
        
        # Create gabor at each orientation and store measured activity of each V1 gabor filter in tensor
        with torch.no_grad():
            for i in tqdm(range(self.tuning_curve_sample)):
                for orientation in range(self.v1_orientation_number):
                    for sf in range(self.sfs):
                        for phi in range(self.phis):
                            for horizontal in range(self.v1_dimensions):
                                for vertical in range(self.v1_dimensions):

                                    # Create list of angles between -pi/2 and pi/2 for units tuned to first and last phases, and between -pi/2 and 3pi/2 for second and third phases

                                    if phi == 0 or phi == self.phis - 1:
                                        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
                                    else:
                                        x = np.linspace(-np.pi/2, 3*np.pi/2, self.tuning_curve_sample)

                                    # Create gabor
                                    test = self.generate_gabor(self.v1_size, x[i], self.phis_range[phi], self.sfs_range[sf]).view(
                                        self.v1_size, self.v1_size)#.to(self.device)

                                    # Present to specific gabor after training
                                    result = torch.sum(
                                        self.simple_weight[0][self.phis * self.sfs * orientation + self.phis * sf + phi][0][horizontal][vertical].view(
                                            self.v1_size, self.v1_size) * test)
                                    result = F.relu(result)

                                    # Present to specific gabor before training
                                    initial_result = torch.sum(
                                        self.before_v1weight[0][self.phis * self.sfs * orientation + self.phis * sf + phi][0][horizontal][vertical].view(
                                            self.v1_size, self.v1_size) * test)
                                    initial_result = F.relu(initial_result)

                                    # Save activity in tensor
                                    self.results[orientation][sf][phi][horizontal][vertical][i] = result 
                                    self.initial_tuning_curves[orientation][sf][phi][horizontal][vertical][i] = initial_result 
                                
            # Normalize tuning curves
            self.results = self.results / self.initial_tuning_curves.max()
            self.initial_tuning_curves = self.initial_tuning_curves / self.initial_tuning_curves.max()
                                

    def plot_v1_tuning_curve(self, orientation, sf, phi, position, orientations = False, phis = False, sfs = False, differences = False, color = False):
        
        """
        Plot tuning curves at a particular orientation index or phase/spatial frequeny index and at a particular 
        filter position. Setting orientations = True plots tuning curves at all orientations at a specified phase and
        sf. Setting sf = True plots tuning curves at all sfs at a specified orientation and phase. Setting phi = True plots 
        tuning curves at all phases at a specified orientation and sf. Setting differences = True plots difference in
        responses using initial V1 simple cell weights and trained weights. Setting color = True plots the curves in a gradient 
        of colors.
        """
        
        # Ensure orientation is between 0 and v1_orientation_number
        if not 0 <= orientation <= self.v1_orientation_number - 1:
            return("orientation needs to be between 0 and " + str(self.v1_orientation_number - 1))
        
        # Ensure phi is between 0 and phis
        if not 0 <= phi <= self.phis - 1:
            return("phase needs to be between 0 and " + str(self.phis - 1))
        
        # Ensure sf is between 0 and phisfss
        if not 0 <= sf <= self.sfs - 1:
            return("sf needs to be between 0 and " + str(self.sfs - 1))
        
        # Ensure position is between 0 and v1_dimensions
        if not 0 <= position <= self.v1_dimensions - 1:
            return("position needs to be between 0 and " + str(self.v1_dimensions - 1))
        
        # Create list of angles between -pi/2 and pi/2
        if phi == 0 or phi == self.phis - 1:
            x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
            x = (x * 180) / np.pi
        else:
            x = np.linspace(-np.pi/2, 3*np.pi/2, self.tuning_curve_sample)
            x = (x * 180) / np.pi
        
        colors = pl.cm.jet(np.linspace(0, 1, self.v1_orientation_number))
        
        # Calculate difference in tuning curves before and after training
        difference = self.results - self.initial_tuning_curves
        
        if orientations == True and differences == False:
            
            # Plot each tuning curve at different orientations with specified phase/sf and position
            for i in range(self.v1_orientation_number):
                if color == True:
                    plt.plot(x, self.results[i, sf, phi, position, position, :], color = colors[i])
                else:
                    plt.plot(x, self.results[i, sf, phi, position, position, :])
                
            # Create legend
            plt.legend([round(self.v1_angles[i] * 180 / np.pi, 1) for i in range(self.v1_orientation_number)])
            
            plt.ylabel("Response")
            plt.title("V1 tuning curves selective for different orientations", loc = 'center')
                
        if sfs == True and differences == False:
            
            # Plot each tuning curve at different phase/sfs with specified orientation and position
            for i in range(self.sfs):
                if color == True:
                    plt.plot(x, self.results[orientation, i, phi, position, position, :], color = colors[i])
                else:
                    plt.plot(x, self.results[orientation, i, phi, position, position, :])
                
            # Create legend
            plt.legend([5, 9])
            
            plt.ylabel("Response")
            plt.title("V1 tuning curves selective for different SFs", loc = 'center');        
                
        if phis == True and differences == False:
            
            # Plot each tuning curve at different phase/sfs with specified orientation and position
            for i in range(self.phis):
                if color == True:
                    plt.plot(x, self.results[orientation, sf, i, position, position, :], color = colors[i])
                else:
                    plt.plot(x, self.results[orientation, sf, i, position, position, :])
                
            # Create legend
            ranges = np.linspace(self.phis_range[0], self.phis_range[-1], self.phis)
            plt.legend([round(ranges[i], 1) for i in range(self.phis)])
            
            plt.ylabel("Response")
            plt.title("V1 tuning curves selective for different phase/SFs", loc = 'center');
            
        if orientations == True and differences == True:
            
            # Plot each difference in tuning curve at different orientations with specified phase/sf and position
            for i in range(self.v1_orientation_number):
                if color == True:
                    plt.plot(x, difference[i, sf, phi, position, position, :], color = colors[i])
                else:
                    plt.plot(x, difference[i, sf, phi, position, position, :])
            
            # Create legend
            plt.legend([round(self.v1_angles[i] * 180 / np.pi, 1) for i in range(self.v1_orientation_number)])
            
            plt.ylabel("Difference in response")
            plt.title("Difference in V1 tuning curves selective for different orientations", loc = 'center');
            
            
        if sfs == True and differences == True:
            
            # Plot each tuning curve at different phase/sfs with specified orientation and position
            for i in range(self.sfs):
                if color == True:
                    plt.plot(x, difference[orientation, i, phi, position, position, :], color = colors[i])
                else:
                    plt.plot(x, difference[orientation, i, phi, position, position, :])
                
            # Create legend
            plt.legend([5, 9])
            
            plt.ylabel("Difference in response")
            plt.title("Difference in V1 tuning curves selective for different SFs", loc = 'center');    
        
        if phis == True and differences == True:
            
            # Plot each difference in tuning curve at different phase/sfs with specified orientation and position
            for i in range(self.phis):
                if color == True:
                    plt.plot(x, difference[orientation, sf, i, position, position, :], color = colors[i])
                else:
                    plt.plot(x, difference[orientation, sf, i, position, position, :])

            # Create legend
            ranges = np.linspace(self.phis_range[0], self.phis_range[-1], self.phis)
            plt.legend([round(ranges[i], 1) for i in range(self.phis)])
            
            plt.ylabel("Difference in response")
            plt.title("Difference in V1 tuning curves selective for different phases", loc = 'center');
        
        plt.xlabel("Angle (Degrees)")
        
    def v4_tuning_curve(self):

        """
        Creates tuning curves for each gabor filter in V4 layer by testing each filter with a number of
        orientations at a fixed phase and spatial frequency. Returns a torch tensor consisting of all tuning curves 
        organized with dimensions orientation, horizontal position, vertical position and the 
        tuning curve data plot.
        """
        
        # Create list of angles between -pi/2 and pi/2
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        
        # Initialise tensor for all tuning curves after training, organized into orientations, horizontal position, vertical position, tuning curve data
        self.v4_results = torch.empty(self.v4_orientation_number, self.v4_dimensions, self.v4_dimensions, self.tuning_curve_sample)
        
        # Initialise tensor for all tuning curves before training, organized into orientations, horizontal position, vertical position, tuning curve data
        self.v4_initial_tuning_curves = torch.empty(
            self.v4_orientation_number, self.v4_dimensions, self.v4_dimensions, self.tuning_curve_sample)
        
        # Create gabor at each orientation, present activity to network and measure activity at each V4 filter
        with torch.no_grad():
            for i in range(len(x)):
                
                # Create gabor
                test = self.generate_gabor(self.input_size, x[i], 0, 5).view(1, 1, self.input_size, self.input_size)
                
                # Forward pass through V1 
                out = test.unfold(2, self.v1_size, 1).unfold(3, self.v1_size, 1)#.to(self.device)
                out = out.reshape(1, 1, self.v1_dimensions, self.v1_dimensions, self.v1_size * self.v1_size)
                out_after = (out.unsqueeze(1) * self.simple_weight).sum([2, -1])
                out_after = F.relu(out_after)
                out_before = (out.unsqueeze(1) * self.before_v1weight).sum([2, -1])
                out_before = F.relu(out_before)
                
                phase_pools_after = []
                phase_pools_before = []
                for j in range(0, self.v1_orientation_number*self.sfs*self.phis, self.phis):
                    pool = out_after[0][j:j+self.phis] ** 2
                    pool = (torch.sqrt(torch.sum(pool, dim = 0)) * self.phase_rescale/self.phis).view(1, self.v1_dimensions, self.v1_dimensions)
                    phase_pools_after.append(pool)
                    
                    pool = out_before[0][j:j+self.phis] ** 2
                    pool = (torch.sqrt(torch.sum(pool, dim = 0)) * self.phase_rescale/self.phis).view(1, self.v1_dimensions, self.v1_dimensions)
                    phase_pools_before.append(pool)
                
                phase_pools_after = torch.stack(phase_pools_after).view(self.v1_orientation_number * self.sfs, self.v1_dimensions, self.v1_dimensions)
                phase_pools_before = torch.stack(phase_pools_before).view(self.v1_orientation_number * self.sfs, self.v1_dimensions, self.v1_dimensions)
                
                sf_pools_after = []
                sf_pools_before = []                
                for k in range(0, self.v1_orientation_number * self.sfs, self.sfs):
                    pool = phase_pools_after[k:k+self.sfs]
                    pool = (torch.sum(pool, dim = 0) / self.sfs).view(1, self.v1_dimensions, self.v1_dimensions)
                    sf_pools_after.append(pool)
                    
                    pool = phase_pools_before[k:k+self.sfs]
                    pool = (torch.sum(pool, dim = 0) / self.sfs).view(1, self.v1_dimensions, self.v1_dimensions)
                    sf_pools_before.append(pool) 
                
                sf_pools_after = torch.stack(sf_pools_after).view(self.v1_orientation_number, self.v1_dimensions, self.v1_dimensions)
                sf_pools_before = torch.stack(sf_pools_before).view(self.v1_orientation_number, self.v1_dimensions, self.v1_dimensions)
                    
                
                v4_pool_after = torch.zeros(self.v4_orientation_number, self.v4_dimensions, self.v4_dimensions)
                v4_pool_before = torch.zeros(self.v4_orientation_number, self.v4_dimensions, self.v4_dimensions)
                    
                # Forward pass through V4 
                out_a = sf_pools_after.unfold(1, self.v4_size, self.v4_stride).unfold(2, self.v4_size, self.v4_stride)
                out_a = out_a.reshape(self.v1_orientation_number, self.v4_dimensions, self.v4_dimensions, self.v4_size * self.v4_size)
                
                out_b = sf_pools_before.unfold(1, self.v4_size, self.v4_stride).unfold(2, self.v4_size, self.v4_stride)
                out_b = out_b.reshape(self.v1_orientation_number, self.v4_dimensions, self.v4_dimensions, self.v4_size * self.v4_size)
                
                # For each V4 orientation and spatial position, present the centre gabor test input to the V4 weight and measure activity
                for k in range(self.v4_orientation_number):
                    for a in range(self.v4_dimensions):
                        for b in range(self.v4_dimensions):
                            out_after = (out_a[:, 1, 1, :] * self.v4_weight[k, :, a, b, :]).sum([-1, 0])
                            out_after = F.relu(out_after)
                            v4_pool_after[k][a][b] = out_after

                            out_before = (out_b[:, 1, 1, :] * self.before_v4weight[k, :, a, b, :]).sum([-1, 0])
                            out_before = F.relu(out_before)
                            v4_pool_before[k][a][b] = out_before


                # Store measured activity of each filter in tensor
                self.v4_results[:, :, :, i] = v4_pool_after
                self.v4_initial_tuning_curves[:, :, :, i] = v4_pool_before
                       
        # Normalize tuning curves
        self.v4_results = self.v4_results / self.v4_initial_tuning_curves.max()
        self.v4_initial_tuning_curves = self.v4_initial_tuning_curves / self.v4_initial_tuning_curves.max()
                       
    def plot_v4_tuning_curve(self, position, differences = False, color = False):
        
        """
        Plot tuning curves at a particular orientation index and at a particular filter position. Setting differences = True 
        plots difference in responses using initial V1 simple cell weights and trained weights. 
        """
        
        # Ensure position is between 0 and v4_dimensions
        if not 0 <= position <= self.v4_dimensions - 1:
            return("position needs to be between 0 and " + str(self.v4_dimensions - 1))
        
        # Create list of angles between -pi/2 and pi/2
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = (x * 180) / np.pi
        
        colors = pl.cm.jet(np.linspace(0, 1, self.v4_orientation_number))
        
        if differences == False:
            
            # Plot each tuning curve at different positions with specified position
            for i in range(self.v4_orientation_number):
                if color == True:
                    plt.plot(x, self.v4_results[i, position, position, :], color = colors[i])
                else:
                    plt.plot(x, self.v4_results[i, position, position, :])
            
            # Find preferred orientation of V4 curves for legend
            plt.legend([round(x[self.v4_results[i][position][position][:].argmax()], 1) for i in range(self.v4_orientation_number)])
            plt.ylabel("Response")
            plt.title("V4 tuning curves selective for different orientations", loc = 'center');

        if differences == True:

            # Calculate difference in tuning curves before and after training
            difference = self.v4_results - self.v4_initial_tuning_curves
            
            # Plot each difference in tuning curve at different orientations with specified phase/sf and position
            for i in range(self.v4_orientation_number):
                if color == True:
                    plt.plot(x, difference[i, position, position, :], color = colors[i])
                else:
                    plt.plot(x, difference[i, position, position, :])
            
            # Find preferred orientation of V4 curves for legend
            plt.legend([round(x[self.v4_results[i][position][position][:].argmax()], 1) for i in range(self.v4_orientation_number)])
            plt.ylabel("Difference in response")
            plt.title("Difference in V4 tuning curves selective for different orientations", loc = 'center');

        plt.xlabel("Angle (Degrees)")
    
    
    def v1_tuning_params(self, position):
        
        """
        Calculates the amplitude and bandwidth of V1 tuning curves. Takes position of gabor filter as argument.  
        """
        
        # Ensure position is between 0 and v1_dimensions
        if not 0 <= position <= self.v1_dimensions - 1:
            return("position needs to be between 0 and " + str(self.v1_dimensions - 1))
        

        self.after_amplitudes = []
        self.after_bandwidths = []
        self.after_baselines = []
        self.v1_after_pos = []
        
        self.before_amplitudes = []
        self.before_bandwidths = []
        self.before_baselines = []
        self.v1_before_pos = []
        
        self.xs = [i for i in range(self.tuning_curve_sample)]
        
        angles = np.linspace(-np.pi/2 + np.pi/(2 * self.v1_orientation_number), np.pi/2 - np.pi/(2 * self.v1_orientation_number), self.v1_orientation_number)
        threshold1 = self.angle1 - np.pi/8
        threshold2 = self.angle2 + np.pi/8
        
        for i in range(self.v1_orientation_number):
            
            # Only calculate parameters between trained angle ± 22.5°
            if not threshold1 <= angles[i] <= threshold2:
                continue
                
            for sf in range(self.sfs):
                for j in range(self.phis):
                
                
                    # Calculate tuning curve parameters after training

                    # Set tuning curve at particular orientation, phase/sf and position after training
                    curve = self.results[i, sf, j, position, position, :]

                    # Create list of angles between -pi/2 and pi/2 for units tuned to first and last phases, and between -pi/2 and 3pi/2 for second and third phases

                    if j == 0 or j == self.phis - 1:
                        a = -np.pi/2
                        b = np.pi/2
                    else:
                        a = -np.pi/2
                        b = 3*np.pi/2
                    x = np.linspace(a, b, self.tuning_curve_sample)
                    x = (x * 180) / np.pi

                    # Measure baseline using min of tuning curve
                    baseline = curve.min()
                    self.after_baselines.append(baseline)
                    
                    # Measure preferred orientation of tuning curve
                    after_preferred_orientation = curve.argmax().item()   
                    self.v1_after_pos.append(x[after_preferred_orientation])
                    
                    # Measure amplitude using max and min of tuning curve
                    amplitude = curve.max() - curve.min()
                    self.after_amplitudes.append(amplitude)

                    # Find amplitude of half the difference between max and min
                    halfmax_amplitude = torch.abs(curve.max()) - torch.abs(curve.min())
                    halfmax = halfmax_amplitude/2 + curve.min()

                    # Find the first index closest to halfmax
                    halfmax_index1 = self.find_nearest(curve, halfmax)

                    # Remove that point to find the next closest index, ensuring that it is on the other side of the curve
                    temporary = torch.cat([curve[0:halfmax_index1], curve[halfmax_index1+1:]])
                    halfmax_index2 = self.find_nearest(temporary, halfmax)

                    # While loop to prevent choosing 2 halfmax indices next to each other (check -20 and -5 to prevent edge cases)
                    add = 0
                    while self.xs[halfmax_index1 - 22] <= self.xs[halfmax_index2 - 20] <= self.xs[halfmax_index1 - 18] or self.xs[halfmax_index1 - 7] <= self.xs[halfmax_index2 - 5] <= self.xs[halfmax_index1 - 3]:
                        temporary = torch.cat([temporary[0:halfmax_index2], temporary[halfmax_index2+1:]])
                        halfmax_index2 = self.find_nearest(temporary, halfmax)
                        add += 1

                    # Add the indices taken away back in if needed to get 2 correct indices at halfmax
                    if halfmax_index1 < halfmax_index2:
                        halfmax_index2 = halfmax_index2 + 1 + add
                    # Find closest difference between the two points at halfmax to calculate bandwidth 
                    if curve[self.xs[halfmax_index1 - 1]] < curve[self.xs[halfmax_index1]] < curve[self.xs[(halfmax_index1 + 1) % self.tuning_curve_sample]] and curve[self.xs[halfmax_index2 - 1]] > curve[self.xs[halfmax_index2]] > curve[self.xs[(halfmax_index2 + 1) % self.tuning_curve_sample]] and halfmax_index1<halfmax_index2:
                        bandwidth = x[halfmax_index2] - x[halfmax_index1]
                    elif curve[self.xs[halfmax_index2 - 1]] < curve[self.xs[halfmax_index2]] < curve[self.xs[(halfmax_index2 + 1) % self.tuning_curve_sample]] and curve[self.xs[halfmax_index1 - 1]] > curve[self.xs[halfmax_index1]] > curve[self.xs[(halfmax_index1 + 1) % self.tuning_curve_sample]] and halfmax_index2 < halfmax_index1:
                        bandwidth = x[halfmax_index1] - x[halfmax_index2]
                    else:
                        bandwidth = ((b-a)*180/np.pi) - np.abs(x[halfmax_index1] - x[halfmax_index2])
                    self.after_bandwidths.append(bandwidth/2)

                    
                    # Calculate tuning curve parameters before training

                    # Set tuning curve at particular orientation, phase/sf and position before training
                    initial_params = self.initial_tuning_curves[i, sf, j, position, position, :]

                    # Measure baseline using min of tuning curve
                    baseline = initial_params.min()
                    self.before_baselines.append(baseline)
                    
                    # Measure preferred orientation of tuning curve
                    before_preferred_orientation = initial_params.argmax().item()   
                    self.v1_before_pos.append(x[before_preferred_orientation])
                    
                    # Measure amplitude using max and min of tuning curve
                    amplitude2 = initial_params.max() - initial_params.min()
                    self.before_amplitudes.append(amplitude2)

                    # Find amplitude of half the difference between max and min
                    halfmax2_amplitude2 = torch.abs(initial_params.max()) - torch.abs(initial_params.min())
                    halfmax2 = halfmax2_amplitude2/2 + initial_params.min()

                    # Find the first index closest to halfmax
                    halfmax2_index1 = self.find_nearest(initial_params, halfmax2)

                    # Remove that point to find the next closest index, ensuring that it is on the other side of the curve
                    temporary2 = torch.cat([initial_params[0:halfmax2_index1], initial_params[halfmax2_index1+1:]])
                    halfmax2_index2 = self.find_nearest(temporary2, halfmax2)

                    # While loop to prevent choosing 2 halfmax indices next to each other (check -20 and -5 to prevent edge cases) 
                    add = 0
                    while self.xs[halfmax2_index1 - 22] <= self.xs[halfmax2_index2 - 20] <= self.xs[halfmax2_index1 - 18] or self.xs[halfmax2_index1 - 7] <= self.xs[halfmax2_index2 - 5] <= self.xs[halfmax2_index1 - 3]:
                        temporary2 = torch.cat([temporary2[0:halfmax2_index2], temporary2[halfmax2_index2+1:]])
                        halfmax2_index2 = self.find_nearest(temporary2, halfmax2)
                        add += 1

                    # Add the indices taken away back in if needed to get 2 correct indices at halfmax
                    if halfmax2_index1 < halfmax2_index2:
                        halfmax2_index2 = halfmax2_index2 + 1 + add

                    # Find closest difference between the two points at halfmax to calculate bandwidth
                    if curve[self.xs[halfmax2_index1 - 1]] < curve[self.xs[halfmax2_index1]] < curve[self.xs[(halfmax2_index1 + 1) % self.tuning_curve_sample]] and curve[self.xs[halfmax2_index2 - 1]] > curve[self.xs[halfmax2_index2]] > curve[self.xs[(halfmax2_index2 + 1) % self.tuning_curve_sample]] and halfmax2_index1<halfmax2_index2:
                        bandwidth2 = x[halfmax2_index2] - x[halfmax2_index1]
                    elif curve[self.xs[halfmax2_index2 - 1]] < curve[self.xs[halfmax2_index2]] < curve[self.xs[(halfmax2_index2 + 1) % self.tuning_curve_sample]] and curve[self.xs[halfmax2_index1 - 1]] > curve[self.xs[halfmax2_index1]] > curve[self.xs[(halfmax2_index1 + 1) % self.tuning_curve_sample]] and halfmax2_index2 < halfmax2_index1:
                        bandwidth2 = x[halfmax2_index1] - x[halfmax2_index2]
                    else:
                        bandwidth2 = ((b-a)*180/np.pi) - np.abs(x[halfmax2_index1] - x[halfmax2_index2])
                    self.before_bandwidths.append(bandwidth2/2)
                
        # Calculate mean and standard deviation of the ampltiude and bandwidth of each v1 curves before and after training
        self.v1_mean_after_amplitude = np.mean(self.after_amplitudes)
        self.v1_std_after_amplitude = np.std(self.after_amplitudes)

        self.v1_mean_after_bandwidth = np.mean(self.after_bandwidths)
        self.v1_std_after_bandwidth = np.std(self.after_bandwidths)

        self.v1_mean_before_amplitude = np.mean(self.before_amplitudes)
        self.v1_std_before_amplitude = np.std(self.before_amplitudes)

        self.v1_mean_before_bandwidth = np.mean(self.before_bandwidths)
        self.v1_std_before_bandwidth = np.std(self.before_bandwidths)
        
        self.v1_mean_after_baseline = np.mean(self.after_baselines)
        self.v1_mean_before_baseline = np.mean(self.before_baselines)

        # Calculate percentage differences
        self.v1_amplitude_difference = ((self.v1_mean_after_amplitude - self.v1_mean_before_amplitude) / self.v1_mean_before_amplitude) * 100      
        self.v1_bandwidth_difference = ((self.v1_mean_after_bandwidth - self.v1_mean_before_bandwidth) / self.v1_mean_before_bandwidth) * 100
        
        
    def v4_tuning_params(self, position):
        
        """
        Calculates the amplitude and bandwidth of V4 tuning curves. Takes position of gabor filter as argument.  
        """
        
        # Ensure position is between 0 and v4_dimensions
        if not 0 <= position <= self.v4_dimensions - 1:
            return("position needs to be between 0 and " + str(self.v4_dimensions - 1))
        
        # Create list of angles between -pi/2 and pi/2
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = (x * 180) / np.pi
        self.after_amplitudes = []
        self.after_bandwidths = []
        self.after_baselines = []
        self.v4_after_pos = []
        
        self.before_amplitudes = []
        self.before_bandwidths = []
        self.before_baselines = []
        self.v4_before_pos = []
        
        angles = np.linspace(-np.pi/2 + np.pi/(2 * self.v4_orientation_number), np.pi/2 - np.pi/(2 * self.v4_orientation_number), self.v4_orientation_number)
        threshold1 = self.angle1 - np.pi/4
        threshold2 = self.angle2 + np.pi/4
        
        self.xs = [i for i in range(self.tuning_curve_sample)]
        for i in range(self.v4_orientation_number):
            
            # Only calculate parameters between trained angle ± 45°
            if not threshold1 <= angles[i] <= threshold2:
                continue
            
            # Calculate tuning curve parameters after training
                
            # Set tuning curve at particular orientation and position after training
            curve = self.v4_results[i, position, position, :]
            
            # Measure baseline using min of tuning curve
            baseline = curve.min()
            self.after_baselines.append(baseline)

            # Measure preferred orientation of tuning curve
            after_preferred_orientation = curve.argmax().item()   
            self.v4_after_pos.append(x[after_preferred_orientation])
            
            # Measure amplitude using max and min of tuning curve
            amplitude = curve.max() - curve.min()
            self.after_amplitudes.append(amplitude)

            # Find amplitude of half the difference between max and min
            halfmax_amplitude = torch.abs(curve.max()) - torch.abs(curve.min())
            halfmax = halfmax_amplitude/2 + curve.min()
            
            # Find the first index closest to halfmax
            halfmax_index1 = self.find_nearest(curve, halfmax)
            
            # Remove that point to find the next closest index, ensuring that it is on the other side of the curve
            temporary = torch.cat([curve[0:halfmax_index1], curve[halfmax_index1+1:]])
            halfmax_index2 = self.find_nearest(temporary, halfmax)

            # While loop to prevent choosing 2 halfmax indices next to each other (check -20 -5 to prevent edge case)
            add = 0
            while self.xs[halfmax_index1 - 22] <= self.xs[halfmax_index2 - 20] <= self.xs[halfmax_index1 - 18] or self.xs[halfmax_index1 - 7] <= self.xs[halfmax_index2 - 5] <= self.xs[halfmax_index1 - 3]:
                temporary = torch.cat([temporary[0:halfmax_index2], temporary[halfmax_index2+1:]])
                halfmax_index2 = self.find_nearest(temporary, halfmax)
                add += 1
            
            # Add the indices taken away back in if needed to get 2 correct indices at halfmax
            if halfmax_index1 < halfmax_index2:
                halfmax_index2 = halfmax_index2 + 1 + add
            
            
            # Find closest difference between the two points at halfmax to calculate bandwidth
            if curve[self.xs[halfmax_index1 - 1]] < curve[self.xs[halfmax_index1]] < curve[self.xs[(halfmax_index1 + 1) % self.tuning_curve_sample]] and curve[self.xs[halfmax_index2 - 1]] > curve[self.xs[halfmax_index2]] > curve[self.xs[(halfmax_index2 + 1) % self.tuning_curve_sample]] and halfmax_index1<halfmax_index2:
                bandwidth = x[halfmax_index2] - x[halfmax_index1]
            elif curve[self.xs[halfmax_index2 - 1]] < curve[self.xs[halfmax_index2]] < curve[self.xs[(halfmax_index2 + 1) % self.tuning_curve_sample]] and curve[self.xs[halfmax_index1 - 1]] > curve[self.xs[halfmax_index1]] > curve[self.xs[(halfmax_index1 + 1) % self.tuning_curve_sample]] and halfmax_index2 < halfmax_index1:
                bandwidth = x[halfmax_index1] - x[halfmax_index2]
            else:
                bandwidth = 180 - np.abs(x[halfmax_index1] - x[halfmax_index2])
            self.after_bandwidths.append(bandwidth/2)

            # Set tuning curve at particular orientation and position before training
            initial_params = self.v4_initial_tuning_curves[i, position, position, :]
            
            # Measure baseline using min of tuning curve
            baseline = initial_params.min()
            self.before_baselines.append(baseline)

            # Measure preferred orientation of tuning curve
            before_preferred_orientation = initial_params.argmax().item()   
            self.v4_before_pos.append(x[before_preferred_orientation])
            
            # Measure amplitude using max and min of tuning curve
            amplitude2 = initial_params.max() - initial_params.min()
            self.before_amplitudes.append(amplitude2)

            # Find amplitude of half the difference between max and min
            halfmax2_amplitude2 = torch.abs(initial_params.max()) - torch.abs(initial_params.min())
            halfmax2 = halfmax2_amplitude2/2 + initial_params.min()
            
            # Find the first index closest to halfmax
            halfmax2_index1 = self.find_nearest(initial_params, halfmax2)
            
            # Remove that point to find the next closest index, ensuring that it is on the other side of the curv
            temporary2 = torch.cat([initial_params[0:halfmax2_index1], initial_params[halfmax2_index1+1:]])
            halfmax2_index2 = self.find_nearest(temporary2, halfmax2)

            # While loop to prevent choosing 2 halfmax indices next to each other (check -20 and -5 to prevent edge case)
            add = 0
            while self.xs[halfmax2_index1 - 22] <= self.xs[halfmax2_index2 - 20] <= self.xs[halfmax2_index1 - 18] or self.xs[halfmax2_index1 - 7] <= self.xs[halfmax2_index2 - 5] <= self.xs[halfmax2_index1 - 3]:
                temporary2 = torch.cat([temporary2[0:halfmax2_index2], temporary2[halfmax2_index2+1:]])
                halfmax2_index2 = self.find_nearest(temporary2, halfmax2)
                add += 1
            
            # Add the indices taken away back in if needed to get 2 correct indices at halfmax
            if halfmax2_index1 < halfmax2_index2:
                halfmax2_index2 = halfmax2_index2 + 1 + add
            
            # Find closest difference between the two points at halfmax to calculate bandwidth
            if curve[self.xs[halfmax2_index1 - 1]] < curve[self.xs[halfmax2_index1]] < curve[self.xs[(halfmax2_index1 + 1) % self.tuning_curve_sample]] and curve[self.xs[halfmax2_index2 - 1]] > curve[self.xs[halfmax2_index2]] > curve[self.xs[(halfmax2_index2 + 1) % self.tuning_curve_sample]] and halfmax2_index1<halfmax2_index2:
                bandwidth2 = x[halfmax2_index2] - x[halfmax2_index1]
            elif curve[self.xs[halfmax2_index2 - 1]] < curve[self.xs[halfmax2_index2]] < curve[self.xs[(halfmax2_index2 + 1) % self.tuning_curve_sample]] and curve[self.xs[halfmax2_index1 - 1]] > curve[self.xs[halfmax2_index1]] > curve[self.xs[(halfmax2_index1 + 1) % self.tuning_curve_sample]] and halfmax2_index2 < halfmax2_index1:
                bandwidth2 = x[halfmax2_index1] - x[halfmax2_index2]
            else:
                bandwidth2 = 180 - np.abs(x[halfmax2_index1] - x[halfmax2_index2])
            self.before_bandwidths.append(bandwidth2/2)
        
        # Calculate mean and standard deviation of the ampltiude and bandwidth of each v4 curves before and after training
        self.v4_mean_after_amplitude = np.mean(self.after_amplitudes)
        self.v4_std_after_amplitude = np.std(self.after_amplitudes)

        self.v4_mean_after_bandwidth = np.mean(self.after_bandwidths)
        self.v4_std_after_bandwidth = np.std(self.after_bandwidths)

        self.v4_mean_before_amplitude = np.mean(self.before_amplitudes)
        self.v4_std_before_amplitude = np.std(self.before_amplitudes)

        self.v4_mean_before_bandwidth = np.mean(self.before_bandwidths)
        self.v4_std_before_bandwidth = np.std(self.before_bandwidths)
        
        self.v4_mean_after_baseline = np.mean(self.after_baselines)
        self.v4_mean_before_baseline = np.mean(self.before_baselines)

        # Calculate percentage differences
        self.v4_amplitude_difference = ((self.v4_mean_after_amplitude - self.v4_mean_before_amplitude) / self.v4_mean_before_amplitude) * 100      
        self.v4_bandwidth_difference = ((self.v4_mean_after_bandwidth - self.v4_mean_before_bandwidth) / self.v4_mean_before_bandwidth) * 100
    
    def otc_curve(self, v1_position_1, v1_position_2, v4_position_1, v4_position_2):
        
        """
        Calculates slope of each tuning curve before and after training at the trained angle. Takes in trained angle, V1 position 
        and V4 position as arguments. 
        """
        
        # Ensure v1_position is between 0 and v4_dimensions
        if not 0 <= v1_position_1 <= self.v1_dimensions - 1 or not 0 <= v1_position_2 <= self.v1_dimensions - 1:
            return("position needs to be between 0 and " + str(self.v4_dimensions - 1))
        
        # Ensure v4_position is between 0 and v4_dimensions
        if not 0 <= v4_position_1 <= self.v4_dimensions - 1 or not 0 <= v4_position_2 <= self.v4_dimensions - 1:
            return("position needs to be between 0 and " + str(self.v4_dimensions - 1))
        

        
        trained_angle1 = self.angle1 * 180 / np.pi
        trained_angle2 = self.angle2 * 180 / np.pi
        
        self.v1_mean_after_slopes = []
        self.v1_mean_before_slopes = []
        
        self.v1_after_range = []
        self.v1_before_range = []
        
        for i in range(self.v1_orientation_number):
            
            after_slopes = []
            before_slopes = []
            after_ranges = []
            before_ranges = []

            for sf in range(self.sfs):
#                 if sf == 1:
#                     # Skipping this because these are broadly tuned
#                     continue
                for j in range(self.phis):

                    # Set V1 tuning curve at particular orientation, phase/sf and position after and before training
                    curve = self.results[i, sf, j, v1_position_1, v1_position_2, :]
                    initial = self.initial_tuning_curves[i, sf, j, v1_position_1, v1_position_2, :]

                    # Create list of angles depending on which phase the unit is selctive for
                    if j == 0 or j == self.phis - 1:
                        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
                        x = x * 180 / np.pi
                    else:
                        x = np.linspace(-np.pi/2, 3*np.pi/2, self.tuning_curve_sample)
                        x = x * 180 / np.pi


                    # Find index closest to trained angle
                    trained_index1 = self.find_nearest(torch.tensor(x), trained_angle1)
                    trained_index2 = self.find_nearest(torch.tensor(x), trained_angle2)

                    # Find index of preferred orientation of curve
                    after_preferred_orientation = curve.argmax().item()   
                    before_preferred_orientation = initial.argmax().item()

                    # Save differences between preferred orientation and trained angle and save slope at trained angle – use negative trained angle if preferred orientation is negative and vice versa

                    if x[after_preferred_orientation] <= 0:
                        # Calculate difference between preferred orientation and trained angle
                        after_ranges.append(x[after_preferred_orientation] - x[trained_index1])
                        # Calculate slope at trained angle
                        after_slope = (curve[trained_index1 + 1] - curve[trained_index1 - 1])/(x[trained_index1 + 1] - x[trained_index1 - 1])


                    else:
                        # Calculate difference between preferred orientation and trained angle
                        after_ranges.append(x[after_preferred_orientation] - x[trained_index2])
                        # Calculate slope at trained angle
                        after_slope = (curve[trained_index2 + 1] - curve[trained_index2 - 1])/(x[trained_index2 + 1] - x[trained_index2 - 1])


                    if x[before_preferred_orientation] <= 0:    
                        # Calculate difference between preferred orientation and trained angle
                        before_ranges.append(x[before_preferred_orientation] - x[trained_index1])

                        # Calculate slope at trained angle
                        before_slope = (initial[trained_index1 + 1] - initial[trained_index1 - 1])/(x[trained_index1 + 1] - x[trained_index1 - 1])

                    else:
                        # Calculate difference between preferred orientation and trained angle
                        before_ranges.append(x[before_preferred_orientation] - x[trained_index2])

                        # Calculate slope at trained angle
                        before_slope = (initial[trained_index2 + 1] - initial[trained_index2 - 1])/(x[trained_index2 + 1] - x[trained_index2 - 1])

                    after_slopes.append(torch.abs(after_slope).item())
                    before_slopes.append(torch.abs(before_slope).item())

                
            # Calculate mean difference between preferred orientation and trained angle of all tuning curves selective for particular orientation but different phase/sfs
            mean_after_range = np.mean(after_ranges)
            mean_before_range = np.mean(before_ranges)
            
            self.v1_after_range.append(mean_after_range)
            self.v1_before_range.append(mean_before_range)

            # Calculate mean slope at trained angle of all tuning curves selective for particular orientation but different phase/sfs
            mean_after_slope = np.mean(after_slopes)
            mean_before_slope = np.mean(before_slopes)
            
            self.v1_mean_after_slopes.append(mean_after_slope)
            self.v1_mean_before_slopes.append(mean_before_slope)
            
        self.v1_after_range, self.v1_mean_after_slopes = self.sort_lists(self.v1_after_range, self.v1_mean_after_slopes)
        self.v1_before_range, self.v1_mean_before_slopes = self.sort_lists(self.v1_before_range, self.v1_mean_before_slopes)
        
        self.v4_after_slopes = []
        self.v4_before_slopes = []
        
        self.v4_after_range = []
        self.v4_before_range = []
        
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = x * 180 / np.pi
        
        # Find index closest to trained angle
        trained_index1 = self.find_nearest(torch.tensor(x), trained_angle1)
        trained_index2 = self.find_nearest(torch.tensor(x), trained_angle2)
        
        for k in range(self.v4_orientation_number):
            
            
            # Set V4 tuning curve at particular orientation and position after and before training
            curve = self.v4_results[k, v4_position_1, v4_position_2, :]
            initial = self.v4_initial_tuning_curves[k, v4_position_1, v4_position_2, :]
            
            # Find index of preferred orientation of curve
            after_preferred_orientation = curve.argmax().item()
            before_preferred_orientation = initial.argmax().item()
            
            # Save differences between preferred orientation and trained angle and save slope at trained angle – use negative trained angle if preferred orientation is negative and vice versa
            if x[after_preferred_orientation] <= 0:
                    
                # Calculate difference between preferred orientation and trained angle
                self.v4_after_range.append(x[after_preferred_orientation] - x[trained_index1])
                
                # Calculate slope at trained angle
                after_slope = (curve[trained_index1 + 1] - curve[trained_index1 - 1])/(x[trained_index1 + 1] - x[trained_index1 - 1])

            else:
                # Calculate difference between preferred orientation and trained angle
                self.v4_after_range.append(x[after_preferred_orientation] - x[trained_index2])
                

                # Calculate slope at trained angle
                after_slope = (curve[trained_index2 + 1] - curve[trained_index2 - 1])/(x[trained_index2 + 1] - x[trained_index2 - 1])
                
                
            if x[before_preferred_orientation] <= 0:
                
                # Calculate difference between preferred orientation and trained angle
                self.v4_before_range.append(x[before_preferred_orientation] - x[trained_index1])
                
                
                # Calculate slope at trained angle
                before_slope = (initial[trained_index1 + 1] - initial[trained_index1 - 1])/(x[trained_index1 + 1] - x[trained_index1 - 1])
                
            else:
                # Calculate difference between preferred orientation and trained angle
                self.v4_before_range.append(x[before_preferred_orientation] - x[trained_index2])
                
                # Calculate slope at trained angle
                before_slope = (initial[trained_index2 + 1] - initial[trained_index2 - 1])/(x[trained_index2 + 1] - x[trained_index2 - 1])
                
            self.v4_after_slopes.append(torch.abs(after_slope).item())
            self.v4_before_slopes.append(torch.abs(before_slope).item())
            
        # Sort in ascending order
        self.v4_after_range, self.v4_after_slopes = self.sort_lists(self.v4_after_range, self.v4_after_slopes)
        self.v4_before_range, self.v4_before_slopes = self.sort_lists(self.v4_before_range, self.v4_before_slopes)
        
        self.v1_mean_after_slope = np.mean(self.v1_mean_after_slopes)
        self.v1_mean_before_slope = np.mean(self.v1_mean_before_slopes)
        self.v4_mean_after_slope = np.mean(self.v4_after_slopes)
        self.v4_mean_before_slope = np.mean(self.v4_before_slopes)
        
        # Calculate slope differences before and after training
        self.v1_diff = [self.v1_mean_after_slopes[i] - self.v1_mean_before_slopes[i] for i in range(self.v1_orientation_number)]
        self.v4_diff = [self.v4_after_slopes[i] - self.v4_before_slopes[i] for i in range(self.v4_orientation_number)]
        
        self.v1_max_diff = np.max(self.v1_mean_after_slope) - np.max(self.v1_mean_before_slope)
        self.v4_max_diff = np.max(self.v4_mean_after_slope) - np.max(self.v4_mean_before_slope)
        
        v1_linspace = np.linspace(-np.pi/2, np.pi/2, self.v1_orientation_number)
        v1_linspace = v1_linspace * 180/np.pi
        v4_linspace = np.linspace(-np.pi/2, np.pi/2, self.v4_orientation_number)
        v4_linspace = v4_linspace * 180/np.pi
        self.v1_max_diff_angle = np.abs(v1_linspace[np.argmax(self.v1_diff)])
        self.v4_max_diff_angle = np.abs(v4_linspace[np.argmax(self.v4_diff)])
        
        # Check asymmetry – if range distribution is different or if peaks are more than 5% different or if there are sharp fluctuations in slope or if difference in curve magnitude is too much
        if np.abs(self.v1_after_range[0] - self.v1_before_range[0]) >= 10 or np.abs(self.v1_after_range[-1] - self.v1_before_range[-1]) >= 10 or 0.05 * np.max(self.v1_mean_after_slopes) < np.abs(np.max(self.v1_mean_after_slopes[:int(len(self.v1_mean_after_slopes)/2)]) - np.max(self.v1_mean_after_slopes[int(len(self.v1_mean_after_slopes)/2):])):
            self.v1_binary_heatmap = 0

        else:
            self.v1_binary_heatmap = 1
            
        if np.abs(self.v4_after_range[0] - self.v4_before_range[0]) >= 10 or 0.05 * np.max(self.v4_after_slopes) < np.abs(np.max(self.v4_after_slopes[:int(len(self.v4_after_slopes)/2)]) - np.max(self.v4_after_slopes[int(len(self.v4_after_slopes)/2):])):
            self.v4_binary_heatmap = 0
        else:
            self.v4_binary_heatmap = 1
        
        for i in range(1,len(self.v1_mean_after_slopes) - 2):
            if self.v1_mean_after_slopes[i-1] < self.v1_mean_after_slopes[i] and self.v1_mean_after_slopes[i] > self.v1_mean_after_slopes[i+1] and self.v1_mean_after_slopes[i+1] < self.v1_mean_after_slopes[i+2]:
                self.v1_binary_heatmap = 0
            elif self.v1_mean_after_slopes[i-1] > self.v1_mean_after_slopes[i] and self.v1_mean_after_slopes[i] < self.v1_mean_after_slopes[i+1] and self.v1_mean_after_slopes[i+1] > self.v1_mean_after_slopes[i+2]:
                self.v1_binary_heatmap = 0
        for i in range(1,len(self.v4_after_slopes) - 2):
            if self.v4_after_slopes[i-1] < self.v4_after_slopes[i] and self.v4_after_slopes[i] > self.v4_after_slopes[i+1] and self.v4_after_slopes[i+1] < self.v4_after_slopes[i+2]:
                self.v4_binary_heatmap = 0
            elif self.v4_after_slopes[i-1] > self.v4_after_slopes[i] and self.v4_after_slopes[i] < self.v4_after_slopes[i+1] and self.v4_after_slopes[i+1] > self.v4_after_slopes[i+2]:
                self.v4_binary_heatmap = 0
                
        if np.max(self.v1_mean_before_slopes) > 10 * np.max(self.v4_before_slopes):
            self.v4_binary_heatmap = 0
                
    def plot_otc_curve(self):
        
        """
        Plots slopes of orientation tuning curves against difference between trained angle and preferred angle.  
        """
       
        plt.plot(self.v1_before_range, self.v1_mean_before_slopes)
        plt.plot(self.v1_after_range, self.v1_mean_after_slopes)
        
        plt.plot(self.v4_before_range, self.v4_before_slopes)
        plt.plot(self.v4_after_range, self.v4_after_slopes)
        
        plt.legend(["V1 pre", "V1 post", "V4 pre", "V4 post"])
        plt.xlabel("Preferred orientation - trained orientation (Degrees)")
        plt.ylabel("Slope at Trained Orientation (Change/Degrees)")

        
    def plot_otc_curve_diff(self, absolute_diff = False):
        
        """
        Plots difference between slopes of orientation tuning curves against absolute difference between trained angle and 
        preferred angle before and after training.  
        """
        
        if absolute_diff == True:
            # Calculate absolute difference between trained angle and preferred angle and sort lists
            self.v1_diff_range = np.abs(self.v1_after_range)
            self.v1_diff_range, self.v1_diff = self.sort_lists(self.v1_diff_range, self.v1_diff)

            self.v4_diff_range = np.abs(self.v4_after_range)
            self.v4_diff_range, self.v4_diff = self.sort_lists(self.v4_diff_range, self.v4_diff)
        
            # Plot differences
            plt.plot(self.v1_diff_range, self.v1_diff)
            plt.plot(self.v4_diff_range, self.v4_diff)
            
            plt.xlabel("Absolute difference between preferred orientation and trained orientation (Degrees)")
            
        else:
            # Plot differences
            plt.plot(self.v1_before_range, self.v1_diff)
            plt.plot(self.v4_before_range, self.v4_diff)
            
            plt.xlabel("Difference between preferred orientation and trained orientation (Degrees)")
        
        plt.legend(["V1 difference", "V4 difference"])
        
        plt.ylabel("Difference between slope at trained orientation before and after training (Change/Degrees)")
     
    
    def transfer_test(self, x_location, y_location):
        
        """
        Creates test gabors at new location using the same angle trained on and calculates performance of network. Takes in x and 
        y coordinates of gabor location as arguments.
        """
       
        # Ensure locations are within v1_dimensions
        if not 0 <= x_location <= self.v1_dimensions - 1 or not 0 <= y_location <= self.v1_dimensions - 1:
            return("x_location and y_location needs to be between 0 and " + str(self.v1_dimensions - 1))
        
        
        transfer_score = 0
        self.kernels = []
        
        # For each orientation, create input gabor stimulus at that orientation at particular location
        losses = []
#         phases = np.linspace(0, np.pi, self.training_size)
        phases = self.trained_phis
        sfs = self.trained_sfs
        for j in range(self.training_size):
            if j < 10:
                angle = self.angle1

            else: 
                angle = self.angle2
            kernel = self.generate_location_gabor(angle, phases[j // 10], sfs[j // 10], x_location, y_location)
            self.kernels.append(kernel)
        # Label stimulus as 0 if it is clockwise to reference orientation of 0 and 1 if counterclockwise
            if 0 < angle < np.pi/2: 
                label = torch.tensor([0])
            else:
                label = torch.tensor([1])

            with torch.no_grad():
                pred = self.forward(kernel)
                if torch.argmax(pred) == label:
                    transfer_score += 1
                loss = self.binary_loss(pred, label) 
                losses.append(float(loss))
        
        # Calculate transfer performance and error
        self.transfer_score = (transfer_score/self.training_size) * 100
        self.transfer_error = np.mean(losses)
        
        return self.transfer_score, self.transfer_error
    
    def plot_transfer_score(self, performance = True, grid = False):
        
        """
        Calculates transfer performance and error at every input location by testing the 2 angles with phase_number of phases. 
        Setting performance = True plots transfer performance and performance = False plots error performance. Setting grid = 
        True plots transfer performance/error on a grid and grid = False plots transfer performance/error against distance 
        between transfer location and trained location. 
        """
        
        scores = []
        errors = []
        distances = []
        
        self.grid_score = torch.empty(self.v1_dimensions, self.v1_dimensions)
        self.grid_error = torch.empty(self.v1_dimensions, self.v1_dimensions)
        
        # For each location on x and y axis, calculate transfer performance and error
        for i in tqdm(range(self.v1_dimensions)):
            for j in range(self.v1_dimensions):
                score, error = self.transfer_test(i, j)
                scores.append(score)
                errors.append(error)

                # Calculate distance between transfer location and trained location
                distance = np.sqrt(((i - self.train_x_location) ** 2) + ((j - self.train_y_location) ** 2))
                distances.append(distance)
                
                self.grid_score[i][j] = score
                self.grid_error[i][j] = torch.tensor(error)
                
        # Plot results
        if performance == True and grid == True:
            plt.imshow(self.grid_score, cmap = 'PiYG', vmin = 0, vmax = 100)
            plt.colorbar(label = 'Performance (%)')
            plt.title("Heat map of performance at different untrained locations")
        
        if performance == True and grid == False:
            distances, scores = self.sort_lists(distances, scores)
            plt.plot(distances, scores)
            plt.ylim(-5, 105)
            plt.xlabel("Distance from trained angle")
            plt.ylabel("Performance (%)")
            plt.title("Performance on untrained locations")
        
        if performance == False and grid == True:
            plt.imshow(self.grid_error, cmap = 'PiYG', vmin = 0, vmax = 1)
            plt.colorbar(label = 'error')
            plt.title("Heat map of error at different untrained locations")
        
        if performance == False and grid == False:
            distances, errors = self.sort_lists(distances, errors)
            plt.plot(distances, errors)
            plt.ylim(-5, 105)
            plt.xlabel("Distance from trained angle")
            plt.ylabel("Error")
            plt.title("Error on untrained locations")
    
    # Helper functions
    
    def generate_gabor(self, size, theta, phi, lamda, gamma = 0.5):
        
        """
        Generates one gabor matrix. Takes in gabor size, angle, phase, spatial frequency as arguments and returns a torch
        tensor gabor matrix.
        """
        
        ksize = size # Size of gabor filter
        sigma = 1.6 # Standard deviation of Gaussian envelope
        self.lamda = lamda
        gamma = gamma # Spatial aspect ratio and ellipticity 
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, self.lamda, gamma, phi)
        return torch.tensor(kernel).view(1, 1, size, size).float()
    
    def generate_location_gabor(self, theta, phi, lamda, x_location, y_location):
        
        """
        Generates one gabor matrix in a specific location. Takes in angle, phase, spatial frequency, x and y coordinates as 
        arguments and returns a torch tensor gabor matrix.
        """
        
        # Generate gabor and noise
        kernel = self.generate_gabor(self.v1_size, theta, phi, lamda).view(self.v1_size, self.v1_size)
#         kernel_noise = torch.normal(0, 0.03, (self.input_size, self.input_size))
        kernel_noise = torch.zeros(self.input_size, self.input_size)
        
        # Add gabor to noise at particular location
        for i in range(self.v1_size):
            for j in range(self.v1_size):
                kernel_noise[i + y_location][j + x_location] = kernel[i][j] + kernel_noise[i + y_location][j + x_location]
        return kernel_noise.view(1, 1, self.input_size, self.input_size)
    
    def generate_gaussian(self, mean, std, kernlen):
        
        """
        Generates a gaussian list. Takes in mean, length of list and standard deviation and returns a list. 
        """
        
        n = torch.linspace(-np.pi/2, np.pi/2, kernlen) - mean
        sig2 = 2 * std * std
        w = torch.exp(-n ** 2 / sig2)
        return w
    
    def generate_3d_gaussian(self, mean, spatial_std, orientation_std, roll):
        
        """
        Generates a v1_orientation_number x v4_size x v4_size 3D gaussian. Takes in mean and standard deviation of 
        orientation gaussian and returns a 3D torch tensor.
        """
        # Create 1D gaussian
        gkern1d = self.generate_gaussian(0, spatial_std, self.v4_size) 
        
        # Compile 2 1D gaussians to form 2D gaussian for spatial pooling
        gkern2d = torch.outer(gkern1d, gkern1d) 
        
        # Clone 2D gaussians by number of V1 orientations and stack them
        twod = (gkern2d, ) * self.v1_orientation_number 
        twods = torch.stack(twod) 
        
        # Scaling each 2D gaussian at different orientations for orientation pooling using 1D gaussian
        
        # Generate a 1D gaussian and roll the x-axis to change the peak
        scale = self.generate_gaussian(mean, orientation_std, self.v1_orientation_number) 
        a, b = self.sort_lists(np.roll(self.v1_angles, roll), scale)
        
        scales = torch.empty(self.v1_orientation_number, self.v4_size, self.v4_size) 
        
        # Multiply each 2D gaussian by the orientation gaussian scalar
        for i in range(self.v1_orientation_number):
            scales[i] = twods[i] * b[i]
        # Return 3D gaussian with gaussian weighted 2D gaussians to perform spatial and orientation pooling
        return scales 
            
    
    def remove_ambiguous_stimuli(self, angle1, angle2, size, even_space = False):
        
        """
        Function returns a numpy list of angles that are unambiguously clockwise or anticlockwise relative to 0°. 
        Ambiguous angles include -pi/2, 0, pi/2 and pi which are not clockwise or anticlockwise relative to 0°.
        A constant is either added or subtracted from these angles to make them unambiguous. Takes in the two angles 
        and the number of angles as arguments. Setting even_space = True returns a numpy list of angles that are evenly 
        spaced across 180° by removing the last element in the list.
        """
        
        if even_space == True:
            x = np.linspace(angle1, angle2, size+1)
            x = np.delete(x, -1) # Remove last element in list so that all elements are evenly spaced
            ambiguous = True
            
            # Add 0.1 to -pi/2 or 0 and subtract 0.5 to pi/2 or pi and repeat until no more ambiguous stimuli in list
            while ambiguous == True: 
                for i in self.ambiguous[0:2]:
                    x = np.where(x == i, i+0.01, x)
                for i in self.ambiguous[2:4]:
                    x = np.where(x == i, i-0.01, x)            
                x = np.linspace(x[0], x[-1], size)
                if -np.pi/2 in x or 0 in x or np.pi/2 in x:
                    ambiguous = True
                else:
                    ambiguous = False

        if even_space == False:
            x = np.linspace(angle1, angle2, size)
            ambiguous = True
            
            # Add 0.1 to -pi/2 or 0 and subtract 0.5 to pi/2 or pi and repeat until no more ambiguous stimuli in list
            while ambiguous == True: 
                for i in self.ambiguous[0:2]:
                    x = np.where(x == i, i+0.01, x)
                for i in self.ambiguous[2:4]:
                    x = np.where(x == i, i-0.01, x)            
                x = np.linspace(x[0], x[-1], size)
                if -np.pi/2 in x or 0 in x or np.pi/2 in x:
                    ambiguous = True
                else:
                    ambiguous = False
        return x
    
    
    def find_nearest(self, tensor, value):
        
        """
        Finds index closest to point on curve. Takes in curve as tensor and scalar value and returns index of point on curve 
        closest to the scalar value. 
        """
        
        idx = (torch.abs(tensor - value)).argmin()
        return idx.item()
    
    def sort_lists(self, list1, list2):
        
        """
        Takes in 2 lists as arguments and sorts them in parallel based on first list in ascending order. Returns tuple of sorted 
        lists. 
        """
        
        zipped_lists = zip(list1, list2)
        sorted_pairs = sorted(zipped_lists)
        tuples = zip(*sorted_pairs)
        list1, list2 = [list(tuple) for tuple in tuples] 
        return list1, list2
