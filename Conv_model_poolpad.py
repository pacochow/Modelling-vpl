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

class convnet(nn.Module):
    
    def __init__(self, input_size, v1_size, v1_orientation_number, v4_size, v4_stride, v4_orientation_number, phis_sfs, 
                 training_size, phis = True, sfs = False, alpha = 0.01, v1_rescale = 1, phase_rescale = 1, v4_rescale = 1,
                 v1_gamma = 0.5, v4_orientation_std = 0.7):
        
        """
        Initialize network parameters.
        """
        
        super(convnet, self).__init__()
        
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
        self.v1_padding = int((self.v1_size - 1) / 2)
        self.phis_sfs = phis_sfs # Number of V1 gabor filters at different phases/sfs depending on phis = True or sfs = True
        self.phis = phis # Boolean – True if pooling over phase
        self.sfs = sfs # Boolean – True if pooling over sf
        self.v1_rescale = v1_rescale # scalar multiplier for scaling of V1 weights, defaults to 1 (no scaling)
        self.phase_rescale = phase_rescale # scalar multiplier for scaling of phase pooling, defaults to 1 (no scaling)
        self.v4_rescale = v4_rescale # scalar multiplier for scaling of V4 weights, defaults to 1 (no scaling)
        self.v1_gamma = v1_gamma
        self.v4_orientation_std = v4_orientation_std
        
        self.training_size = training_size # Batch size
        self.alpha = alpha # Learning rate
        self.v1_dimensions = self.input_size - self.v1_size + 2 * self.v1_padding + 1 # Dimension of activation map after V1 simple cell filtering
        
        self.v4_size = v4_size # Size of 2D V4 gaussian filter (pixels)
        self.v4_stride = v4_stride # V4 filter stride
        self.v4_padding = int((self.v4_size - 1) / 2)
        self.v4_orientation_number = v4_orientation_number # Number of V4 filters selective for different orientations
        self.v4_dimensions = int(((self.v1_dimensions - self.v4_size + (2 * self.v4_padding))/self.v4_stride) + 1) # Dimension of activation map after V4 filtering
        
        self.ambiguous = [-np.pi/2, 0, np.pi/2, np.pi] # List of ambiguous gabor angles that needs to be removed
        
        # Initialising V1 weights – learnable parameter 
        self.simple_weight = nn.Conv2d(1, self.v1_orientation_number * self.phis_sfs, self.v1_size, padding = self.v1_padding, bias = False)
        self.simple_weight.weight.data = self.init_weights()
        
        # Initialising V4 weights – learnable parameter
        self.v4_weight = nn.Conv3d(self.v1_orientation_number, self.v4_orientation_number, (self.v1_orientation_number, self.v4_size, self.v4_size), stride = self.v4_stride, padding = (0, self.v4_padding, self.v4_padding), bias = False)
        self.v4_weight.weight.data = self.init_gaussian_weights()
        
        # Initialising decision layer
        self.decision = nn.Linear(self.v4_orientation_number, 2)
        self.decision.weight = torch.nn.Parameter(torch.zeros((2, self.v4_orientation_number)))
        
        # Saving initial weights
        self.before_v1weight = self.simple_weight.weight.clone()
        self.simple_weight_before = nn.Conv2d(1, self.v1_orientation_number * self.phis_sfs, self.v1_size, padding = 5, bias = False)
        self.simple_weight_before.weight.data = self.before_v1weight
        
        self.before_v4weight = self.v4_weight.weight.clone()
        self.v4_weight_before = nn.Conv3d(self.v1_orientation_number, self.v4_orientation_number, (self.v1_orientation_number, self.v4_size, self.v4_size), stride = self.v4_stride, padding = (0, 5, 5), bias = False)
        self.v4_weight_before.weight.data = self.before_v4weight
        self.before_decision_weight = self.decision.weight.clone()
        
        
    # Network training functions
    
    def init_weights(self):
      
        """
        Initialize V1 simple cell weights with gabor filters at different orientations and different phases/spatial 
        frequencies, depending on what is chosen when initializing network. Returns torch tensor weights.
        """   
        
        # Create range of orientations between -pi/2 and pi/2 for each V1 gabor that are equally spaced and symmetrical around 0
        self.v1_angles = np.linspace(-np.pi/2 + np.pi/(2 * self.v1_orientation_number), np.pi/2 - np.pi/(2 * self.v1_orientation_number), self.v1_orientation_number)
        # Create range of phases between 0 and pi for each V1 gabor
        self.phis_sfs_range = np.linspace(0, np.pi, self.phis_sfs) 
        weights = []

         # For each orientation and phase, create gabor filter with those parameters
        for i in range(self.v1_orientation_number):
            for j in range(self.phis_sfs):
                theta = self.v1_angles[i]
                phi = self.phis_sfs_range[j]
                kernel = self.generate_gabor(self.v1_size, theta, phi, 5, gamma = self.v1_gamma)
                kernel = kernel * self.v1_rescale

                    # Add random noise from normal distribution 
#                 noise = torch.normal(0, 0.03, (self.v1_size, self.v1_size)) 
#                 kernel = kernel + noise 
                weights.append(kernel)

        # Return torch tensor weights for each V1 gabor filter
        weight = torch.stack(weights).view(
           self.v1_orientation_number * self.phis_sfs, 1, self.v1_size, self.v1_size) 
        return weight 
        
    def init_gaussian_weights(self):
        
        """
        Initialize V4 weights with gaussian filters. Returns torch tensor weights.
        """
        # Create range of orientations between -pi/2 and pi/2 for each V4 filter
        self.v4_angles = np.linspace(-np.pi/2 + np.pi/(2 * self.v4_orientation_number), np.pi/2 - np.pi/(2 * self.v4_orientation_number), self.v4_orientation_number)
        
        x = np.linspace(-np.pi/2, np.pi/2, self.v1_orientation_number)
        
        # For each V4 orientation, create 3D V4 gaussian filter that has dimensions v1_orientation_number x v4_dimensions x v4_dimensions
        v4 = torch.empty(self.v4_orientation_number, 1, self.v1_orientation_number, self.v4_size, self.v4_size) 
        
        for pool_orientation in range(self.v4_orientation_number):
            for orientation in range(self.v1_orientation_number):
                # Find index in v1_angles using desired v4_angle to initialise preferred orientation of V4 orientation tuning curve
                index = self.find_nearest(torch.tensor(x), self.v4_angles[pool_orientation])

                # Generate 3D gaussian filter and roll the orientation gaussian to change the peaks
                kernel = self.generate_3d_gaussian(mean = self.v1_angles[int(round(self.v1_orientation_number/2, 0))], spatial_std = 0.5, orientation_std = self.v4_orientation_std, roll = (int(round(self.v1_orientation_number/2, 0)) - index))[orientation] 
                kernel = kernel * self.v4_rescale

                # Add random noise from normal distribution scaled by mean of each gaussian filter so noise does not cover up gaussian
#                         noise = torch.normal(0, 0.015, (self.v4_size, self.v4_size)) * kernel.mean() 
#                         kernel = kernel + noise
                v4[pool_orientation][0][orientation] = kernel
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
        
        self.trained_phis = np.linspace(0, np.pi, 10) # Use phases regularly spaced across 0 to pi
        self.trained_sfs = []
    
        self.angle1 = angle1
        self.angle2 = angle2
        
        # For each orientation, create input gabor stimulus at that orientation with test_size/2 random phases
        for angle in [angle1, angle2]:
            for i in range(int(self.training_size/2)):
                theta = angle
                phi = self.trained_phis[i]
                kernel = self.generate_gabor(self.input_size, theta, phi, 5, random_sf = random_sf) 
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
        
        self.trained_phis = np.linspace(0, np.pi, 10) # Use phases regularly spaced across 0 to pi
        self.trained_sfs = []

        # For each orientation, create input gabor stimulus at that orientation at particular location
        for angle in [angle1, angle2]:
            for i in range(int(self.training_size/2)):
                theta = angle
                phi = self.trained_phis[i]
                kernel = self.generate_location_gabor(theta, phi, 5, x_location, y_location, random_sf = random_sf)
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
        
        # V1 simple cell convolution
        out = self.simple_weight(x)
        out.view(1, self.v1_orientation_number * self.phis_sfs, self.v1_dimensions, self.v1_dimensions)
    
        # V1 complex cell pooling over phase/sf
        pools = torch.zeros(1, 1, self.v1_orientation_number, self.v1_dimensions, self.v1_dimensions)
        
        for i in range(0, self.v1_orientation_number*self.phis_sfs, self.phis_sfs):
            # Apply relu activation function on all activation maps with same orientation but different phase/sfs 
            relu = F.relu(out[0][i:i+self.phis_sfs]) 
            
            # Sum all of these activation maps after relu 
#             pool = (torch.sum(relu, dim = 0) * self.phase_rescale / self.phis_sfs).view(1, self.v1_dimensions, self.v1_dimensions)
            pool = (torch.sum(relu, dim = 0) * self.phase_rescale / self.phis_sfs).view(1, self.input_size, self.input_size)
            pools[0][0][i//4] = pool
        
        # V4 cell convolution
        # out = self.v4_weight(pools).view(1, self.v4_orientation_number * self.v4_dimensions * self.v4_dimensions)
        out = self.v4_weight(pools).view(self.v4_orientation_number, self.v4_dimensions, self.v4_dimensions)
        
        # Spatial mean pool
        v4_spatial_pool = out.sum([1, 2]) / (self.v4_dimensions * self.v4_dimensions)
        v4_spatial_pool.view(self.v4_orientation_number)
        
        # Feed flattened activation maps into decision layer
        # out = self.decision(out)
        out = self.decision(v4_spatial_pool.view(1, self.v4_orientation_number))
        return out

    def train(self, iterations, optimizer):
        
        """
        Training loop. Takes in the number of iteractions and optimizer as arguments and trains the network over the 
        number of iteractions. Calculates mean loss over batch size and optimizes over this loss. 
        """

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
                self.scores = self.forward(self.input[j].view(1, 1, self.input_size, self.input_size))

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
            self.v1_weight_changes.append(self.v1_weight_change(self.before_v1weight, self.simple_weight.weight)) 
            self.v4_weight_changes.append(self.v4_weight_change(self.before_v4weight, self.v4_weight.weight)) 
            self.decision_weight_changes.append(self.decision_weight_change(self.before_decision_weight, self.decision.weight))

            # Backpropagate error and update weights
            loss.backward() 
            optimizer.step() 

            self.generalization(self.angle2, self.training_size)
            self.generalize_perform.append(self.generalization_score)
            self.generalize_error.append(self.general_mean_error)

            if i % 500 == 0 or i == iterations - 1:
                self.v1_tuning_curve()
                self.v4_tuning_curve()
                self.otc_curve()
                self.v1_otc_max_diffs.append(self.v1_max_diff)
                self.v4_otc_max_diffs.append(self.v4_max_diff)
                    

#         Generate tuning curves    
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
                
                kernel = self.generate_location_gabor(angle, phases[j], sfs[j], int((self.v1_dimensions - 1)/2), int((self.v1_dimensions - 1)/2), random_sf = False)
                
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
        
        # For each V1 simple cell gabor selective for different orientation, and phase, calculate difference before and after training
        diff = after - before
        net_diff = []
        
        # Calculate frobenius norm of each gabor difference and return mean magnitude of change
        for i in diff.view(self.v1_orientation_number*self.phis_sfs, self.v1_size, self.v1_size):
            net_diff.append(torch.linalg.norm(i, ord = 'fro').item())
        return np.mean(net_diff)
    
    def v4_weight_change(self, before, after):
        
        """
        Calculates frobenius norm of difference between weights in V4 cells and initial weights for each iteraction 
        of training. Takes in initial weights and current weights as arguments and returns scalar value.
        """
        
        # For each V4 gaussian filter selective for different orientations, calculate difference before and after training
        diff = after - before
        net_diff = []
        
        # Calculate frobenius norm of each filter difference and return mean magnitude of change
        for v4_orientation in diff.view(self.v4_orientation_number, self.v1_orientation_number, self.v4_size, self.v4_size):
            for simple in v4_orientation.view(self.v1_orientation_number, self.v4_size, self.v4_size):
                net_diff.append(torch.linalg.norm(simple, ord = 'fro').item())
        return np.mean(net_diff)
    
    def decision_weight_change(self, before, after):
        
        """
        Calculates frobenius norm of difference between weights in decision layer and initial weights for each iteraction
        of training. Takes in initial weights and current weights as arguments and returns scalar value. 
        """
        
        # Calculate frobenius norm of decision layer weight changes before and after training
        diff = after - before
        net_diff = (torch.linalg.norm(
            diff.view(2, self.v4_orientation_number), ord = 'fro').item())
        return net_diff
    
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
        organized with dimensions orientation, spatial frequency and the 
        tuning curve data plot.
        """
        
        # Create list of angles between -pi/2 and pi/2 to generate tuning curves
        self.tuning_curve_sample = 100
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        
        # Initialise tensor for all tuning curves after training, organized into orientations, phase/sfs, tuning curve data
        self.results = torch.empty(self.v1_orientation_number, self.phis_sfs, len(x))
        
        # Initialise tensor for all tuning curves before training
        self.initial_tuning_curves = torch.empty(self.v1_orientation_number, self.phis_sfs, len(x))
        
        # Create gabor at each orientation and store measured activity of each V1 gabor filter in tensor
        with torch.no_grad():
            for i in tqdm(range(len(x))):
                for orientation in range(self.v1_orientation_number):
                    for phi in range(self.phis_sfs):

                        # Create gabor
                        test = self.generate_gabor(self.v1_size, x[i], self.phis_sfs_range[phi], 5).view(
                            self.v1_size, self.v1_size)#.to(self.device)

                        # Present to specific gabor after training
                        result = torch.sum(
                            self.simple_weight.weight[self.phis_sfs * orientation + phi][0].view(
                                self.v1_size, self.v1_size) * test)

                        # Present to specific gabor before training
                        initial_result = torch.sum(
                            self.before_v1weight[self.phis_sfs * orientation + phi][0].view(
                                self.v1_size, self.v1_size) * test)

                        # Save activity in tensor
                        self.results[orientation][phi][i] = result 
                        self.initial_tuning_curves[orientation][phi][i] = initial_result 
                                
            # Normalize tuning curves
            self.results = self.results / self.initial_tuning_curves.max()
            self.initial_tuning_curves = self.initial_tuning_curves / self.initial_tuning_curves.max()
                                

    def plot_v1_tuning_curve(self, orientation, phi_sf, orientations = False, phi_sfs = False, differences = False):
        
        """
        Plot tuning curves at a particular orientation index or phase/spatial frequeny index. Setting orientations = True plots 
        tuning curves at all orientations at a specified phase/spatial frequency. Setting phi_sfs = True plots tuning curves at 
        all phases/spatial frequencies (determined during network initialization) at a specified orientation. Setting differences 
        = True plots difference in responses using initial V1 simple cell weights and trained weights. 
        """
        
        # Ensure orientation is between 0 and v1_orientation_number
        if not 0 <= orientation <= self.v1_orientation_number - 1:
            return("orientation needs to be between 0 and " + str(self.v1_orientation_number - 1))
        
        # Ensure phi is between 0 and phis_sfs
        if not 0 <= phi_sf <= self.v1_dimensions - 1:
            return("phi/sf needs to be between 0 and " + str(self.phis_sfs - 1))
        
        # Create list of angles between -pi/2 and pi/2
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = (x * 180) / np.pi
        
        if orientations == True and differences == False:
            
            # Plot each tuning curve at different orientations with specified phase/sf 
            for i in range(self.v1_orientation_number):
                plt.plot(x, self.results[i, phi_sf, :])
            
            # Create legend
            plt.legend([round(self.v1_angles[i] * 180 / np.pi, 1) for i in range(self.v1_orientation_number)])
            
            plt.ylabel("Response")
            plt.title("V1 tuning curves selective for different orientations", loc = 'center')
                
        if phi_sfs == True and differences == False:
            
            # Plot each tuning curve at different phase/sfs with specified orientation 
            for i in range(self.phis_sfs):
                plt.plot(x, self.results[orientation, i, :])
            
            # Create legend
            ranges = np.linspace(self.phis_sfs_range[0], self.phis_sfs_range[-1], self.phis_sfs)
            plt.legend([round(ranges[i], 1) for i in range(self.phis_sfs)])
            
            plt.ylabel("Response")
            plt.title("V1 tuning curves selective for different phase/SFs", loc = 'center');
            
        if orientations == True and differences == True:
            
            # Calculate difference in tuning curves before and after training
            difference = self.results - self.initial_tuning_curves
            
            # Plot each difference in tuning curve at different orientations with specified phase/sf 
            for i in range(self.v1_orientation_number):
                plt.plot(x, difference[i, phi_sf, :])
            
            # Create legend
            plt.legend([round(self.v1_angles[i] * 180 / np.pi, 1) for i in range(self.v1_orientation_number)])
            
            plt.ylabel("Difference in response")
            plt.title("Difference in V1 tuning curves selective for different orientations", loc = 'center');
            
        
        if phi_sfs == True and differences == True:
            
            # Calculate difference in tuning curves before and after training
            difference = self.results - self.initial_tuning_curves
            
            # Plot each difference in tuning curve at different phase/sfs with specified orientation 
            for i in range(self.phis_sfs):
                plt.plot(x, difference[orientation, i, :])

            # Create legend
            ranges = np.linspace(self.phis_sfs_range[0], self.phis_sfs_range[-1], self.phis_sfs)
            plt.legend([round(ranges[i], 1) for i in range(self.phis_sfs)])
            
            plt.ylabel("Difference in response")
            plt.title("Difference in V1 tuning curves selective for different phase/SFs", loc = 'center');
        
        plt.xlabel("Angle (Degrees)")
        
    def v4_tuning_curve(self):

        """
        Creates tuning curves for each gabor filter in V4 layer by testing each filter with a number of
        orientations at a fixed phase and spatial frequency. Returns a torch tensor consisting of all tuning curves 
        organized with dimensions orientation, and the tuning curve data plot.
        """
        
        # Create list of angles between -pi/2 and pi/2
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        
        # Initialise tensor for all tuning curves after training, organized into orientations, tuning curve data
        self.v4_results = torch.empty(self.v4_orientation_number, len(x))
        
        # Initialise tensor for all tuning curves before training, organized into orientations, tuning curve data
        self.v4_initial_tuning_curves = torch.empty(self.v4_orientation_number, len(x))
        
        # Create gabor at each orientation, present activity to network and measure activity at each V4 filter
        with torch.no_grad():
            for i in range(len(x)):
                
                # Create gabor
                test = self.generate_gabor(self.input_size, x[i], 0, 5).view(1, 1, self.input_size, self.input_size)
                
                
                # Forward pass through V1 
                out_after = self.simple_weight(test).view(1, self.v1_orientation_number * self.phis_sfs, self.v1_dimensions, self.v1_dimensions)
                
                out_before = self.simple_weight_before(test).view(1, self.v1_orientation_number * self.phis_sfs, self.v1_dimensions, self.v1_dimensions)
                
                pools_after = torch.zeros(self.v1_orientation_number, self.v1_dimensions, self.v1_dimensions)
                pools_before = torch.zeros(self.v1_orientation_number, self.v1_dimensions, self.v1_dimensions)
                
                for j in range(0, self.v1_orientation_number*self.phis_sfs, self.phis_sfs):
                    relu_after = F.relu(out_after[0][j:j+self.phis_sfs]) 
                    pool_after = (torch.sum(relu_after, dim = 0) * self.phase_rescale / self.phis_sfs).view(1, self.v1_dimensions, self.v1_dimensions) 
                    pools_after[j//4] = pool_after
                    
                    relu_before = F.relu(out_before[0][j:j+self.phis_sfs]) 
                    pool_before = (torch.sum(relu_before, dim = 0) * self.phase_rescale / self.phis_sfs).view(1, self.v1_dimensions, self.v1_dimensions) 
                    pools_before[j//4] = pool_before
                
                # For each V4 orientation, present the centre gabor test input to the V4 weight and measure activity
                v4_pool_after = torch.zeros(self.v4_orientation_number)
                v4_pool_before = torch.zeros(self.v4_orientation_number)
                
                index1 = int((self.v1_dimensions - 1)/ 2 - (self.v4_size - 1)/ 2)
                index2 = int((self.v1_dimensions - 1)/ 2 + (self.v4_size - 1)/ 2 + 1)
                for k in range(self.v4_orientation_number):
                    out_after = (pools_after[:, index1:index2, index1:index2] * self.v4_weight.weight[k, 0, :, :, :]).sum([-1, 0, 1])
                    v4_pool_after[k] = out_after

                    out_before = (
                        pools_before[:, index1:index2, index1:index2] * self.before_v4weight[k, 0, :, :, :]).sum([-1, 0, 1])
                    v4_pool_before[k] = out_before


                # Store measured activity of each filter in tensor
                self.v4_results[:, i] = v4_pool_after
                self.v4_initial_tuning_curves[:, i] = v4_pool_before
                       
        # Normalize tuning curves
        self.v4_results = self.v4_results / self.v4_initial_tuning_curves.max()
        self.v4_initial_tuning_curves = self.v4_initial_tuning_curves / self.v4_initial_tuning_curves.max()
                       
    def plot_v4_tuning_curve(self, differences = False):
        
        """
        Plot tuning curves at a particular orientation index. Setting differences = True 
        plots difference in responses using initial V1 simple cell weights and trained weights. 
        """

        # Create list of angles between -pi/2 and pi/2
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = (x * 180) / np.pi
        
        if differences == False:
            
            # Plot each tuning curve
            for i in range(self.v4_orientation_number):
                plt.plot(x, self.v4_results[i, :])
            
            # Find preferred orientation of V4 curves for legend
            plt.legend([round(x[self.v4_results[i][:].argmax()], 1) for i in range(self.v4_orientation_number)])
            plt.ylabel("Response")
            plt.title("V4 tuning curves selective for different orientations", loc = 'center');

        if differences == True:

            # Calculate difference in tuning curves before and after training
            difference = self.v4_results - self.v4_initial_tuning_curves
            
            # Plot each difference in tuning curve at different orientations with specified phase/sf
            for i in range(self.v4_orientation_number):
                plt.plot(x, difference[i, :])
            
            # Find preferred orientation of V4 curves for legend
            plt.legend([round(x[self.v4_results[i][:].argmax()], 1) for i in range(self.v4_orientation_number)])
            plt.ylabel("Difference in response")
            plt.title("Difference in V4 tuning curves selective for different orientations", loc = 'center');

        plt.xlabel("Angle (Degrees)")
    
    
    def v1_tuning_params(self):
        
        """
        Calculates the amplitude and bandwidth of V1 tuning curves.
        """
        
        # Create list of angles between -pi/2 and pi/2
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = (x * 180) / np.pi
        self.after_amplitudes = []
        self.after_bandwidths = []
        
        self.before_amplitudes = []
        self.before_bandwidths = []
        
        self.xs = [i for i in range(self.tuning_curve_sample)]
        for i in range(self.v1_orientation_number):
            for j in range(self.phis_sfs):
                
                # Calculate tuning curve parameters after training
                
                # Set tuning curve at particular orientation, and phase/sf after training
                curve = self.results[i, j, :]

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
                    bandwidth = 180 - np.abs(x[halfmax_index1] - x[halfmax_index2])
                self.before_bandwidths.append(bandwidth/2)
                
                # Calculate tuning curve parameters before training
                
                # Set tuning curve at particular orientation, and phase/sf before training
                initial_params = self.initial_tuning_curves[i, j, :]
                
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
                    bandwidth2 = 180 - np.abs(x[halfmax2_index1] - x[halfmax2_index2])
                self.after_bandwidths.append(bandwidth2/2)
          
        # Calculate mean and standard deviation of the ampltiude and bandwidth of each v1 curves before and after training
        self.v1_mean_after_amplitude = np.mean(self.after_amplitudes)
        self.v1_std_after_amplitude = np.std(self.after_amplitudes)

        self.v1_mean_after_bandwidth = np.mean(self.after_bandwidths)
        self.v1_std_after_bandwidth = np.std(self.after_bandwidths)

        self.v1_mean_before_amplitude = np.mean(self.before_amplitudes)
        self.v1_std_before_amplitude = np.std(self.before_amplitudes)

        self.v1_mean_before_bandwidth = np.mean(self.before_bandwidths)
        self.v1_std_before_bandwidth = np.std(self.before_bandwidths)

        # Calculate percentage differences
        self.v1_amplitude_difference = ((self.v1_mean_after_amplitude - self.v1_mean_before_amplitude) / self.v1_mean_before_amplitude) * 100      
        self.v1_bandwidth_difference = ((self.v1_mean_after_bandwidth - self.v1_mean_before_bandwidth) / self.v1_mean_before_bandwidth) * 100
        
        
    def v4_tuning_params(self):
        
        """
        Calculates the amplitude and bandwidth of V4 tuning curves.  
        """
        
        # Create list of angles between -pi/2 and pi/2
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = (x * 180) / np.pi
        self.after_amplitudes = []
        self.after_bandwidths = []
        
        self.before_amplitudes = []
        self.before_bandwidths = []
        
        self.xs = [i for i in range(self.tuning_curve_sample)]
        for i in range(self.v4_orientation_number):

            # Calculate tuning curve parameters after training
                
            # Set tuning curve at particular orientation after training
            curve = self.v4_results[i, :]
            
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

            # Set tuning curve at particular orientation before training
            initial_params = self.v4_initial_tuning_curves[i, :]
            
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

        # Calculate percentage differences
        self.v4_amplitude_difference = ((self.v4_mean_after_amplitude - self.v4_mean_before_amplitude) / self.v4_mean_before_amplitude) * 100      
        self.v4_bandwidth_difference = ((self.v4_mean_after_bandwidth - self.v4_mean_before_bandwidth) / self.v4_mean_before_bandwidth) * 100
    
    def otc_curve(self):
        
        """
        Calculates slope of each tuning curve before and after training at the trained angle.
        """
        
        # Create list of angles between -pi/2 and pi/2
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = x * 180 / np.pi
        
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
            
            # Find index closest to trained angle
            trained_index1 = self.find_nearest(torch.tensor(x), trained_angle1)
            trained_index2 = self.find_nearest(torch.tensor(x), trained_angle2)

            for j in range(self.phis_sfs):
                
                # Set V1 tuning curve at particular orientation, and phase/sf after and before training
                curve = self.results[i, j]
                initial = self.initial_tuning_curves[i, j]
                
                # Find index of preferred orientation of curve
                if torch.abs(curve.max()) > torch.abs(curve.min()):
                    after_preferred_orientation = curve.argmax().item()
                
                if torch.abs(initial.max()) > torch.abs(initial.min()):
                    before_preferred_orientation = initial.argmax().item()
                
                if torch.abs(curve.max()) < torch.abs(curve.min()):
                    after_preferred_orientation = curve.argmin().item()
                    
                if torch.abs(initial.max()) < torch.abs(initial.min()):
                    after_preferred_orientation = initial.argmin().item()
                
                # Save differences between preferred orientation and trained angle and save slope at trained angle – use negative trained angle if preferred orientation is negative and vice versa
                
                if x[after_preferred_orientation] <= 0:
                    # Calculate difference between preferred orientation and trained angle
                    after_ranges.append(x[after_preferred_orientation] - x[trained_index1])
                    
                    # Calculate slope at trained angle
                    after_slope = (curve[trained_index1 + 1] - curve[trained_index1 - 1])/(2 * 180/self.tuning_curve_sample)
                    

                else:
                    # Calculate difference between preferred orientation and trained angle
                    after_ranges.append(x[after_preferred_orientation] - x[trained_index2])
                    
                    # Calculate slope at trained angle
                    after_slope = (curve[trained_index2 + 1] - curve[trained_index2 - 1])/(2 * 180/self.tuning_curve_sample)
                    
                
                if x[before_preferred_orientation] <= 0:    
                    # Calculate difference between preferred orientation and trained angle
                    before_ranges.append(x[before_preferred_orientation] - x[trained_index1])
                    
                    # Calculate slope at trained angle
                    before_slope = (initial[trained_index1 + 1] - initial[trained_index1 - 1])/(2 * 180/self.tuning_curve_sample)
                
                else:
                    # Calculate difference between preferred orientation and trained angle
                    before_ranges.append(x[before_preferred_orientation] - x[trained_index2])
                    
                    # Calculate slope at trained angle
                    before_slope = (initial[trained_index2 + 1] - initial[trained_index2 - 1])/(2 * 180/self.tuning_curve_sample)
                    
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
        
        for k in range(self.v4_orientation_number):
            
            # Set V4 tuning curve at particular orientation after and before training
            curve = self.v4_results[k, :]
            initial = self.v4_initial_tuning_curves[k, :]
            
            # Find index of preferred orientation of curve
            after_preferred_orientation = curve.argmax().item()
            before_preferred_orientation = initial.argmax().item()
            
            # Save differences between preferred orientation and trained angle and save slope at trained angle – use negative trained angle if preferred orientation is negative and vice versa
            if x[after_preferred_orientation] <= 0:
                    
                # Calculate difference between preferred orientation and trained angle
                self.v4_after_range.append(x[after_preferred_orientation] - x[trained_index1])
                self.v4_before_range.append(x[before_preferred_orientation] - x[trained_index1])


                # Calculate slope at trained angle
                after_slope = (curve[trained_index1 + 1] - curve[trained_index1 - 1])/(2 * 180/self.tuning_curve_sample)
                before_slope = (initial[trained_index1 + 1] - initial[trained_index1 - 1])/(2 * 180/self.tuning_curve_sample)


            else:
                # Calculate difference between preferred orientation and trained angle
                self.v4_after_range.append(x[after_preferred_orientation] - x[trained_index2])
                self.v4_before_range.append(x[before_preferred_orientation] - x[trained_index2])

                # Calculate slope at trained angle
                after_slope = (curve[trained_index2 + 1] - curve[trained_index2 - 1])/(2 * 180/self.tuning_curve_sample)
                before_slope = (initial[trained_index2 + 1] - initial[trained_index2 - 1])/(2 * 180/self.tuning_curve_sample)

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
#         if not 0 <= x_location <= self.v1_dimensions - 1 or not 0 <= y_location <= self.v1_dimensions - 1:
#             return("x_location and y_location needs to be between 0 and " + str(self.v1_dimensions - 1))
        
        
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
            kernel = self.generate_location_gabor(angle, phases[j // 10], sfs[j // 10], x_location, y_location, random_sf = False)
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
        
        grid_score = torch.empty(self.v1_dimensions - 2 * self.v1_padding, self.v1_dimensions - 2 * self.v1_padding)
        grid_error = torch.empty(self.v1_dimensions - 2 * self.v1_padding, self.v1_dimensions - 2 * self.v1_padding)
        
        # For each location on x and y axis, calculate transfer performance and error
        for i in tqdm(range(self.v1_dimensions - 2 * self.v1_padding)):
            for j in range(self.v1_dimensions - 2 * self.v1_padding):
                score, error = self.transfer_test(i, j)
                scores.append(score)
                errors.append(error)

                # Calculate distance between transfer location and trained location
                distance = np.sqrt(((i - self.train_x_location) ** 2) + ((j - self.train_y_location) ** 2))
                distances.append(distance)
                
                grid_score[i][j] = score
                grid_error[i][j] = torch.tensor(error)
                
        # Plot results
        if performance == True and grid == True:
            plt.imshow(grid_score, cmap = 'PiYG', vmin = 0, vmax = 100)
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
            plt.imshow(grid_error, cmap = 'PiYG', vmin = 0, vmax = 1)
            plt.colorbar(label = 'Error')
            plt.title("Heat map of error at different untrained locations")
        
        if performance == False and grid == False:
            distances, errors = self.sort_lists(distances, errors)
            plt.plot(distances, errors)
            plt.ylim(-5, 105)
            plt.xlabel("Distance from trained angle")
            plt.ylabel("Error")
            plt.title("Error on untrained locations")        
    
    
# Helper functions
    
    def generate_gabor(self, size, theta, phi, lamda, random_sf = False, gamma = 0.5):
        
        ksize = size # Size of gabor filter
        sigma = 1.6 # Standard deviation of Gaussian envelope
        self.lamda = lamda
        if random_sf == True:
            self.lamda = np.random.uniform(1.1, 13) # Wavelength of sinusoidal component so determines sf
        gamma = gamma # Spatial aspect ratio and ellipticity 
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, self.lamda, gamma, phi)
        return torch.tensor(kernel).view(1, 1, size, size).float()
    
    def generate_location_gabor(self, theta, phi, lamda, x_location, y_location, random_sf = False):
       
        
        # Generate gabor and noise
        kernel = self.generate_gabor(self.v1_size, theta, phi, lamda, random_sf = random_sf).view(self.v1_size, self.v1_size)
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
