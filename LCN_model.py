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



class LCN(nn.Module):
    
    def __init__(self, input_size, simple_number, simple_size, v4_size, v4_stride, v4_orientation_number, phis_sfs, training_size, phis = True, sfs = False, alpha = 0.0001):

        """
        Initialize network parameters.
        """
        
        super(LCN, self).__init__()
        self.input_size = input_size
        self.simple_number = simple_number
        self.simple_size = simple_size
        self.phis_sfs = phis_sfs
        self.phis = phis
        self.sfs = sfs
        
        self.training_size = training_size
        self.alpha = alpha
        self.dimensions = self.input_size - self.simple_size + 1
        
        self.v4_size = v4_size
        self.v4_stride = v4_stride
        self.v4_orientation_number = v4_orientation_number
        self.v4_dimensions = int(((self.dimensions - self.v4_size)/self.v4_stride) + 1)
        
        self.simple_weight = torch.nn.Parameter(self.init_weights())
        
        self.v4_weight = torch.nn.Parameter(self.init_gaussian_weights().view(
            self.v4_orientation_number, self.simple_number, self.v4_dimensions, self.v4_dimensions, self.v4_size ** 2))
        
        self.decision = nn.Linear(self.v4_dimensions*self.v4_dimensions*v4_orientation_number, 2)
        self.decision.weight = torch.nn.Parameter(
            torch.zeros((2, self.v4_dimensions*self.v4_dimensions*self.v4_orientation_number)))
        
        self.before_v1weight = self.simple_weight.clone()
        self.before_v4weight = self.v4_weight.clone()
        self.before_decision_weight = self.decision.weight.clone()
        
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
    
    def init_weights(self):
        
        """
        Initialize V1 simple cell weights with gabor filters at different orientations and different phases/spatial 
        frequencies, depending on what is chosen when initializing network. Returns torch tensor weights.
        """        
        angles = self.remove_ambiguous_stimuli(-np.pi/2, np.pi/2, self.simple_number, even_space = True)
        if self.phis == True:
            self.phis_sfs_range = np.linspace(0, np.pi, self.phis_sfs)
            weights = []
            for i in range(self.simple_number):
                for j in range(self.phis_sfs):
                    for k in range(self.dimensions ** 2):
                        theta = angles[i]
                        phi = self.phis_sfs_range[j]
                        kernel = self.generate_gabor(self.simple_size, theta, phi, 5)
                        noise = np.random.normal(0, 0.05, (self.simple_size, self.simple_size))
                        kernel = kernel + noise
                        weights.append(kernel)

            weight = torch.stack(weights).view(
                1, self.simple_number*self.phis_sfs, 1, self.dimensions, self.dimensions, self.simple_size ** 2)
            return weight
        if self.sfs == True:
            self.phis_sfs_range = np.linspace(1.1, 14, self.phis_sfs)
            weights = []
            for i in range(self.simple_number):
                for j in range(self.phis_sfs):
                    for k in range(self.dimensions ** 2):
                        theta = angles[i]
                        lamda = self.phis_sfs_range[j]
                        kernel = self.generate_gabor(self.simple_size, theta, 0, lamda)
                        noise = torch.normal(0, 0.05, (self.simple_size, self.simple_size))
                        kernel = kernel + noise
                        weights.append(kernel)

            weight = torch.stack(weights).view(
                1, self.simple_number*self.phis_sfs, 1, self.dimensions, self.dimensions, self.simple_size ** 2)
            return weight
    
    
    def init_gaussian_weights(self):
        
        """
        Initialize V4 weights with gaussian filters. Returns torch tensor weights.
        """
        
        angles = self.remove_ambiguous_stimuli(-np.pi/2, np.pi/2, self.v4_orientation_number, even_space = True)
        
        v4 = torch.empty(self.v4_orientation_number, self.simple_number, self.v4_dimensions, self.v4_dimensions, self.v4_size, self.v4_size)
        for pool_orientation in range(self.v4_orientation_number):
            for orientation in range(self.simple_number):
                for i in range(self.v4_dimensions):
                    for j in range(self.v4_dimensions):
                        v4[pool_orientation][orientation][i][j] = self.generate_3d_gaussian(angles[pool_orientation], 0.5)[orientation]
        return v4
    
    def generate_gabor(self, size, theta, phi, lamda, random_sf = False):
        
        """
        Generates one gabor matrix. Takes in gabor size, angle, phase, spatial frequency as arguments and returns a torch
        tensor gabor matrix. Setting random_sf = True creates a gabor with a random spatial frequency.
        """
        
        ksize = size
        sigma = 3
        if random_sf == True:
            lamda = np.random.uniform(1.1, 13)
        gamma = 0.5
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi)
        return torch.tensor(kernel).view(1, 1, size, size).float()
    
    def generate_gaussian(self, mean, std, kernlen):
        
        """
        Generates a gaussian list. Takes in mean, length of list and standard deviation and returns a list. 
        """
        
        n = torch.linspace(-np.pi/2, np.pi/2, kernlen) - mean
        sig2 = 2 * std * std
        w = torch.exp(-n ** 2 / sig2)
        
        return w
    
    def generate_3d_gaussian(self, mean, std):
        
        """
        Generates a self.simple_number x self.v4_size x self.v4_size 3D gaussian. Takes in mean and standard deviation of 
        orientation gaussian and returns a 3D torch tensor.
        """
        
        gkern1d = self.generate_gaussian(0, std, self.v4_size)
        gkern2d = torch.outer(gkern1d, gkern1d)
        
        twod = (gkern2d, ) * self.simple_number
        twods = torch.stack(twod)
        
        scale = self.generate_gaussian(mean, std, self.simple_number)
        scales = torch.empty(self.simple_number, self.v4_size, self.v4_size)
        for i in range(self.simple_number):
            scales[i] = twods[i] * scale[i]
        
        return scales
            
    
    def remove_ambiguous_stimuli(self, angle1, angle2, size, even_space = False):
        
        """
        Function returns a numpy list of angles that are unambiguously clockwise or anticlockwise relative to 0°. 
        Ambiguous angles include -pi/2, 0, pi/2 and pi which are not clockwise or anticlockwise relative to 0°.
        A constant is either added or subtracted from these angles to make them unambiguous. Takes in the two angles 
        and the number of angles as arguments. Setting even_space = True returns a numpy list of angles that are evenly 
        spaced across 360° by removing the last element in the list.
        """
        
        stimuli = [-np.pi/2, 0, np.pi/2, np.pi]
        if even_space == True:
            x = np.linspace(angle1, angle2, size+1)
            x = np.delete(x, -1)
            ambiguous = True
            while ambiguous == True:
                for i in stimuli[0:2]:
                    x = np.where(x == i, i+0.1, x)
                for i in stimuli[2:4]:
                    x = np.where(x == i, i-0.1, x)            
                x = np.linspace(x[0], x[-1], size)
                if -np.pi/2 in x or 0 in x or np.pi/2 in x:
                    ambiguous = True
                else:
                    ambiguous = False

        if even_space == False:
            x = np.linspace(angle1, angle2, size)
            ambiguous = True
            while ambiguous == True:
                for i in stimuli[0:2]:
                    x = np.where(x == i, i+0.1, x)
                for i in stimuli[2:4]:
                    x = np.where(x == i, i-0.1, x)            
                x = np.linspace(x[0], x[-1], size)
                if -np.pi/2 in x or 0 in x or np.pi/2 in x:
                    ambiguous = True
                else:
                    ambiguous = False
        return x
    
    def binary_loss(self, scores, desired_output):
        
        """
        Loss function. Takes in prediction and desired output as arguments and returns cross entropy loss. 
        """
        
        loss = nn.CrossEntropyLoss()
        return loss(scores, desired_output)
    
    def inputting(self, angle1, angle2, random_sf):
        
        """
        Creates gabor filters to train network with. Takes in two angles that give the range of orientation of gabors
        and a Boolean random_sf to determine if spatial frequency of gabors is randomized. Returns a torch tensor 
        consisting of all the training gabors. 
        """
        
        self.labels = []
        self.inputs = []
    
        x = self.remove_ambiguous_stimuli(angle1, angle2, self.training_size, even_space = False)

        for i in range(self.training_size):
            theta = x[i]
            phi = np.random.uniform(0, np.pi)
            kernel = self.generate_gabor(self.input_size, theta, phi, 5, random_sf = random_sf)
            self.inputs.append(kernel)
            if 0 < theta < np.pi/2:
                label = torch.tensor([0])
            else:
                label = torch.tensor([1])
            self.labels.append(label)
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
        
        x = x.unfold(2, self.simple_size, 1).unfold(3, self.simple_size, 1)
        x = x.contiguous().view(*x.size()[:-2], -1)
        out = (x.unsqueeze(1) * self.simple_weight).sum([2, -1])
        pools = []
        for i in range(0, self.simple_number*self.phis_sfs, self.phis_sfs):
            relu = F.relu(out[0][i:i+self.phis_sfs])
            pool = (torch.sum(relu, dim = 0)/(self.phis_sfs*200)).view(1, self.dimensions, self.dimensions)
            pools.append(pool)
        pools = torch.stack(pools).view(self.simple_number, self.dimensions, self.dimensions)
        v4_pools = []
        out = pools.unfold(1, self.v4_size, self.v4_stride).unfold(2, self.v4_size, self.v4_stride)
        out = out.contiguous().view(*out.size()[:-2], -1)
        for j in range(self.v4_orientation_number):
            pooled = (out.unsqueeze(1)*self.v4_weight[j]).sum([1, -1, 0])
            v4_pools.append(pooled)
        v4_pool = torch.stack(v4_pools).view(self.v4_orientation_number, self.v4_dimensions, self.v4_dimensions)
        out = v4_pool.view(1, self.v4_dimensions*self.v4_dimensions*self.v4_orientation_number)
        out = self.decision(out.float())
        return out
    
    def mean_train(self, iterations, optimizer):
        
        """
        Training loop. Takes in the number of iteractions and optimizer as arguments and trains the network over the 
        number of iteractions. Calculates mean loss over batch size and optimizes over this loss. 
        """
        
        self.before_v1weights = []
        self.before_v4weights = []
        self.before_decision_weights = []
        
        self.losses = []
        self.training_scores = []

        for i in tqdm(range(iterations)):
            optimizer.zero_grad()
            self.training_score = 0
            loss2 = torch.empty(self.training_size)
            
            for j in range(self.training_size):
                self.scores = self.forward(self.input[j][0].view(1, 1, self.input_size, self.input_size))
                loss1 = self.binary_loss(self.scores, self.desired_output[j].view(1))
                if torch.argmax(self.scores) == self.desired_output[j]:
                    self.training_score += 1
                loss2[j] = loss1
            
            
            loss = torch.sum(loss2)/self.training_size
            self.losses.append(loss)
            
            self.training_score = self.training_score/self.training_size * 100
            self.training_scores.append(self.training_score)
            
            self.before_v1weights.append(self.v1_weight_change(self.before_v1weight, self.simple_weight))
            self.before_v4weights.append(self.v4_weight_change(self.before_v4weight, self.v4_weight))
            self.before_decision_weights.append(self.decision_weight_change(self.before_decision_weight, self.decision.weight))
            
            loss.backward()
            optimizer.step()
            
        self.v1_tuning_curve()
        self.v4_tuning_curve()
        
    def double_train(self, iterations, optimizer, angle1, angle2, test_angle, test_size):
        
        """
        Training loop for sequential curriculum. Takes in the number of iteractions, optimizer, the first angle to be
        trained on (will be trained on -angle and angle), the second angle to be trained on after training on first 
        angle, the test angle and the test size. For each iteraction, the network is tested on the test_size number of 
        gabors between -test_angle and test_angle to see how well it generalizes. 
        """
        
        self.before_v1weights = []
        self.before_v4weights = []
        self.before_decision_weights = []
        
        self.losses = []
        self.training_scores = []
        self.generalize_error = []
        self.generalize_perform = []
        for angle in [angle1, angle2]:
            input = self.inputting(-angle, angle, random_sf = False)
            desired_output = self.desired_outputting()
            for i in tqdm(range(iterations)):
                optimizer.zero_grad()
                self.training_score = 0
                loss2 = torch.empty(self.training_size)

                for j in range(self.training_size):
                    self.scores = self.forward(input[j][0].view(1, 1, self.input_size, self.input_size))
                    loss1 = self.binary_loss(self.scores, desired_output[j].view([1]))
                    if torch.argmax(self.scores) == desired_output[j]:
                        self.training_score += 1
                    loss2[j] = loss1

                loss = torch.sum(loss2)/self.training_size
                self.losses.append(loss)

                self.training_score = self.training_score/self.training_size * 100
                self.training_scores.append(self.training_score)

                generalize = self.generalization(test_angle, test_size)
                self.generalize_error.append(self.general_mean_error)
                self.generalize_perform.append(self.generalization_score)

                self.before_v1weights.append(self.v1_weight_change(self.before_v1weight, self.simple_weight))
                self.before_v4weights.append(self.v4_weight_change(self.before_v4weight, self.v4_weight))
                self.before_decision_weights.append(self.decision_weight_change(self.before_decision_weight, self.decision.weight))

                loss.backward()
                optimizer.step()
    
        self.v1_tuning_curve()
        self.v4_tuning_curve()
        
    def plot_training_error(self, color):
        
        """
        Plots the training error for each iteraction. Calculated by mean loss between prediction and desired output
        for each batch of training data. Takes in color as argument. 
        """
        
        plt.plot([loss.detach().cpu().numpy() for loss in self.losses], color = color)
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
        
    def generalization(self, angle, test_size):
        
        """
        Function used to test model against test_size number of gabors between -angle and angle to measure 
        generalization error and performance.
        """
        
        angles = self.remove_ambiguous_stimuli(-angle, angle, test_size, even_space = False)
        self.generalization_score = 0
        general_error = []
        for i in range(test_size):
            gabor = self.generate_gabor(self.input_size, angles[i], phi = 0, lamda = 5).clone().detach()
            if 0 < angles[i] < np.pi/2:
                label = torch.tensor([0])
            else:
                label = torch.tensor([1])
            with torch.no_grad():
                a = self.forward(gabor)
                if torch.argmax(a) == label:
                    self.generalization_score += 1
                general_error.append(self.binary_loss(a, label))
        self.generalization_score = self.generalization_score/test_size * 100
        self.general_mean_error = np.mean(general_error)

    def plot_generalization_performance(self, color):
        
        """
        Plots generalization performance. Takes in color as argument.
        """
        
        plt.plot(self.generalize_perform, color = color)
        plt.xlabel("Time (epochs)")
        plt.ylabel("Performance (%)")

    def plot_generalization_error(self, color):
        
        """
        Plots generalization error. Takes in color as argument. 
        """
        
        plt.plot(self.generalize_error, color = color)
        plt.xlabel("Time (epochs)")
        plt.ylabel("Error")
   
    def plot_angle_performance(self, number, color):
        
        """
        Plots the angle/performance graph for the model. Takes in the number of test angles and plot color as 
        arguments. Creates the number of angles between 0 and pi/2 and tests model with test angles between -angle 
        and angle. 
        """
        
        angles = self.remove_ambiguous_stimuli(0, np.pi/2, number, even_space = False)
        scores = []
        for i in range(number):
            self.generalization(angles[i], 50)
            scores.append(self.generalization_score)
        plt.plot((180 * angles)/np.pi, scores, color = color)
        plt.xlabel("Separation angle (Degrees)")
        plt.ylabel("Performance (%)")

    def v1_weight_change(self, before, after):
        
        """
        Calculates frobenius norm of difference between weights in V1 simple cells and initial weights for each iteraction 
        of training. Takes in initial weights and current weights as arguments and returns scalar value.
        """
        
        diff = after - before
        net_diff = []
        for i in diff.view(self.simple_number*self.phis_sfs, 1, self.dimensions, self.dimensions, self.simple_size ** 2):
            for j in i.view(self.dimensions, self.dimensions, self.simple_size ** 2):
                for k in j:
                    net_diff.append(torch.linalg.norm(k.view(self.simple_size, self.simple_size), ord = 'fro').item())
        return np.mean(net_diff)
    
    def v4_weight_change(self, before, after):
        
        """
        Calculates frobenius norm of difference between weights in V4 cells and initial weights for each iteraction 
        of training. Takes in initial weights and current weights as arguments and returns scalar value.
        """
        
        diff = after - before
        net_diff = []
        for v4_orientation in diff.view(self.v4_orientation_number, self.simple_number, self.v4_dimensions, self.v4_dimensions, self.v4_size ** 2):
            for simple in v4_orientation.view(self.simple_number, self.v4_dimensions, self.v4_dimensions, self.v4_size ** 2):
                for j in simple.view(self.v4_dimensions, self.v4_dimensions, self.v4_size ** 2):
                    for k in j:
                        net_diff.append(torch.linalg.norm(k.view(self.v4_size, self.v4_size), ord = 'fro').item())
        return np.mean(net_diff)
    
    def decision_weight_change(self, before, after):
        
        """
        Calculates frobenius norm of difference between weights in decision layer and initial weights for each iteraction
        of training. Takes in initial weights and current weights as arguments and returns scalar value. 
        """
        
        diff = after - before
        net_diff = (torch.linalg.norm(
            diff.view(2, self.v4_dimensions*self.v4_dimensions*self.v4_orientation_number), ord = 'fro').item())
        return net_diff
    
    def plot_weight_change(self, color, v1 = False, v4 = False, decision = False):
        
        """
        Plot weight changes. Takes in plot color as argument. Setting v1 = True plots V1 simple cell weight change and
        setting decision = True plots deicision layer weight changes. 
        """
        
        if v1 == True:
            plt.plot(self.before_v1weights, color = color)
        if v4 == True:
            plt.plot(self.before_v4weights, color = color)
        if decision == True:
            plt.plot(self.before_decision_weights, color = color)
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
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        self.results = torch.empty(self.simple_number, self.phis_sfs, self.dimensions, self.dimensions, len(x))
        self.initial_tuning_curves = torch.empty(
            self.simple_number, self.phis_sfs, self.dimensions, self.dimensions, len(x))
        with torch.no_grad():
            for i in tqdm(range(len(x))):
                for orientation in range(self.simple_number):
                    for sf in range(self.phis_sfs):
                        for horizontal in range(len(self.simple_weight[0][self.phis_sfs * orientation + sf][0])):
                            for vertical in range(
                                len(self.simple_weight[0][self.phis_sfs * orientation + sf][0][horizontal])):
                                test = self.generate_gabor(self.simple_size, x[i], 0, 5).view(
                                    self.simple_size, self.simple_size)
                                result = torch.sum(
                                    self.simple_weight[0][self.phis_sfs * orientation + sf][0][horizontal][vertical].view(
                                        self.simple_size, self.simple_size) * test)
                                self.results[orientation][sf][horizontal][vertical][i] = result
                                
                                initial_result = torch.sum(
                                    self.before_v1weight[0][self.phis_sfs * orientation + sf][0][horizontal][vertical].view(
                                        self.simple_size, self.simple_size) * test)
                                self.initial_tuning_curves[orientation][sf][horizontal][vertical][i] = initial_result
                        
    def plot_v1_tuning_curve(self, orientation, phi_sf, position, orientations = False, phi_sfs = False, differences = False):
        
        """
        Plot tuning curves at a particular orientation index or phase/spatial frequeny index and at a particular 
        filter position. Setting orientations = True plots tuning curves at all orientations at a specified phase/
        spatial frequency. Setting phi_sfs = True plots tuning curves at all phases/spatial frequencies (determined
        during network initialization) at a specified orientation. Setting differences = True plots difference in
        responses using initial V1 simple cell weights and trained weights. 
        """
        
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = (x * 180) / np.pi
        if orientations == True and differences == False:
            ranges = self.remove_ambiguous_stimuli(-np.pi/2, np.pi/2, self.simple_number, even_space = True)
            ranges = (ranges * 180) / np.pi
            for i in range(self.simple_number):
                plt.plot(x, self.results[i, phi_sf, position, position, :])
            plt.legend([round(ranges[i], 1) for i in range(self.simple_number)])
            plt.ylabel("Response")
                
        if phi_sfs == True and differences == False:
            ranges = np.linspace(self.phis_sfs_range[0], self.phis_sfs_range[-1], self.phis_sfs)
            for i in range(self.phis_sfs):
                plt.plot(x, self.results[orientation, i, position, position, :])
            plt.legend([round(ranges[i], 1) for i in range(self.phis_sfs)])
            plt.ylabel("Response")
            
        if orientations == True and differences == True:
            ranges = self.remove_ambiguous_stimuli(-np.pi/2, np.pi/2, self.simple_number, even_space = True)
            ranges = (ranges * 180) / np.pi
            difference = self.results - self.initial_tuning_curves
            for i in range(self.simple_number):
                plt.plot(x, difference[i, phi_sf, position, position, :])
            plt.legend([round(ranges[i], 1) for i in range(self.simple_number)])
            plt.ylabel("Difference in response")
            
        
        if phi_sfs == True and differences == True:
            ranges = np.linspace(self.phis_sfs_range[0], self.phis_sfs_range[-1], self.phis_sfs)
            difference = self.results - self.initial_tuning_curves
            for i in range(self.phis_sfs):
                plt.plot(x, difference[orientation, i, position, position, :])
            plt.legend([round(ranges[i], 1) for i in range(self.phis_sfs)])
            plt.ylabel("Difference in response")
        
        plt.xlabel("Angle (Degrees)")
        
    def v4_tuning_curve(self):

        """
        Creates tuning curves for each gabor filter in V4 layer by testing each filter with a number of
        orientations at a fixed phase and spatial frequency. Returns a torch tensor consisting of all tuning curves 
        organized with dimensions orientation, horizontal position, vertical position and the 
        tuning curve data plot.
        """
        
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        self.v4_results = torch.empty(self.v4_orientation_number, self.v4_dimensions, self.v4_dimensions, len(x))
        self.v4_initial_tuning_curves = torch.empty(
            self.v4_orientation_number, self.v4_dimensions, self.v4_dimensions, len(x))
        with torch.no_grad():
            for i in range(len(x)):
                test = self.generate_gabor(self.input_size, x[i], 0, 5).view(1, 1, self.input_size, self.input_size)
                out = test.unfold(2, self.simple_size, 1).unfold(3, self.simple_size, 1)
                out = out.contiguous().view(*out.size()[:-2], -1)
                out = (out.unsqueeze(1) * self.simple_weight).sum([2, -1])
                pools = []
                for j in range(0, self.simple_number*self.phis_sfs, self.phis_sfs):
                    relu = F.relu(out[0][j:j+self.phis_sfs])
                    pool = (torch.sum(relu, dim = 0)/(self.phis_sfs*200)).view(1, self.dimensions, self.dimensions)
                    pools.append(pool)
                pools = torch.stack(pools).view(self.simple_number, self.dimensions, self.dimensions)
                v4_pools_after = []
                v4_pools_before = []
                    
                out = pools.unfold(1, self.v4_size, self.v4_stride).unfold(2, self.v4_size, self.v4_stride)
                out = out.contiguous().view(*out.size()[:-2], -1)
                for k in range(self.v4_orientation_number):
                    out_after = (out.unsqueeze(1)*self.v4_weight[k]).sum([1, -1, 0])
                    v4_pools_after.append(out_after)
                    
                    out_before = (out.unsqueeze(1)*self.before_v4weight[k]).sum([1, -1, 0])
                    v4_pools_before.append(out_before)
                v4_pool_after = torch.stack(v4_pools_after).view(self.v4_orientation_number, self.v4_dimensions, self.v4_dimensions)
                v4_pool_before = torch.stack(v4_pools_before).view(self.v4_orientation_number, self.v4_dimensions, self.v4_dimensions)
                            
                self.v4_results[:, :, :, i] = v4_pool_after
                self.v4_initial_tuning_curves[:, :, :, i] = v4_pool_before

    def plot_v4_tuning_curve(self, position, differences = False):
        
        """
        Plot tuning curves at a particular orientation index and at a particular filter position. Setting differences = True plots 
        difference in responses using initial V1 simple cell weights and trained weights. 
        """
        
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = (x * 180) / np.pi
        if differences == False:
            ranges = self.remove_ambiguous_stimuli(-np.pi/2, np.pi/2, self.v4_orientation_number, even_space = True)
            ranges = (ranges * 180) / np.pi
            for i in range(self.v4_orientation_number):
                plt.plot(x, self.v4_results[i, position, position, :])
            plt.legend([round(ranges[i], 1) for i in range(self.v4_orientation_number)])
            plt.ylabel("Response")

            
        if differences == True:
            ranges = self.remove_ambiguous_stimuli(np.pi/2, np.pi/2, self.v4_orientation_number, even_space = True)
            ranges = (ranges * 180) / np.pi
            difference = self.v4_results - self.v4_initial_tuning_curves
            for i in range(self.v4_orientation_number):
                plt.plot(x, difference[i, position, position, :])
            plt.legend([round(ranges[i], 1) for i in range(self.v4_orientation_number)])
            plt.ylabel("Difference in response")

        plt.xlabel("Angle (Degrees)")
    
    
    def find_nearest(self, tensor, value):
        
        """
        Finds index closest to point on curve. Takes in curve as tensor and scalar value and returns index of point on curve 
        closest to the scalar value. 
        """
        
        idx = (torch.abs(tensor - value)).argmin()
        return idx.item()
    
    def v1_tuning_params(self, position):
        
        """
        Calculates the amplitude and bandwidth of V1 tuning curves. Takes position of gabor filter as argument.  
        """
        
        
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = (x * 180) / np.pi
        self.after_amplitudes = []
        self.after_bandwidths = []
        
        self.before_amplitudes = []
        self.before_bandwidths = []
        
        
        for i in range(self.simple_number):
            for j in range(self.phis_sfs):
                
                curve = self.results[i, j, position, position, :]
                initial_params = self.initial_tuning_curves[i, j, position, position, :]
                amplitude = curve.max() - curve.min()
                self.after_amplitudes.append(amplitude)
                
                halfmax_amplitude = torch.abs(curve.max()) - torch.abs(curve.min())
                halfmax = halfmax_amplitude/2
                halfmax_index1 = self.find_nearest(curve, halfmax)
                temporary = torch.cat([curve[0:halfmax_index1], curve[halfmax_index1+1:]])
                halfmax_index2 = self.find_nearest(temporary, halfmax)
                
                add = 0
                while halfmax_index1 - 2 <= halfmax_index2 <= halfmax_index1 + 2:
                    temporary = torch.cat([temporary[0:halfmax_index2], temporary[halfmax_index2+1:]])
                    halfmax_index2 = self.find_nearest(temporary, halfmax)
                    add += 1
                if halfmax_index1 < halfmax_index2:
                    halfmax_index2 = halfmax_index2 + 1 + add
                
                try1 = np.abs(x[halfmax_index1] - x[halfmax_index2])
                try2 = 180 - np.abs(x[halfmax_index1] - x[halfmax_index2])
                if try1 < try2:
                    bandwidth = try1
                else:
                    bandwidth = try2
                self.after_bandwidths.append(bandwidth)
                
                amplitude2 = initial_params.max() - initial_params.min()
                self.before_amplitudes.append(amplitude2)
            
                halfmax2_amplitude2 = torch.abs(initial_params.max()) - torch.abs(initial_params.min())
                halfmax2 = halfmax2_amplitude2/2
                halfmax2_index1 = self.find_nearest(initial_params, halfmax2)
                temporary2 = torch.cat([initial_params[0:halfmax2_index1], initial_params[halfmax2_index1+1:]])
                halfmax2_index2 = self.find_nearest(temporary2, halfmax2)
                
                add = 0
                while halfmax2_index1 - 2 <= halfmax2_index2 <= halfmax2_index1 + 2:
                    temporary2 = torch.cat([temporary2[0:halfmax2_index2], temporary2[halfmax2_index2+1:]])
                    halfmax2_index2 = self.find_nearest(temporary2, halfmax2)
                    add += 1
                if halfmax2_index1 < halfmax2_index2:
                    halfmax2_index2 = halfmax2_index2 + 1 + add
                try1 = np.abs(x[halfmax2_index1] - x[halfmax2_index2])
                try2 = 180 - np.abs(x[halfmax2_index1] - x[halfmax2_index2])
                if try1 < try2:
                    bandwidth2 = try1
                else:
                    bandwidth2 = try2
                self.before_bandwidths.append(bandwidth2)
          
        
        self.v1_mean_after_amplitude = np.mean(self.after_amplitudes)
        self.v1_std_after_amplitude = np.std(self.after_amplitudes)

        self.v1_mean_after_bandwidth = np.mean(self.after_bandwidths)
        self.v1_std_after_bandwidth = np.std(self.after_bandwidths)

        self.v1_mean_before_amplitude = np.mean(self.before_amplitudes)
        self.v1_std_before_amplitude = np.std(self.before_amplitudes)

        self.v1_mean_before_bandwidth = np.mean(self.before_bandwidths)
        self.v1_std_before_bandwidth = np.std(self.before_bandwidths)

        self.v1_amplitude_difference = ((self.v1_mean_after_amplitude - self.v1_mean_before_amplitude)/self.v1_mean_before_amplitude)*100      
        self.v1_bandwidth_difference = ((self.v1_mean_after_bandwidth - self.v1_mean_before_bandwidth)/self.v1_mean_before_bandwidth)*100
        
        
    def v4_tuning_params(self, position):
        
        """
        Calculates the amplitude and bandwidth of V4 tuning curves. Takes position of gabor filter as argument.  
        """
        
        
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = (x * 180) / np.pi
        self.after_amplitudes = []
        self.after_bandwidths = []
        
        self.before_amplitudes = []
        self.before_bandwidths = []
        
        
        for i in range(self.v4_orientation_number):

            curve = self.v4_results[i, position, position, :]
            initial_params = self.v4_initial_tuning_curves[i, position, position, :]
            amplitude = curve.max() - curve.min()
            self.after_amplitudes.append(amplitude)

            halfmax_amplitude = torch.abs(curve.max()) - torch.abs(curve.min())
            halfmax = halfmax_amplitude/2
            halfmax_index1 = self.find_nearest(curve, halfmax)
            temporary = torch.cat([curve[0:halfmax_index1], curve[halfmax_index1+1:]])
            halfmax_index2 = self.find_nearest(temporary, halfmax)

            add = 0
            while halfmax_index1 - 2 <= halfmax_index2 <= halfmax_index1 + 2:
                temporary = torch.cat([temporary[0:halfmax_index2], temporary[halfmax_index2+1:]])
                halfmax_index2 = self.find_nearest(temporary, halfmax)
                add += 1
            if halfmax_index1 < halfmax_index2:
                halfmax_index2 = halfmax_index2 + 1 + add

            try1 = np.abs(x[halfmax_index1] - x[halfmax_index2])
            try2 = 180 - np.abs(x[halfmax_index1] - x[halfmax_index2])
            if try1 < try2:
                bandwidth = try1
            else:
                bandwidth = try2
            self.after_bandwidths.append(bandwidth)

            amplitude2 = initial_params.max() - initial_params.min()
            self.before_amplitudes.append(amplitude2)

            halfmax2_amplitude2 = torch.abs(initial_params.max()) - torch.abs(initial_params.min())
            halfmax2 = halfmax2_amplitude2/2
            halfmax2_index1 = self.find_nearest(initial_params, halfmax2)
            temporary2 = torch.cat([initial_params[0:halfmax2_index1], initial_params[halfmax2_index1+1:]])
            halfmax2_index2 = self.find_nearest(temporary2, halfmax2)

            add = 0
            while halfmax2_index1 - 2 <= halfmax2_index2 <= halfmax2_index1 + 2:
                temporary2 = torch.cat([temporary2[0:halfmax2_index2], temporary2[halfmax2_index2+1:]])
                halfmax2_index2 = self.find_nearest(temporary2, halfmax2)
                add += 1
            if halfmax2_index1 < halfmax2_index2:
                halfmax2_index2 = halfmax2_index2 + 1 + add
            try1 = np.abs(x[halfmax2_index1] - x[halfmax2_index2])
            try2 = 180 - np.abs(x[halfmax2_index1] - x[halfmax2_index2])
            if try1 < try2:
                bandwidth2 = try1
            else:
                bandwidth2 = try2
            self.before_bandwidths.append(bandwidth2)
        
        self.v4_mean_after_amplitude = np.mean(self.after_amplitudes)
        self.v4_std_after_amplitude = np.std(self.after_amplitudes)

        self.v4_mean_after_bandwidth = np.mean(self.after_bandwidths)
        self.v4_std_after_bandwidth = np.std(self.after_bandwidths)

        self.v4_mean_before_amplitude = np.mean(self.before_amplitudes)
        self.v4_std_before_amplitude = np.std(self.before_amplitudes)

        self.v4_mean_before_bandwidth = np.mean(self.before_bandwidths)
        self.v4_std_before_bandwidth = np.std(self.before_bandwidths)

        self.v4_amplitude_difference = ((self.v4_mean_after_amplitude - self.v4_mean_before_amplitude)/self.v4_mean_before_amplitude)*100      
        self.v4_bandwidth_difference = ((self.v4_mean_after_bandwidth - self.v4_mean_before_bandwidth)/self.v4_mean_before_bandwidth)*100
        
    def otc_curve(self, trained_angle, v1_position, v4_position):
        
        """
        Calculates slope of each tuning curve before and after training at the trained angle. Takes in trained angle, V1 position 
        and V4 position as arguments. 
        """
        
        x = np.linspace(-np.pi/2, np.pi/2, self.tuning_curve_sample)
        x = x * 180 / np.pi
        trained_angle = trained_angle * 180 / np.pi
        
        self.v1_mean_after_slopes = []
        self.v1_mean_before_slopes = []
        
        self.v1_after_range = []
        self.v1_before_range = []
        
        for i in range(self.simple_number):
            
            after_slopes = []
            before_slopes = []
            
            after_ranges = []
            before_ranges = []
            
            trained_index = self.find_nearest(torch.tensor(x), trained_angle)
            
            for j in range(self.phis_sfs):
                curve = self.results[i, j, v1_position, v1_position, :]
                initial = self.initial_tuning_curves[i, j, v1_position, v1_position, :]
                
                if torch.abs(curve.max()) > torch.abs(curve.min()):
                    after_preferred_orientation = curve.argmax().item()
                    before_preferred_orientation = initial.argmax().item()
                
                else:
                    after_preferred_orientation = curve.argmin().item()
                    before_preferred_orientation = initial.argmin().item()
                
                after_ranges.append(x[after_preferred_orientation] - x[trained_index])
                before_ranges.append(x[before_preferred_orientation] - x[trained_index])
                
                after_slope = (curve[trained_index + 1] - curve[trained_index])/(180/self.tuning_curve_sample)
                before_slope = (initial[trained_index + 1] - initial[trained_index])/(180/self.tuning_curve_sample)
                after_slopes.append(torch.abs(after_slope).item())
                before_slopes.append(torch.abs(before_slope).item())
            
            mean_after_range = np.mean(after_ranges)
            mean_before_range = np.mean(before_ranges)
            
            self.v1_after_range.append(mean_after_range)
            self.v1_before_range.append(mean_before_range)

            mean_after_slope = np.mean(after_slopes)
            mean_before_slope = np.mean(before_slopes)
            
            self.v1_mean_after_slopes.append(mean_after_slope)
            self.v1_mean_before_slopes.append(mean_before_slope)
        
            
        self.v4_after_slopes = []
        self.v4_before_slopes = []
        
        self.v4_after_range = []
        self.v4_before_range = []
        
        
        for k in range(self.v4_orientation_number):
            
            curve = self.v4_results[k, v4_position, v4_position, :]
            initial = self.v4_initial_tuning_curves[k, v4_position, v4_position, :]
            
            after_preferred_orientation = curve.argmax().item()
            before_preferred_orientation = initial.argmax().item()

            self.v4_after_range.append(x[after_preferred_orientation] - x[trained_index])
            self.v4_before_range.append(x[before_preferred_orientation] - x[trained_index])
            
            after_slope = (curve[trained_index + 1] - curve[trained_index])/(180/self.tuning_curve_sample)
            before_slope = (initial[trained_index + 1] - initial[trained_index])/(180/self.tuning_curve_sample)
            self.v4_after_slopes.append(torch.abs(after_slope).item())
            self.v4_before_slopes.append(torch.abs(before_slope).item())
        
        
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

