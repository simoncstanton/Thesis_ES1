#!/usr/bin/env python3
# File: watkins_q_lfa.py

import os

import random
from random import randint

import copy
from collections import deque
import numpy as np

from .strategy import Strategy


class Watkins_q_lfa(Strategy):
    
    def __init__(self, strategy, strategy_options):
        super().__init__(strategy, strategy_options)

        self.state_length = 2 * self.options["memory_depth"]
        '''
        RL method
            Watkins Linear Function Approximation, Sutton&Barto 1st edition
                with binary features, e-greedy policy, accumulating traces
        
        '''
        
        self.memory = deque(maxlen=self.options["memory_depth"])
        self.state = None
        self.prev_state = None
        
        self.weights_initial = 0
        
        self.features = self.get_feature_vector()
        
        self.weights = [self.weights_initial for i in self.features]
        #print(os.path.basename(__file__) + ":: features at instantiation", self.features)
        #print(os.path.basename(__file__) + ":: weights at instantiation", self.weights)

        # making some of the internal strategy state variables as object vars to enable testing for inf values, and applying guards.
        self.prev_q = 0
        self.next_q = 0
        self.td_error = 0
        
        self.actions = [0,1]
        
        self.qs = list(np.dot(self.get_feature_vector(self.state, i), self.weights) for i in self.actions)

        #   i forget why this noop is here, was probably checking in print debug but left the noop behind. 
        #       commenting out for now, TODO remove if no observed negative outcome
        # self.get_maximal_action(self.qs)
        
        # to catch warnings as exceptions so we can catch and suppress RuntimeWarning:OverflowError
        np.seterr(all = 'raise')

        self.overflow = False

    def action(self, agent, previous_step) -> int:
        '''     
            
            
            t       : timestep
            r       : reward at t
            
            
            
        '''
        super().action(agent, previous_step)
        
        
        ''' first move, act according to distribution supplied by option '''
        if len(agent.action_history) == 0:
            rnd = random.random()
            if rnd > (1 - self.options["initial_action"]):
                action = 1
            else:
                action = 0
        
        else:
                        
            # agent_id maps to the tuple position to get that (this) agent's action out of the __previous_step__ data structure (which is a tuple)
            prev_action = previous_step[agent.agent_id]
            
            '''
                obtain reward from last action
                    type(r) == int
                r is stored in agent object as matrix payoff from last action
                if no previous reward (i.e. this is the first round, then previous_reward == 0
            '''
             
            r = agent.previous_reward
            

            # note state
            ''' 
            append previous_step actions to agent memory
            previous_step is a tuple (n, n); n = (0|1)
                where 
                    position indicates agent (0 or 1)
                    value indicates action (0, or 1)
            '''
            self.memory.append(previous_step)
            
            '''
                translate n most recent paired actions to str
                    where 
                        n = self.memory_depth
                    gives 
                        len(self.state) == 2 * self.memory_depth
            '''
            self.state = self.memory_to_state_key()
            
            
            self.qs = list(np.dot(self.get_feature_vector(self.prev_state, i), self.weights) for i in self.actions)
            
            '''
                epsilon policy
            '''
            rnd = random.random() 
            if rnd > self.options["epsilon"]: 
                action = self.get_maximal_action(self.qs)

            else:
                action = randint(0,1)
            
            
            # added np.isinf() and try/catch blocks to suppress overflow warnings, and inhibit propogation of invalid values.
            #   no doubt, cleaner logic possible. atm, preserves algorithm, and does the above. (020921)
            #   note, these issues have only been observed to arise in long episodes, i.e., over 20k timesteps
            
            phi = self.get_feature_vector(self.prev_state, prev_action)
            
            prev_q = np.dot(phi, self.weights)
            
            if not np.isinf(prev_q):
                self.prev_q = prev_q
            
            phi_next = self.get_feature_vector(self.state, action)
            
            next_q = np.dot(phi_next, self.weights)
            
            if not np.isinf(next_q):
                self.next_q = next_q
            
            try:
                td_error = r + (self.options["gamma"] * self.next_q) - self.prev_q

            except FloatingPointError:
                if not self.overflow:
                    print("FP error in " + self.options["name"] + " at " + str(len(agent.action_history)))
                    self.overflow = True
            
                # if this occurs, just carry on, using old values .. but print to stdout (will be in OU file) on first event only
                #   otherwise log files grow and computation slows
                # need to propagate this catch to agent and/or trial to get further visibility [TODO 090921]
            
            else:
                self.td_error = td_error

                
            # in long (~10,000 < ts < ~100,000) episodes td_error can overflow. if this happens the weights are *probably* so extreme that leaving them as they are is *maybe* the best course of action.
            #    so, try the two operations that can overflow, if ok update self.weights, otherwise write message to STD_OUT and leave self.weights as is.  

            for i in range(len(self.weights)):
                
                try:
                    weight_i = self.options["alpha"] * self.td_error * phi_next[i]
                    weight_j = self.weights[i] + weight_i
                    
                except FloatingPointError:
                    if not self.overflow:
                        # turn off for tournament (hack)
                    # print("FP error in " + self.options["name"] + " at " + str(len(agent.action_history)))
                        self.overflow = True
                
                    # if this occurs, just carry on, using old values .. but print to stdout (will be in OU file) on first event only
                    #   otherwise log files grow and computation slows
                    # need to propagate this catch to agent and/or trial to get further visibility [TODO 090921]
                
                else:
                    self.weights[i] = weight_j
                            

        # 7. state -> previous_state
        '''
            push current state into the past
        '''
        self.prev_state = self.state
        
        return action

    def get_maximal_action(self, q_a):
        #print("q_a", q_a)
        maxq = max(q_a)
        #print("index",  q_a.index(maxq))
        return q_a.index(maxq)
        
    def get_feature_vector(self, state=None, action=None):
    
        # artint.info/html/ArtInt_272.html 
        #   w_0 shoud always be 1.
        
        if state == None and action == None:
            return [1, 1, 1, 1, 1]
        
        f1 = 1 if state == '00' and action == 0 else 0 # CC, C
        f2 = 1 if state == '01' and action == 0 else 0 # CD, C
        f3 = 1 if state == '10' and action == 1 else 0 # DC, D
        f4 = 1 if state == '11' and action == 1 else 0 # DD, D
        
        
        self.features = [1, f1, f2, f3, f4]
        return self.features


    def get_strategy_internal_state(self):
        return { "state" : self.state, "f" : self.features.copy(), "weights" : copy.deepcopy(self.weights) }  # , "weights" : copy.deepcopy(self.weights)
       
if __name__ == '__main__':
    print(self.options["name"])