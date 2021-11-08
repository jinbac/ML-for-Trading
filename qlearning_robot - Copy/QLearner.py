""" 			  		 			     			  	   		   	  			  	
Template for implementing QLearner  (c) 2015 Tucker Balch 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Copyright 2018, Georgia Institute of Technology (Georgia Tech) 			  		 			     			  	   		   	  			  	
Atlanta, Georgia 30332 			  		 			     			  	   		   	  			  	
All Rights Reserved 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Template code for CS 4646/7646 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Georgia Tech asserts copyright ownership of this template and all derivative 			  		 			     			  	   		   	  			  	
works, including solutions to the projects assigned in this course. Students 			  		 			     			  	   		   	  			  	
and other users of this template code are advised not to share it with others 			  		 			     			  	   		   	  			  	
or to make it available on publicly viewable websites including repositories 			  		 			     			  	   		   	  			  	
such as github and gitlab.  This copyright statement should not be removed 			  		 			     			  	   		   	  			  	
or edited. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
We do grant permission to share solutions privately with non-students such 			  		 			     			  	   		   	  			  	
as potential employers. However, sharing with other current or future 			  		 			     			  	   		   	  			  	
students of CS 7646 is prohibited and subject to being investigated as a 			  		 			     			  	   		   	  			  	
GT honor code violation. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
-----do not edit anything above this line--- 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Student Name: Jarod Kennedy (replace with your name)
GT User ID: jKennedy76 (replace with your User ID)
GT ID: 903369277 (replace with your GT ID)
""" 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
import numpy as np 			  		 			     			  	   		   	  			  	
import random as rand 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
class QLearner(object):


 			  		 			     			  	   		   	  			  	
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 100, \
        verbose = False): 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
        self.verbose = verbose 			  		 			     			  	   		   	  			  	
        self.num_actions = num_actions 			  		 			     			  	   		   	  			  	
        self.s = 0 			  		 			     			  	   		   	  			  	
        self.a = 0

        self.qtable = np.zeros((num_states,num_actions)) # initialize Q as zeros
        self.Tdyna =  np.zeros((dyna,3)) # dyna Transition model as zeros
        self.Tcount = 0.00001 # initialize Tcount as small number

        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna

    def author(self):
        return 'jKennedy76'
 			  		 			     			  	   		   	  			  	
    def querysetstate(self, s): 			  		 			     			  	   		   	  			  	
        """ 			  		 			     			  	   		   	  			  	
        @summary: Update the state without updating the Q-table 			  		 			     			  	   		   	  			  	
        @param s: The new state 			  		 			     			  	   		   	  			  	
        @returns: The selected action 			  		 			     			  	   		   	  			  	
        """ 			  		 			     			  	   		   	  			  	
        self.s = s 			  		 			     			  	   		   	  			  	
        action = rand.randint(0, self.num_actions-1) 			  		 			     			  	   		   	  			  	
        if self.verbose: print "s =", s,"a =",action 			  		 			     			  	   		   	  			  	
        return action 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    def query(self,s_prime,r): 			  		 			     			  	   		   	  			  	
        """ 			  		 			     			  	   		   	  			  	
        @summary: Update the Q table and return an action 			  		 			     			  	   		   	  			  	
        @param s_prime: The new state 			  		 			     			  	   		   	  			  	
        @param r: The ne state 			  		 			     			  	   		   	  			  	
        @returns: The selected action 			  		 			     			  	   		   	  			  	
        """

        action = np.argmax(self.qtable[s_prime, :]) # set action to whatever action in the Q table at state S_prime is highest

        # maybe ignore Q table and choose a random action
        if np.random.random() < self.rar:
            action = rand.randint(0, self.num_actions-1) 	# selects a random action

        LaterReward = np.max(self.qtable[s_prime, :])  # later reward is best choice in NEXT STEP ONLY
        self.qtable[self.s,self.a] = (1 - self.alpha) * self.qtable[self.s,self.a] + self.alpha * (r + LaterReward * self.gamma) # update Q table

        # Decay random rate
        self.rar = self.rar * self.radr

        #update state and action, not sure if this does anything
        self.s = s_prime
        self.a = action


        #dyna Q
        i = 0.
        while i<self.dyna:
            i +=1
            #learn T




        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
if __name__=="__main__": 			  		 			     			  	   		   	  			  	
    print "Remember Q from Star Trek? Well, this isn't him" 			  		 			     			  	   		   	  			  	
