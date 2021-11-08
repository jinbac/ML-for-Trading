u"""
template for generating data to fool learners (c) 2016 Tucker Balch 			  		 			     			  	   		   	  			  	
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
import math 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
# this function should return a dataset (X and Y) that will work 			  		 			     			  	   		   	  			  	
# better for linear regression than decision trees 			  		 			     			  	   		   	  			  	
def best4LinReg(seed):
    np.random.seed(seed) 			  		 			     			  	   		   	  			  	
    # X = np.zeros((100,2))
    # Y = np.random.random(size = (100,))*200-100
    # # Here's is an example of creating a Y from randomly generated
    # # X with multiple columns
    # # Y = X[:,0] + np.sin(X[:,1]) + X[:,2]**2 + X[:,3]**3

    noise = np.random.random(size=(100,)) / 10
    x1 = np.linspace(0, 10, 100) - noise
    x2 = np.linspace(0, 10, 100) + noise
    X = np.array([x1, x2]).transpose()
    Y = np.copy(x1)
    idx = np.random.random(size=(10,)) * 100
    idx = idx.astype(int)
    Y[idx] = Y[idx] * -1   # throw off DT with strong linear trend with some outliers




    return X, Y 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
def best4DT(seed):
    np.random.seed(seed) 			  		 			     			  	   		   	  			  	
    # X = np.zeros((100,2))
    # Y = np.random.random(size = (100,))*200-100

    x1 = np.linspace(0, 10, 100)
    x2 = np.linspace(0, 10, 100) * -1
    X = np.array([x1, x2]).transpose()
    y = np.sin(x1)            # Lin reg will be bad at prediction on sin
    noise = np.random.random(size=(100,)) / 10
    Y = y + noise


    return X, Y 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
def author(): 			  		 			     			  	   		   	  			  	
    return 'jKennedy76' #Change this to your user ID
 			  		 			     			  	   		   	  			  	
if __name__=="__main__": 			  		 			     			  	   		   	  			  	
    print "they call me Tim."

