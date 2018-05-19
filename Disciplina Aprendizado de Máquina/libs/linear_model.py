import numpy as np
from scipy.optimize import fmin_bfgs
import sys

class LogisticRegression(object):
    
    def fit(self, X, y):
        self.X = np.hstack((np.ones((X.shape[0],1)),X))
        self.y = y
        self.nData = X.shape[0]
        self.nEta = X.shape[1]+1
        self.eta = fmin_bfgs(self.costFunction,np.zeros(self.nEta),fprime=self.gradient)
   
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)),X))
        return np.round(self.sigmoid(np.dot(X,self.eta)))
    
    def sigmoid(self,z):
        return 1.0 / ( 1.0+np.exp(-z) )

    def costFunction(self,eta):
        prob = self.sigmoid(np.dot(self.X,eta))
        prob_ = 1.0-prob
        np.place(prob,prob==0.0,sys.float_info.min)
        np.place(prob_,prob_==0.0,sys.float_info.min)
        self.cost = np.dot(self.y,np.log(prob))
        self.cost += np.dot((1.0-self.y),np.log(prob_))
        return -self.cost

    def gradient(self,eta):
        grad = np.zeros(self.nEta)
        for i in range(grad.shape[0]):
            grad[i] = ( (self.sigmoid(np.dot(self.X,eta))-self.y)*self.X[:,i] ).sum()
        return grad

# y = b0 + b1 * x
# b1 é a inclinação, b0 é y-intercept
def compute_error_for_line_given_points(b0, b1, x, y):
    totalError = np.sum((y - (b1 * x + b0)) ** 2)
    return totalError / float(len(y))

def step_gradient(b0_current, b1_current, x, y, learning_rate):
    N = float(len(y))
    b0_gradient = 2/N * np.sum(-(y - ((b1_current * x) + b0_current)))
    b1_gradient = 2/N * np.sum(-x * (y - ((b1_current * x) + b0_current)))
    new_b0 = b0_current - (learning_rate * b0_gradient)
    new_b1 = b1_current - (learning_rate * b1_gradient)
    return new_b0, new_b1

def gradient_descent_runner(x, y, b0, b1, learning_rate, num_iterations):
    for _ in range(num_iterations):
        b0, b1 = step_gradient(b0, b1, x, y, learning_rate)
    return b0, b1

class SimpleLinearRegression(object):
    
    def fit(self, X_, y):
        X = X_[:,0]
        self.b1_ = np.sum(((X - np.mean(X))*(y-np.mean(y)))) / np.sum((X - np.mean(X))**2)
        self.b0_ = np.mean(y) - self.b1_*np.mean(X)
    
    def predict(self,X):
        return self.b1_*X[:,0] + self.b0_
    

    
    
    