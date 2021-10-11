import numpy as np
from mtuq.util.cmaes import Repair

# class CMA_ES(object):


class CMA_ES(object):

    def __init__(self, parameters_list, lmbda=None): # Initialise with the parameters to be used in optimisation.
        # Initialize parameters-tied variables.
        self._parameters = parameters_list
        self.n = len(self._parameters)
        self.xmean = np.random.randn(self.n,1)
        self.sigma = 2

        # Parameter setting
        if not lmbda == None:
            self.lmbda = lmbda
        else:
            self.lmbda = 40
        self.mu = np.floor(self.lmbda/2)
        a = 1 # Original author uses 1/2 in tutorial and 1 in publication
        self.weights = np.array([np.log(self.mu+a) - np.log(np.arange(1, self.mu+1))]).T
        self.weights /= sum(self.weights)
        self.mueff = sum(self.weights)**2/sum(self.weights**2)

        # Step-size control
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1)/(self.n + 1)) - 1) + self.cs

        # Covariance matrix adaptation
        self.cc = (4 + self.mueff / self.n)/(self.n + 4 + 2 * self.mueff / self.n)
        self.acov = 2
        self.c1 = self.acov / ((self.n + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, self.acov * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2)**2 + self.acov*self.mueff/2))

        # INITIALIZATION
        self.ps = np.zeros_like(self.xmean)
        self.pc = np.zeros_like(self.xmean)
        self.B = np.eye(self.n,self.n)
        self.D = np.ones((self.n, 1))
        self.C = self.B @ np.diag(self.D[:,0]**2) @ self.B.T
        self.invsqrtC = self.B @ np.diag(self.D[:,0]**-1) @ self.B.T
        self.eigeneval = 0
        self.chin = self.n**0.5 * (1 - 1 / ( 4 * self.n) + 1 / (21 * self.n**2))
        self.mutants = np.zeros((self.n, self.lmbda))

    def draw_muants(self):
        bounds = [0,10]
        for i in range(self.lmbda):
            mutant = self.xmean + self.sigma * self.B @ (self.D * np.random.randn(self.n,1))
            self.mutants[:,i] = mutant.T
        for _i, param in enumerate(self.mutants):
            while array_in_bounds(self.mutants[_i], 0, 10) == False:
                print('repairing '+self._parameters[_i].name+' with '+self._parameters[_i].repair+' method')
                Repair(self._parameters[_i].repair, self.mutants[_i])
