import numpy as np
from mtuq.util.cmaes import *
from mtuq.util.math import to_mij, to_rtp, to_rho
from mtuq.grid.force import to_force



# class CMA_ES(object):


class CMA_ES(object):

    def __init__(self, parameters_list, lmbda=None, data=None, GFclient=None, origin=None, callback_function=None): # Initialise with the parameters to be used in optimisation.

        # Initialize parameters-tied variables.
        self._parameters = parameters_list
        self.n = len(self._parameters)
        self.xmean = np.asarray([[val.initial for val in self._parameters]]).T
        self.sigma = 2
        self.origin = origin
        if not callback_function == None:
            self.callback = callback_function
        elif len(parameters_list) == 6 or len(parameters_list) == 9:
            self.callback = to_mij
            self.mij_args = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
        elif len(parameters_list) == 3:
            self.callback = to_force
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

    def draw_mutants(self):
        bounds = [0,10]
        for i in range(self.lmbda):
            mutant = self.xmean + self.sigma * self.B @ (self.D * np.random.randn(self.n,1))
            self.mutants[:,i] = mutant.T
        for _i, param in enumerate(self.mutants):
            if not self._parameters[_i].repair == None:
                while array_in_bounds(self.mutants[_i], 0, 10) == False:
                    print('repairing '+self._parameters[_i].name+' with '+self._parameters[_i].repair+' method')
                    Repair(self._parameters[_i].repair, self.mutants[_i], self.xmean[_i])

    def eval_fitness(self):
        # Project each parameter in their respective physical domain, according to their `scaling` property
        self.transformed_mutants = np.zeros_like(self.mutants)
        for _i, param in enumerate(self._parameters):
            print(param.scaling)
            if param.scaling == 'linear':
                self.transformed_mutants[_i] = linear_transform(self.mutants[_i], param.lower_bound, param.upper_bound)
            elif param.scaling == 'log':
                self.transformed_mutants[_i] = logarithmic_transform(self.mutants[_i], param.lower_bound, param.upper_bound)
            else:
                raise ValueError("Unrecognized scaling, must be linear or log")
            # Apply optional projection operator to each parameter
            if not param.projection is None:
                self.transformed_mutants[_i] = np.asarray(list(map(param.projection, self.transformed_mutants[_i])))
        self.sources = np.ascontiguousarray(self.callback(*self.transformed_mutants))

        # Evaluate the misfit for each mutant of the population.

    def create_origins(self, **kwargs):
        catalog_origin = self.origin

        depths = np.array(
             # depth in meters
            [25000., 30000., 35000., 40000.,
             45000., 50000., 55000., 60000.])

        origins = []
        for depth in depths:
            origins += [catalog_origin.copy()]
            setattr(origins[-1], 'depth_in_m', depth)
