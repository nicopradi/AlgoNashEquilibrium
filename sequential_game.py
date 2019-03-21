# Modelisation of the sequential game with continuous price and capacity

# General
import sys
import time
import copy
import matplotlib.pyplot as plt
# CPLEX
import cplex
from cplex.exceptions import CplexSolverError
# numpy
import numpy as np
# data
import Data.Stackelberg.MILPLogit_n10r100 as data_file
import Data.Non_linear_Stackelberg.ProbLogit_n10 as data_file_2
# Stackelberg
import stackelberg_game
import non_linear_stackelberg

class Sequential:

    def __init__(self, **kwargs):
        ''' Construct a sequential game.
            KeywordArgs:
                K          Number of operators
                Operator   Mapping between alternatives and operators
                maxIter    Maximum number of iterations
                Optimizer  Current operator index
                p_fixed    Current prices of the other operator's alternatives
                p_fixed    Current availabilty of the other operator's alternatives
        '''
        self.K = kwargs.get('K', 1)
        self.Operator = kwargs.get('Operator', np.full((1, self.K + 1), 1))
        self.maxIter = kwargs.get('maxIter', 10)
        self.Optimizer = kwargs.get('Optimizer', 1)
        self.p_fixed = kwargs.get('p_fixed', np.full((1, len(self.Operator)), 1.0))
        self.y_fixed = kwargs.get('y_fixed', np.full((1, len(self.Operator)), 1.0))
        self.p_history = np.full((self.maxIter, len(self.Operator)), -1.0)
        # Make a copy of the initial optimizer
        self.InitialOptimizer = self.Optimizer
        self.InitialPrice = copy.deepcopy(self.p_fixed)

    def run(self, data, linearized=True):
        ''' Run the sequential game with the Stackelberg game
            Args:
                data:          dictionary containing data to instanciate the Stackelberg game
        '''
        iter = 0
        cycle = False
        cycle_iter = 0
        while (iter < self.maxIter) and cycle is False:
            print('\n--- ITERATION %r ----' %iter)
            for (index, p) in enumerate(self.p_fixed):
                if p != -1.0:
                    print('The operator %r fixed the price of alternative %r to %r'
                           %(self.Operator[index], index, p))
            # Run the game for the current optimizer
            if linearized is True:
                sub_game = stackelberg_game.Stackelberg(**data)
                model = sub_game.getModel()
                model = sub_game.solveModel(model)
                prices = []
                for i in range(len(self.Operator)):
                    prices.append(model.solution.get_values('p[' + str(i) + ']'))
            else:
                prices = non_linear_stackelberg.main(data)
            # Update the price history
            for i in range(len(self.Operator)):
                self.p_history[iter, i] = prices[i]
            # Check for the cycle
            if iter >= self.K:
                iteration_to_check = range(iter-self.K, -1, -self.K)
                print('ITERATION TO CHECK: %r' %list(iteration_to_check))
                for j in iteration_to_check:
                    cycle = True
                    for i in range(len(self.Operator)):
                        # Ignore the price of the alternative managed by the current optimizer
                        if self.Operator[i] != self.Optimizer:
                            # TODO: Numerical error ? Tolerance 1e-6
                            if abs(self.p_history[j, i] - self.p_history[iter, i]) > 1e-3:
                                cycle = False
                    if cycle is True:
                        cycle_iter = j
                        if iter-cycle_iter == self.K:
                            print('\nNash equilibrium detected')
                        else:
                            print('\nCycle detected')
                        break
            # Update the data for the next iteration
            data['Optimizer'] = ((data['Optimizer']) % self.K) + 1
            self.Optimizer = data['Optimizer']
            # Fix the price of the operators exept for the next optimizer
            data['p_fixed'] = copy.deepcopy(self.p_history[iter])
            self.p_fixed = copy.deepcopy(self.p_history[iter])
            for (i, k) in enumerate(self.Operator):
                if k == data['Optimizer']:
                    data['p_fixed'][i] = -1.0
                    self.p_fixed[i] = -1.0
            # Go to the next iteration
            iter += 1

        print('Price history: %r' %self.p_history)

    def plotGraphs(self, title):
        ''' Plot the value of the fixed prices for each optimizer as a function
            of the iterations number.
        '''

        # Get the price history for each operators
        p_history_1 = [prices[1] for prices in self.p_history if prices[1] != -1]
        p_history_2 = [prices[2] for prices in self.p_history if prices[2] != -1]
        # Plot them
        plt.plot(p_history_2, label='Operator 2 price', color='blue')
        plt.plot(p_history_1, label='Operator 1 price', color='red')
        plt.ylabel('Price')
        plt.title("Operator's prices as a function of the iteration number. \
        \n The initial prices are: Operator 1: %r and Operator 2: %r" %(self.InitialPrice[1], self.InitialPrice[2]))
        plt.legend()
        plt.savefig('price_history_%r.png' %(title))
        plt.close()


if __name__ == '__main__':
    # LINEAR
    stackelberg_dict = data_file.getData()
    data_file.preprocess(stackelberg_dict)

    for initial_price in np.arange(0.1, 1.01, 0.1):
        initial_price = round(float(initial_price), 3)
        sequential_dict = {'K': 2,
                        'Operator': [0, 1, 2],
                        'maxIter': 50,
                        'Optimizer': 1,
                        'p_fixed': [0.0, -1.0, initial_price],
                        'y_fixed': [1.0, 1.0, 1.0]}
        sequential_game = Sequential(**sequential_dict)
        # Update the dict with the attributes of the Stackelberg game
        sequential_dict.update(stackelberg_dict)
        sequential_game.run(sequential_dict, linearized=True)
        sequential_game.plotGraphs(initial_price)
    '''
    # NON LINEAR
    stackelberg_dict = data_file_2.getData()
    data_file_2.preprocess(stackelberg_dict)

    for initial_price in np.arange(0.1, 1.01, 0.1):
        initial_price = float(round(float(initial_price), 3))
        sequential_dict = {'K': 2,
                        'Operator': [0, 1, 2],
                        'maxIter': 50,
                        'Optimizer': 2,
                        'p_fixed': [0.0, initial_price, -1.0],
                        'y_fixed': [1.0, 1.0, 1.0]}
        sequential_game = Sequential(**sequential_dict)
        # Update the dict with the attributes for the Stackelberg game
        sequential_dict.update(stackelberg_dict)
        sequential_game.run(sequential_dict, linearized=False)
        sequential_game.plotGraphs(initial_price)
    '''
