# Modelisation of the sequential game with continuous price and capacity
# The linear or non linear formulation of the Stackelberg game can be used

# General
import sys
import time
import copy
import matplotlib.pyplot as plt
import warnings
# CPLEX
import cplex
from cplex.exceptions import CplexSolverError
# numpy
import numpy as np
# data
import Data.Parking_lot.Stackelberg.MILPLogit_n10r100 as data_file
import Data.Parking_lot.Non_linear_Stackelberg.ProbMixedLogit_n10r50 as data_file_2
# Stackelberg
import stackelberg_game
import non_linear_stackelberg

class Sequential:

    def __init__(self, **kwargs):
        ''' Construct a sequential game.
            KeywordArgs:
                K            Number of operators [int]
                operator     Mapping between alternatives and operators [list]
                max_iter     Maximum number of iterations [int]
                p_fixed      Current prices of the operators [list]
                y_fixed      Current availabilty of the other operator's alternatives [list]
                tolerance    Tolerance used for the cycle detection [float]
                name_mapping Mapping between alternatives id and their names [dict]
        '''
        # Keyword arguments
        self.K = kwargs.get('K', 1)
        self.operator = kwargs.get('operator', np.full((1, self.K + 1), 1))
        self.max_iter = kwargs.get('max_iter', 10)
        self.p_fixed = kwargs.get('p_fixed', np.full((1, len(self.operator)), 1.0))
        self.y_fixed = kwargs.get('y_fixed', np.full((1, len(self.operator)), 1.0))
        self.tolerance = kwargs.get('tolerance', 1e-3)
        # Keep track of the price, benefit, market share and demand at each iteration
        self.p_history = np.full((self.max_iter, len(self.operator)), -1.0)
        self.benefit = np.full((self.max_iter, self.K + 1), -1.0)
        self.market_share = np.full((self.max_iter, len(self.operator)), -1.0)
        self.demand = np.full((self.max_iter, len(self.operator)), -1.0)
        # Keep track of the iteration number from which the cycle starts
        self.cycle_iter = None

    def run(self, data, linearized=True):
        ''' Run the sequential game with the Stackelberg game
            Args:
                data           dictionary containing data to instanciate the Stackelberg game [dict]
                linearized     choice between the linear or non linear formulation fo the Stackelberg game [boolean]
        '''
        # Copy the mapping between alternatives index and their names
        self.name_mapping = copy.deepcopy(data['name_mapping'])
        # Start the sequential game
        iter = 0
        cycle = False
        while (iter < self.max_iter) and cycle is False:
            print('\n---- ITERATION %r ----' %iter)
            # Print the fixed prices
            for (index, p) in enumerate(data['p_fixed']):
                if p != -1.0:
                    print('The operator %r fixed the price of alternative %r to %r'
                           %(self.operator[index], index, p))
            ### Run the Stackelberg game for the current optimizer
            if linearized is True:
                # Linear formulation
                sub_game = stackelberg_game.Stackelberg(**data)
                model = sub_game.getModel()
                model = sub_game.solveModel(model)
                prices = []
                demand = []
                for i in range(len(self.operator)):
                    prices.append(model.solution.get_values('p[' + str(i) + ']'))
                    demand.append(model.solution.get_values('d[' + str(i) + ']'))
            else:
                # Non-Linear formulation
                prices, choice, x0, status, status_msg = non_linear_stackelberg.main(data)
                demand = []
                for i in range(len(self.operator)):
                    demand.append(np.sum([customer for customer in choice[i*data['N']:(i+1)*data['N']]]))
            # Update the price history
            for i in range(len(self.operator)):
                self.p_history[iter, i] = prices[i]
            # Update the benefit history
            for k in range(self.K + 1):
                self.benefit[iter][k] = 0.0
                alternatives = [i for i in range(len(self.operator)) if self.operator[i] == k]
                for i in alternatives:
                    self.benefit[iter][k] += demand[i]*prices[i]
                    if 'fixed_cost' in data.keys():
                        self.benefit[iter][k] += -data['fixed_cost'][i]
                    if 'customer_cost' in data.keys():
                        self.benefit[iter][k] += -data['customer_cost'][i]*demand[i]
            # Update the market share history
            for i in range(len(self.operator)):
                self.market_share[iter, i] = float(demand[i])/data['N']
            # Update the demand history
            for i in range(len(self.operator)):
                self.demand[iter, i] = demand[i]

            ### Cycle detection
            if iter >= self.K:
                # Check the prices at the iteration for which the current optimizer played
                iterations_to_check = range(iter-self.K, -1, -self.K)
                print('Iteration number to check: %r' %list(iterations_to_check))
                for j in iterations_to_check:
                    cycle = True
                    for i in range(len(self.operator)):
                        if abs(self.p_history[j, i] - self.p_history[iter, i]) > self.tolerance:
                            cycle = False
                    if cycle is True:
                        # Cycle detected, update the cycle iteration number
                        self.cycle_iter = j
                        if iter-self.cycle_iter == self.K:
                            print('\nNash equilibrium detected')
                        else:
                            print('\nCycle detected')
                        break

            ### Update the data for the next iteration
            data['optimizer'] = ((data['optimizer']) % self.K) + 1
            # Set the current solution to be the starting point in the next interation
            # except if it is unfeasible
            if linearized is False:
                print('\nIPOTP Status: %r\n' %status_msg)
                if status != 2:
                    data['x0'] = copy.deepcopy(x0)

            data['p_fixed'] = copy.deepcopy(self.p_history[iter])
            for (i, k) in enumerate(self.operator):
                if k == data['optimizer']:
                    data['p_fixed'][i] = -1.0
            # Go to the next iteration
            iter += 1

        print('Price history: %r' %self.p_history)

    def plotGraphs(self, title):
        ''' Plot the value of the fixed prices for each optimizer as a function
            of the iterations number.
            Args:
                title          identification string used to save the graphs [string]
        '''

        ### Price graph
        # Get the price history for each operators
        p_history = [[] for i in range(len(self.operator))]
        for i in range(len(self.operator)):
            p_history[i] = [prices for prices in self.p_history[:,i] if prices != -1]
        # Plot them
        color = {4: '0', 5: '0', 6: '0.7', 7: '0.7'}
        linestyle = {4: '-', 5: '--', 6: '-', 7: '--'}
        #color = {1: '0', 2: '0.7'}
        #linestyle = {1: '-', 2: '-'}
        for i in range(len(self.operator)):
            if self.operator[i] != 0:
                plt.plot(p_history[i], label=self.name_mapping[i] + ' price',
                         color=color[i], linestyle=linestyle[i], marker='x')
        # Plot vertical line to indicate the beginning of the cycle
        if self.cycle_iter is not None:
            plt.axvline(x=self.cycle_iter, linestyle=':', color='black', linewidth=0.5)
        plt.ylabel('Price')
        plt.xlabel('Iteration number')
        #plt.title("Alternatives' price as a function of the iteration number. \
        #\n The initial prices are: %r" %(self.p_fixed))
        plt.legend(loc='upper left')
        plt.savefig('price_history_%r.png' %(title))
        plt.close()

        ### Market share graph
        # Get the demand history for each operators
        market_history = [[] for i in range(len(self.operator))]
        nb_opt_out = self.operator.count(0)
        for i in range(len(self.operator)):
            market_history[i] = [market for market in self.market_share[:,i] if market != -1]
        market_sum = []
        for iter in range(len(market_history[0])):
            market_sum.append(sum([market_history[i][iter] for i in range(nb_opt_out, len(self.operator))]))

        # Plot them
        for i in range(len(self.operator)):
            if self.operator[i] != 0:
                plt.plot(market_history[i], label=self.name_mapping[i] + ' market share',
                         color=color[i], linestyle=linestyle[i], marker='x')
        plt.plot(market_sum, ':' ,label='Market total', color='black')
        # Plot vertical line to indicate the beginning of the cycle
        if self.cycle_iter is not None:
            plt.axvline(x=self.cycle_iter, linestyle=':', color='black', linewidth=0.5)
        plt.ylabel('Market share')
        plt.xlabel('Iteration number')
        #plt.title("Operator's market share as a function of the iteration number. \
        #\n The initial prices are: %r" %(self.p_fixed))
        plt.legend(loc='upper left')
        plt.savefig('market_history_%r.png' %(title))
        plt.close()

        ### Benefit graph
        # Get the benefit history for each operators
        benefit_history = [[] for k in range(self.K + 1)]
        for k in range(self.K + 1):
            benefit_history[k] = [benefits for benefits in self.benefit[:, k] if benefits != -1]
        benefit_sum = []
        for iter in range(len(benefit_history[0])):
            benefit_sum.append(sum([benefit_history[k][iter] for k in range(1, self.K + 1)]))

        # Plot them
        color = {1: '0', 2: '0.7'}
        for k in range(1, self.K + 1):
            if k == 1:
                operator = 'AV'
                #operator = '1'
            else:
                operator = 'NTV'
                #operator = '2'
            plt.plot(benefit_history[k], label=operator + ' benefit',
                    color=color[k], marker='x')
        plt.plot(benefit_sum, ':' ,label='Benefit total', color='black')
        # Plot vertical line to indicate the beginning of the cycle
        if self.cycle_iter is not None:
            plt.axvline(x=self.cycle_iter, linestyle=':', color='black', linewidth=0.5)
        plt.ylabel('Benefit')
        plt.xlabel('Iteration number')
        #plt.title("Operator's benefit as a function of the iteration number. \
        #\n The initial prices are: %r" %(self.p_fixed))
        plt.legend(loc='upper left')
        plt.savefig('benefit_history_%r.png' %(title))
        plt.close()

        # The following commented plot is not relevant
        '''
        ### Price elasticity graphs
        # Get the demand history for each operators
        demand_1 = [demand[1] for demand in self.demand if demand[1] != -1]
        demand_2 = [demand[2] for demand in self.demand if demand[2] != -1]
        #market_sum = [m1 + m2 for m1, m2 in zip(market_history_1, market_history_2)]
        # Plot them
        plt.plot(p_history_2[1::2], demand_2[1::2], 's' ,label='Alt.2 demand', color='blue')
        plt.plot(p_history_1[::2], demand_1[::2], 's' ,label='Alt.1 demand', color='red')
        #plt.plot(market_sum, '--' ,label='Market total', color='green')
        # Plot vertical line to indicate the beginning of the cycle
        plt.ylabel('Demand curve')
        plt.xlabel('Price')
        plt.title('Demand of the alternative as a function of their price.\n')
        plt.legend(loc='upper left')
        plt.savefig('demand_%r.png' %(title))
        plt.close()
        '''

if __name__ == '__main__':

    ### Linear formulation
    t_0 = time.time()
    stackelberg_dict = data_file.getData()
    data_file.preprocess(stackelberg_dict)

    t_1 = time.time()

    sequential_dict = {'K': 2,
                    'operator': [0, 1, 2],
                    'max_iter': 50,
                    'optimizer': 1.0,
                    'p_fixed': [0.0, -1.0, 0.5],
                    'y_fixed': [1.0, 1.0, 1.0]}
    '''
    sequential_dict = {'K': 2,
                    'operator': [0, 0, 0, 0, 1, 1, 2, 2],
                    'max_iter': 50,
                    'optimizer': 1.0,
                    'p_fixed': [stackelberg_dict['PRICE_CAR'], stackelberg_dict['PRICE_PLANE'], stackelberg_dict['PRICE_IC_1'], stackelberg_dict['PRICE_IC_2'], -1.0, -1.0, 70.0, 70.0],
                    'y_fixed': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
    '''
    sequential_game = Sequential(**sequential_dict)
    # Update the dict with the attributes of the Stackelberg game
    sequential_dict.update(stackelberg_dict)

    t_2 = time.time()
    sequential_game.run(sequential_dict, linearized=True)

    t_3 = time.time()
    print('\n -- TIMING -- ')
    print('Get data + Preprocess: %r sec' %(t_1 - t_0))
    print('Update dictionary, initiate sequential game: %r sec' %(t_2 - t_1))
    print('Run the game: %r sec' %(t_3 - t_2))
    nb_iter = len([price[0] for price in sequential_game.p_history if price[0] != -1])
    print('Total number of iterations: %r' %nb_iter)

    print('n: %r and r: %r' %(sequential_dict['N'], sequential_dict['R']))
    #sequential_game.plotGraphs('new_fixed_cost')
    '''
    ### Non linear formulation
    t_0 = time.time()
    stackelberg_dict = data_file_2.getData()
    data_file_2.preprocess(stackelberg_dict)

    t_1 = time.time()
    sequential_dict = {'K': 2,
                    'operator': [0, 1, 2],
                    'max_iter': 50,
                    'optimizer': 1,
                    'p_fixed': [0.0, -1.0, 0.5],
                    'y_fixed': [1.0, 1.0, 1.0]}
    sequential_game = Sequential(**sequential_dict)
    # Update the dict with the attributes for the Stackelberg game
    sequential_dict.update(stackelberg_dict)

    t_2 = time.time()
    sequential_game.run(sequential_dict, linearized=False)

    t_3 = time.time()
    print('\n -- TIMING -- ')
    print('Get data + Preprocess: %r sec' %(t_1 - t_0))
    print('Update dictionary, initiate sequential game: %r sec' %(t_2 - t_1))
    print('Run the game: %r sec' %(t_3 - t_2))
    nb_iter = len([price[0] for price in sequential_game.p_history if price[0] != -1])
    print('Total number of iterations: %r' %nb_iter)
    sequential_game.plotGraphs('test')
    '''
