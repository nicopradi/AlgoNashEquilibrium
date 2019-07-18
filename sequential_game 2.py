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
import Data.Italian.Stackelberg.MILPNested_n40r50 as data_file
import Data.Italian.Non_linear_Stackelberg.ProbLogit_n10 as data_file_2
import Data.Italian.Non_linear_Stackelberg.Nested_Logit.nested_n01 as df

# Stackelberg
import stackelberg_game
import non_linear_stackelberg
import non_linear_nested_stackelberg

class Sequential:

    def __init__(self, **kwargs):
        ''' Construct a sequential game.
            KeywordArgs:
                K          Number of operators
                operator   Mapping between alternatives and operators
                max_iter    Maximum number of iterations
                optimizer  Current operator index
                p_fixed    Current prices of the other operator's alternatives
                y_fixed    Current availabilty of the other operator's alternatives
        '''
        self.K = kwargs.get('K', 1)
        self.operator = kwargs.get('operator', np.full((1, self.K + 1), 1))
        self.max_iter = kwargs.get('max_iter', 10)
        self.optimizer = kwargs.get('optimizer', 1)
        self.p_fixed = kwargs.get('p_fixed', np.full((1, len(self.operator)), 1.0))
        self.y_fixed = kwargs.get('y_fixed', np.full((1, len(self.operator)), 1.0))
        self.p_history = np.full((self.max_iter, len(self.operator)), -1.0)
        self.revenue = np.full((self.max_iter, self.K + 1), -1.0)
        self.market_share = np.full((self.max_iter, len(self.operator)), -1.0)
        self.total_share = np.full((self.max_iter, len(self.operator)), -1.0)

        # Make a copy of the initial prices
        self.initial_price = copy.deepcopy(self.p_fixed)

    # Run the sequential game
    def run(self, data, linearized=True, nested=False):
        ''' Run the sequential game with the Stackelberg game
            Args:
                data:          dictionary containing data to instanciate the Stackelberg game
        '''
        iter = 0
        cycle = False
        cycle_iter = 0

        if( linearized is True ):
            cust_choice = np.empty((self.max_iter, data['I'] + data['I_opt_out'], data['N'], data['R']))
            cust_U = np.empty((self.max_iter, data['I'] + data['I_opt_out'], data['N'], data['R']))

        while (iter < self.max_iter) and cycle is False:
            print('\n--- ITERATION %r ----' %iter)

            a = [i for i in range(len(self.operator)) if self.operator[i] == self.optimizer]
            print('Optimizer %r is fixing prices of alternatives %r' %(self.optimizer, a))

            # for (index, p) in enumerate(self.p_fixed):
            #     if p != -1.0:
            #         print('The operator %r fixed the price of alternative %r to %r'
            #                %(self.operator[index], index, p))

            # Run the game for the current optimizer
            if linearized is True:
                sub_game = stackelberg_game.Stackelberg(**data)
                model = sub_game.getModel()
                model = sub_game.solveModel(model)
                prices = []
                demand = []
                for i in range(len(self.operator)):
                    prices.append(model.solution.get_values('p[' + str(i) + ']'))
                    demand.append(model.solution.get_values('d[' + str(i) + ']'))
                    for n in range(data['N']):
                        for r in range(data['R']):
                            cust_choice[iter, i, n, r] = model.solution.get_values('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                            cust_U[iter, i, n, r] = model.solution.get_values('U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
            else:
                if( nested is False ):
                    # Solves the non-linear logit version
                    prices, choice, x, status, status_msg = non_linear_stackelberg.main(data)

                    # Computes and print objective function of the non-linear stackelberg game of current optimizer
                    non_linear_stackelberg.objective(data, x)
                else:
                    prices, choice, x, status, status_msg = non_linear_nested_stackelberg.main(data)


                demand = []
                for i in range(len(self.operator)):
                    demand.append(np.sum([customer for customer in choice[i*data['N']:(i+1)*data['N']]]))
                    #print(choice[i*data['N']:(i+1)*data['N']])

            # Update the price history
            for i in range(len(self.operator)):
                self.p_history[iter, i] = prices[i]

            # Update the revenue history
            for k in range(self.K + 1):
                self.revenue[iter][k] = 0.0
                alternatives = [i for i in range(len(self.operator)) if self.operator[i] == k]
                for i in alternatives:
                    self.revenue[iter][k] += demand[i]*prices[i]

            # Check that optimizer increased his revenue after optimization
            if( iter > 1 ):
                print('New revenue: %r / Old revenue : %r' %(self.revenue[iter, self.optimizer], self.revenue[iter - 1, self.optimizer]))
                assert(self.revenue[iter, self.optimizer] >= self.revenue[iter - 1, self.optimizer] - 1e-3)

            # Update the market share history
            for i in range(len(self.operator)):
                self.market_share[iter, i] = float(demand[i])/data['N']

            # print some informations about current iteration
            print(' ----------- Problem solved ------------------ ')
            print(' *----------* New prices *------------*')
            for (index, p) in enumerate(prices):
                if( self.operator[index] == self.optimizer ):
                    print('The operator %r fixed the price of alternative %r to %r'
                          %(self.operator[index], index, p))
            print(' *--------* New market shares *----------*')
            for k in range(self.K + 1):
                self.total_share[iter][k] = 0.0
                alternatives = [i for i in range(len(self.operator)) if self.operator[i] == k]
                for i in alternatives:
                    self.total_share[iter][k] += demand[i]
                print('The operator %r has %r of the market' %(k, self.total_share[iter, k]))
            print(' *----------* New revenues *--------------*')
            for k in range(self.K + 1):
                print('The operator %r has a new revenue equal to %r' %(k, self.revenue[iter, k]))
            if( (linearized is True) and (data['R'] == 1) ):
                for i in range(len(self.operator)):
                    for n in range(data['N']):
                        if( iter == 0 and cust_choice[iter, i, n, 0] == 1):
                            print('Customer %r choose alternative %r with utility %r' %(n, i, cust_U[iter, i, n, 0]))
                        elif( iter >= 1 and cust_choice[iter, i, n, 0] - cust_choice[iter - 1, i, n, 0] == 1):
                            print('Customer %r switches to alternative %r with utility %r' %(n, i, cust_U[iter, i, n, 0]))

            # Check for the cycle
            # First check if all operators have played.
            if( iter >= self.K ):
                iteration_to_check = range(iter-self.K, -1, -self.K)
                print('ITERATION TO CHECK: %r' %list(iteration_to_check))
                for j in iteration_to_check:
                    cycle = True
                    for i in range(len(self.operator)):
                        # TODO: Numerical error ? Tolerance 1e-6
                        if abs(self.p_history[j, i] - self.p_history[iter, i]) > 1e-4:
                            cycle = False
                    if cycle is True:
                        cycle_iter = j
                        if iter-cycle_iter == self.K:
                            print('\nNash equilibrium detected')
                        else:
                            print('\nCycle detected')
                        break

            # Update the data for the next iteration
            data['optimizer'] = ((data['optimizer']) % self.K) + 1
            self.optimizer = data['optimizer']
            # Fix the price of the operators exept for the next optimizer
            data['p_fixed'] = copy.deepcopy(self.p_history[iter])
            self.p_fixed = copy.deepcopy(self.p_history[iter])
            for (i, k) in enumerate(self.operator):
                if k == data['optimizer']:
                    data['p_fixed'][i] = -1.0
                    self.p_fixed[i] = -1.0
            # Go to the next iteration
            iter += 1

        print('Price history: %r' %self.p_history)

    def plotGraphs(self, title):
        ''' Plot the value of the fixed prices for each optimizer as a function
            of the iterations number.
        '''

        ### Price graph
        # Get the price history for each operators
        p_history_1 = [prices[1] for prices in self.p_history if prices[1] != -1]
        p_history_2 = [prices[2] for prices in self.p_history if prices[2] != -1]
        # Plot them
        plt.plot(p_history_2, label='Operator 2 price', color='blue')
        plt.plot(p_history_1, label='Operator 1 price', color='red')
        plt.ylabel('Price')
        plt.title("Operator's prices as a function of the iteration number. \
        \n The initial prices are: Operator 1: %r and Operator 2: %r" %(self.initial_price[1], self.initial_price[2]))
        plt.legend()
        plt.savefig('price_history_%r.png' %(title))
        plt.close()

        ### Revenue graph
        # Get the revenue history for each operators
        revenue_history_1 = [revenue[1] for revenue in self.revenue if revenue[1] != -1]
        revenue_history_2 = [revenue[2] for revenue in self.revenue if revenue[2] != -1]
        # Plot them
        plt.plot(revenue_history_2, label='Operator 2 revenue', color='blue')
        plt.plot(revenue_history_1, label='Operator 1 revenue', color='red')
        plt.ylabel('Revenue')
        plt.title("Operator's revenue as a function of the iteration number. \
        \n The initial prices are: Operator 1: %r and Operator 2: %r" %(self.initial_price[1], self.initial_price[2]))
        plt.legend()
        plt.savefig('revenue_history_%r.png' %(title))
        plt.close()

        ### Market share graph
        # Get the demand history for each operators
        market_history_1 = [market[1] for market in self.market_share if market[1] != -1]
        market_history_2 = [market[2] for market in self.market_share if market[2] != -1]
        # Plot them
        plt.plot(market_history_2, label='Operator 2 market share', color='blue')
        plt.plot(market_history_1, label='Operator 1 market share', color='red')
        plt.ylabel('Market share')
        plt.title("Operator's market share as a function of the iteration number. \
        \n The initial prices are: Operator 1: %r and Operator 2: %r" %(self.initial_price[1], self.initial_price[2]))
        plt.legend()
        plt.savefig('market_history_%r.png' %(title))
        plt.close()

# Main function -- read data file and solve the sequential game (linearized or non-linearized)
if __name__ == '__main__':

    # LINEAR
    t_0 = time.time()
    stackelberg_dict = data_file.getData()
    data_file.preprocess(stackelberg_dict)
    t_1 = time.time()
    sequential_dict = {'K': 2,
                    'operator': [0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 1, 1, 2, 2],
                    'max_iter': 50,
                    'optimizer': 1,
                    'p_fixed': [-1.0, -1.0, -1.0, -1.0,
                                 125,
                                  80,
                                 105,
                                  60,
                                -1.0,-1.0,  105,  60],
                    'y_fixed': [1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0]}
    sequential_game = Sequential(**sequential_dict)
    # Update the dict with the attributes of the Stackelberg game
    sequential_dict.update(stackelberg_dict)
    t_2 = time.time()
    sequential_game.run(sequential_dict, linearized=True, nested=False)
    t_3 = time.time()
    print('\n -- TIMING -- ')
    print('Get data + Preprocess: %r sec' %(t_1 - t_0))
    print('Update dictionary, initiate sequential game: %r sec' %(t_2 - t_1))
    print('Run the game: %r sec' %(t_3 - t_2))
    nb_iter = len([price[0] for price in sequential_game.p_history if price[0] != -1])
    print('Total number of iterations: %r' %nb_iter)

    print('n: %r and r: %r' %(sequential_dict['N'], sequential_dict['R']))
    sequential_game.plotGraphs('test')
    sequential_game.plotGraphs(0.2)

    '''
    # NON LINEAR
    t_0 = time.time()
    stackelberg_dict = data_file_2.getData()
    data_file_2.preprocess(stackelberg_dict)
    t_1 = time.time()
    sequential_dict = {'K': 2,
                    'operator': [0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 1, 1, 2, 2],
                    'max_iter': 50,
                    'optimizer': 1,
                    'p_fixed': [-1.0, -1.0, -1.0, -1.0,
                                 125,
                                  80,
                                 105,
                                  60,
                                -1.0,-1.0,  105,  60],
                    'y_fixed': [1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0]}
    sequential_game = Sequential(**sequential_dict)
    # Update the dict with the attributes for the Stackelberg game
    sequential_dict.update(stackelberg_dict)
    #sequential_dict['x0'] = non_linear_stackelberg.getInitialPoint(sequential_dict)
    #print('x0:', sequential_dict['x0'])
    t_2 = time.time()
    #stackelberg_dict['x0'] = non_linear_stackelberg.getInitialPoint(stackelberg_dict)
    sequential_game.run(sequential_dict, linearized=False, nested=False)
    t_3 = time.time()
    print('\n -- TIMING -- ')
    print('Get data + Preprocess: %r sec' %(t_1 - t_0))
    print('Update dictionary, initiate sequential game: %r sec' %(t_2 - t_1))
    print('Run the game: %r sec' %(t_3 - t_2))
    nb_iter = len([price[0] for price in sequential_game.p_history if price[0] != -1])
    print('Total number of iterations: %r' %nb_iter)
    sequential_game.plotGraphs('italy')
    '''

    '''
    # NON LINEAR NESTED
    t_0 = time.time()
    stackelberg_dict = df.getData()
    df.preprocess(stackelberg_dict)
    t_1 = time.time()
    sequential_dict = {'K': 2,
                    'operator': [0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 1, 1, 2, 2],
                    'max_iter': 50,
                    'optimizer': 1,
                    'p_fixed': [-1.0, -1.0, -1.0, -1.0,
                                 125,
                                  80,
                                 105,
                                  60,
                                -1.0,-1.0,  105,  60],
                    'y_fixed': [1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0]}
    sequential_game = Sequential(**sequential_dict)
    # Update the dict with the attributes for the Stackelberg game
    sequential_dict.update(stackelberg_dict)
    #sequential_dict['x0'] = non_linear_stackelberg.getInitialPoint(sequential_dict)
    #print('x0:', sequential_dict['x0'])
    t_2 = time.time()
    #stackelberg_dict['x0'] = non_linear_stackelberg.getInitialPoint(stackelberg_dict)
    sequential_game.run(sequential_dict, linearized=False, nested=True)
    t_3 = time.time()
    print('\n -- TIMING -- ')
    print('Get data + Preprocess: %r sec' %(t_1 - t_0))
    print('Update dictionary, initiate sequential game: %r sec' %(t_2 - t_1))
    print('Run the game: %r sec' %(t_3 - t_2))
    nb_iter = len([price[0] for price in sequential_game.p_history if price[0] != -1])
    print('Total number of iterations: %r' %nb_iter)
    sequential_game.plotGraphs('italy')
    '''