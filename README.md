# Project Title

Heuristic algorithms to find market equilibria in oligopolies.
Project developed within the scope of a Master's project at EPFL.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to run the algorithmic framework.
Follow the instructions in the links below to install them.

* [IPOPT](https://github.com/matthias-k/cyipopt) - The open source NLP solver used
* [CPLEX](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.6.1/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html) - The MILP solver used


### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Contents

Algorithmic framework developed within the scope of a Master's project.
The main folder contains the different blocks of the algorithmic framework

* Stackelberg games
  * Non-linear formulation: *non_linear_stackelberg.py* (https://github.com/nicopradi/Master_Project/blob/master/non_linear_stackelberg.py)
  * Linear formulation: *stackelberg_game.py* (https://github.com/nicopradi/Master_Project/blob/master/stackelberg_game.py)
* Nash Heuristic Algorithmic: *nash_heuristic.py* (https://github.com/nicopradi/Master_Project/blob/master/nash_heuristic.py)
* Sequential game: *sequential_game.py* (https://github.com/nicopradi/Master_Project/blob/master/sequential_game.py)
* Fixed-point MILP model
  * Capacitated formulation: *fixed_point.py* (https://github.com/nicopradi/Master_Project/blob/master/fixed_point.py)
  * Uncapacitated formulation: *fixed_point_no_capacity.py* (https://github.com/nicopradi/Master_Project/blob/master/fixed_point_no_capacity.py)
  * Discretized previous prices: *fixed_point_discretized_strat.py* (https://github.com/nicopradi/Master_Project/blob/master/fixed_point_discretized_strat.py)
* Methods combining different blocks: *main.py* (https://github.com/nicopradi/Master_Project/blob/master/main.py)

The data folder contains the instances used by the different blocks.
It contains two datasets: Parking choice instances and Italian railway instances.

### How to run the algorithmic framework

Import the instance in the algorithmic blocks you want to use:

```
# data
import path_to_instance as data_file
```

Run/Adapt the code in the following block of the file you want to run:

```
if __name__ == '__main__':
```

## Contributing

Please contact the author or the contributors for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Nicolas Pradignac** (https://www.linkedin.com/in/nicolas-pradignac-b13298159/)
  * *Email:* nicolas.pradignac.epfl@gmail.com

### Contributors

* **Stefano Bortolomiol** (https://www.epfl.ch/labs/transp-or/people/)
* **Nikola Obrenovic** (https://www.epfl.ch/labs/transp-or/people/)
* **Theophile Thiery** (https://www.epfl.ch/labs/transp-or/people/)
