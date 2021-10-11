# Markov Decision Process Simulator

A Python library that solves and simulates Markov decision process.

## Requirements

+ python 3.8 or newer
+ numpy

## Running

The library comes with a ready-to-run file ```gridworld.py```, which can be executed by simply typing:

```
python gridworld.py
```

Alternatively, you can write your own Python script and include the library with:

```
import simulator
```

# User-defined Problems

As a generic MDP solver, it is able to solve any problems that are defined by anyone, as long as they abide to several rules.

## Modeling and Implementing the Process

A MDP, involving rewards and discounting factors, can be described by a 6-tuple: (S, A, P, R, H, γ) where

+ S is the state space, the set of all possible states of the environment.
+ A is the action space, the set of all possible actions of the agent.
+ P is the transition probability, the probability of the next state given current state and action.
+ R is the reward function, the reward given to the agent given current state and action.
+ H is the horizon, the number of steps the agent can take.
+ γ is the discounting factor that discounts rewards further in the future.

There are some mathematical concepts that can be reused in Python, and we implement these functions while abiding by the following constraints:

+ ```P``` must be callable, receives two arguments ```s``` and ```a```, and returns a probabilistic distribution of type ```list``` and length equal to ```len(S)```. The library provides a class ```Transition``` for inheritence.
+ ```S``` and ```A``` must be of type ```list``` but their content is not restricted as long as ```P``` handles their elements well.
+ ```R``` must be a callable object that takes three arguments ```s'```, ```s```, and ```a```, and returns a number.
+ ```H``` and ```gamma``` are constant numbers. ```gamma``` should be in the range [0, 1).

## Interface

The library provides the interface ```Environment``` that users can interact with. The methods available are:

```
Environment(S, A, P, O, R)
```

The object instantiation method that creates an ```Environment``` object with necessary information. Note that there is a slot for observation ```O``` but it will be of no use at current time.

```
Environment.step(a) -> type(R)
```

A method that changes the current state of the environment according to the given action ```a```. The next state is computed by calling ```P```, and the method returns the corresponding reward. The state of the environment can be accessed with ```Environment.state```.

```
Environment.initialize(s)
```

A method that sets the initial state of the environment to the given argument ```s```.

```
Environment.task(H, gamma)
```

A method that provides the environment with information necessary for policy iteration and value iteration.

```
Environment.policyIteration()
```

A method that computes value and policy based on policy iteration, both of which can be accessed with ```Environment.V``` and ```Environment.pi```.

```
Environment.valueIteration()
```

A method that computes value and policy based on value iteration, both of which can be accessed with ```Environment.V``` and ```Environment.pi```.

# Known Issues

+ The order of actions in the list ```A``` affects the output of either policy iteration or value iteration. This has to do with how a maximum is picked if there are multiple maxima in a list.
