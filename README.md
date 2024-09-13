Code for assessing the statistics of first passage times through kinetic networks.

Matthew Gerry

_utils files and rate_matrics.py include useful functions.

The remaining scripts examine the first passage time distribution for a Markov process in continuous time through a network of states, with structure determined by the choice of rate matrix. Types of networks studied include a uniform one-dimensional chain, a chain with a modular structure (spatial periodicity in the transition rates), an impurity (one transition with rates that differ from the rest of the chain), and side-chains of variable lengths. The first passage time distribution is calculated, in some cases, fitted to a multiexponential function, and derived quantities such as the moments of the distribution and randomness parameter.