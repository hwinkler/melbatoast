melbatoast
==========
All files copyright 2013 Hugh Winkler and licensed to everyone under the
GPL v3. See the file LICENSE.

This is a Bayesian network Gibbs sampler, using CUDA, designed to run on NVIDIA GPUs.

It accepts as input two files: a network description, and an initial
state.

Here is an example network description:

`A 2
0.4 0.6

B|A 2
0.3 0.7 0.8 0.2

C|A 2
0.7 0.3 0.4 0.6

D|B 2
0.5 0.5 0.1 0.9

E|D,C 2
0.9 0.1 0.999 0.001
0.999 0.001 0.999 0.001`

The example above describes five nodes, named A, B, C, D, and E. A is
a root node. B and C are child nodes of A; D is a child node of B, and
E is a child node of D and of C.

Each of the nodes takes on two states, indicated by the integer
following the node's name.

The array of numbers describes the conditional probability table for
the node. You can separate the numbers by spaces or commas, and the
line endings aren't significant. The numbers are the elements of a
multidimensional table having number of dimensions equal to one plus
the number of the node's parents. The first dimension is for the
node's variable itself, and the subsequent dimensions are for each of
its parents read left to right. The most rapidly varying dimension is
the node's own dimension, that is, the first dimension, followed by
the parent dimensions left to right.

Here is a sample state file for the network above:

`0
-1
0
0
-1`

Each state must be on a line by itself, and the order of the lines
matches the order of its node in the network file. The integer takes
on a value from zero to the number of states for that node, minus
one. The minus sign is only a flag to indicate that the node's value
is evidence -- its value remains fixed during the sampling.




