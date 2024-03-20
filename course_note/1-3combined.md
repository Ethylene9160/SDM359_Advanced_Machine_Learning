

Perceptron: single layer network.
$$
v = \pmb {W}^T\pmb{X}
$$
with $w_0x_0=b$

Apply activate function:
$$
y=g(v)
$$
Linearly separable vs. linerly unseparable.



Example: 

class 0: > 0, (0,1), (1,0)

class 1: <0, (-1,0), (0,-1)

i.e.
$$
W=[0,0,0]^T
$$
Step 1:
$$
(0,1)\rightarrow v=0\\
(1,0)\rightarrow v=0\\
\pmb W = \pmb W+\Delta \pmb W
$$
with the 4 points provided, this means:
$$
\pmb W^* = \pmb W+[0,1,0]^T = [0,1,0]^T
$$
class 0 and v > 0, but. $\rightarrow~ wp+b<0,~w\leftarrow w+p$

class 1 and v < 0, but. $\rightarrow~ wp+b>0,~w\leftarrow w-p$