from model import dynamics, cost
import numpy as np
from numpy.linalg import inv, norm


stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor

total_costs = []

A = np.random.rand(4,4)
B = np.random.rand(4,2)
Q = np.eye(4)
R = np.eye(2)

for n in range(N):
    costs = []
    
    x = dynfun.reset()
    for t in range(T):
        # TODO compute policy
        n = A.shape[0]
        P = np.zeros((n,n))
        BT = np.transpose(B)
        AT = np.transpose(A)

        l0_1 = inv(R + np.matmul(np.matmul(BT, P), B)) 
        l0_2 = np.matmul(np.matmul(BT, P), A)
        L = np.matmul(l0_1, l0_2) 

        p1 = np.matmul(np.matmul(AT, P), A)
        p2 = np.matmul(np.matmul(AT, P), B)
        p3 = inv(R + np.matmul(np.matmul(BT, P) , B))
        p4 = np.matmul(np.matmul(BT, P), A)
        P = p1 + np.matmul(np.matmul(p2, p3), p4) + Q
        
        # TODO compute action
        u = (-L @ x)

        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)
        
        # TODO implement recursive least squares update


        x = xp.copy()
        
    total_costs.append(sum(costs))

print(total_costs)
