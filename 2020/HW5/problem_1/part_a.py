from model import dynamics, cost
import numpy as np
from numpy.linalg import inv, norm

# Riccati recursion
def Riccati(A,B,Q,R):

    # implement infinite horizon riccati recursion
    n = A.shape[0]
    P = np.zeros((n,n))
    BT = np.transpose(B)
    AT = np.transpose(A)

    print(A.shape)
    print(B.shape)
    print(Q.shape)
    print(R.shape)


    l0_1 = inv(R + np.matmul(np.matmul(BT, P), B)) 
    l0_2 = np.matmul(np.matmul(BT, P), A)
    L = np.matmul(l0_1, l0_2) 

    L_old = np.ones((2,4))

    while norm(L - L_old,2) > 1e-4:
        p1 = np.matmul(np.matmul(AT, P), A)
        p2 = np.matmul(np.matmul(AT, P), B)
        p3 = inv(R + np.matmul(np.matmul(BT, P) , B))
        p4 = np.matmul(np.matmul(BT, P), A)
        P = p1 + np.matmul(np.matmul(p2, p3), p4) + Q
        L_old = L
        l1 = inv(R + np.matmul(np.matmul(BT, P), B))
        l2 = np.matmul(np.matmul(BT, P), A)
        L = np.matmul(l1, l2)
    
    return L,P

def simulate():
    # dynfun = dynamics(stochastic=False)
    dynfun = dynamics(stochastic=True) # uncomment for stochastic dynamics

    costfun = cost()

    T = 100 # episode length
    N = 100 # number of episodes
    gamma = 0.95 # discount factor
    A = dynfun.A
    B = dynfun.B
    Q = costfun.Q
    R = costfun.R

    L,P = Riccati(A,B,Q,R)

    total_costs = []

    for n in range(N):
        costs = []
        
        x = dynfun.reset()
        for t in range(T):
            
            # policy 
            u = (-L @ x)
            
            # get reward
            c = costfun.evaluate(x,u)
            costs.append((gamma**t)*c)
        
            # dynamics step
            x = dynfun.step(u)
            
        total_costs.append(sum(costs))
        
    return np.mean(total_costs)

cost_list = []
trail_num = 20
for i in range(trail_num):
    c = simulate()
    cost_list.append(c)
    print(c)

print("average cost over {} trails is {}".format(trail_num, np.mean(cost_list)))