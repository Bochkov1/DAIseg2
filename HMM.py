import numpy as np
import pandas as pd

import math


#number of states
N=3
cover_cut=0.8


#Ti: Introgression of Nd
def initA(t1, t2, p1, p2, L=1000, r=10 ** -8) -> np.array:
    A = np.zeros((3,3))
    
    A[0][1]=np.exp(-L*r*t2)*(1 - np.exp(-L*r*(t1-t2)))*p1 + (1 - np.exp(-L*r*t2))*p1*(1-p2)
    A[0][2]=(1 - np.exp(-L*r*t2))*p2
    A[0][0] = 1 - A[0][1] - A[0][2]
 
    A[1][0]=np.exp(-L*r*t2)*(1 - np.exp(-L*r*(t1-t2)))*(1-p1)+(1 - np.exp(-L*r*t2))*(1-p1)*(1-p2)
    A[1][2]=(1 - np.exp(-L*r*t2))*p2
    A[1][1]=1-A[1][0] - A[1][2]
    
    A[2][0]=np.exp(-L*r*t2)*(1 - np.exp(-L*r*(t1-t2)))*(1-p2)+(1 - np.exp(-L*r*t2))*(1-p1)*(1-p2)
    A[2][1]=(1 - np.exp(-L*r*t2))*p1
    A[2][2]=1-A[2][0] - A[2][1]
    
    return A

#Ti: Introgression of Nd
#Taf: Time out of Africa
#Tn: Time of Split between Nd and Sapiens

def initB(mu, L, t_OOA, t_ANC, t1, t2, n_st) -> np.array:
    
    B = np.zeros((3, n_st, n_st))
    # Calculate lambda values for each distribution
    lambda_OOA = mu * L * t_OOA
    lambda_ANC = mu * L * t_ANC
    lambda_t1 = mu * L * t1
    lambda_t2 = mu * L * t2

    # Precompute Poisson distributions
    P_OOA = np.zeros(n_st)
    P_ANC = np.zeros(n_st)
    P_t1 = np.zeros(n_st)
    P_t2 = np.zeros(n_st)

    # Base probabilities
    P_OOA[0] = np.exp(-lambda_OOA)
    P_ANC[0] = np.exp(-lambda_ANC)
    P_t1[0] = np.exp(-lambda_t1)
    P_t2[0] = np.exp(-lambda_t2)

    # Iteratively compute Poisson probabilities
    for i in range(1, n_st):
        P_OOA[i] = P_OOA[i - 1] * lambda_OOA / i
        P_ANC[i] = P_ANC[i - 1] * lambda_ANC / i
        P_t1[i] = P_t1[i - 1] * lambda_t1 / i
        P_t2[i] = P_t2[i - 1] * lambda_t2 / i

    # Compute emission probabilities
    for i in range(n_st):
        for j in range(n_st):
            # EU: P_OOA(i) * P_ANC(j)
            B[0][i][j] = P_OOA[i] * P_ANC[j]
            
            # ND1: P_ANC(i) * P_t1(j)
            B[1][i][j] = P_ANC[i] * P_t1[j]
            
            # ND2: P_ANC(i) * P_t2(j)
            B[2][i][j] = P_ANC[i] * P_t2[j]

    return B

    

        
    
    



    
    
def viterbi_modified(V, initial_distribution, a,  b):
    
    T = len(V)
    M = a.shape[0]
 
    omega = np.zeros((T, M))

    omega[0, :] = np.log(initial_distribution * b[:, V[0][0],V[0][1]])

 
    prev = np.zeros((T - 1, M))
 
    for t in range(1, T):
        for j in range(M):

            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t][0], V[t][1]])

                
               
 
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)
 
            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)
 
    # Path Array
    S = np.zeros(T)
 
    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])
 
    S[0] = last_state
 
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
 
    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)
 
    # Convert numeric values to actual hidden states
 
    result = []
    for s in S:
        if s == 0:
            result.append(0)
        elif s == 1:
            result.append(1)
        elif s == 2:
            result.append(2)

    return result
    




def get_HMM_tracts(seq):
    
    migrating_tracts = []
    for i in range(N):
        migrating_tracts.append([])
    start=0
    for i in range(1,len(seq)):
        if seq[i]!=seq[i-1]:
            migrating_tracts[seq[i-1]].append([start,i-1])
            start=i
    migrating_tracts[seq[len(seq)-1]].append([start,len(seq)-1])
    return migrating_tracts
















