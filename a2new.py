import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

A    = 0.25*np.array([[1.0, 3.0], [3.0, 1.0]])
cat1 = 1.0/6.0*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
cat2 = np.array([0.5, 0.25, 0.12, 0.06, 0.03, 0.04])
cat3 = np.array([0.25, 0.5, 0.12, 0.06, 0.03, 0.04])

test = np.random.multinomial(1, cat1)

# sample sum of two dices
def S(cat1, cat2, cat3, l):
    # cat1 = distribution of dice on T_i
    # cat2 = distribution of dice on T'_i
    # cat3 = distribution of player dice
    # l = 0 or 1 describing if we are on T_i or T'_i
    if l==0:
        return (np.random.choice(6, 1, p = cat1)+np.random.choice(6, 1, p=cat3)+2)[0]
    return (np.random.choice(6, 1, p = cat2)+np.random.choice(6, 1, p=cat3)+2)[0]

# go to next table, Z_t -> Z_(t+1)
# returns the new state l
def next_table(l):
    if l==0:
        return bernoulli.rvs(0.75)
    return bernoulli.rvs(0.25)

# sample from K tables
def sample_hmm(cat1, cat2, cat3, K, l):
    sums = np.zeros(K)
    for k in range(K):
        sums[k] = S(cat1, cat2, cat3, l)
        l = next_table(l)
    return sums

# sample from N players
def players(cat1, cat2, cat3, K, l, N=1):
    sums = np.zeros((N, K))
    for n in range(N):
        sums[n, :] = sample_hmm(cat1, cat2, cat3, K, l)
    return sums


# probability distribution of sum of two distributions
def sumprob(p1, p2):
    A = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            A[i][j] = p1[i]*p2[j]
    p = np.zeros(11)
    for n in range(-5, 6):
        p[n+5] = np.sum(np.flipud(A).diagonal(n))
    return p

def f(S, kstart, cat1, cat2, cat3, A):
    T = len(S)
    K = 2  # number of states
    fs = np.zeros((T, K))
    fs[0, kstart] = 1
    p13 = sumprob(cat1, cat3)
    p23 = sumprob(cat2, cat3)
    print(p13)
    for t in range(1, T):
        for k in range(K):
            fs[t, k] = fs[t-1, 0]*p13[S[t-1]-2]*A[0, k]+fs[t-1, 1]*p23[S[t-1]-2]*A[1, k]
    return fs

def sample_z(prob):
    return np.random.choice(2, 1, p=prob)[0]
def sample_table(fs, S,cat1, cat2, cat3, A):
    T = len(fs[:, 0])
    K = len(fs[0, :])
    Z = np.zeros(T)
    p13 = sumprob(cat1, cat3)
    p23 = sumprob(cat2, cat3)
    pT = np.zeros(2)
    for l in range(K):
        pT[0] += A[0, l]*p13[S[T-1]]*fs[T-1, 0]   # CHECK AGAIN
        pT[1] += A[1, l]*p23[S[T-1]]*fs[T-1, 1]   # CHECK AGAIN
    pT = pT/sum(pT)
    Z[T-1] = sample_z(pT)
    for k in list(reversed(range(T-1))):
        print(k)
        pk = np.zeros(2)
        p = [p13, p23]
        for l in range(K):
            pk[l] += A[l, Z[k+1]]*p[l][S[k]-2]*fs[k, l] # CHECK AGAIN
        pk = pk/fs[k+1, Z[k+1]]
        print(sum(pk))
        Z[k] = sample_z(pk)
    return Z
kstart = 0
K = 200
S = sample_hmm(cat1, cat1, cat1, K, kstart)
fs = f(S, kstart, cat1, cat1, cat1, A)
print(f(S, kstart, cat1, cat1, cat1, A))
print(sample_table(fs, S, cat1, cat1, cat1, A))
Z = sample_table(fs, S, cat1, cat1, cat1, A)
plt.plot(Z)
plt.plot(S)
plt.show()
