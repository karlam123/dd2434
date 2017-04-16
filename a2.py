import numpy as np
from scipy.stats import bernoulli

A    = 0.25*np.array([[1.0, 3.0], [3.0, 1.0]])
cat1 = 1.0/6.0*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
cat2 = np.array([0.5, 0.25, 0.12, 0.06, 0.03, 0.04])

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
    print(A)
    p = np.zeros(11)
    for n in range(-5, 6):
        print(np.flipud(A).diagonal(n))
        p[n+5] = np.sum(np.flipud(A).diagonal(n))
    return p
print(sumprob(cat1, cat1))