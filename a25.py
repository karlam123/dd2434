import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
PATH = "DataICA.txt"
def readx(path):
    with open(path) as f:
        line = f.read().splitlines()
    l1 = line[0].split()
    N = int(l1[2])
    x = np.zeros((len(line)-2, 2))
    i = 0
    for l in line[2:]:
        (x1str, x2str) = l.split()
        (x1, x2) = (float(x1str), float(x2str))
        x[i, :] = np.array([x1, x2])
        i = i+1
    return (N, x)

(N, x) = readx(PATH)
#plt.plot(x[:, 0], x[:, 1], '*')
#plt.show()

def centering(x):
    print(np.mean(x[0, :]))
    x[:, 0] = x[:, 0] - np.mean(x[:, 0])
    x[:, 1] = x[:, 1] - np.mean(x[:, 1])
    return x

def whitening(X):
    X = centering(X)
    n = len(X[:, 0])
    cov = np.cov(X.T)
        ##print(x.shape)
        #print(x.dot(x.T))
        #print(x)
    (E, D, v) = np.linalg.svd(cov, full_matrices=True)
    D2 = D**(-0.5)
    Dsqrt = np.diag(D2)
    X = E.dot(Dsqrt.dot(E.T.dot(X.T)))
    return (X, D2, E)

def plot12(x):
    (X, D, E) = whitening(x)
    print(x.shape)
    plt.plot(x[:, 0].T, x[:, 1].T, '*')
    plt.plot(X[0, :], X[1, :], '*')
    plt.plot([0, D[1]*E[0,1]], [0, D[1]*E[1, 1]])
    plt.plot([0, D[0]*E[0,0]], [0, D[0]*E[1, 0]])
    plt.show()

def fastICA(X, tol):
    # X = 2 x T
    T = len(X)
    def g(w):
        wX = w.T.dot(X)  # 1 x T
        print(wX[0, 1])
        for i in range(T):
            wX[0,i] = np.tanh(wX[0, i])
        tmp1 = X.dot(wX.T) # 2 x T
        ret = np.mean(tmp1, axis=1) # 2 x 1
        return ret
    def gp(w):
        wX2 = w.T.dot(X)
        for i in range(T):
            wX2[0, i] = 1-np.tanh(wX2[0, i])
        return np.mean(wX2)*w
    w1 = np.matrix([[0.5], [0.5]])
    w2 = np.matrix([[0.5], [0.5]])
    diff1 = tol + 1
    diff2 = diff1
    while (diff1 > tol):
        wold1 = w1
        w1 = g(w1)-gp(w1)
        w1 = w1/np.linalg.norm(w1)
        diff1 = np.linalg.norm(wold1-w1)

    while (diff2 > tol):
        wold2 = w2
        w2 = g(w2)-gp(w2)
        w2 = w2/np.linalg.norm(w2)
        w2 = w2 - (w2.T.dot(w1))[0,0]*w1
        w2 = w2/np.linalg.norm(w2)
        diff2 = np.linalg.norm(wold2-w2)
        #print(wold2.shape)
    W = np.zeros((2, 2))
    print(w1.shape)
    print(np.asarray(w1).shape)

    W[0, :] = np.asarray(w1).ravel()
    W[1, :] = np.asarray(w2).ravel()
    return W
(X, D2, E) = whitening(x)
W=fastICA(X, 10**(-6))
S = W.dot(X)
plt.plot(S.T)
plt.show()
