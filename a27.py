from a26 import *

def fileP(file, K):
        alpha = 0.3
        sigma = 0.3
        return file+":K="+str(K)+",alpha="+str(alpha)+",sigma="+str(sigma)+",iterations=500.p"
K = 15
PATH = "R3-tst-all_run.txt"
file1 = fileP("dics", K)
file2 = fileP("nvk", K)
file3 = fileP("nmk", K)
file4 = fileP("nk", K)
file5 = fileP("nm", K)
(dicstrn, nvk, nmk, nk, nm) = load_files(file1, file2, file3, file4, file5)
# infer the topic distribution
def infer(path, K, dicstrn, nvk, nmk, nk, nm):
    dics = readDict(path, 0.3, 3)  # 0.3 and 3 not used in this case
    M = len(dics)
    #print(M)
    T = np.zeros((M, K))
    B = betakv(dicstrn, nvk, nk, 0.3, K)
    for m, doc in dics.items():
        for wordID, count in doc.items():  # count vector, only need first element
            if wordID < B.shape[1]:
                #print(sum(B[:, wordID]))
                T[m-1, :] += np.random.multinomial(count[0]*100000, B[:, wordID]/sum(B[:, wordID]))
            else:
                T[m-1, :] += np.random.multinomial(count[0]*100000, 1/K*np.ones(K))
    return T/T.sum(axis=1, keepdims=True)
#print(PATH1)
#print(infer(PATH1, K, dicstrn, nvk, nmk, nk, nm)[159])
#print(thetamk(dicstrn, nmk, nm, 0.3, K)[159])


def dist(t1, t2, K):
    for k in range(K):
        d += (t1[k]**0.5-t2[k]**0.5)**2
    return d

LABELPATH = "R3-Label.txt"
def readLabels(path):
    with open(path) as f:
        label = f.read().splitlines()
        return list(map(int, label))
def k_NN(T1, T2, k, labels):
    # T1 = inferred theta using K classes, T = M x K
    # T2 = trained theta using K classes
    T1sqrt = T1**0.5
    T2sqrt = T2**0.5
    C = {} # the class of every training document
    Mtrn = len(labels)  # number of train documents
    Mtst = len(T1[:, 0])
    for m in range(Mtrn):
        C[m] = labels[m]

    # k-nearest points for test document theta
    def tpnts(mtest):

        Tdiff = (T1sqrt[mtest, :]-T2sqrt)**2   # M x K
        Tsum = np.sum(Tdiff, axis=1)           # M x 1
        indices = np.argsort(Tsum)[0:k]        # indices of smallest elements in Tsum, each index corresponds to row in T2
        return indices

    def classify(indices):
        count = np.zeros(3)
        for i in indices:
            count[C[i]] += 1
        return np.argsort(count)[2]
    classes = np.zeros(Mtst)
    for mtst in range(Mtst):
        indices = tpnts(mtst)
        classes[mtst] = classify(indices)
    return classes

labels = readLabels(LABELPATH)
K = 15
load_files(file1, file2, file3, file4, file5)
T1 = infer(PATH, K, dicstrn, nvk, nmk, nk, nm)
T2 = thetamk(dics, nmk, nm, alpha, K)
for k in [11, 30, 50, 150]:
    print("k="+str(k))
    print(((np.nonzero(k_NN(T1, T2, k, labels)-np.array(readLabels(TRUTH))))[0]))
    print(k_NN(T1, T2, k, labels))
TRUTH = "R3-GT.txt"
