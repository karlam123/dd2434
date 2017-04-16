import numpy as np
# import scipy
# import scipy.stats.dirichlet
# import matplotlib.pyplot as plt
import pickle

PATH1 = "R3-trn-all_run.txt"
def readDict(path, alpha, K):
    dics = {}
    def spt(s):
        s = s.split(":")
        return (int(s[0]), int(s[1]))

    # count = 0
    with open(path) as f:
        for line in f:
    #        if count >= 5:
    #            break
            theta = np.random.dirichlet(K*[alpha])
            theta[-1] = theta[-1] + 1 - sum(theta)  # must sum to 1
            words = line.split()
            dic = {}
            for word in words[1:]:
                (key, value) = spt(word)
                dic[key] = [value, np.random.choice(K, 1, p=theta)[0]]
            dics[int(words[0])] = dic
    #        count+=1
    return dics
dics = readDict(PATH1, 0.3, 3)
def readDic(path):
    with open(path) as f:
        dic = f.read().splitlines()
    return dic
PATH2 = "R3_all_Dictionary.txt"
DIC = readDic(PATH2)
def nstart(dics, K):
    nvk = {}          # number of words v assigned to topic k
    nk = K*[0]        # number of words assigned to topic K = sum of nvk over all v
    nmk = {}          # number of words which are assigned to the topic k from the document m
    nm  = {}          # number of words in document m
    for m, doc in dics.items():
        nmk[m] = np.zeros(K)
        for v, word in doc.items():
            if v not in nvk:
                nvk[v] = K*[0]
            nvk[v][word[1]] += word[0]
            nk[word[1]] += word[0]
            nmk[m][word[1]] += word[0]
        nm[m] = sum(nmk[m])
    return (nvk, nk, nmk, nm)

def pz(dics, K, alpha, sigma):
    (nvk, nk, nmk, nm) = nstart(dics, K)
    N = 500  # number of iterations
    V = max(nvk)
    Vsigma = V*sigma
    Kalpha = K*alpha
    for n in range(N):
        for m, doc in dics.items():
            for i, word in doc.items():
                n_wmi = word[0]
                topic = word[1]
                nvk[i][topic] -= n_wmi
                nmk[m][topic] -= n_wmi
                nk[topic]     -= n_wmi
                prob = np.zeros(K)

                for k in range(K):
                    prob[k] = (nvk[i][k]+sigma)/(nk[k]+Vsigma)*(nmk[m][k]+alpha)/(nm[m]+Kalpha-1)
                topic = np.random.choice(K, p=prob/sum(prob))
                dics[m][i][1] = topic
                nvk[i][topic] += n_wmi
                nmk[m][topic] += n_wmi
                nk[topic]     += n_wmi
        print(n)
    return (dics, nvk, nmk, nk, nm)

def run_and_save(dics, K, alpha, sigma):
    (dics, nvk, nmk, nk, nm) = pz(dics, K, alpha, sigma)
    pickle.dump(dics, open("dics:K="+str(K)+",alpha="+str(alpha)+",sigma="+str(sigma)+",iterations=500.p", "wb"))
    pickle.dump(nvk, open("nvk:K="+str(K)+",alpha="+str(alpha)+",sigma="+str(sigma)+",iterations=500.p", "wb"))
    pickle.dump(nmk, open("nmk:K="+str(K)+",alpha="+str(alpha)+",sigma="+str(sigma)+",iterations=500.p", "wb"))
    pickle.dump(nk, open("nk:K="+str(K)+",alpha="+str(alpha)+",sigma="+str(sigma)+",iterations=500.p", "wb"))
    pickle.dump(nm, open("nm:K="+str(K)+",alpha="+str(alpha)+",sigma="+str(sigma)+",iterations=500.p", "wb"))


def load_files(file1="dics", file2="nvk", file3="nmk", file4="nk", file5="nm"):
    dics = pickle.load(open(file1, "rb"))
    nvk  = pickle.load(open(file2, "rb"))
    nmk  = pickle.load(open(file3, "rb"))
    nk   = pickle.load(open(file4, "rb"))
    nm   = pickle.load(open(file5, "rb"))
    return (dics, nvk, nmk, nk, nm)
file1 = "dics:K=3,alpha=0.3,sigma=0.3,iterations=500.p"
file2 = "nvk:K=3,alpha=0.3,sigma=0.3,iterations=500.p"
file3 = "nmk:K=3,alpha=0.3,sigma=0.3,iterations=500.p"
file4 = "nk:K=3,alpha=0.3,sigma=0.3,iterations=500.p"
file5 = "nm:K=3,alpha=0.3,sigma=0.3,iterations=500.p"

def thetamk(dics, nmk, nm, alpha, K):
    Kalpha = K*alpha
    M = len(dics)
    #print(dics)
    T = np.zeros((M, K))
    for m, doc in dics.items():
        #print(m)
        for k in range(K):
            T[m-1][k] = (nmk[m][k]+alpha)/(nm[m]+Kalpha)
    return T


def betakv(dics, nvk, nk, sigma, K):
    V = max(nvk)
    B = np.zeros((K, V))
    for v in range(V):
        for k in range(K):
            if v in nvk:
                B[k][v] = (nvk[v][k]+sigma)/(nk[k]+V*sigma)
    return B
    #the number of words wmi assigned to topic k and where the ith word in the mth document is not counted


def common_words(dics, nvk, nk, sigma, K):
    B = betakv(dics, nvk, nk, sigma, K)
    N = 20
    indices = np.zeros((K, N))
    #print(B[0, :])
    #print(np.argsort(B[0, :]))
    #print(np.argsort(B[0, :])[::-1])
    for k in range(K):
        indices[k, :] = (np.argsort(B[k, :])[::-1])[0:N]
    words = [[0]*N]*K
    for k in range(K):
        words[k] = list(map(lambda i:DIC[i], list(map(int, indices[k, :]))))
    common = np.zeros((K, N))
    for k in range(K):
        common[k, :] = B[k, list(map(int,indices[k, :]))]
    return (indices, words, common)

def doc_distr(dics, nmk, nm, alpha, K, m):
    T = thetamk(dics, nmk, nm, alpha, K)   # document topic distribution theta_m
    return T[m, :]

def manyruns():
    Ks = [10, 15, 5, 3]
    for K in Ks:
        dics = readDict(PATH1, 0.3, K)
        run_and_save(dics, K, 0.3, 0.3)
    alphas = [0.1, 0.5]
    for alpha in alphas:
        dics = readDict(PATH1, alpha, 3)
        run_and_save(dics, 3, alpha, 0.3)
    sigmas = [0.1, 0.5]
    for sigma in sigmas:
        dics = readDict(PATH1, 0.3, 3)
        run_and_save(dics, 3, 0.3, sigma)

def handle_results():
    Ks = [10, 15, 5, 3]
    def fileP(file, K):
        alpha = 0.3
        sigma = 0.3
        return file+":K="+str(K)+",alpha="+str(alpha)+",sigma="+str(sigma)+",iterations=500.p"
    for K in Ks:
        file1 = fileP("dics", K)
        file2 = fileP("nvk", K)
        file3 = fileP("nmk", K)
        file4 = fileP("nk", K)
        file5 = fileP("nm", K)
        (dics, nvk, nmk, nk, nm) = load_files(file1, file2, file3, file4, file5)
        (indices, words, common) = common_words(dics, nvk, nk, 0.3, K)
        for k in range(K):
            c1 = words[k]
            c2 = common[k, :]
            print("-----------K="+str(K)+"-----------")
            for c1, c2 in zip(c1, c2):
                print("%-9s %s" % (c1, c2))
            print("-------------------------")
        thetam = doc_distr(dics, nmk, nm, 0.3, K, 9)
handle_results()
#manyruns()
#run_and_save(dics, 3, 0.3, 0.3)
#(dics, nvk, nmk, nk, nm) = pz(dics, 3, 0.3, 0.3)
#print(nvk)
#(dics, nvk, nmk, nk, nm) = load_files(file1, file2, file3, file4, file5)
#print(type(nvk))
#B = betakv(dics, nvk, nk, 0.3, 3)
#print(common_words(dics, nvk, nk, 0.3, 3))
#print(doc_distr(dics, nmk, nm, 0.3, 3, 3))
#print(thetamk(dics, nmk, nm, 0.3, 3))
#print(sum(B[1, :]))
sigma = 0.3
alpha = 0.3


