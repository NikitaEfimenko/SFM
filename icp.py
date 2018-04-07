from sklearn.neighbors import NearestNeighbors
import numpy as np

def estimateRigitTransform(a, b):
    aMean = a.mean(axis=0)
    bMean = b.mean(axis=0)
    a = a - aMean
    b = b - bMean
    m = a.shape[1]
    H = np.dot(a.T, b)
    U, E, VT = np.linalg.svd(H)
    R = np.dot(VT.T, U.T)
    if np.linalg.det(R) < 0:
        VT[m - 1, :] *= -1
        R = np.dot(VT.T, U.T)
    T = bMean - np.dot(R, aMean)
    return R, np.reshape(T,(3,1))



def icp(src, dst, init_pose=(0,0,0), tolerance=0.00001):
    maxIter = 200
    srcCur = src.copy()
    prevEr = 0
    delta = np.reshape(dst.mean(axis=1) - src.mean(axis=1), (3,1))
    src = src + delta
    b = dst.T
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(b)
    for i in range(maxIter):
        a = src.T
        distances, indices = nbrs.kneighbors(a)
        A, t = estimateRigitTransform(a, b[indices.T][0])
        src = np.dot(A, src) + t
        curEr = distances.mean(axis=0)
        if (np.abs(prevEr - curEr)) < tolerance:
            break
        prevEr = curEr
    print(curEr)
    A, T = estimateRigitTransform(srcCur.T, src.T)
    return A ,T
