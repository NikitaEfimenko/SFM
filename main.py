import math as m
import cv2
import matplotlib.tri as tri
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from pointCloudViewer import PCViewer

import icp

global xrot, yrot,cameraCord, clrCord, wWin, hWin, cO,cT,cor,triag


def getS(S):
    x = []
    y = []
    z = []
    for p in S:
        x.append(p[0])
        y.append(p[1])
        z.append(p[2])
    return  x, y, z



def getA(M, t0):
    def getA2():
        A2 = np.zeros((3, 3))
        A2[0][0] = Q[0]
        A2[0][1] = Q[1]
        A2[1][0] = Q[1]
        A2[0][2] = Q[2]
        A2[2][0] = Q[2]
        A2[1][1] = Q[3]
        A2[1][2] = Q[4]
        A2[2][1] = Q[4]
        A2[2][2] = Q[5]
        return A2

    def g(a, b):
        return np.reshape([a[0] * b[0], a[0] * b[1] + a[1] * b[0], b[0] * a[2] + a[0] * b[2], a[1] * b[1], a[2] * b[1] + a[1] * b[2], a[2] * b[2]], (1, 6))
    def getS():
        p = 1
        e = 0
        S[0] = g(M[e], M[e])
        for i in range(n):
            S[p] = g(M[e], M[e]) - g(M[e + 1], M[e + 1])
            S[p + 1] = g(M[e], M[e + 1])
            p += 2
            e += 2

    n, m = M.shape
    n /= 2
    S = np.zeros((2 * n + 1, 2 * m))
    count, _ = M.shape
    bb = np.reshape([0, 0], (2, 1))
    getS()
    b = np.concatenate(([[1]], np.tile(bb, (count/2, 1))), axis=0)
    ST = S.T
    STS = np.dot(ST, S)
    STb = np.dot(ST, b)
    Q = np.linalg.solve(STS, STb)
    A2 = getA2()
    u, s, v = np.linalg.svd(A2, False)
    sq = np.sqrt(s)
    return np.dot(u, np.diag(sq))


def showAllScene(X, Y, Z, clrCord, camOrds, camT):
    fig = plt.figure()
    axes = Axes3D(fig)
    r = 400
    axes.set_aspect('equal')
    axes.set_xlim3d([-r, r])
    axes.set_ylim3d([-r, r])
    axes.set_zlim3d([-r, r])
    axes.scatter(X, Y, Z, 'ro', c=clrCord, s=1, alpha=1.0)

    x, y, z = getS(camT)
    u, v, w = getS(camOrds)

    clr = range(0, len(x))
    axes.scatter(x, y, z, 'ro', s=5, c=clr, alpha=1.0)

    ux, vx, wx = getS(u)
    uy, vy, wy = getS(v)
    uz, vz, wz = getS(w)

    axes.quiver(x, y, z, ux, vx, wx, length=25, color="blue")
    axes.quiver(x, y, z, uy, vy, wy, length=25, color="green")
    axes.quiver(x, y, z, uz, vz, wz, length=50, color="red")

    fig.show()


def showPointsCloud(X, Y, Z, clrCord):
    fig = plt.figure()
    axes = Axes3D(fig)
    r = 300
    axes.set_aspect('equal')
    axes.set_xlim3d([-r, r])
    axes.set_ylim3d([-r, r])
    axes.set_zlim3d([-r, r])
    axes.scatter(X, Y, Z, 'ro', c=clrCord, s=3, alpha=1.0)
    fig.show()



def showCameras(camOrds, camT):
    fig = plt.figure()
    axes = Axes3D(fig)
    r = 400
    axes.set_aspect('equal')
    axes.set_xlim3d([-r, r])
    axes.set_ylim3d([-r, r])
    axes.set_zlim3d([-r, r])
    x, y, z = getS(camT)
    u, v, w = getS(camOrds)

    ux, vx, wx = getS(u)
    uy, vy, wy = getS(v)
    uz, vz, wz = getS(w)


    axes.quiver(x, y, z, ux, vx, wx, length=50, color="blue")
    axes.quiver(x, y, z, uy, vy, wy, length=50, color="green")
    axes.quiver(x, y, z, uz, wz, vz, length=5, color="red")

    fig.show()


def showMatchPoints(ls, w):
    fig = plt.figure()
    for i in range(w):
        g = map(lambda x: x[0], ls)
        ax = fig.add_subplot(w, 1, i + 1)
        img = cv2.drawKeypoints(frames[i], g, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show(fig)


def toCameraShape(sdvig):
    def B(cords):
        return sdvig + cords
    return B

def toCameraRotate(pre, cur):
    def to():
        a = np.linalg.solve(cur, np.reshape(pre[0], (3, 1)))
        b = np.linalg.solve(cur, np.reshape(pre[1], (3, 1)))
        c = np.linalg.solve(cur, np.reshape(pre[2], (3, 1)))
        K = np.concatenate((a, b, c), axis=1)
        return K
    return to().T


def collectPoint(frames, algorithm):
    def matches():
        l = range(0, n - 1)
        currentKP, currentDes = pointsAndDetectors[crnt]
        match = [[] for _ in l]

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        for j in l:
            m = bf.match(currentDes, pointsAndDetectors[j + 1][1])
            match[j] = m
        return match

    def showMatches():
        fig = plt.figure()
        l = range(0, n - 1)
        currentKP, currentDes = pointsAndDetectors[crnt]
        for i in l:
            ax = fig.add_subplot(n - 1, 1, i + 1)
            img = cv2.drawMatches(frames[crnt], currentKP, frames[i + 1], pointsAndDetectors[i + 1][0], matchesInFrames[i], None, flags=2)
            ax.imshow(img)
        plt.show()

    def pointList():
        frame = frames[crnt]
        current = pointsAndDetectors[crnt]
        grafKp = [[current[0][o]] for o in range(len(current[0]))]
        sP = current[1][0].shape
        grafDes = [[current[1][o]] for o in range(len(current[1]))]
        for im in range(len(matchesInFrames)):
            for m in matchesInFrames[im]:
                c = m.queryIdx
                k = m.trainIdx
                kp = pointsAndDetectors[im + 1][0][k]
                des = pointsAndDetectors[im + 1][1][k]
                grafDes[c].append(des)
                grafKp[c].append(kp)

        dl = filter(lambda x: len(x) == n, grafKp)
        dll = np.reshape(filter(lambda x: len(x) == n, grafDes),(len(dl), n, sP[0]))
        return dl, map(lambda x: tuple(frame[map(int, x[0].pt)[1]][map(int, x[0].pt)[0]][::-1] / 255.0), dl), dll


    crnt = 0
    method = algorithm()
    pointsAndDetectors = map(lambda x: method.detectAndCompute(x, None), frames)
    n = len(pointsAndDetectors)
    matchesInFrames = matches()
    return pointList()


path = '../tests/house.wmv'

cap = cv2.VideoCapture(path)

#zombie 100
#house 50

cor = np.reshape([],(0,3))
window = 10
frames = []
shift = window - 1
k = 0
flag = True
sp = 2

X = []
Y = []
Z = []

clrCord = []


q = 0
cO = []
cT = []
gf = 1/m.tan(np.pi/3)

def space(sp):
    for i in range(sp):
        _, _ = cap.read()


def M2Ords(M, t):
    camOrds = []
    nn, _ = M.shape
    n0 = nn / 2
    for c in range(0, nn, 2):
        mn = np.linalg.norm(M[c])
        nn = np.linalg.norm(M[c + 1])
        i = M[c] / mn
        j = M[c + 1] / nn
        k = np.cross(i, j)
        camOrds.append(normalize([i, j, k]))
    return np.reshape(camOrds, (n0, 3, 3))


def OrdsTrans2Pos(Ords, Tr):
    camT = []
    n = len(Ords)
    for c in range(n):
        pos = np.linalg.solve(Ords[c], Tr[c])
        camT.append(pos)
    return np.reshape(camT, (n, 3))


def t2Trans(M, t):
    camT = []
    nn = len(t)
    n = nn / 2
    for c in range(0, nn, 2):
        mm = np.linalg.norm(M[c])
        nn = np.linalg.norm(M[c + 1])
        zf = np.sqrt(1./2 * (1./mm/mm + 1./nn/nn))
        tx = -t[c] * zf
        ty = -t[c + 1] * zf
        tz = -zf
        camT.append([tz, ty, tx])
    return np.reshape(camT, (n, 3))


def trans(H, t):
    def A(x):
        return np.dot(H,x) + t
    return A


def normalize(v):
    return np.reshape(map(lambda x: x/np.finfo(v.dtype).eps if np.linalg.norm(x) == 0 else x/np.linalg.norm(x), v), (3, 3))
LS = []
sz = 100000


def SFM(ls, clr, des):
    m = len(ls)
    n = len(ls[0])
    W0 = np.zeros((2 * n, m))
    for i in range(0, 2 * n, 2):
        for j in range(m):
            c = ls[j][i / 2].pt
            W0[i][j] = c[1]
            W0[i + 1][j] = ww - c[0]
    t0 = W0.mean(axis=1)
    t = np.reshape(t0, (len(t0), 1))
    T = np.dot(t, np.ones((1, m)))
    W = W0 - T
    u, s, v = np.linalg.svd(W, False)
    r = 3
    U, s, V = u[:, :r], np.diag(s[:r]), v[:r, :]
    M, S = np.dot(U, s), V
    A = getA(M, t0)
    A = np.dot(np.diag([1., 1., -1.]), A)
    M, S = np.dot(M, A), np.dot(np.linalg.inv(A), S)
    camOrds = M2Ords(M, t0)
    ords = camOrds[0]
    R0 = ords.T
    M, S = np.dot(M, R0), np.dot(R0.T, S)
    camOrds = M2Ords(M, t0)
    TT = t2Trans(M, t0)
    camT = OrdsTrans2Pos(camOrds, TT)
    return M, S, camOrds, camT


#space(100)
#k + window < sz
def triangulate(cor):
    x = cor[:, 0]
    y = cor[:, 1]
    x = x.ravel()
    y = y.ravel()
    triag = tri.Triangulation(x, y)
    triag = triag.triangles.ravel()
    return triag



def pointFilter(S, count = 10, n = 2, axes=1):
    if axes == 1:
        SS = S.T[:,-1:]
    elif axes == 2:
        SS = S.T[:,-2:]
    else:
        SS = S.T[:,-3:]
    nbrs = NearestNeighbors(n_neighbors=count, algorithm="kd_tree").fit(SS)
    dis, ind = nbrs.kneighbors(SS)
    l = reduce(lambda a,x: a + x,dis.mean(axis=0),0)
    fl = filter(lambda i: reduce(lambda a,x: a + x, dis[i],0) < n*l, range(len(dis)))
    return fl

while q < 1:
    for i in range(window):
        ret, frame = cap.read()
        space(sp)
        if frame is not None:
            frames.append(frame)
            ww, hh, _ = frame.shape
        else:
            sz = len(frames)

    data = frames[k: k + window]
    ls, clr, des = collectPoint(data, cv2.AKAZE_create)

    M, S, camOrds, camT = SFM(ls, clr, des)

    idx = pointFilter(S)
    S = S[:,idx]
    des = des[idx]
    clr = [ clr[idx[i]]  for i in range(len(idx))]
    ST = S.T
    clrCord = clrCord + clr
    if q == 0:
        cor = np.concatenate((cor, ST), axis=0)
        cO = camOrds
        cT = camT
        testT = np.reshape([], (0, 3))
        testO = np.reshape([], (0, 3, 3))
        Sprev = S
        preDes = des
        triag = triangulate(ST)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        match = bf.match(preDes[:,-(window - shift) + 2,:], des[:,0,:])
        idx = np.reshape(map(lambda x : [x.queryIdx ,x.trainIdx],match),(len(match), 2))
        preIdx = idx[:,0]
        curIdx = idx[:,1]
        K, t = icp.icp(S[:, curIdx], Sprev[:, preIdx])
        S = np.dot(K,S) + t
        camT = (np.dot(K,camT.T) + t).T
        camOrds = np.reshape(map(lambda x: normalize(np.dot(K, x.T).T), camOrds),camOrds.shape)
        ST = S.T
        cO = np.concatenate((cO, camOrds), axis=0)
        cT = np.concatenate((cT, camT), axis=0)
        cor = np.concatenate((cor, ST), axis=0)
        Sprev = S
        preDes = des
        tr = triangulate(ST)
        triag = np.concatenate((triag, tr), axis=0)
    q += 1
    k += shift
    xx, yy, zz = getS(ST)
    X = X + xx
    Y = Y + yy
    Z = Z + zz

#d1, d2 = cv2.cvtColor(data[u - 5],cv2.COLOR_RGB2GRAY), cv2.cvtColor(data[u + 5],cv2.COLOR_RGB2GRAY)
#disp = cv2.calcOpticalFlowFarneback(d1,d2, None, 0.5, 3, 15, 3, 10, 1.2, 0)
#zf, _ = cv2.polarToCart(disp[:,:,0],disp[:,:,1])


fl = pointFilter(cor.T,count = 30, n = 2, axes=3)
cor = cor[fl]
clrCord = [clrCord[fl[i]] for i in range(len(fl))]

#showAllScene(X, Y, Z, clrCord, cO, cT)
#showAllScene(X, Y, Z, clrCord, testO, testT)
'''
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
ax = fig.add_subplot(2, 1, 2)
ax.imshow(cv2.cvtColor(frames[k + window - shift - 1  ], cv2.COLOR_BGR2RGB))
plt.show()
'''
viewer = PCViewer()
viewer.initPoints(cor, clrCord, triag)
viewer.unmeshPC()
viewer.on()