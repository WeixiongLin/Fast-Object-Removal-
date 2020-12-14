'''
use while loop to delete seams,
this means we have to construct graphs in multiple times
'''

import numpy as np
from queue import Queue
from collections import defaultdict
import imutils
import cv2
from utils import *
# from git_utils import *
import imutils,cv2

INF = 1e9
constant = 1000
connectionWidth=3
maxSeamNum=10

def constructGraph(img,mask):
    '''
    :param img: numpy array
    :param mask: numpy array
    :return:
        edges(looks like [[t,v1,v2,s],[...],...]),
        s1(source,type is int),
        t(sink, type is int)
    '''

    H = img.shape[0]
    W = img.shape[1]
    eMap = calc_energy_map(img).astype(np.int32)
    eMap[np.where(mask[:, :] > 0)] *= -constant
    s1, s2 = 2 * W * H , 2 * W * H +1  # caution
    t = 2 * W * H+2

    adj = [[] for i in range(2 * W * H + 3)]
    cap,costDic=dict(),dict()
    # The following part is to add edges
    # I don't add any reversed edges to boost speed
    # add edge (s1,s2)
    adj[s1].append(s2)
    cap[(s1,s2)]=maxSeamNum
    costDic[(s1,s2)]=0
    # add edges from s2 to all nodes in the first row
    for y in range(W):
        cost = eMap[0][y] >> 1
        nodeNum=2*y
        adj[s2].append(nodeNum)
        cap[(s2, nodeNum)] = 1
        costDic[(s1, s2)] = cost
    # add edges in the following rows
    for x in range(H - 1):
        for y in range(connectionWidth + 1):
            nodeNum=2 * W * x + 2 * y
            adj[nodeNum].append(nodeNum+1)
            cap[(nodeNum,nodeNum+1)]=1
            costDic[(nodeNum,nodeNum+1)]=0 # split the node in layer x
            for newY in range(y + connectionWidth + 1):
                cost = (eMap[x][y] + eMap[x + 1][newY]) >> 1
                newNodeNum=2 * W * (x + 1) + 2 * newY
                adj[nodeNum+1].append(newNodeNum)
                cap[(nodeNum+1, newNodeNum)] = 1
                costDic[(nodeNum+1, newNodeNum)] = cost
        for y in range(connectionWidth + 1, W - connectionWidth - 1):
            nodeNum = 2 * W * x + 2 * y
            adj[nodeNum].append(nodeNum + 1)
            cap[(nodeNum, nodeNum + 1)] = 1
            costDic[(nodeNum, nodeNum + 1)] = 0
            for newY in range(y - connectionWidth, y + connectionWidth + 1):
                cost = (eMap[x][y] + eMap[x + 1][newY]) >> 1
                newNodeNum = 2 * W * (x + 1) + 2 * newY
                adj[nodeNum + 1].append(newNodeNum)
                cap[(nodeNum + 1, newNodeNum)] = 1
                costDic[(nodeNum + 1, newNodeNum)] = cost
        for y in range(W - connectionWidth - 1, W):
            nodeNum = 2 * W * x + 2 * y
            adj[nodeNum].append(nodeNum + 1)
            cap[(nodeNum, nodeNum + 1)] = 1
            costDic[(nodeNum, nodeNum + 1)] = 0
            for newY in range(y - connectionWidth, W):
                cost = (eMap[x][y] + eMap[x + 1][newY]) >> 1
                newNodeNum = 2 * W * (x + 1) + 2 * newY
                adj[nodeNum + 1].append(newNodeNum)
                cap[(nodeNum + 1, newNodeNum)] = 1
                costDic[(nodeNum + 1, newNodeNum)] = cost
    # add edges from all nodes in the last row to sink node t
    for y in range(W):
        nodeNum=2 * W * (H - 1) + 2 * y
        adj[nodeNum].append(nodeNum+1)
        cap[(nodeNum, nodeNum+1)] = 1
        costDic[(nodeNum,nodeNum+1)] = 0

        cost = eMap[H - 1][y] >> 1
        adj[nodeNum+1].append(t)
        cap[(nodeNum+1, t)] = 1
        costDic[(nodeNum+1, t)] = cost

    return adj,costDic,cap,2 * W * H + 3,s1,t,H,W

def shortestPath(n, v0, t, adj, cap, cost, H, W):
    dist = [INF] * n
    prev = [0] * n
    dist[v0] = 0
    inq = [False] * n
    q = Queue()
    q.put(v0)

    while (not q.empty()):
        out = q.get()
        inq[out] = False
        for v in adj[out]:
            if cap[(out, v)] > 0 and dist[v] > dist[out] + cost[(out, v)]:
                dist[v] = dist[out] + cost[(out, v)]
                prev[v] = out
                if (not inq[v]):
                    inq[v] = True
                    q.put(v)
    if dist[t] == INF:
        return -1, []
    # find path
    cur = t
    path = []  # path is from t to s, which is in reversed order. It doesn't include t and v0.
    _W = 2 * W
    while cur != v0:
        cap[(prev[cur], cur)] -= 1
        cur = prev[cur]
        if cur % 2 == 0:
            x = cur // _W
            y = (cur - _W * x) // 2
            path.append((x, y))
    return 0, path

def minCostFlow(adj,cost,cap, N, K, s, t, H, W):
    flow = 0
    paths = []
    while flow < K:
        flow += 1
        print(f'flow={flow}')
        flag, path = shortestPath(N, s, t, adj, cap, cost, H, W)
        print('finish calc path')
        if flag == -1:
            break
        paths.append(path)
    return flow, paths

def objectRemoval(imagePath,maskPath):
    img=cv2.imread(imagePath)
    mask=cv2.imread(maskPath, 0)
    flag = 0  # if flag=1, rotation has been performed

    ratio = 1
    maskWidth = max_width(mask) // ratio
    rotatedMask = imutils.rotate(mask, angle=90)
    rotatedMaskWidth = max_width(rotatedMask) // ratio
    if rotatedMaskWidth < maskWidth:
        img = imutils.rotate(img, angle=90)
        mask = rotatedMask
        flag = 1
    while len(np.where(mask[:, :] > 0)[0]) > 0:
        energy_map = calc_energy_map(img)
        print(type(np.where(mask[:, :] > 0)[0][0]))
        energy_map[np.where(mask[:, :] > 0)] *= -constant
        adj,cost,cap,n,s1,t,H,W=constructGraph(img,mask)
        flow,paths=minCostFlow(adj,cost,cap,n,maxSeamNum,s1,t,H,W)
        img,mask=delete(img,mask,paths)


objectRemoval('../figures/pic.jpg', '../figures/mask.jpg')