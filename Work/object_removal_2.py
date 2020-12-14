'''
some changes were made to getRemovalPaths()
edge:[from,to,capacity,cost]
usage: getRemovalPaths(image,mask)
'''
import cv2, imageio, imutils
import numpy as np
from queue import Queue
from collections import defaultdict
from utils import *

INF = 1e9
constant = 1000


def getRemovalPaths(image, maskPath):
    '''return: number of paths, paths, flag
    for each path in paths,the sequence is from sink to source
    if flag=1, rotation has been performed'''

    mask = cv2.imread(maskPath, 0)
    print(mask.shape)
    flag = 0  # if flag=1, rotation has been performed

    # ratio divided by bottleneck
    ratio = 1
    bottleNeck = max_width(maskPath) // ratio
    print(f'1:{bottleNeck}')

    rotatedMask = "../figures/ro.jpg"
    bottleNeck2 = max_width(rotatedMask) // ratio
    print(f'2:{bottleNeck2}')
    mat=cv2.imread(image)
    if bottleNeck > bottleNeck2:
        mat=imutils.rotate(mat,angle=90)
        flag=1
        bottleNeck=bottleNeck2
        mask=rotatedMask
    H = mat.shape[0]
    W = mat.shape[1]
    
    eMap = calc_energy_map(mat).astype(np.int32)
    eMap[np.where(mask[:, :] > 0)] *= -constant
    print(f'bottleneck={bottleNeck}')
    edges = []

    s1, s2 = 2 * W * H + 1, 2 * W * H + 3
    t = 2 * W * H + 5
    # The following part is to add edges
    # I don't added any reversed edges to boost speed
    # add edge (s1,s2)
    edges.append([s1, s2, bottleNeck, 0])
    # add edges from s2 to all nodes in the first row
    for y in range(W):
        cost = eMap[0][y] >> 1
        edges.append([s2, 2 * y, 1, cost])
    # add edges in the following rows
    for x in range(H - 1):
        for y in range(bottleNeck + 1):
            edges.append([2 * W * x + 2 * y, 2 * W * x + 2 * y + 1, 1, 0])
            for newY in range(y + bottleNeck + 1):
                cost = (eMap[x][y] + eMap[x + 1][newY]) >> 1
                edges.append([2 * W * x + 2 * y + 1, 2 * W * (x + 1) + 2 * newY, 1, cost])
        for y in range(bottleNeck + 1, W - bottleNeck - 1):
            edges.append([2 * W * x + 2 * y, 2 * W * x + 2 * y + 1, 1, 0])
            for newY in range(y - bottleNeck, y + bottleNeck + 1):
                cost = (eMap[x][y] + eMap[x + 1][newY]) >> 1
                edges.append([2 * W * x + 2 * y + 1, 2 * W * (x + 1) + 2 * newY, 1, cost])
        for y in range(W - bottleNeck - 1, W):
            edges.append([2 * W * x + 2 * y, 2 * W * x + 2 * y + 1, 1, 0])
            for newY in range(y - bottleNeck, W):
                cost = (eMap[x][y] + eMap[x + 1][newY]) >> 1
                edges.append([2 * W * x + 2 * y + 1, 2 * W * (x + 1) + 2 * newY, 1, cost])
    # add edges from all nodes in the last row to sink node t
    for y in range(W):
        edges.append([2 * W * (H - 1) + 2 * y, 2 * W * (H - 1) + 2 * y + 1, 1, 0])
        cost = eMap[H - 1][y] >> 1
        edges.append([2 * W * (H - 1) + 2 * y + 1, t, 1, cost])

    num = H * W * 2 + 6
    numOfSeam, paths = minCostFlow(num, edges, bottleNeck, s1, t, H,
                                   W)  # path doesn't inlcude t, 21, s2. but it is in reversed order.
    print(f'number of seams={numOfSeam}')
    # for path in paths:
    #     print(f'{path}\n')
    return numOfSeam, paths, flag


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


def minCostFlow(N, edges, K, s, t, H, W):
    adj = [[] for i in range(N)]
    cost = defaultdict(int)
    cap = defaultdict(int)

    for (source, desti, capacity, _cost) in edges:
        adj[source].append(desti)
        cap[(source, desti)] = capacity
        cost[(source, desti)] = _cost

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


image='../figures/pic.jpg'
mask='../figures/mask.jpg'

numOfSeam, paths, flag = getRemovalPaths(image, mask)
new_img = delete_seams(image, paths)
cv2.imwrite("deleted.png", new_img)
