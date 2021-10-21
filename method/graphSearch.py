import queue, heapq

def argOne(l):
    return l.index(1.0)

def DFS(G: list, v: int):
    visited = [0 for i in G]
    visited[v] = 1
    parent = [(-1,-1) for i in G]
    parent[v] = (v, 0)
    for n, a in G[v]:
        if not visited[n]:
            visited[n] = 1
            parent[n] = (v, a) # parent v + action a = n
            DFS_(G, n, visited, parent)
    return parent

def DFS_(G: list, v: int, visited: list, parent: list):
    for n, a in G[v]:
        if not visited[n]:
            visited[n] = 1
            parent[n] = (v, a)
            DFS_(G, n, visited, parent)

def BFS(G: list, v:int):
    Q = queue.Queue()
    Q.put(v)
    visited = [0 for i in G]
    visited[v] = 1
    parent = [(-1,-1) for i in G]
    parent[v] = (v, 0)
    while not Q.empty():
        v_ = Q.get()
        for n, a in G[v_]:
            if not visited[n]:
                visited[n] = 1
                parent[n] = (v_, a)
                Q.put(n)
    return parent

def AStar(G: list, si: int, sf: int, S: list):
    def h(S, s1, s2):
        # heuristic that estimates the distance btw s and sf
        # !!! this is problem-specific !!!
        # L1-norm
        return abs(S[s1][0] - S[s2][0]) + abs(S[s1][1] - S[s2][1])

    C = [] # closed set
    O = [(0, si)] # open set, sorted as a heap according to F of each element
    g = [0.0 for s in G]
    parent = [(-1,-1) for i in G]
    parent[si] = (si, 0)
    while len(O):
        v = heapq.heappop(O)[1]
        if v == sf:
            return parent
        for n, a in G[v]:
            if v == n: continue
            if n in C: continue
            elif n in O:
                g_ = g[v] + 3 # g = G(current_node) + weight(curr, neigh)
                f = g_ + h(S, n, sf)
                if g[n] > g_:
                    g[n] = g_
                    parent[n] = (v, a)
                    # update f of n in open set
                    for i, _, v_ in enumerate(O):
                        if v_ == n:
                            O.remove(i)
                            heapq.heapify(O)
                            heapq.heappush(O, (f, n))
                            break
            else:
                g_ = g[v] + 3
                f = g_ + h(S, n, sf)
                g[n] = g_
                parent[n] = (v, a)
                heapq.heappush(O, (f, n))
        C.append(v)
    raise RuntimeError("Goal not reachable.")

def solve(env: dict, H: int, gamma: float, thr: float):
    S = env['S']
    A = env['A']
    P = env['P']
    si, sf = env['sisf'] # initial state and goal state
    nS = len(S)

    G = [[] for s in S] # adjacency list: int -> [(int, int)]
    for i, s in enumerate(S):
         for j, a in enumerate(A):
             s_ = argOne(P(s, a))
             G[i].append((s_, j))

    #parent = DFS(G, si)
    #parent = BFS(G, si)
    parent = AStar(G, si, sf, S)
    pi = []
    s = sf
    while s != si:
        p, a = parent[s]
        s = p
        pi.append((p, a))
    return pi, []
