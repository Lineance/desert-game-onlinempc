from collections import deque
from typing import List, Tuple

import numpy as np


def get_adjacency_matrix(num_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    conn = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    for u, v in edges:
        u0, v0 = u - 1, v - 1
        if 0 <= u0 < num_nodes and 0 <= v0 < num_nodes:
            conn[u0, v0] = conn[v0, u0] = 1
    return conn


def get_neighbors(conn: np.ndarray, node: int) -> List[int]:
    return [i for i in range(conn.shape[0]) if conn[node, i] == 1]


def compute_shortest_distances(conn: np.ndarray, start: int) -> np.ndarray:
    num_nodes = conn.shape[0]
    distances = np.full(num_nodes, -1, dtype=np.int32)
    distances[start] = 0
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in range(num_nodes):
            if conn[current, neighbor] == 1 and distances[neighbor] == -1:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances


def shortest_next_hop(conn: np.ndarray, start: int, end: int) -> int:
    if start == end:
        return end
    queue = deque([start])
    parent = {start: None}

    while queue:
        node = queue.popleft()
        if node == end:
            break
        for nxt in get_neighbors(conn, node):
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)

    if end not in parent:
        return start

    cur = end
    while parent[cur] != start:
        cur = parent[cur]
        if cur is None:
            return start
    return cur
