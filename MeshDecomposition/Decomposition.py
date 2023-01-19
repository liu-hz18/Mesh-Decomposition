import os
import copy
from pathlib import Path
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from .Mesh import Mesh
from .utils import get_color, export_ply


INIT_NUM_RESPS = 20


# 生成带权对偶图, 考虑角距离和测地线距离，得出最短距离矩阵 (preprocess)
# 对于层次化递归树的每个节点（每个区域）：
# 选择第一个代表点：该点到其他所有点的最短路和 是 最小的
# 添加其他代表点：到之前所有代表点的 最短的最短路径 最大 ？
# 计算 G(k) = min_{i<k} Dist(REP_k, REP_i)
# 当 G(k) - G(k+1) 最大的时候，选择k。象征k路分解. 得到了k个代表点
# 1.图上任一点属于第i个patch的概率由归一化公式得出
# 2.使用kmeans算法重新计算k个代表点，直到k个代表点在相邻两次迭代中保持不变。否则回到1
# 概率由 \epsilon 确定为3部分：A B C
# 使用最大流最小割算法确定C的最小割，以确定边界

# 层次化K路分解
class KWayDecomposer(object):

    def __init__(self, eps: float=0.1, max_iter: int=100):
        self.eps = eps
        self.max_iter = max_iter
        self.distance: np.ndarray
        self.selected_k = 0
        self.selected_resps = []
        self.probability_matrix: np.ndarray
        self.belong_matrix: np.ndarray
        self.fuzzy_matrix: np.ndarray
        self.graphs = {}
        self.colors: np.ndarray

    def dijkstra(self, mesh: Mesh):
        print("running dijkstra...")
        distance_dict = dict(nx.all_pairs_dijkstra_path_length(mesh.dual_graph))
        self.distance = np.zeros((mesh.num_faces, mesh.num_faces), dtype=float)
        for i in range(mesh.num_faces):
            for j in range(mesh.num_faces):
                self.distance[i][j] = distance_dict[i][j]
        print("dijkstra done.")

    def init_representatives(self, mesh: Mesh, display: bool=False):
        print("selecting initial representatives...")
        sum_distance = np.sum(self.distance, axis=1)
        first_resp = np.argmin(sum_distance)
        # print("1-st representative id = ", first_resp)
        G = []
        resps = [first_resp]
        for k in range(INIT_NUM_RESPS-1):
            # select next representative
            dists = []
            valid_nodes = []
            for i in range(mesh.num_faces):
                if i not in resps:
                    dist_per_node = []
                    for resp in resps:
                        dist_per_node.append(self.distance[i][resp])
                    dists.append(dist_per_node)
                    valid_nodes.append(i)
            dists = np.array(dists)
            min_dists = np.min(dists, axis=1)
            max_node_rank = valid_nodes[np.argmax(min_dists)]
            # print(f"{k+2}-th representative id = {max_node_rank}: min distance = {min_dists[np.argmax(min_dists)]:.3f}")
            resps.append(max_node_rank)
            G.append(min_dists[np.argmax(min_dists)])
        print("resps: ", resps)
        # print("G:", G)
        dG = []
        for i in range(len(G)-1):
            dG.append(G[i] - G[i+1])
        print("dG:", dG)
        self.selected_k = np.argmax(dG) + 2
        print(f"seleted {self.selected_k} patches.")
        self.selected_resps = np.sort(resps[:self.selected_k])
        print("\nselected initial respresentatives: ", self.selected_resps)
        self.colors = np.array(get_color(self.selected_k)) # [K, 3]
        # plot G and dG
        if display:
            plt.figure()
            plt.subplot(121)
            plt.plot(list(range(2, 21)), G)
            plt.xlabel('k')
            plt.ylabel('G(k)')
            plt.xticks(list(range(2, 21)))
            plt.subplot(122)
            plt.plot(list(range(2, 20)), dG)
            plt.xlabel('k')
            plt.ylabel('dG(k)')
            plt.xticks(list(range(2, 20)))
            plt.show()

    def fuzzy_kmeans(self, mesh: Mesh, maxiter: int=20):
        # select K representatives
        print("Fuzzy K-means running...")
        last_F = float('inf')
        last_prob_mat = np.zeros(mesh.num_faces, self.selected_k)
        # O(T N^2)
        for t in range(maxiter):
            print(f"# iter {t}:")
            ### calculate each node's probability to patch i
            probability_matrix = []
            for i in range(mesh.num_faces):
                dists = np.zeros(self.selected_k)
                if i not in self.selected_resps:
                    for j, resp in enumerate(self.selected_resps):
                        dists[j] = 1.0 / (self.distance[i][resp] + 1e-30)
                    dists = dists / np.sum(dists)
                else:
                    dists[np.where(self.selected_resps == i)] = 1.0
                probability_matrix.append(dists)
            probability_matrix = np.array(probability_matrix) # [N, K]
            # print(probability_matrix)
            ### minimize the target function and select new representatives
            new_resps = copy.deepcopy(self.selected_resps)
            tmp = np.dot(self.distance, probability_matrix) # [N, K]
            F = 0.0
            for k, resp in enumerate(self.selected_resps):
                F += tmp[resp][k]
            print(f"\tF = {F:.3f}")
            # print(f"F' = {np.mean(np.sum(distance, axis=1))}")
            for k, resp in enumerate(self.selected_resps):
                min_i = np.argmin(tmp[:, k])
                if tmp[min_i][k] < tmp[resp][k]:
                    new_resps[k] = min_i
                    tmp[min_i, :] = float('inf')
            new_resps = np.sort(new_resps)
            if (self.selected_resps == new_resps).all() or F > last_F:
                break
            last_F = F
            self.selected_resps = copy.deepcopy(new_resps)
            last_prob_mat = copy.deepcopy(probability_matrix)
            print("\tresps changed to: ", self.selected_resps)

        # final representatives
        # compute final probability matrix
        self.probability_matrix = last_prob_mat
        # print(probability_matrix) # [N, K]
        print("Final representatives: ", self.selected_resps)
        
    def determine_fuzzy_patchs(self, mesh: Mesh, outdir: str, filename: str, eps: float=0.1):
        ### determine fuzzy areas
        self.fuzzy_matrix = np.zeros(mesh.num_faces)
        self.belong_matrix = np.zeros((mesh.num_faces, self.selected_k))
        for i in range(mesh.num_faces):
            probs, prob_idxs = np.sort(self.probability_matrix[i])[::-1], np.argsort(self.probability_matrix[i])[::-1]
            prob0 = probs[0] / (probs[0] + probs[1])
            # prob1 = 1.0 - prob0
            if prob0 > 0.5 - eps and prob0 < 0.5 + eps:
                self.fuzzy_matrix[i] = 1
            if prob0 <= 0.5 - eps:
                self.belong_matrix[i][prob_idxs[1]] = 1.0
            elif prob0 >= 0.5 + eps:
                self.belong_matrix[i][prob_idxs[0]] = 1.0
        ### paint fuzzy areas and output intermediate results (in top-2 probs)
        # belong_matrix: [N, K]
        face_colors = ((1.0 - np.dot(self.belong_matrix, self.colors)) * 255.0).astype(int)
        # face_colors[np.where(np.sum(face_colors, axis=1) < 1e-3)] = np.array([255, 255, 255])
        os.makedirs(outdir, exist_ok=True)
        output_prefix = Path(filename).name.split('.')[0]
        export_ply(os.path.join(outdir, output_prefix + "_fuzzy.ply"), mesh=mesh, face_colors=face_colors)

        # do it until convergence
        while True:
            last_belong_matrix = copy.deepcopy(self.belong_matrix)
            for i in range(mesh.num_faces):
                if self.fuzzy_matrix[i] == 1:
                    probs, prob_idxs = np.sort(self.probability_matrix[i])[::-1], np.argsort(self.probability_matrix[i])[::-1]
                    possible_idxs = []
                    for idx in prob_idxs:
                        t = False
                        for neighbor in mesh.dual_graph.neighbors(i):
                            if self.belong_matrix[neighbor][idx] > 0:
                                t = True
                                break
                        if t:
                            possible_idxs.append(idx)
                    for idx in possible_idxs:
                        self.belong_matrix[i][idx] = 0.5
            if (last_belong_matrix == self.belong_matrix).all():
                break
        # delete some places where have more than two 0.5
        for i in range(mesh.num_faces):
            if self.fuzzy_matrix[i] == 1:
                if np.sum(self.belong_matrix[i]) > 1.0:
                    probs, prob_idxs = np.sort(self.probability_matrix[i])[::-1], np.argsort(self.probability_matrix[i])[::-1]
                    count = 0
                    for idx in prob_idxs:
                        if self.belong_matrix[i][idx] == 0.5:
                            count += 1
                        if count > 2:
                            self.belong_matrix[i][idx] = 0.0
                elif np.sum(self.belong_matrix[i]) == 0.5:
                    self.fuzzy_matrix[i] = 0
                    self.belong_matrix[i] *= 2.0
                elif np.sum(self.belong_matrix[i]) == 0.0:
                    self.fuzzy_matrix[i] = 0
                    self.belong_matrix[i][prob_idxs[0]] = 1.0
        # check result
        for i in range(mesh.num_faces):
            if np.sum(self.belong_matrix[i]) != 1.0:
                raise ValueError(f"{i}: {self.belong_matrix[i]} {self.fuzzy_matrix[i]}")

    def build_flowgraph(self, mesh: Mesh):
        self.graphs = {}
        for i in range(self.selected_k):
            for j in range(i+1, self.selected_k):
                self.graphs[(i, j)] = nx.Graph()
                self.graphs[(i, j)].add_node(-1, name='S')
                self.graphs[(i, j)].add_node(-2, name='T')
        # build subgraph
        for i in range(mesh.num_faces):
            if self.fuzzy_matrix[i] == 1:
                patches = np.where(self.belong_matrix[i] == 0.5)[0]
                self.graphs[(patches[0], patches[1])].add_node(i)
                for neighbor in mesh.dual_graph.neighbors(i):
                    if (self.fuzzy_matrix[neighbor] == 0) and (neighbor not in self.graphs[(patches[0], patches[1])].nodes()) and (self.belong_matrix[neighbor][patches[0]] == 1.0 or self.belong_matrix[neighbor][patches[1]] == 1.0):
                        self.graphs[(patches[0], patches[1])].add_node(neighbor)
        for i in range(mesh.num_faces):
            if self.fuzzy_matrix[i] == 1:
                patches = np.where(self.belong_matrix[i] == 0.5)[0]
                for neighbor in mesh.dual_graph.neighbors(i):
                    if neighbor in self.graphs[(patches[0], patches[1])].nodes():
                        # print(f"({patches[0]}, {patches[1]}): self.belong_matrix[{neighbor}] = {self.belong_matrix[neighbor]}")
                        if self.belong_matrix[neighbor][patches[0]] == 1.0:
                            self.graphs[(patches[0], patches[1])].add_edge(-1, neighbor, weight=0.0, capacity=float('inf'))
                        elif self.belong_matrix[neighbor][patches[1]] == 1.0:
                            self.graphs[(patches[0], patches[1])].add_edge(neighbor, -2, weight=0.0, capacity=float('inf'))
                        self.graphs[(patches[0], patches[1])].add_edge(i, neighbor, weight=mesh.dual_graph.edges[i, neighbor]["weight"], capacity=1.0/(1.0+mesh.angle_dists[i][neighbor]))

    def mincut(self):
        for (i, j), g in self.graphs.items():
            if len(g.nodes()) > 2:
                if nx.is_connected(g):
                    # nx.draw(g, with_labels=True)
                    # plt.show()
                    _, partition = nx.minimum_cut(g, _s=-1, _t=-2)
                    reachables, unreachables = partition
                    for reachable in reachables:
                        self.belong_matrix[reachable] = 0.0
                        self.belong_matrix[reachable][i] = 1.0
                    for unreachable in unreachables:
                        self.belong_matrix[unreachable] = 0.0
                        self.belong_matrix[unreachable][j] = 1.0
                else:
                    for n in g.nodes():
                        self.belong_matrix[n] = 0.0
                        selected_patch = np.argmax(self.probability_matrix[n])
                        self.belong_matrix[n][selected_patch] = 1.0

    def save_decomposition(self, mesh: Mesh, outdir: str, filename: str):
        face_colors = ((1.0 - np.dot(self.belong_matrix, self.colors)) * 255.0).astype(int)
        os.makedirs(outdir, exist_ok=True)
        output_prefix = Path(filename).name.split('.')[0]
        export_ply(os.path.join(outdir, output_prefix + "_final.ply"), mesh=mesh, face_colors=face_colors)

    def decompose(self, mesh: Mesh, outdir: str, filename: str, display: bool=False):
        self.dijkstra(mesh)
        self.init_representatives(mesh, display=display)
        self.fuzzy_kmeans(mesh, maxiter=self.max_iter)
        self.determine_fuzzy_patchs(mesh, outdir=outdir, filename=filename, eps=self.eps)
        self.build_flowgraph(mesh)
        self.mincut()
        self.save_decomposition(mesh, outdir=outdir, filename=filename)
