import os
import copy
from pathlib import Path
import numpy as np

import networkx as nx

from .Mesh import Mesh
from .utils import get_color, export_ply


INIT_NUM_RESPS = 20


class HierarchicalKwayDecomposer(object):

    def __init__(self, mesh: Mesh, eps: float=0.1, maxiter: int=10, threshold: float=1.0):
        self.eps = eps
        self.maxiter = maxiter
        self.threshold = threshold
        self.colors = get_color(50)
        self.mesh = mesh
        ### dijkstra
        print("running dijkstra...")
        self.distance_dict = dict(nx.all_pairs_dijkstra_path_length(self.mesh.dual_graph))

    def decompose_submesh(self, subgraph: nx.Graph, root: bool=False):
        nodeids = list(subgraph.nodes())
        num_faces = len(nodeids)
        print("\nnum_faces: ", num_faces)
        if num_faces < INIT_NUM_RESPS:
            print("return mat of shape: ", (num_faces, 1))
            return np.ones((num_faces, 1))
        noderank = {}
        for i, nodeid in enumerate(nodeids):
            noderank[nodeid] = i
        distance = np.zeros((num_faces, num_faces), dtype=float)
        for i in range(num_faces):
            for j in range(num_faces):
                # print(nodeids[i], nodeids[j])
                # print(self.distance_dict[nodeids[i]])
                # print(self.distance_dict[nodeids[i]][nodeids[j]])
                distance[i][j] = self.distance_dict[nodeids[i]][nodeids[j]]

        ### init representatives
        print("init representatives...")
        sum_distance = np.sum(distance, axis=1)
        first_resp = np.argmin(sum_distance)
        G = []
        resps = [first_resp]
        for _ in range(min(INIT_NUM_RESPS-1, num_faces-1)):
            # select next representative
            dists = []
            valid_nodes = []
            for i in range(num_faces):
                if i not in resps:
                    dist_per_node = []
                    for resp in resps:
                        dist_per_node.append(distance[i][resp])
                    dists.append(dist_per_node)
                    valid_nodes.append(i)
            dists = np.array(dists)
            min_dists = np.min(dists, axis=1)
            max_node_rank = valid_nodes[np.argmax(min_dists)]
            # print(f"{k+2}-th representative id = {max_node_rank}: min distance = {min_dists[np.argmax(min_dists)]:.3f}")
            resps.append(max_node_rank)
            G.append(min_dists[np.argmax(min_dists)])
        # print("resps: ", resps)
        # print("G:", G)
        dG = []
        for i in range(len(G)-1):
            dG.append(G[i] - G[i+1])
        # print("dG:", dG)
        selected_k = np.argmax(dG) + 2
        # print(f"seleted {selected_k} patches.")
        selected_resps = np.sort(resps[:selected_k])
        # print("\nselected initial respresentatives: ", selected_resps)

        ### select K representatives
        print(f"select {selected_k} representatives...")
        last_F = float('inf')
        last_prob_mat = np.zeros(num_faces, selected_k)
        # O(T N^2)
        for t in range(self.maxiter):
            # print(f"# iter {t}:")
            ### calculate each node's probability to patch i
            probability_matrix = []
            for i in range(num_faces):
                dists = np.zeros(selected_k)
                if i not in selected_resps:
                    for j, resp in enumerate(selected_resps):
                        dists[j] = 1.0 / (distance[i][resp] + 1e-30)
                    dists = dists / np.sum(dists)
                else:
                    dists[np.where(selected_resps == i)] = 1.0
                probability_matrix.append(dists)
            probability_matrix = np.array(probability_matrix) # [N, K]
            # print(probability_matrix)
            ### minimize the target function and select new representatives
            new_resps = copy.deepcopy(selected_resps)
            tmp = np.dot(distance, probability_matrix) # [N, K]
            F = 0.0
            for k, resp in enumerate(selected_resps):
                F += tmp[resp][k]
            # print(f"\tF = {F:.3f}")
            # print(f"F' = {np.mean(np.sum(distance, axis=1))}")
            for k, resp in enumerate(selected_resps):
                min_i = np.argmin(tmp[:, k])
                if tmp[min_i][k] < tmp[resp][k]:
                    new_resps[k] = min_i
                    tmp[min_i, :] = float('inf')
            new_resps = np.sort(new_resps)
            if (selected_resps == new_resps).all() or F > last_F:
                if t < 1:
                    probability_matrix = last_prob_mat
                break
            last_F = F
            selected_resps = copy.deepcopy(new_resps)
            last_prob_mat = copy.deepcopy(probability_matrix)

        ### stop condition
        if not root:
            avg_dist = 0.0
            for i, respi in enumerate(new_resps):
                for j, respj in enumerate(new_resps[i+1:]):
                    avg_dist += distance[respi][respj]
            avg_dist /= (selected_k*(selected_k-1) / 2.0)
            print(f"check stop condition: avg dist = {avg_dist:.3f}")
            if avg_dist < self.threshold:
                print(f"stop")
                print("return mat of shape: ", (num_faces, 1))
                return np.ones((num_faces, 1))
        
        ### determine fuzzy patches
        print("determine fuzzy patches...")
        fuzzy_matrix = np.zeros(num_faces)
        belong_matrix = np.zeros((num_faces, selected_k))
        for i in range(num_faces):
            probs, prob_idxs = np.sort(probability_matrix[i])[::-1], np.argsort(probability_matrix[i])[::-1]
            prob0 = probs[0] / (probs[0] + probs[1])
            # prob1 = 1.0 - prob0
            if prob0 > 0.5 - self.eps and prob0 < 0.5 + self.eps:
                fuzzy_matrix[i] = 1
            if prob0 <= 0.5 - self.eps:
                belong_matrix[i][prob_idxs[1]] = 1.0
            elif prob0 >= 0.5 + self.eps:
                belong_matrix[i][prob_idxs[0]] = 1.0

        # do it until convergence
        while True:
            last_belong_matrix = copy.deepcopy(belong_matrix)
            for i in range(num_faces):
                if fuzzy_matrix[i] == 1:
                    probs, prob_idxs = np.sort(probability_matrix[i])[::-1], np.argsort(probability_matrix[i])[::-1]
                    possible_idxs = []
                    for idx in prob_idxs:
                        t = False
                        for neighbor in subgraph.neighbors(nodeids[i]):
                            if belong_matrix[noderank[neighbor]][idx] > 0:
                                t = True
                                break
                        if t:
                            possible_idxs.append(idx)
                    for idx in possible_idxs:
                        belong_matrix[i][idx] = 0.5
            if (last_belong_matrix == belong_matrix).all():
                break
        # delete some places where have more than two 0.5
        for i in range(num_faces):
            if fuzzy_matrix[i] == 1:
                probs, prob_idxs = np.sort(probability_matrix[i])[::-1], np.argsort(probability_matrix[i])[::-1]
                if np.sum(belong_matrix[i]) > 1.0:
                    count = 0
                    for idx in prob_idxs:
                        if belong_matrix[i][idx] == 0.5:
                            count += 1
                        if count > 2:
                            belong_matrix[i][idx] = 0.0
                elif np.sum(belong_matrix[i]) == 0.5:
                    fuzzy_matrix[i] = 0
                    belong_matrix[i] *= 2.0
                elif np.sum(belong_matrix[i]) == 0.0:
                    fuzzy_matrix[i] = 0
                    belong_matrix[i][prob_idxs[0]] = 1.0
        graphs = {}
        for i in range(selected_k):
            for j in range(i+1, selected_k):
                graphs[(i, j)] = nx.Graph()
                graphs[(i, j)].add_node(-1, name='S')
                graphs[(i, j)].add_node(-2, name='T')
        ### build subgraph between patches
        print("building subgraph between patches...")
        for i in range(num_faces):
            if fuzzy_matrix[i] == 1:
                patches = np.where(belong_matrix[i] == 0.5)[0]
                graphs[(patches[0], patches[1])].add_node(i)
                for neighbor in subgraph.neighbors(nodeids[i]):
                    if (fuzzy_matrix[noderank[neighbor]] == 0) and (noderank[neighbor] not in graphs[(patches[0], patches[1])].nodes()) and (belong_matrix[noderank[neighbor]][patches[0]] == 1.0 or belong_matrix[noderank[neighbor]][patches[1]] == 1.0):
                        graphs[(patches[0], patches[1])].add_node(noderank[neighbor])
        for i in range(num_faces):
            if fuzzy_matrix[i] == 1:
                patches = np.where(belong_matrix[i] == 0.5)[0]
                for neighbor in subgraph.neighbors(nodeids[i]):
                    if noderank[neighbor] in graphs[(patches[0], patches[1])].nodes():
                        # print(f"({patches[0]}, {patches[1]}): belong_matrix[{neighbor}] = {belong_matrix[neighbor]}")
                        if belong_matrix[noderank[neighbor]][patches[0]] == 1.0:
                            graphs[(patches[0], patches[1])].add_edge(-1, noderank[neighbor], weight=0.0, capacity=float('inf'))
                        elif belong_matrix[noderank[neighbor]][patches[1]] == 1.0:
                            graphs[(patches[0], patches[1])].add_edge(noderank[neighbor], -2, weight=0.0, capacity=float('inf'))
                        graphs[(patches[0], patches[1])].add_edge(i, noderank[neighbor], weight=subgraph.edges[nodeids[i], neighbor]["weight"], capacity=1.0/(1.0+self.mesh.angle_dists[nodeids[i]][neighbor]))
        ### apply mincut algorithm
        print("mincut...")
        for (i, j), g in graphs.items():
            if len(g.nodes()) > 2:
                if nx.is_connected(g):
                    _, partition = nx.minimum_cut(g, _s=-1, _t=-2)
                    reachables, unreachables = partition
                    for reachable in reachables:
                        belong_matrix[reachable] = 0.0
                        belong_matrix[reachable][i] = 1.0
                    for unreachable in unreachables:
                        belong_matrix[unreachable] = 0.0
                        belong_matrix[unreachable][j] = 1.0
                else:
                    for n in g.nodes():
                        belong_matrix[n] = 0.0
                        selected_patch = np.argmax(probability_matrix[n])
                        belong_matrix[n][selected_patch] = 1.0

        #### final hierarchical implementation
        ### split into subgraphs
        print("split into subgraphs...")
        belong_matrixs = []
        subgraphs = []
        for k in range(selected_k):
            tmp = np.where(belong_matrix[:, k] == 1.0)[0]
            patch_k = [nodeids[i] for i in tmp]
            _subgraph = nx.subgraph(subgraph, nbunch=patch_k)
            _nodeids = list(_subgraph.nodes())
            # print(f"{k}-th subgraph fake_nodes: ", tmp, "real nodes: ", patch_k)
            if len(_nodeids) > 0:
                subgraphs.append(_subgraph)
                mat = self.decompose_submesh(_subgraph, root=False)
                belong_matrixs.append(mat)
        ### construct final `belong_matrix`
        sum_k = 0
        for sub_belong_mat in belong_matrixs:
            sum_k += sub_belong_mat.shape[1]
        belong_matrix = np.zeros((num_faces, sum_k))
        tmp_sum = 0
        for k, (_subgraph, sub_belong_mat) in enumerate(zip(subgraphs, belong_matrixs)):
            nodes = list(_subgraph.nodes())
            for i in range(sub_belong_mat.shape[0]):
                for j in range(sub_belong_mat.shape[1]):
                    belong_matrix[noderank[nodes[i]]][tmp_sum+j] = sub_belong_mat[i][j]
            tmp_sum += sub_belong_mat.shape[1]
        ### check belong_matrix:
        for i in range(num_faces):
            if np.sum(belong_matrix[i]) != 1.0:
                print(f"[ERROR] sum[{i}] = {np.sum(belong_matrix[i])}!")
        ### return `belong_matrix`
        print("return mat of shape: ", belong_matrix.shape)
        return belong_matrix

    def decompose(self, outdir: str, filename: str):
        belong_matrix = self.decompose_submesh(self.mesh.dual_graph, root=True)
        num_class = belong_matrix.shape[1]
        colors = get_color(num_class)
        face_colors = ((1.0 - np.dot(belong_matrix, colors)) * 255.0).astype(int)
        os.makedirs(outdir, exist_ok=True)
        output_prefix = Path(filename).name.split('.')[0]
        # fuzzy_mesh_colored = trimesh.Trimesh(vertices=self.mesh.verts, faces=self.mesh.faces, face_colors=face_colors)
        export_ply(os.path.join(outdir, output_prefix + "_hie_kway.ply"), mesh=self.mesh, face_colors=face_colors)
