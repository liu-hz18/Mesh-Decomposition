import numpy as np

import trimesh
import networkx as nx
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class Mesh(object):

    def __init__(self, filename: str):
        # to keep the raw data intact, disable any automatic processing "process=False"
        self.mesh = trimesh.load_mesh(filename, process=True, validate=True)
        
        # `faces`: (n, 3) int64, The faces(triangles) of the mesh.
        self.faces = np.array(self.mesh.faces)
        print("Number of faces: ", self.faces.shape[0])
        # `vertices`: (n, 3) float, The vertices of the mesh. (Points in space referenced by self.faces)
        self.verts = np.array(self.mesh.vertices)
        # `face_normals`: (len(self.faces), 3) np.float64, Return the unit normal vector for each face.
        self.norms = np.array(self.mesh.face_normals) # 法向量
        
        # meta information
        self.num_faces = self.faces.shape[0]

        # adjacent faces
        # `face_adjacency`: Find faces that share an edge, i.e. ‘adjacent’ faces.
        self.face_adj = self.mesh.face_adjacency
        print("Number of edges(adjacent faces pairs): ", self.face_adj.shape[0])
        # `face_adjacency_convex`: (len(self.face_adjacency), ) bool, Return faces which are adjacent and locally convex.
        self.face_adj_convex = self.mesh.face_adjacency_convex
        # `face_adjacency_edges`: (len(self.face_adjacency),2) int, Returns the edges that are shared by the adjacent faces.
        self.face_adj_edges = np.array(self.mesh.face_adjacency_edges)
        # `face_adjacency_unshared`: (len(mesh.face_adjacency), 2) int, Return the vertex index of the two vertices not in the shared edge between two adjacent faces
        self.face_adj_unshared = np.array(self.mesh.face_adjacency_unshared)

        # weights
        self.angle_dists = np.zeros((self.num_faces, self.num_faces))
        self.geode_dists = np.zeros((self.num_faces, self.num_faces))
        self.avg_angle_dists: float
        self.avg_geode_dists: float
        
        # dual graph
        self.G = nx.Graph()

    def build_dual_graph(self, eta=0.1, delta=0.5):
        ang_list, geo_list = [], []
        for pair_id, (i, j) in enumerate(self.face_adj):
            ang = self.angle_distance(self.norms[i], self.norms[j], self.face_adj_convex[pair_id], eta=eta)
            geo = self.geode_distance(
                verts_shared=self.verts[self.face_adj_edges[pair_id]],
                verts_unshared=self.verts[self.face_adj_unshared[pair_id]]
            )
            ang_list.append(ang)
            geo_list.append(geo)
        ang_weight = np.array(ang_list)
        geo_weight = np.array(geo_list)
        self.avg_angle_dists = ang_weight.mean()
        ang_weight /= self.avg_angle_dists
        self.avg_geode_dists = geo_weight.mean()
        geo_weight /= self.avg_geode_dists
        # add node to dual graph
        for i in range(len(self.faces)):
            self.G.add_node(i)
        # add edge to dual graph
        for pair_id, (i, j) in enumerate(self.face_adj):
            self.G.add_edge(i, j, weight=delta * ang_weight[pair_id] + (1.0-delta) * geo_weight[pair_id])
            self.angle_dists[i][j] = self.angle_dists[j][i] = ang_weight[pair_id]
            self.geode_dists[i][j] = self.geode_dists[j][i] = geo_weight[pair_id]

    @staticmethod
    def angle_distance(norm_a, norm_b, is_convex, eta=0.1):
        ang_dist = 1 - np.dot(norm_a, norm_b)
        ang_dist = eta * ang_dist if is_convex else ang_dist
        return ang_dist

    @staticmethod
    def geode_distance(verts_shared, verts_unshared):
        """
        :param:verts_shared:   2x3 float
        :param:verts_unshared: 2x3 float
        """
        # switch to origin point
        offset = verts_shared[1]
        verts_shared = verts_shared - offset
        verts_unshared = verts_unshared - offset
        # get the shared edge
        edge_adj = verts_shared[0] - verts_shared[1]
        edge_adj = edge_adj / np.linalg.norm(edge_adj)
        # calculate the normal of faces
        edge_a = verts_unshared[0] - verts_shared[1]
        edge_b = verts_unshared[1] - verts_shared[1]
        norm_a = np.cross(edge_adj, edge_a)
        norm_a = norm_a / np.linalg.norm(norm_a)
        norm_b = -np.cross(edge_adj, edge_b)
        norm_b = norm_b / np.linalg.norm(norm_b)
        # rotation angle
        # convex: What this means is that given faces A and B, the one vertex in B 
        #         that is not shared with A, projected onto the plane of A has a projection 
        #         that is zero or negative.
        is_convex = np.dot(norm_a, edge_b) <= 0 # NOTE: convex is not determined by the dot of `norm_a` and `norm_b` !
        angle = np.arccos(np.dot(norm_a, norm_b))
        angle = angle if is_convex else -angle
        # construct rotation from the axis and angle
        mrp = edge_adj * np.tan(angle / 4) # `edge_adj` is the rotation axis (旋转轴)
        rot = R.from_mrp(mrp)
        # calculate centroids(质心)
        centroids = [np.mean([*verts_shared, v], axis=0) for v in verts_unshared]
        centroids[0] = rot.apply(centroids[0])
        # check if centroids[0], centroids[1] and edge_adj are coplanar
        assert np.absolute(np.linalg.det(np.array([centroids[0], centroids[1], edge_adj]))) < 1e-6, f"not coplanar: {np.linalg.det(np.array([centroids[0], centroids[1], edge_adj]))}"
        # get geode_distance
        geode_dist = np.linalg.norm(centroids[0] - centroids[1])
        return geode_dist

    def show_mesh(self):
        self.mesh.show()

    def show_dual_graph(self):
        nx.draw(self.G)
        plt.show()

    @property
    def dual_graph(self):
        return self.G