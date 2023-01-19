import argparse

from MeshDecomposition import Mesh, HierarchicalKwayDecomposer, KWayDecomposer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hierarchical Mesh Decomposition")
    parser.add_argument("--file", type=str, required=True, help=".ply file to be processed.")
    parser.add_argument("--eta", type=float, default=0.09, help="parameter `eta` for Hierarchical Mesh Decomposition.")
    parser.add_argument("--delta", type=float, default=0.5, help="parameter `delta` for Hierarchical Mesh Decomposition.")
    parser.add_argument("--eps", type=float, default=0.1, help="parameter `eps` for Hierarchical Mesh Decomposition.")
    parser.add_argument("--maxiter", type=int, default=100, help="max iteration for K-means in Hierarchical Mesh Decomposition.")
    parser.add_argument("--display", action='store_true', default=False, help="whether to display the intermediate results.")
    parser.add_argument("--outdir", type=str, default="./output", help="output directory to save results.")
    parser.add_argument("--threshold", type=float, default=8.0, help="stop condition: Distance between representatives < threshold.")
    parser.add_argument("--hi", default=False, action='store_true', help="use hierarchical kway decomposition.")
    args = parser.parse_args()

    ### initialization
    print(f"---------------------------------------------------------")
    print(f"Processing Mesh Decomposition on file {args.file}")
    mesh = Mesh(filename=args.file)
    mesh.build_dual_graph(eta=args.eta, delta=args.delta)

    if args.hi:
        decomposer = HierarchicalKwayDecomposer(mesh=mesh, eps=args.eps, maxiter=args.maxiter, threshold=args.threshold)
        decomposer.decompose(outdir=args.outdir, filename=args.file)
    else:
        decomposer = KWayDecomposer(eps=args.eps, max_iter=args.maxiter)
        decomposer.decompose(mesh, args.outdir, args.file, display=args.display)
