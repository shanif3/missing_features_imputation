import pandas as pd
import torch
import numpy as np
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_scatter import scatter_add
from torch_geometric.data import Data
import scipy

def find_mask(dataset: pd.DataFrame) -> torch.Tensor:
    """

    :param dataset: the dataset.
    :return: the mask.
    note: the dataset itself is always a Pandas.DataFrame. the mask is a torch.Tensor.

    """
    # find nan values.
    z = dataset.notna().astype(int)  # 1 for numeric values, 0 for nan values.
    mask = torch.from_numpy(z.values).bool()  # True for numeric values, False for nan values.
    return mask

def fill_data(dataset):
    """
    this method takes the name of the dataset, then gets the missing values mask, and then fill the data accordingly.
    :param data_name: the dataset name (tabular or graph).
    :return: the data with filled values.
    """

    edges = dataset.edge_index
    x = pd.DataFrame(dataset.x.numpy())
    # TODO: Watch out! I randomize put nan values because my dataset is already filled. otherwise, comment the next line.
    x = x.mask(np.random.rand(*x.shape) < 0.1)

    mask = find_mask(x)
    x = torch.from_numpy(x.values.astype(np.float32))
    x = FeaturePropagation(num_iterations=40).propagate(x, edges, mask)
    return pd.concat([pd.DataFrame(x.cpu().detach().numpy()), pd.DataFrame(dataset.y.numpy())], axis=1)





class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def propagate(self, x: Tensor, edge_index, mask: Tensor) -> Tensor:
        # out is initialized to 0 for missing values. However, its initialization does not matter for the final
        # value at convergence
        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]

        n_nodes = x.shape[0]
        adj = self.get_propagation_matrix(edge_index, n_nodes)
        for _ in range(self.num_iterations):
            # Diffuse current features
            out = torch.sparse.mm(adj, out)
            # Reset original known features
            out[mask] = x[mask]

        return out

    def get_propagation_matrix(self, edge_index, n_nodes):
        # Initialize all edge weights to ones if the graph is unweighted.

        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)

        return adj

def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """

    :param edge_index: a tensor of shape [2, num_edges] containing the indices of the edges in the graph.
    :param n_nodes: the number of nodes in the graph (rows in dataset).
    :return: the symmetrically normalized adjacency matrix of the graph, noted as D^-1/2 * A * D^-1/2.
    :return: the propagation matrix, and the weights of that matrix.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD

def main():
    #load your graph and feature table
    dataset = Planetoid(root='/tmp/Cora', name='Cora')[0]  # Load Cora dataset
    filled_data = fill_data(dataset)
    print(filled_data.head())

if __name__ == "__main__":
    main()