import networkx as nx
from torch.utils.data import Dataset


class TreeSet(Dataset):
    def __init__(self, G, names, descriptions, batch_size=300):
        super(TreeSet, self).__init__()
        self._tree = G
        self._undigraph = G.to_undirected()
        self.names = names
        self.descriptions = descriptions
        self.leaves = []
        self.paths = {}
        self.min_depth = 0
        self.max_depth = 0
        self.mean_depth = 0

        self._process()

    def _get_leaves(self):
        self.leaves = list([node for node in self._tree.nodes.keys() if self._tree.out_degree(node) == 0])

    def _get_all_paths(self):
        for l in self.leaves:
            self.paths[l] = nx.shortest_path(self._tree, 0, l)

    def _stats(self):
        depths = map(lambda k, v: len(v), self.paths.items())
        self.min_depth = min(depths)
        self.max_depth = max(depths)
        self.mean_depth = sum(depths) / len(depths)

    def _process(self):
        self._get_leaves()
        self._get_all_paths()
        self._stats()

    def distance(self, a, b):
        return nx.shortest_path_length(self._undigraph, a, b)

    def path_between(self, a, b):
        return nx.shortest_path(self._undigraph, a, b)

    def __getitem__(self, idx):
        pass
