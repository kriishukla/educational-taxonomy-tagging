class TaxonomyNode:
    def __init__(self, name, level_index=None, parent=None, taxonomy=None):
        self.name = name
        self.level_index = level_index
        self.parent = parent
        self.children = {}
        self.taxonomy = taxonomy

    def add_child(self, child_name, level_index):
        if child_name not in self.children:
            self.children[child_name] = TaxonomyNode(
                child_name,
                level_index=level_index,
                parent=self,
                taxonomy=self.taxonomy
            )
        return self.children[child_name]

    def get_path(self):
        """Return full taxonomy path as a list [Level1, Level2, ...]."""
        node = self
        path = []
        while node and node.name != "ROOT":
            path.append(node.name)
            node = node.parent
        return list(reversed(path))

    def to_json(self):
        """
        Convert taxonomy node to semantic JSON representation.

        Example:
        {
            "Subject": "Physics",
            "Chapter": "Magnetism",
            "Topic": "Bar Magnet"
        }
        """

        if self.taxonomy is None:
            raise ValueError(
                "TaxonomyNode is not linked to a Taxonomy object."
            )

        path = self.get_path()
        levels = self.taxonomy.levels

        taxonomy_json = {}

        for i, value in enumerate(path):
            if i < len(levels):
                taxonomy_json[levels[i]] = value
            else:
                taxonomy_json[f"Level_{i}"] = value

        return taxonomy_json

    def __repr__(self):
        path = " >> ".join(self.get_path())
        return f"TaxonomyNode({path})"


class Taxonomy:
    def __init__(self, file_path:str, name:str = "<Taxonomy>"):
        self.root = TaxonomyNode("ROOT", level_index=-1, taxonomy=self)
        self.name = name
        self.levels = []
        self.leaf_nodes: list[TaxonomyNode] = []
        self.__load(file_path)
        self.leaf_nodes = self.__load_leaf_nodes()

    def __load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # First line contains level names
        self.levels = [lvl.strip() for lvl in lines[0].split(">>")]
        max_depth = len(self.levels)

        # Build the tree
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(">>")]

            if len(parts) > max_depth:
                raise ValueError(
                    f"Line exceeds defined taxonomy depth ({max_depth}): {line}"
                )

            current = self.root
            for idx, part in enumerate(parts):
                current = current.add_child(part, level_index=idx)

    def __load_leaf_nodes(self):
        """Return a list of all leaf TaxonomyNode objects."""
        leaves = []

        def dfs(node):
            if not node.children and node.name != "ROOT":
                leaves.append(node)
            for c in node.children.values():
                dfs(c)

        dfs(self.root)
        return leaves
    
    def to_leaf(self, path_str: str) -> TaxonomyNode:
        """
        Convert a taxonomy string into a TaxonomyNode leaf.

        Expected format:
            "Level1 >> Level2 >> Level3"

        Returns:
            TaxonomyNode

        Raises:
            ValueError if path is invalid or not a leaf node.
        """

        if not path_str or not isinstance(path_str, str):
            raise ValueError("Invalid taxonomy path string.")

        parts = [p.strip() for p in path_str.split(">>") if p.strip()]

        if not parts:
            raise ValueError("Empty taxonomy path.")

        current = self.root

        for idx, part in enumerate(parts):
            if part not in current.children:
                raise ValueError(
                    f"Invalid taxonomy path at level {idx}: '{part}' not found."
                )
            current = current.children[part]

        # Ensure it is a leaf node
        if current.children:
            raise ValueError(
                f"Path does not point to a leaf node: {path_str}"
            )

        return current

    def print_tree(self, node=None, indent=0):
        if node is None:
            node = self.root

        for child in node.children.values():
            print("  " * indent + f"- {child.name}")
            self.print_tree(child, indent + 1)


if __name__ == "__main__":
    taxonomy = Taxonomy("PCv3.txt")
    taxonomy.print_tree()
    node = taxonomy.to_leaf(
    "Chemistry >> The d- and f-Block Elements >> Lanthanoids"
)

    print(node)
    print(node.to_json())

    leaves = taxonomy.leaf_nodes
    print("\nLeaf Nodes:")
    print(leaves[0].get_path())
