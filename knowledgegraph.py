class KnowledgeGraph:
    def __init__(self):
        # Store nodes as a dictionary where key is node ID and value is node attributes
        self.nodes = {}
        # Store edges as a dictionary of dictionaries for source -> target -> relationship
        self.edges = {}

    def clear(self):
        self.nodes = {}
        self.edges = {}        
    
    def add_node(self, node_id, attributes=None):
        """Add a node to the graph with optional attributes."""
        self.nodes[node_id] = attributes or {}
        if node_id not in self.edges:
            self.edges[node_id] = {}
            
    def add_edge(self, source, target, relationship):
        """Add a directed edge between two nodes with a specific relationship."""
        # Ensure both nodes exist
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
            
        # Add the relationship
        if source not in self.edges:
            self.edges[source] = {}
        if target not in self.edges[source]:
            self.edges[source][target] = set()
        self.edges[source][target].add(relationship)
    
    def dump(self):
        return {
            "nodes": self.nodes,
            "edges": {
                source: {
                    target: list(rel)
                    for target, rel in targets.items()
                } 
                for source, targets in self.edges.items()
            }
        }