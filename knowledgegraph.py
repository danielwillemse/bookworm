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
        self.edges[source][target] = relationship
        
    def get_relationships(self, source):
        """Get all relationships from a source node."""
        return self.edges.get(source, {})
    
    def query(self, source, relationship):
        """Find all nodes that have a specific relationship with the source node."""
        results = []
        for target, rel in self.edges.get(source, {}).items():
            if rel == relationship:
                results.append(target)
        return results
    
    def find_path(self, start, end, max_depth=3):
        """Find a path between two nodes using breadth-first search."""
        if start not in self.nodes or end not in self.nodes:
            return None
            
        queue = [(start, [start])]
        visited = set([start])
        
        while queue:
            (vertex, path) = queue.pop(0)
            
            # Don't explore paths longer than max_depth
            if len(path) > max_depth:
                continue
                
            # Check all adjacent nodes
            for next_vertex in self.edges.get(vertex, {}):
                if next_vertex == end:
                    return path + [next_vertex]
                if next_vertex not in visited:
                    visited.add(next_vertex)
                    queue.append((next_vertex, path + [next_vertex]))
        
        return None

    def dump(self):
        return {
            "people": self.nodes,
            "relations": {
                source: {
                    target: rel 
                    for target, rel in targets.items()
                } 
                for source, targets in self.edges.items()
            }
        }