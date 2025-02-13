import networkx as nx
import matplotlib.pyplot as plt
import io
from PIL import Image

def to_image(nodes, edges, node_labels=None, edge_labels=None, title="Characters and their relations"):
    G = nx.Graph()
    
    # Add nodes and edges
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # Create a figure with a reasonable size
    plt.figure(figsize=(20, 8))
    
    # Set the layout for the graph
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          width=1, alpha=0.5)
    
    # Add node labels if provided
    if node_labels is None:
        node_labels = {node: str(node) for node in nodes}
    nx.draw_networkx_labels(G, pos, node_labels)
    
    # Add edge labels if provided
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    # Add title and remove axes
    plt.title(title)
    plt.axis('off')
    
    # Show the plot
    # plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    img = Image.open(buf)
    
    return img