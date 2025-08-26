import networkx as nx
from pyvis.network import Network
import os
import sys

# Add the parent directory to the Python path to allow importing modules from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_manager import load_knowledge_graph, DOCUMENT_KG_FILE, OUTPUT_DIR

def visualize_kg(graph):
    """
    Visualizes a networkx knowledge graph using pyvis and saves it as an HTML file.
    Also prints basic information about the graph.
    """
    if graph is None:
        print("No graph provided for visualization.")
        return

    # Print graph information
    print("--- Knowledge Graph Information ---")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    
    # Display some example nodes and edges
    print("\n--- Example Nodes (first 5) ---")
    for i, node in enumerate(graph.nodes(data=True)):
        if i >= 5: break
        print(f"Node: {node[0]}, Attributes: {node[1]}")

    print("\n--- Example Edges (first 5) ---")
    for i, edge in enumerate(graph.edges(data=True)):
        if i >= 5: break
        print(f"Edge: {edge[0]} --({edge[2].get('relation', 'UNKNOWN')})--> {edge[1]}")

    # Create a pyvis network
    net = Network(notebook=True, height="750px", width="100%", cdn_resources='remote')
    
    # Add nodes and edges to pyvis network
    for node, attrs in graph.nodes(data=True):
        title = f"Type: {attrs.get('type','N/A')}<br>Label: {attrs.get('label','N/A')}<br>Page: {attrs.get('page','N/A')}"
        net.add_node(str(node), title=title, label=str(node), size=15, group=attrs.get('type', 'entity'))
    
    for s, t, attrs in graph.edges(data=True):
        net.add_edge(str(s), str(t), title=attrs.get('relation', 'UNKNOWN'), label=attrs.get('relation', 'UNKNOWN'))

    # Save the visualization
    output_html_file = os.path.join(OUTPUT_DIR, "knowledge_graph_visualization.html")
    net.show(output_html_file)
    print(f"\nKnowledge graph visualization saved to: {output_html_file}")

if __name__ == "__main__":
    print("Loading document knowledge graph...")
    document_kg = load_knowledge_graph(DOCUMENT_KG_FILE, graph_type="document")
    visualize_kg(document_kg)
