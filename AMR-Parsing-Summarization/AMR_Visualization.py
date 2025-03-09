# %% [markdown]
# # AMR Parsing and Visualization
# 
# This notebook demonstrates Abstract Meaning Representation (AMR) parsing and visualization using LangGraph for graph-based visualization.
# 
# ## Setup and Dependencies

# %%
# Install required packages
# pip install -q networkx matplotlib penman spacy langgraph langchain

# %%
import networkx as nx
import matplotlib.pyplot as plt
import re
from typing import Dict, List, Any

# Try importing non-standard libraries, with fallbacks
try:
    import penman
except ImportError:
    print("Penman not installed. Install with: pip install penman")

try:
    import spacy
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import sys
        print("Downloading spaCy model...")
        print(f"Running: {sys.executable} -m spacy download en_core_web_sm")
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    print("spaCy not installed. Install with: pip install spacy")
    
try:
    from langgraph.graph import StateGraph
    from langchain.schema import HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("LangGraph/LangChain not installed. Install with: pip install langgraph langchain")
    LANGGRAPH_AVAILABLE = False

# %% [markdown]
# ## Sample AMR Representations
# 
# We'll work with some sample AMR notations to demonstrate parsing and visualization.

# %%
# Sample AMR notations
amr_samples = {
    "simple": """
(w / want-01
   :ARG0 (b / boy)
   :ARG1 (g / go-01
            :ARG0 b))
""",
    "medium": """
(r / recommend-01
      :ARG0 (i / i)
      :ARG1 (a / advocate-01
            :ARG0 (p / person
                  :ARG0-of (h / have-rel-role-91
                        :ARG1 (p2 / person
                              :ARG1-of (s / smoke-01))
                        :ARG2 (d / doctor)))
            :ARG1 (c / cease-01
                  :ARG1 s)))
""",
    "complex": """
(m / multi-sentence
      :snt1 (r / read-01
            :ARG0 (g / girl)
            :ARG1 (b / book
                  :topic (v / vaccine
                        :mod (c / covid-19))))
      :snt2 (l / learn-01
            :ARG0 g
            :ARG1 (i / important-01
                  :ARG1 v)
            :ARG2 (h / health
                  :poss (p / public))))
"""
}

# Corresponding natural language sentences
amr_sentences = {
    "simple": "The boy wants to go.",
    "medium": "I recommend that doctors advocate for smokers to cease smoking.",
    "complex": "The girl read a book about COVID-19 vaccines. She learned about their importance to public health."
}

# %% [markdown]
# ## Basic AMR Parsing with Penman
# 
# First, we'll use the Penman library to parse AMR notations into graph structures.

# %%
def parse_amr(amr_string):
    """Parse AMR notation into a Penman graph"""
    try:
        return penman.decode(amr_string)
    except Exception as e:
        print(f"Error parsing AMR: {e}")
        return None

# Parse the sample AMRs
parsed_amrs = {key: parse_amr(amr) for key, amr in amr_samples.items()}

# Display the parsed simple AMR
print("Parsed Simple AMR:")
print(parsed_amrs["simple"])

# %% [markdown]
# ## Convert AMR to NetworkX Graph
# 
# To visualize the AMR, we need to convert the Penman graph to a NetworkX graph.

# %%
def amr_to_networkx(amr_graph):
    """Convert Penman AMR graph to NetworkX for visualization"""
    G = nx.DiGraph()
    
    # Get triples from the AMR graph
    triples = []
    for triple in amr_graph.triples:
        source, relation, target = triple
        # Make sure the target is a string
        if isinstance(target, str) and target.startswith('(') and target.endswith(')'):
            target = target[1:-1]  # Remove parentheses
        triples.append((source, relation, target))
    
    # Add nodes to the graph
    for triple in triples:
        source, relation, target = triple
        if source not in G.nodes:
            # Get the instance name if available
            instance = next((t[2] for t in triples if t[0] == source and t[1] == ':instance'), source)
            G.add_node(source, label=instance)
        
        if isinstance(target, str) and target not in G.nodes and relation != ':instance':
            G.add_node(target, label=target)
    
    # Add edges to the graph
    for source, relation, target in triples:
        if relation != ':instance' and isinstance(target, str):
            G.add_edge(source, target, label=relation)
    
    return G

# %% [markdown]
# ## Basic Graph Visualization

# %%
def visualize_amr(nx_graph, title="AMR Graph"):
    """Visualize the AMR as a NetworkX graph"""
    plt.figure(figsize=(12, 8))
    
    # Use a spring layout for the graph
    pos = nx.spring_layout(nx_graph, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(nx_graph, pos, node_size=2000, node_color="lightblue", alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(nx_graph, pos, width=1.5, alpha=0.7, edge_color="gray")
    
    # Add node labels
    node_labels = {node: nx_graph.nodes[node].get('label', node) for node in nx_graph.nodes}
    nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=12)
    
    # Add edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').replace(':', '').replace('\'', '')}.png")
    plt.show()

# Convert and visualize the simple AMR
simple_nx = amr_to_networkx(parsed_amrs["simple"])
visualize_amr(simple_nx, title=f"Simple AMR: '{amr_sentences['simple']}'")

# %% [markdown]
# ## LangGraph-style Visualization
# 
# Now let's create a more advanced visualization inspired by LangGraph.

# %%
def create_langgraph_state(amr_graph, sentence):
    """Create a state structure similar to LangGraph's format"""
    # Create nodes dictionary from AMR triples
    nodes = {}
    
    # Get instance triples
    instance_triples = [(s, t) for s, r, t in amr_graph.triples if r == ':instance']
    
    for node_id, instance in instance_triples:
        # Find all edges from this node
        outgoing_edges = [(r, t) for s, r, t in amr_graph.triples 
                         if s == node_id and r != ':instance']
        
        nodes[node_id] = {
            "id": node_id,
            "instance": instance,
            "edges": outgoing_edges
        }
    
    # Create state dictionary
    state = {
        "sentence": sentence,
        "amr_nodes": nodes,
        "amr_root": amr_graph.metadata.get('snt', amr_graph.top),
        "parse_metadata": dict(amr_graph.metadata)
    }
    
    return state

# Create LangGraph-style states
amr_states = {}
for key in amr_samples.keys():
    if parsed_amrs[key]:
        amr_states[key] = create_langgraph_state(parsed_amrs[key], amr_sentences[key])

# %%
if LANGGRAPH_AVAILABLE:
    def create_langgraph_visualization(state):
        """Create a LangGraph-style visualization of the AMR"""
        # Create a state graph for visualization
        graph = StateGraph(state_type=Dict)
        
        # Add nodes for each AMR node
        for node_id, node_data in state["amr_nodes"].items():
            graph.add_node(node_id, lambda x, node_id=node_id: {"result": state["amr_nodes"][node_id]["instance"]})
        
        # Add edges between nodes
        for node_id, node_data in state["amr_nodes"].items():
            for relation, target in node_data["edges"]:
                if isinstance(target, str) and target in state["amr_nodes"]:
                    graph.add_edge(node_id, target)
        
        # Set entry points
        graph.set_entry_point(state["amr_root"])
        
        # Compile the graph
        compiled_graph = graph.compile()
        
        # Display the graph (would normally be interactive in LangGraph)
        print("LangGraph visualization created")
        
        return compiled_graph
    
    # Create the LangGraph visualization
    try:
        simple_graph = create_langgraph_visualization(amr_states["simple"])
        print("Simple graph visualization created with LangGraph")
    except Exception as e:
        print(f"Error creating LangGraph visualization: {e}")
else:
    print("LangGraph visualization skipped (library not available)")

# %% [markdown]
# ## Enhanced AMR Visualization
# 
# Creating more visually appealing visualizations of the AMR graphs.

# %%
def enhanced_visualize_amr(nx_graph, title="AMR Graph"):
    """Create an enhanced visualization of the AMR graph"""
    plt.figure(figsize=(12, 8))
    
    # Use a spring layout for the graph with more space
    pos = nx.spring_layout(nx_graph, seed=42, k=2.0)
    
    # Draw nodes with improved styling
    nx.draw_networkx_nodes(
        nx_graph, 
        pos, 
        node_size=2500, 
        node_color="#AED6F1",  # Light blue
        edgecolors="#2E86C1",  # Darker blue border
        linewidths=2,
        alpha=0.9
    )
    
    # Draw edges with arrows and improved styling
    nx.draw_networkx_edges(
        nx_graph, 
        pos, 
        width=1.5, 
        alpha=0.8, 
        edge_color="#95A5A6",  # Gray
        connectionstyle="arc3,rad=0.1",  # Curved edges
        arrowsize=15
    )
    
    # Add node labels with better font
    node_labels = {node: nx_graph.nodes[node].get('label', node) for node in nx_graph.nodes}
    nx.draw_networkx_labels(
        nx_graph, 
        pos, 
        labels=node_labels, 
        font_size=12,
        font_weight="bold",
        font_family="sans-serif"
    )
    
    # Add edge labels with better positioning
    edge_labels = {(u, v): d['label'] for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(
        nx_graph, 
        pos, 
        edge_labels=edge_labels, 
        font_size=10,
        font_family="sans-serif",
        bbox=dict(alpha=0.6, pad=0.3, facecolor="white"),
        rotate=False
    )
    
    plt.title(title, fontsize=16, fontweight="bold", fontfamily="sans-serif")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"enhanced_{title.replace(' ', '_').replace(':', '').replace('\'', '')}.png", dpi=300, bbox_inches="tight")
    plt.show()

# Visualize all AMRs with enhanced visualization
for key, amr in parsed_amrs.items():
    if amr:
        nx_graph = amr_to_networkx(amr)
        enhanced_visualize_amr(nx_graph, title=f"{key.capitalize()} AMR: '{amr_sentences[key]}'")

# %% [markdown]
# ## AMR and Summarization
# 
# AMR can be used for text summarization by analyzing the semantic structure and extracting the most important concepts.

# %%
def identify_key_concepts(amr_graph):
    """Identify key concepts from an AMR graph for summarization"""
    # Convert to NetworkX for analysis
    G = amr_to_networkx(amr_graph)
    
    # Calculate node centrality to find important concepts
    centrality = nx.betweenness_centrality(G)
    
    # Sort nodes by centrality
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top nodes
    top_nodes = [node for node, _ in sorted_nodes[:3]]
    
    # Get the labels of the top nodes
    top_concepts = [G.nodes[node].get('label', node) for node in top_nodes]
    
    return top_concepts

# Identify key concepts for each AMR
for key, amr_graph in parsed_amrs.items():
    if amr_graph:
        concepts = identify_key_concepts(amr_graph)
        print(f"Key concepts in {key} AMR: {', '.join(concepts)}")

# %% [markdown]
# ## Comparison Visualization
# 
# Let's compare the different AMR structures side by side.

# %%
def compare_amr_complexity():
    """Compare the complexity of different AMR graphs"""
    # Collect statistics
    stats = []
    for key, amr in parsed_amrs.items():
        if amr:
            G = amr_to_networkx(amr)
            stats.append({
                'name': key,
                'nodes': len(G.nodes),
                'edges': len(G.edges),
                'density': nx.density(G)
            })
    
    # Create a grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define bar positions
    names = [stat['name'] for stat in stats]
    x = range(len(names))
    width = 0.25
    
    # Plot bars
    ax.bar([i - width for i in x], [stat['nodes'] for stat in stats], width, label='Nodes', color='#3498DB')
    ax.bar(x, [stat['edges'] for stat in stats], width, label='Edges', color='#E74C3C')
    ax.bar([i + width for i in x], [stat['density'] * 10 for stat in stats], width, label='Density (×10)', color='#2ECC71')
    
    # Add labels and legend
    ax.set_xlabel('AMR Examples')
    ax.set_ylabel('Count/Value')
    ax.set_title('Comparison of AMR Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('amr_complexity_comparison.png', dpi=300, bbox_inches="tight")
    plt.show()

# Compare AMR complexity
compare_amr_complexity()

# %% [markdown]
# ## Conclusion
# 
# This notebook has demonstrated various ways to visualize AMR graphs, including basic NetworkX visualization, enhanced visualization, and complexity comparison. AMR parsing provides a semantic representation of text that can be useful for various NLP tasks such as summarization.

# %%
# Final message
print("AMR visualization complete. All visualizations have been saved as PNG files.") 