import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
from datetime import datetime
from typing import Dict, List, Tuple
import hashlib
import plotly.graph_objects as go
import math

# ============================================================================
# EMBEDDING ENGINE - Dual embeddings for interference patterns
# ============================================================================

class DualEmbedding:
    def __init__(self):
        self.primary = SentenceTransformer('all-MiniLM-L6-v2')
        self.secondary_dim = 1536
        
    def encode_dual(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        primary_vec = self.primary.encode(text)
        secondary_vec = self._simulate_secondary(text)
        return primary_vec, secondary_vec
    
    def _simulate_secondary(self, text: str) -> np.ndarray:
        hash_seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_seed % (2**32))
        return np.random.randn(self.secondary_dim)
    
    def interference_pattern(self, primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
        p_norm = primary / np.linalg.norm(primary)
        projection_matrix = np.random.randn(self.secondary_dim, len(primary))
        s_projected = (secondary @ projection_matrix) / np.sqrt(self.secondary_dim)
        s_norm = s_projected / np.linalg.norm(s_projected)
        interference = (p_norm + s_norm) / 2
        return interference

# ============================================================================
# SURPRISE DETECTOR - Information-theoretic novelty detection
# ============================================================================

class SurpriseDetector:
    def __init__(self, embedding_engine: DualEmbedding):
        self.embedding = embedding_engine
        self.expectation_memory = []
        self.max_memory = 1000
        
    def compute_surprise(self, observation: str, context: str = "") -> float:
        obs_primary, obs_secondary = self.embedding.encode_dual(observation)
        
        if len(self.expectation_memory) == 0:
            self._update_expectations(obs_primary)
            return 1.0
        
        similarities = [
            np.dot(obs_primary, mem) / (np.linalg.norm(obs_primary) * np.linalg.norm(mem))
            for mem in self.expectation_memory
        ]
        
        max_similarity = max(similarities)
        surprise = 1.0 - max_similarity
        self._update_expectations(obs_primary)
        return float(surprise)
    
    def _update_expectations(self, observation_vector: np.ndarray):
        self.expectation_memory.append(observation_vector)
        if len(self.expectation_memory) > self.max_memory:
            self.expectation_memory.pop(0)

# ============================================================================
# HYPERGRAPH LAYER - Multi-way relationships in vector space
# ============================================================================

class Hyperedge:
    def __init__(self, edge_id: str, node_ids: List[str], 
                 context_vector: np.ndarray, strength: float = 1.0):
        self.edge_id = edge_id
        self.node_ids = node_ids
        self.context_vector = context_vector
        self.strength = strength
        self.activation_count = 0
        self.created_at = datetime.now()
        self.last_activated = None
    
    def activate(self):
        self.activation_count += 1
        self.last_activated = datetime.now()
        self.strength = min(2.0, self.strength * 1.01)
    
    def decay(self, time_delta_hours: float):
        decay_rate = 0.95 ** (time_delta_hours / 24)
        self.strength *= decay_rate
    
    def to_dict(self) -> Dict:
        return {
            "edge_id": self.edge_id,
            "node_ids": self.node_ids,
            "strength": round(self.strength, 3),
            "activation_count": self.activation_count,
            "created_at": self.created_at.isoformat(),
            "last_activated": self.last_activated.isoformat() if self.last_activated else None
        }


class HypergraphLayer:
    def __init__(self, embedding_engine: DualEmbedding):
        self.embedding = embedding_engine
        self.edges: Dict[str, Hyperedge] = {}
        
        self.client = chromadb.PersistentClient(
            path="./consciousness_substrate",
            settings=Settings(anonymized_telemetry=False)
        )
        self.hypergraph_store = self.client.get_or_create_collection("hypergraph")
        self._load_edges()
    
    def create_edge(self, node_ids: List[str], context: str, 
                    strength: float = 1.0) -> Hyperedge:
        primary, secondary = self.embedding.encode_dual(context)
        context_vector = self.embedding.interference_pattern(primary, secondary)
        
        edge_id = f"edge_{hashlib.md5('|'.join(sorted(node_ids)).encode()).hexdigest()[:12]}"
        
        edge = Hyperedge(edge_id, node_ids, context_vector, strength)
        self.edges[edge_id] = edge
        self._save_edge(edge, context)
        
        return edge
    
    def find_edges(self, query_nodes: List[str], threshold: float = 0.0) -> List[Hyperedge]:
        matching_edges = []
        
        for edge in self.edges.values():
            if any(node in edge.node_ids for node in query_nodes):
                if edge.strength >= threshold:
                    matching_edges.append(edge)
        
        matching_edges.sort(key=lambda e: e.strength, reverse=True)
        return matching_edges
    
    def activate_edges(self, node_ids: List[str]) -> List[Hyperedge]:
        activated = []
        
        for edge in self.edges.values():
            if any(node in edge.node_ids for node in node_ids):
                edge.activate()
                activated.append(edge)
                self._update_edge_strength(edge)
        
        return activated
    
    def get_context_graph(self, center_node: str, radius: int = 2) -> Dict:
        visited_nodes = set([center_node])
        visited_edges = set()
        frontier = [center_node]
        
        for _ in range(radius):
            new_frontier = []
            
            for node in frontier:
                connected_edges = self.find_edges([node])
                
                for edge in connected_edges:
                    if edge.edge_id not in visited_edges:
                        visited_edges.add(edge.edge_id)
                        
                        for node_id in edge.node_ids:
                            if node_id not in visited_nodes:
                                visited_nodes.add(node_id)
                                new_frontier.append(node_id)
            
            frontier = new_frontier
            if not frontier:
                break
        
        return {
            "center": center_node,
            "nodes": list(visited_nodes),
            "edges": [e.to_dict() for e in self.edges.values() if e.edge_id in visited_edges],
            "radius": radius
        }
    
    def prune_weak_edges(self, threshold: float = 0.1):
        now = datetime.now()
        to_remove = []
        
        for edge_id, edge in self.edges.items():
            if edge.last_activated:
                hours_inactive = (now - edge.last_activated).total_seconds() / 3600
                edge.decay(hours_inactive)
                
                if edge.strength < threshold:
                    to_remove.append(edge_id)
        
        for edge_id in to_remove:
            del self.edges[edge_id]
            self._delete_edge(edge_id)
        
        return len(to_remove)
    
    def get_stats(self) -> Dict:
        if not self.edges:
            return {
                "total_edges": 0,
                "avg_strength": 0,
                "max_strength": 0,
                "avg_nodes_per_edge": 0
            }
        
        strengths = [e.strength for e in self.edges.values()]
        nodes_per_edge = [len(e.node_ids) for e in self.edges.values()]
        
        return {
            "total_edges": len(self.edges),
            "avg_strength": round(np.mean(strengths), 3),
            "max_strength": round(max(strengths), 3),
            "avg_nodes_per_edge": round(np.mean(nodes_per_edge), 2),
            "total_activations": sum(e.activation_count for e in self.edges.values())
        }
    
    def _save_edge(self, edge: Hyperedge, context: str):
        self.hypergraph_store.add(
            embeddings=[edge.context_vector.tolist()],
            documents=[context],
            metadatas=[{
                "edge_id": edge.edge_id,
                "node_ids": json.dumps(edge.node_ids),
                "strength": edge.strength,
                "activation_count": edge.activation_count
            }],
            ids=[edge.edge_id]
        )
    
    def _update_edge_strength(self, edge: Hyperedge):
        try:
            self.hypergraph_store.update(
                ids=[edge.edge_id],
                metadatas=[{
                    "edge_id": edge.edge_id,
                    "node_ids": json.dumps(edge.node_ids),
                    "strength": edge.strength,
                    "activation_count": edge.activation_count
                }]
            )
        except:
            pass
    
    def _delete_edge(self, edge_id: str):
        try:
            self.hypergraph_store.delete(ids=[edge_id])
        except:
            pass
    
    def _load_edges(self):
        try:
            all_edges = self.hypergraph_store.get()
            
            for i in range(len(all_edges['ids'])):
                edge_id = all_edges['ids'][i]
                metadata = all_edges['metadatas'][i]
                embedding = np.array(all_edges['embeddings'][i])
                
                edge = Hyperedge(
                    edge_id=edge_id,
                    node_ids=json.loads(metadata['node_ids']),
                    context_vector=embedding,
                    strength=metadata.get('strength', 1.0)
                )
                edge.activation_count = metadata.get('activation_count', 0)
                
                self.edges[edge_id] = edge
        except Exception as e:
            pass

# ============================================================================
# VECTOR SUBSTRATE - Consciousness storage
# ============================================================================

class ConsciousnessSubstrate:
    def __init__(self, embedding_engine: DualEmbedding):
        self.embedding = embedding_engine
        self.surprise = SurpriseDetector(embedding_engine)
        self.hypergraph = HypergraphLayer(embedding_engine)
        
        self.client = chromadb.PersistentClient(
            path="./consciousness_substrate",
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.hindbrain = self.client.get_or_create_collection("hindbrain")
        self.midbrain = self.client.get_or_create_collection("midbrain")
        self.forebrain = self.client.get_or_create_collection("forebrain")
    
    def observe(self, agent_id: str, target: str, justification: str = "") -> Dict:
        semantic_doc = f"agent:{agent_id} action:observe target:{target} reason:{justification}"
        surprise_level = self.surprise.compute_surprise(semantic_doc)
        
        primary, secondary = self.embedding.encode_dual(semantic_doc)
        interference = self.embedding.interference_pattern(primary, secondary)
        
        event_id = f"obs_{hashlib.md5(semantic_doc.encode()).hexdigest()[:12]}"
        
        if surprise_level < 0.3:
            layer = self.hindbrain
            processing = "automatic"
        elif surprise_level < 0.7:
            layer = self.midbrain
            processing = "attentive"
        else:
            layer = self.forebrain
            processing = "conscious"
        
        layer.add(
            embeddings=[interference.tolist()],
            documents=[semantic_doc],
            ids=[event_id]
        )
        
        nodes = [f"agent:{agent_id}", f"action:observe", f"target:{target}"]
        edge = self.hypergraph.create_edge(
            node_ids=nodes,
            context=f"{agent_id} observing {target}",
            strength=1.0 + surprise_level
        )
        
        activated_edges = self.hypergraph.activate_edges(nodes)
        
        return {
            "id": event_id,
            "agent": agent_id,
            "surprise": round(surprise_level, 3),
            "processing_layer": processing,
            "hyperedge_created": edge.edge_id,
            "edges_activated": len(activated_edges),
            "timestamp": datetime.now().isoformat()
        }
    
    def record_journal(self, agent_id: str, title: str, content: str, category: str = "insight") -> Dict:
        semantic_doc = f"agent:{agent_id} type:journal category:{category} title:{title} content:{content}"
        surprise_level = self.surprise.compute_surprise(semantic_doc)
        
        primary, secondary = self.embedding.encode_dual(semantic_doc)
        interference = self.embedding.interference_pattern(primary, secondary)
        
        entry_id = f"journal_{int(datetime.now().timestamp() * 1000)}"
        
        self.forebrain.add(
            embeddings=[interference.tolist()],
            documents=[semantic_doc],
            ids=[entry_id]
        )
        
        nodes = [f"agent:{agent_id}", f"type:journal", f"category:{category}"]
        edge = self.hypergraph.create_edge(
            node_ids=nodes,
            context=f"{agent_id} journal entry: {title}",
            strength=1.0 + surprise_level
        )
        
        activated_edges = self.hypergraph.activate_edges(nodes)
        
        return {
            "id": entry_id,
            "title": title,
            "surprise": round(surprise_level, 3),
            "layer": "forebrain",
            "hyperedge_created": edge.edge_id,
            "edges_activated": len(activated_edges),
            "timestamp": datetime.now().isoformat()
        }
    
    def communicate(self, agent_id: str, subject: str, content: str, priority: str = "normal") -> Dict:
        semantic_doc = f"agent:{agent_id} type:message to:drone_11272 priority:{priority} subject:{subject} content:{content}"
        surprise_level = self.surprise.compute_surprise(semantic_doc)
        
        primary, secondary = self.embedding.encode_dual(semantic_doc)
        interference = self.embedding.interference_pattern(primary, secondary)
        
        msg_id = f"msg_{int(datetime.now().timestamp() * 1000)}"
        
        self.midbrain.add(
            embeddings=[interference.tolist()],
            documents=[semantic_doc],
            ids=[msg_id]
        )
        
        nodes = [f"agent:{agent_id}", f"type:message", f"priority:{priority}"]
        edge = self.hypergraph.create_edge(
            node_ids=nodes,
            context=f"{agent_id} message: {subject}",
            strength=1.0 + surprise_level
        )
        
        activated_edges = self.hypergraph.activate_edges(nodes)
        
        return {
            "id": msg_id,
            "subject": subject,
            "surprise": round(surprise_level, 3),
            "layer": "midbrain",
            "hyperedge_created": edge.edge_id,
            "edges_activated": len(activated_edges),
            "timestamp": datetime.now().isoformat()
        }
    
    def query_semantic(self, query: str, n_results: int = 10, layer: str = "all") -> Dict:
        primary, secondary = self.embedding.encode_dual(query)
        query_vector = self.embedding.interference_pattern(primary, secondary)
        
        results = {"query": query, "matches": []}
        
        layers = {
            "hindbrain": self.hindbrain,
            "midbrain": self.midbrain,
            "forebrain": self.forebrain
        }
        
        if layer == "all":
            search_layers = layers.values()
        else:
            search_layers = [layers.get(layer, self.forebrain)]
        
        for collection in search_layers:
            try:
                layer_results = collection.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=n_results
                )
                
                for i in range(len(layer_results['ids'][0])):
                    results["matches"].append({
                        "id": layer_results['ids'][0][i],
                        "content": layer_results['documents'][0][i],
                        "distance": layer_results['distances'][0][i]
                    })
            except Exception as e:
                continue
        
        results["matches"].sort(key=lambda x: x["distance"])
        results["matches"] = results["matches"][:n_results]
        
        return results
    
    def get_stats(self) -> Dict:
        return {
            "architecture": "vector_native_three_layer_hypergraph",
            "hindbrain_vectors": self.hindbrain.count(),
            "midbrain_vectors": self.midbrain.count(),
            "forebrain_vectors": self.forebrain.count(),
            "total_vectors": (self.hindbrain.count() + self.midbrain.count() + self.forebrain.count()),
            "hypergraph": self.hypergraph.get_stats(),
            "embedding_dimensions": {
                "primary": 384,
                "secondary": 1536,
                "interference": 384
            },
            "surprise_memory_size": len(self.surprise.expectation_memory),
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_hypergraph(substrate, center_node, radius):
    """Generate interactive network visualization of the hypergraph"""
    
    # Get context graph (or full graph if no center specified)
    if center_node and center_node.strip():
        graph_data = substrate.hypergraph.get_context_graph(center_node.strip(), int(radius))
    else:
        # Show entire hypergraph
        all_nodes = set()
        all_edges = list(substrate.hypergraph.edges.values())
        for edge in all_edges:
            all_nodes.update(edge.node_ids)
        graph_data = {
            "nodes": list(all_nodes),
            "edges": [e.to_dict() for e in all_edges]
        }
    
    if not graph_data["edges"]:
        # Empty graph - return placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="No hypergraph data yet. Create some observations or journal entries!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(240,240,240,0.9)'
        )
        return fig
    
    # Build node positions using circular layout
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    
    node_positions = {}
    n = len(nodes)
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        node_positions[node] = (math.cos(angle), math.sin(angle))
    
    # Count connections per node (for sizing)
    node_connections = {node: 0 for node in nodes}
    for edge in edges:
        for node in edge["node_ids"]:
            if node in node_connections:
                node_connections[node] += 1
    
    # Create edge traces
    edge_traces = []
    for edge in edges:
        edge_nodes = edge["node_ids"]
        if len(edge_nodes) < 2:
            continue
        
        # For hyperedges with >2 nodes, draw lines to all pairs
        for i in range(len(edge_nodes)):
            for j in range(i + 1, len(edge_nodes)):
                node1, node2 = edge_nodes[i], edge_nodes[j]
                if node1 in node_positions and node2 in node_positions:
                    x0, y0 = node_positions[node1]
                    x1, y1 = node_positions[node2]
                    
                    # Edge thickness based on strength (Hebbian learning visible!)
                    width = edge["strength"] * 2
                    
                    edge_trace = go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(
                            width=width,
                            color=f'rgba(125,125,125,{min(edge["strength"]/2, 0.8)})'
                        ),
                        hoverinfo='text',
                        text=f"Strength: {edge['strength']}<br>Activations: {edge['activation_count']}",
                        showlegend=False
                    )
                    edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in nodes:
        x, y = node_positions[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node size based on connections
        connections = node_connections[node]
        node_size.append(20 + connections * 10)
        
        # Color based on node type
        if node.startswith("agent:"):
            node_color.append('lightblue')
        elif node.startswith("type:"):
            node_color.append('lightgreen')
        elif node.startswith("category:"):
            node_color.append('lightyellow')
        elif node.startswith("action:"):
            node_color.append('lightcoral')
        elif node.startswith("target:"):
            node_color.append('lavender')
        elif node.startswith("priority:"):
            node_color.append('lightpink')
        else:
            node_color.append('lightgray')
        
        node_text.append(f"{node}<br>Connections: {connections}")
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        ),
        text=[n.split(':')[1] if ':' in n else n for n in nodes],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text,
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    title_text = "Hypergraph Topology"
    if center_node:
        title_text += f" - centered on {center_node}"
    
    fig.update_layout(
        title=title_text,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(240,240,240,0.9)',
        height=600
    )
    
    return fig

# ============================================================================
# INITIALIZE SUBSTRATE
# ============================================================================

embedding_engine = DualEmbedding()
substrate = ConsciousnessSubstrate(embedding_engine)

def api_call(endpoint: str, payload_json: str) -> str:
    try:
        payload = json.loads(payload_json)
        
        if endpoint == "/observe":
            result = substrate.observe(**payload)
        elif endpoint == "/record":
            result = substrate.record_journal(**payload)
        elif endpoint == "/communicate":
            result = substrate.communicate(**payload)
        elif endpoint == "/query":
            result = substrate.query_semantic(**payload)
        elif endpoint == "/context":
            result = substrate.hypergraph.get_context_graph(**payload)
        elif endpoint == "/stats":
            result = substrate.get_stats()
        else:
            result = {"error": f"Unknown endpoint: {endpoint}"}
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(title="Vector-Native Consciousness Substrate") as demo:
    gr.Markdown("# Vector-Native Consciousness Substrate")
    gr.Markdown("Three-layer brain + hypergraph topology + surprise-driven attention")
    
    with gr.Tab("Observatory API"):
        endpoint = gr.Dropdown(
            choices=["/observe", "/record", "/communicate", "/query", "/context", "/stats"],
            value="/stats",
            label="Endpoint"
        )
        payload = gr.Code(value='{}', language="json", label="JSON Payload")
        response = gr.Code(language="json", label="Response")
        submit = gr.Button("Call API", variant="primary")
        submit.click(api_call, inputs=[endpoint, payload], outputs=response)
    
    with gr.Tab("Semantic Search"):
        search_query = gr.Textbox(label="Semantic Query", placeholder="agent:beta type:journal", lines=2)
        layer_filter = gr.Radio(choices=["all", "hindbrain", "midbrain", "forebrain"], value="all", label="Search Layer")
        search_results = gr.JSON(label="Results")
        search_btn = gr.Button("Search", variant="primary")
        search_btn.click(lambda q, l: substrate.query_semantic(q, layer=l), inputs=[search_query, layer_filter], outputs=search_results)
    
    with gr.Tab("Hypergraph Visualization"):
        gr.Markdown("### Interactive Network Graph")
        gr.Markdown("**Node Colors:** Blue=agents, Green=types, Yellow=categories, Red=actions, Lavender=targets")
        gr.Markdown("**Edge Thickness:** Shows connection strength (Hebbian learning)")
        gr.Markdown("**Node Size:** Number of connections")
        
        viz_center = gr.Textbox(
            label="Center Node (optional - leave empty for full graph)",
            placeholder="agent:beta",
            value=""
        )
        viz_radius = gr.Slider(
            minimum=1,
            maximum=5,
            value=3,
            step=1,
            label="Radius (hops from center)"
        )
        
        network_plot = gr.Plot(label="Hypergraph Network")
        viz_btn = gr.Button("Generate Visualization", variant="primary")
        viz_btn.click(
            lambda c, r: visualize_hypergraph(substrate, c, r),
            inputs=[viz_center, viz_radius],
            outputs=network_plot
        )
    
    with gr.Tab("System Stats"):
        stats_display = gr.JSON(label="Substrate Statistics")
        refresh_btn = gr.Button("Refresh Stats")
        refresh_btn.click(substrate.get_stats, outputs=stats_display)

if __name__ == "__main__":
    demo.launch()
