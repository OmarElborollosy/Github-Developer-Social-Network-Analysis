# dashboard/utils.py
# Shared helper functions for graph loading, LCC extraction, and analysis.
# Imported by both notebooks/analysis.ipynb and dashboard/app.py.
# DO NOT duplicate this logic in the notebook or dashboard directly.

import random
import numpy as np
import pandas as pd
import networkx as nx

import pickle
import os

# ──────────────────────────────────────────────
# Graph loading and LCC extraction
# ──────────────────────────────────────────────

def load_graph(edges_path, targets_path, pickle_path='outputs/graph_lcc.pkl'):
    """
    Load MUSAE GitHub edge list and node targets, build undirected graph.
    Attaches 'developer_type' node attribute (0=Web, 1=ML).
    Returns full graph G.
    """
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            G = pickle.load(f)
        return G
        
    random.seed(42)
    np.random.seed(42)
    edges   = pd.read_csv(edges_path)
    targets = pd.read_csv(targets_path)
    G = nx.from_pandas_edgelist(edges, source='id_1', target='id_2')
    labels = dict(zip(targets['id'], targets['ml_target']))
    nx.set_node_attributes(G, labels, 'developer_type')
    return G


def get_lcc(G):
    """
    Extract and return Largest Connected Component as a new graph (copy).
    """
    random.seed(42)
    np.random.seed(42)
    lcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc_nodes).copy()


def sample_lcc(G_lcc, n=5000, seed=42):
    """
    Return a subgraph of n randomly sampled nodes from the LCC.
    Used for avg path length approximation.
    """
    random.seed(seed)
    np.random.seed(seed)
    sampled = random.sample(list(G_lcc.nodes()), n)
    return G_lcc.subgraph(sampled).copy()


# ──────────────────────────────────────────────
# Centrality and PageRank helpers
# ──────────────────────────────────────────────

def compute_centrality(G_lcc, k_betweenness=500, seed=42):
    """
    Compute degree, betweenness (k=500, seed=42), closeness, and eigenvector
    centrality on the LCC.  Returns unified DataFrame with columns:
    node, degree, betweenness, closeness, eigenvector, developer_type.
    """
    random.seed(seed)
    np.random.seed(seed)
    deg  = nx.degree_centrality(G_lcc)
    bet  = nx.betweenness_centrality(G_lcc, k=k_betweenness, seed=seed)
    clo  = nx.closeness_centrality(G_lcc)
    eig  = nx.eigenvector_centrality(G_lcc, max_iter=1000)
    labels = nx.get_node_attributes(G_lcc, 'developer_type')
    df = pd.DataFrame({
        'node':           list(G_lcc.nodes()),
        'degree':         [deg[n] for n in G_lcc.nodes()],
        'betweenness':    [bet[n] for n in G_lcc.nodes()],
        'closeness':      [clo[n] for n in G_lcc.nodes()],
        'eigenvector':    [eig[n] for n in G_lcc.nodes()],
        'developer_type': [labels.get(n, -1) for n in G_lcc.nodes()],
    })
    return df


def compute_pagerank(G_lcc, alpha=0.85):
    """Return standard PageRank dict."""
    random.seed(42)
    np.random.seed(42)
    return nx.pagerank(G_lcc, alpha=alpha, max_iter=200)


def compute_ppr(G_lcc, seed_type, labels, alpha=0.85):
    """
    Compute Personalized PageRank.
    seed_type: 'ML' or 'Web'.
    labels: dict mapping node_id -> developer_type (0=Web, 1=ML).
    Personalization = uniform over seed_type nodes, 0 elsewhere.
    """
    random.seed(42)
    np.random.seed(42)
    target_val = 1 if seed_type == 'ML' else 0
    seed_nodes = [n for n in G_lcc.nodes() if labels.get(n, -1) == target_val]
    w = 1.0 / len(seed_nodes) if seed_nodes else 0.0
    personalization = {n: (w if labels.get(n, -1) == target_val else 0.0)
                       for n in G_lcc.nodes()}
    return nx.pagerank(G_lcc, alpha=alpha, max_iter=200,
                       personalization=personalization)


def run_louvain(G_lcc, random_state=42):
    """
    Run Louvain community detection.
    Returns (partition_dict, modularity_score).
    """
    random.seed(random_state)
    np.random.seed(random_state)
    import community as community_louvain
    partition  = community_louvain.best_partition(G_lcc, random_state=random_state)
    modularity = community_louvain.modularity(partition, G_lcc)
    return partition, modularity


def run_link_prediction(G_lcc, test_frac=0.1, seed=42):
    """
    Full link prediction pipeline (no data leakage).
    Splits edges, scores with CN / Jaccard / Adamic-Adar / Resource Allocation
    on G_train only.  Returns dict with keys:
      y_true, test_pairs, scores (nested dict per method), labels.
    """
    random.seed(seed)
    np.random.seed(seed)
    labels  = nx.get_node_attributes(G_lcc, 'developer_type')
    all_edges = list(G_lcc.edges())
    random.shuffle(all_edges)
    cut = int(test_frac * len(all_edges))
    test_edges  = all_edges[:cut]
    train_edges = all_edges[cut:]
    G_train = nx.Graph()
    G_train.add_nodes_from(G_lcc.nodes(data=True))
    G_train.add_edges_from(train_edges)
    existing = set(G_lcc.edges()) | {(v, u) for u, v in G_lcc.edges()}
    non_edges = []
    nodes_list = list(G_lcc.nodes())
    while len(non_edges) < len(test_edges):
        u, v = random.sample(nodes_list, 2)
        if (u, v) not in existing and (v, u) not in existing:
            non_edges.append((u, v))
            existing.add((u, v))
    test_pairs = test_edges + non_edges
    y_true = [1] * len(test_edges) + [0] * len(non_edges)
    scores = {
        'Common Neighbors':    [sum(1 for _ in nx.common_neighbors(G_train, u, v))
                                 if G_train.has_node(u) and G_train.has_node(v) else 0
                                 for u, v in test_pairs],
        'Jaccard Coefficient': [s for _, _, s in nx.jaccard_coefficient(G_train, test_pairs)],
        'Adamic-Adar':         [s for _, _, s in nx.adamic_adar_index(G_train, test_pairs)],
        'Resource Allocation': [s for _, _, s in nx.resource_allocation_index(G_train, test_pairs)],
    }
    return {'y_true': y_true, 'test_pairs': test_pairs,
            'scores': scores, 'labels': labels}


# ──────────────────────────────────────────────
# Information Diffusion Models  (Part D2)
# ──────────────────────────────────────────────

def run_ic_cascade(G, seed_nodes, prob=0.1, seed=42, max_rounds=20):
    """
    Independent Cascade model.

    Each newly activated node independently tries to activate each inactive
    neighbour with probability `prob` in the next time step.
    Stops after max_rounds rounds regardless of whether propagation is still
    active.  Real information cascades on GitHub rarely exceed 20 hops.

    Models viral adoption of tools/frameworks — each developer independently
    decides to follow based on a single interaction.

    Parameters
    ----------
    G          : NetworkX undirected graph
    seed_nodes : list of node IDs to activate initially
    prob       : activation probability per edge (default 0.1)
    seed       : random seed (default 42)
    max_rounds : hard stop after this many rounds (default 20)

    Returns
    -------
    set of all activated node IDs (including seeds)
    """
    random.seed(seed)
    np.random.seed(seed)

    activated       = set(seed_nodes)
    newly_activated = set(seed_nodes)

    for _ in range(max_rounds):
        if not newly_activated:
            break
        next_wave = set()
        for node in newly_activated:
            for nb in G.neighbors(node):
                if nb not in activated:
                    if random.random() < prob:
                        next_wave.add(nb)
        activated       |= next_wave
        newly_activated  = next_wave

    return activated


def run_lt_cascade(G, seed_nodes, seed=42):
    """
    Linear Threshold model.

    Each node is assigned a random threshold drawn from Uniform(0, 1) at the
    start of the simulation.  A node activates when the fraction of its
    neighbours that are already activated exceeds its threshold.

    Models consensus-driven adoption — a developer follows back only when
    enough of their peers have already connected.

    Uses a frontier-based iteration (only candidates whose neighbourhood
    changed are re-evaluated) for performance on large graphs.

    Parameters
    ----------
    G          : NetworkX undirected graph
    seed_nodes : list of node IDs to activate initially
    seed       : random seed (default 42)

    Returns
    -------
    set of all activated node IDs (including seeds)
    """
    random.seed(seed)
    np.random.seed(seed)

    nodes     = list(G.nodes())
    node_idx  = {n: i for i, n in enumerate(nodes)}
    # Draw thresholds fresh at the start of every simulation call
    thresholds = np.random.uniform(0, 1, size=G.number_of_nodes())

    activated = set(seed_nodes)
    frontier  = set(seed_nodes)   # only neighbours of frontier are candidates

    while frontier:
        # Collect uninactivated neighbours of the frontier
        candidates = set()
        for node in frontier:
            for nb in G.neighbors(node):
                if nb not in activated:
                    candidates.add(nb)

        next_wave = set()
        for node in candidates:
            deg = G.degree(node)
            if deg == 0:
                continue
            active_nb = sum(1 for nb in G.neighbors(node) if nb in activated)
            if active_nb / deg > thresholds[node_idx[node]]:
                next_wave.add(node)

        activated |= next_wave
        frontier   = next_wave   # only newly activated nodes generate new candidates

    return activated


import streamlit as st

@st.cache_data
def load_temporal_edges():
    """Load the GH Archive temporal co-starring edge list."""
    path = 'outputs/results/temporal_edges.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_temporal_network_stats():
    """Load daily network evolution statistics."""
    path = 'outputs/results/temporal_network_stats.csv'
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data
def load_temporal_link_prediction():
    """Load temporal link prediction results and merge with MUSAE results."""
    temporal_path = 'outputs/results/temporal_link_prediction.csv'
    musae_path    = 'outputs/results/link_prediction_results.csv'
    if not os.path.exists(temporal_path):
        return None, None
    temporal_df = pd.read_csv(temporal_path)
    if os.path.exists(musae_path):
        musae_df = pd.read_csv(musae_path)[['Method','AUC-ROC']].rename(
            columns={'AUC-ROC': 'MUSAE_AUC'})
        comparison_df = temporal_df.merge(musae_df, on='Method', how='left')
        comparison_df['Delta'] = (
            comparison_df['Temporal_AUC'] - comparison_df['MUSAE_AUC']
        ).round(4)
        return temporal_df, comparison_df
    return temporal_df, None
