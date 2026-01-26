"""
Dependency graph construction.
Builds graph representation of feature dependencies for block formation.
"""

import networkx as nx
from typing import List, Tuple, Dict, Set
import logging

logger = logging.getLogger(__name__)


def build_dependency_graph(
    pairs: List[Tuple[str, str, float]],
    threshold: float = 0.0
) -> nx.Graph:
    """
    Build undirected graph from dependency pairs.
    
    Args:
        pairs: List of (col1, col2, score) tuples
        threshold: Minimum score to include edge
    
    Returns:
        NetworkX graph with features as nodes and dependencies as edges
    """
    G = nx.Graph()
    
    for col1, col2, score in pairs:
        if score >= threshold:
            G.add_edge(col1, col2, weight=score)
    
    logger.info(f"Built dependency graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def get_connected_components(graph: nx.Graph) -> List[Set[str]]:
    """
    Get connected components of dependency graph.
    Each component represents a potential privacy block.
    
    Args:
        graph: Dependency graph
    
    Returns:
        List of sets, each containing features in a component
    """
    components = list(nx.connected_components(graph))
    logger.info(f"Found {len(components)} connected components")
    return components


def get_communities(
    graph: nx.Graph,
    method: str = 'greedy_modularity'
) -> List[Set[str]]:
    """
    Detect communities in dependency graph.
    Alternative to connected components for block formation.
    
    Args:
        graph: Dependency graph
        method: Community detection method ('greedy_modularity' or 'louvain')
    
    Returns:
        List of sets, each containing features in a community
    """
    try:
        if method == 'greedy_modularity':
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(graph, weight='weight')
        elif method == 'louvain':
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(graph, weight='weight')
                # Convert partition to list of sets
                communities_dict = {}
                for node, comm_id in partition.items():
                    if comm_id not in communities_dict:
                        communities_dict[comm_id] = set()
                    communities_dict[comm_id].add(node)
                communities = list(communities_dict.values())
            except ImportError:
                logger.warning("python-louvain not installed. Falling back to greedy_modularity")
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.greedy_modularity_communities(graph, weight='weight')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Detected {len(communities)} communities using {method}")
        return list(communities)
    except Exception as e:
        logger.warning(f"Community detection failed: {e}. Using connected components instead.")
        return get_connected_components(graph)
