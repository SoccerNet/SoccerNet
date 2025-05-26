"""
Graph analysis module for Soccer Visualizer.

This module provides graph-based analysis of player positions and movements,
supporting both distance-based and k-nearest neighbor graph construction.
Uses PyTorch Geometric for efficient graph operations.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx


class SoccerGraph:
    """
    Create and analyze graphs from soccer player positions.
    
    This class builds graph representations of player positions where nodes
    represent players/ball and edges represent spatial relationships.
    Supports two graph types:
    - Distance-based: Connect objects within a threshold distance
    - K-nearest neighbors: Connect each object to its k nearest neighbors
    
    Attributes:
        data (Optional[Data]): PyTorch Geometric Data object
        node_mapping (Dict[Any, int]): Maps objects to node indices
        
    Example:
        >>> graph = SoccerGraph()
        >>> graph.build_distance_based_graph(players, ball, frame=1, distance_threshold=20.0)
        >>> edges = graph.get_edge_list()
    """
    
    def __init__(self) -> None:
        """Initialize empty graph."""
        self.data: Optional[Data] = None  
        self.node_mapping: Dict[Any, int] = {}
        self._node_features: Dict[str, torch.Tensor] = {}
        
    def build_distance_based_graph(
        self, 
        players: List[Any], 
        ball: Any, 
        frame: int = 1, 
        distance_threshold: float = 20.0,
        connect_same_team: bool = True,
        connect_ball_to_players: bool = True,
        min_player_distance: float = 0.0
    ) -> 'SoccerGraph':
        """
        Build graph with distance-based connectivity.
        
        Creates edges between objects that are within a specified distance threshold.
        
        Args:
            players: List of Player objects
            ball: Ball object
            frame: Frame number to analyze
            distance_threshold: Maximum distance (in meters) for edge creation
            connect_same_team: Whether to connect players of the same team
            connect_ball_to_players: Whether to connect ball to nearby players
            min_player_distance: Minimum distance between players for edge creation
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If ball position not available for the frame
        """
        self.node_mapping = {}
        
        # Get ball position
        ball_pos = ball.get_position(frame)
        if not ball_pos:
            raise ValueError(f"Ball position not available for frame {frame}")
        
        # Initialize with ball as first node
        nodes = [ball]
        features = [list(ball_pos)]
        self.node_mapping[ball] = 0
        
        # Collect players within distance threshold of ball
        nearby_players = []
        for player in players:
            pos = player.get_position(frame)
            if pos:
                dist_to_ball = self._euclidean_distance(pos, ball_pos)
                if dist_to_ball <= distance_threshold:
                    nearby_players.append((player, pos))
        
        # Add players to graph
        for i, (player, pos) in enumerate(nearby_players):
            features.append(list(pos))
            self.node_mapping[player] = i + 1
            nodes.append(player)
        
        # Convert features to tensor
        x = torch.tensor(features, dtype=torch.float)
        
        # Build edges
        edges = []
        
        # Connect ball to players
        if connect_ball_to_players:
            for i in range(1, len(nodes)):
                edges.append([0, i])  # Ball to player
                edges.append([i, 0])  # Player to ball (bidirectional)
        
        # Connect players
        for i in range(1, len(nodes)):
            player_i = nodes[i]
            pos_i = features[i]
            
            for j in range(i + 1, len(nodes)):
                player_j = nodes[j]
                pos_j = features[j]
                
                # Calculate distance between players
                dist = self._euclidean_distance(pos_i, pos_j)
                
                # Check if we should connect these players
                should_connect = False
                
                if hasattr(player_i, 'team_id') and hasattr(player_j, 'team_id'):
                    same_team = player_i.team_id == player_j.team_id
                    
                    if connect_same_team and same_team:
                        # Connect teammates within threshold
                        should_connect = (dist <= distance_threshold and 
                                        dist >= min_player_distance)
                    elif not same_team:
                        # Optionally connect opponents
                        should_connect = (dist <= distance_threshold * 0.5 and  # Stricter threshold
                                        dist >= min_player_distance)
                else:
                    # Default: connect if within threshold
                    should_connect = (dist <= distance_threshold and 
                                    dist >= min_player_distance)
                
                if should_connect:
                    edges.append([i, j])
                    edges.append([j, i])  # Bidirectional
        
        # Create edge tensor
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Store additional features
        self._store_node_features(nodes, frame)
        
        self.data = Data(x=x, edge_index=edge_index)
        
        return self
    
    def build_ball_knn_graph(
        self, 
        players: List[Any], 
        ball: Any, 
        frame: int = 1, 
        k: int = 8,
        connect_players: bool = True,
        team_aware: bool = True
    ) -> 'SoccerGraph':
        """
        Build graph with k-nearest neighbors from ball.
        
        Creates a graph where the ball is connected to its k nearest players,
        and optionally connects players based on team relationships.
        
        Args:
            players: List of Player objects
            ball: Ball object
            frame: Frame number to analyze
            k: Number of nearest neighbors to connect
            connect_players: Whether to connect players of the same team
            team_aware: Whether to consider team relationships
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If ball position not available for the frame
        """
        self.node_mapping = {}
        
        # Get ball position
        ball_pos = ball.get_position(frame)
        if not ball_pos:
            raise ValueError(f"Ball position not available for frame {frame}")
        
        # Calculate distances from all players to ball
        player_distances: List[Tuple[Any, float, Tuple[float, float]]] = []
        for player in players:
            pos = player.get_position(frame)
            if pos:
                dist = self._euclidean_distance(pos, ball_pos)
                player_distances.append((player, dist, pos))
        
        # Sort by distance and take k nearest
        player_distances.sort(key=lambda x: x[1])
        k_nearest = player_distances[:min(k, len(player_distances))]
        
        # Build nodes list with ball first
        nodes = [ball]
        features = [list(ball_pos)]
        self.node_mapping[ball] = 0
        
        # Add k nearest players
        for i, (player, dist, pos) in enumerate(k_nearest):
            features.append(list(pos))
            self.node_mapping[player] = i + 1
            nodes.append(player)
        
        # Convert features to tensor
        x = torch.tensor(features, dtype=torch.float)
        
        # Build edges
        edges = []
        
        # Connect ball to each of the k nearest players
        for i in range(1, len(nodes)):
            edges.append([0, i])  # Ball to player
            edges.append([i, 0])  # Player to ball
        
        # Optionally connect players
        if connect_players:
            for i in range(1, len(nodes)):
                player_i = nodes[i]
                
                for j in range(i + 1, len(nodes)):
                    player_j = nodes[j]
                    
                    # Connect based on team relationships
                    if team_aware and hasattr(player_i, 'team_id') and hasattr(player_j, 'team_id'):
                        if player_i.team_id == player_j.team_id:
                            edges.append([i, j])
                            edges.append([j, i])
                    elif not team_aware:
                        # Connect all players
                        edges.append([i, j])
                        edges.append([j, i])
        
        # Create edge tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Store additional features
        self._store_node_features(nodes, frame)
        
        self.data = Data(x=x, edge_index=edge_index)
        
        return self
    
    def get_node_index(self, obj: Any) -> Optional[int]:
        """
        Get node index for a given object.
        
        Args:
            obj: Player or Ball object
            
        Returns:
            Node index or None if object not in graph
        """
        return self.node_mapping.get(obj)
    
    def get_node_positions(self) -> Optional[np.ndarray]:
        """
        Get positions of all nodes.
        
        Returns:
            Array of shape (num_nodes, 2) with x,y positions
        """
        if self.data is None:
            return None
        return self.data.x.numpy()
    
    def get_edge_list(self) -> List[Tuple[Any, Any]]:
        """
        Get list of edges as object pairs.
        
        Returns:
            List of (source_object, target_object) tuples
        """
        if not self.data:
            return []
            
        edges: List[Tuple[Any, Any]] = []
        nodes = list(self.node_mapping.keys())
        
        # Convert edge indices to object pairs
        for i, j in self.data.edge_index.t().tolist():
            if i < len(nodes) and j < len(nodes):
                edges.append((nodes[i], nodes[j]))
        
        return edges
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Convert to NetworkX graph for additional analysis.
        
        Returns:
            NetworkX directed graph
            
        Raises:
            ValueError: If no graph data available
        """
        if not self.data:
            raise ValueError("No graph data available. Build a graph first.")
        return to_networkx(self.data, to_undirected=False)
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Calculate basic graph statistics.
        
        Returns:
            Dictionary with statistics like number of nodes, edges, density, etc.
        """
        if not self.data:
            return {}
        
        num_nodes = self.data.x.size(0)
        num_edges = self.data.edge_index.size(1) // 2  # Divide by 2 for undirected
        
        # Convert to NetworkX for additional metrics
        nx_graph = self.to_networkx()
        
        stats = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": nx.density(nx_graph),
            "num_connected_components": nx.number_weakly_connected_components(nx_graph),
            "average_degree": sum(dict(nx_graph.degree()).values()) / num_nodes if num_nodes > 0 else 0
        }
        
        # Add clustering coefficient if graph is undirected
        try:
            stats["average_clustering"] = nx.average_clustering(nx_graph.to_undirected())
        except:
            stats["average_clustering"] = None
            
        return stats
    
    def _euclidean_distance(self, pos1: Union[List[float], Tuple[float, float]], 
                           pos2: Union[List[float], Tuple[float, float]]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _store_node_features(self, nodes: List[Any], frame: int) -> None:
        """Store additional node features for analysis."""
        # Store team information
        teams = []
        for node in nodes:
            if hasattr(node, 'team_id'):
                teams.append(node.team_id)
            else:
                teams.append('ball')
        
        self._node_features['teams'] = teams
        self._node_features['frame'] = frame
    
    def get_subgraph(self, team: str) -> Optional['SoccerGraph']:
        """
        Extract subgraph for a specific team.
        
        Args:
            team: Team identifier (e.g., "home", "away")
            
        Returns:
            New SoccerGraph containing only specified team's players
        """
        if not self.data or 'teams' not in self._node_features:
            return None
        
        # Find indices of nodes belonging to the team
        team_indices = [i for i, t in enumerate(self._node_features['teams']) 
                       if t == team]
        
        if not team_indices:
            return None
        
        # Create new graph with team nodes
        new_graph = SoccerGraph()
        
        # Create mapping from old to new indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(team_indices)}
        
        # Extract node features
        new_x = self.data.x[team_indices]
        
        # Extract and remap edges
        new_edges = []
        for i, j in self.data.edge_index.t().tolist():
            if i in old_to_new and j in old_to_new:
                new_edges.append([old_to_new[i], old_to_new[j]])
        
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
        else:
            new_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        new_graph.data = Data(x=new_x, edge_index=new_edge_index)
        
        # Update node mapping
        nodes = list(self.node_mapping.keys())
        for old_idx, new_idx in old_to_new.items():
            if old_idx < len(nodes):
                new_graph.node_mapping[nodes[old_idx]] = new_idx
        
        return new_graph
    
    def __repr__(self) -> str:
        """String representation of graph."""
        if self.data:
            num_nodes = self.data.x.size(0)
            num_edges = self.data.edge_index.size(1)
            return f"SoccerGraph(nodes={num_nodes}, edges={num_edges})"
        return "SoccerGraph(empty)"