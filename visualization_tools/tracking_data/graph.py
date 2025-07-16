"""
Graph analysis module for Soccer Visualizer.

This module provides graph-based analysis of player positions and movements,
supporting both distance-based and k-nearest neighbor graph construction.
Uses PyTorch Geometric for efficient graph operations.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import knn_graph, radius_graph


class SoccerGraph:
    """
    Create and analyze graphs from soccer player positions.
    
    This class builds graph representations of player positions where nodes
    represent players/ball and edges represent spatial relationships.
    Supports multiple graph construction methods:
   - Ball-centric graphs (ball as center node)
   - KNN graphs (k-nearest neighbors) (both with spatial coordinates and whole features vectors)
   - Radius graphs (distance-based connections) (both with spatial coordinates and whole features vectors)
   - Fully connected graphs 
   - Tactical graphs (position-based connections)
        
    Example:
        >>> graph = SoccerGraph()
        >>> graph.build_distance_based_graph(players, ball, frame=1, distance_threshold=20.0)
        >>> edges = graph.get_edge_list()
    """
    
    def __init__(self):
        """Initialize empty graph."""
        self.data = None  
        self.node_mapping = {}
        
    def build_distance_based_graph(
        self, 
        players, 
        ball, 
        frame=1, 
        distance_threshold=20.0,
    ):
        """
        Build a graph where ball connects to all players within distance threshold.
        Players of same team also connect if within threshold.
        
        Args:
            players: List of Player objects
            ball: Ball object
            frame: Frame number to analyze
            distance_threshold: Maximum distance (in meters) for edge creation
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If ball position not available for the frame
        """
        self.node_mapping = {}
        
        ball_pos = ball.get_position(frame)
        if not ball_pos:
            raise ValueError("Ball position not available for the specified frame")
        
        # Initialize with ball
        nodes = [ball]
        features = [ball_pos]
        self.node_mapping[ball] = 0
        
        # Find players within distance threshold from ball
        nearby_players = []
        for player in players:
            pos = player.get_position(frame)
            if pos:
                dist = np.sqrt((pos[0] - ball_pos[0])**2 + (pos[1] - ball_pos[1])**2)
                if dist <= distance_threshold:
                    nearby_players.append((player, dist, pos))
                    
        # Add nearby players
        for i, (player, dist, pos) in enumerate(nearby_players):
            features.append(pos)
            self.node_mapping[player] = i + 1
            nodes.append(player)
            
        x = torch.tensor(features, dtype=torch.float)
        
        # Create edges: ball to all nearby players
        edges = []
        for i in range(1, len(nodes)):
            edges.append([0, i])
            edges.append([i, 0])
            
        # Connect same team players within threshold
        for i in range(1, len(nodes)):
            player_i = nodes[i]
            for j in range(i+1, len(nodes)):
                player_j = nodes[j]
                if hasattr(player_i, 'team_id') and hasattr(player_j, 'team_id'):
                    if player_i.team_id == player_j.team_id:
                        pos_i = player_i.get_position(frame)
                        pos_j = player_j.get_position(frame)
                        if pos_i and pos_j:
                            dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                            if dist <= distance_threshold:
                                edges.append([i, j])
                                edges.append([j, i])
                        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        self.data = Data(x=x, edge_index=edge_index)
        
        return self
    
    def build_ball_knn_graph(
        self, 
        players, 
        ball, 
        frame=1, 
        k=8
    ):
        """
        Build graph with k-nearest neighbors from ball.
        
        Creates a graph where the ball is connected to its k nearest players
        and players of same team connect to each other.
        
        Args:
            players: List of Player objects
            ball: Ball object
            frame: Frame number to analyze
            k: Number of nearest neighbors to connect
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If ball position not available for the frame
        """
        self.node_mapping = {}
        
        ball_pos = ball.get_position(frame)
        if not ball_pos:
            raise ValueError("Ball position not available for the specified frame")
        
        # Calculate distances from all players to ball
        player_distances = []
        for player in players:
            pos = player.get_position(frame)
            if pos:
                dist = np.sqrt((pos[0] - ball_pos[0])**2 + (pos[1] - ball_pos[1])**2)
                player_distances.append((player, dist))
        
        # Sort by distance and take k nearest
        player_distances.sort(key=lambda x: x[1])
        k_nearest = player_distances[:min(k, len(player_distances))]
        
        # Build node list with ball first
        nodes = [ball]
        features = [ball_pos]
        self.node_mapping[ball] = 0
        
        # Add k nearest players
        for i, (player, _) in enumerate(k_nearest):
            pos = player.get_position(frame)
            features.append(pos)
            self.node_mapping[player] = i + 1
            nodes.append(player)
        
        x = torch.tensor(features, dtype=torch.float)
        
        # Create edges: ball to each player
        edges = []
        for i in range(1, len(nodes)):
            edges.append([0, i])
            edges.append([i, 0])
            
        # Connect players of same team
        for i in range(1, len(nodes)):
            player_i = nodes[i]
            for j in range(i+1, len(nodes)):
                player_j = nodes[j]
                if hasattr(player_i, 'team_id') and hasattr(player_j, 'team_id'):
                    if player_i.team_id == player_j.team_id:
                        edges.append([i, j])
                        edges.append([j, i])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        self.data = Data(x=x, edge_index=edge_index)
        
        return self
    
    def build_none_graph(self, features):
        """
        Build a graph with no edges.
        
        Args:
            features: Features for the nodes
            
        Returns:
            Self for method chaining
        """
        if isinstance(features, np.ndarray):
            x = torch.tensor(features, dtype=torch.float)
        else:
            x = features
        
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        self.data = Data(x=x, edge_index=edge_index)
        
        return self
    
    def build_fc_graph(self, features):
        """
        Build fully-connected graph where every node connects to every other node.
        
        Args:
            features: Features for the nodes
            
        Returns:
            Self for method chaining
        """
        if isinstance(features, np.ndarray):
            x = torch.tensor(features, dtype=torch.float)
        else:
            x = features
        
        num_nodes = x.shape[0]
        edge_index = torch.cartesian_prod(
            torch.arange(num_nodes), torch.arange(num_nodes)
        ).t()
        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        
        self.data = Data(x=x, edge_index=edge_index)
        return self

    def build_tactical_graph(self, features):
        """
        Build tactical graph based on player positions and formations.
        Connects players based on tactical relationships (GK-DEF, DEF-MID, MID-ATT).
        
        Args:
            features: Features for the nodes
            
        Returns:
            Self for method chaining
            
        This method is created to see how tactical graph works.
        This uses feature vector to identify player positions.
        """
        if isinstance(features, np.ndarray):
            x = torch.tensor(features, dtype=torch.float)
        else:
            x = features
            
        all_positions = [
            "TW", "LV", "RV", "IVL", "IVR", "IVZ", 
            "DML", "DMR", "DMZ",
            "LM", "RM", "MZ", "HL", "HR",
            "OLM", "ORM", "ZO", 
            "LA", "RA", "STL", "STR", "STZ", "HST"
        ]
            
            
        # Create position to idx mapping
        position_to_idx = {pos: i for i, pos in enumerate(all_positions)}
        idx_to_position = {v: k for k, v in position_to_idx.items()}
        
        
        # Position groups for tactical connections
        position_groups = {
            "GK": ["TW"],
            "DEF": ["LV", "RV", "IVL", "IVR", "IVZ"],
            "MID": ["ZO", "HR", "DML", "DMR", "DMZ", "LM", "RM", "MZ", "OLM", "ORM", "HL"],
            "ATT": ["LA", "RA", "STL", "STR", "STZ", "HST"]
        }
        
        pos_to_group = {pos: group for group, pos_list in position_groups.items() for pos in pos_list}
        
        connections = {
            "GK": ["DEF"],
            "DEF": ["GK", "DEF", "MID"],
            "MID": ["DEF", "MID", "ATT"],
            "ATT": ["MID", "ATT"]
        }
        
        # Extract positions and teams from features
        positions = []
        teams = []
        
        for i in range(x.shape[0]):
            player_features = x[i]
            is_ball = player_features[2] == 1.0
            is_home = player_features[3] == 1.0
            is_away = player_features[4] == 1.0
            
            if is_ball:
                positions.append("BALL")
                teams.append("BALL")
            elif is_home or is_away:
                team = "HOME" if is_home else "AWAY"
                teams.append(team)
                
                position = "UNKNOWN"
                position_vector = player_features[8:8+len(all_positions)]
                if torch.sum(position_vector) > 0:
                    pos_idx = torch.argmax(position_vector).item()
                    position = idx_to_position.get(pos_idx, "UNKNOWN")
                positions.append(position)
            else:
                positions.append("UNKNOWN")
                teams.append("UNKNOWN")
        
        # Build edges based on tactical connections
        edges = []
        num_players = len(positions)
        for i in range(num_players):
            for j in range(num_players):
                if i == j:
                    continue
                if teams[i] != teams[j] or teams[i] == "BALL":
                    continue
                
                group_i = pos_to_group.get(positions[i])
                group_j = pos_to_group.get(positions[j])
                
                if not group_i or not group_j:
                    continue
                
                if group_j in connections.get(group_i, []):
                    edges.append([i, j])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            raise ValueError("No edges were created. Check if player positions are being identified correctly.")
            
        self.data = Data(x=x, edge_index=edge_index)
        
        return self

    def build_knn_pyg(self, features, k=8):
        """
        Build KNN graph using PyTorch Geometric's knn_graph function.
        Each node connects to its k nearest neighbors.
        
        Args:
            features: Features for the nodes
            k: Number of nearest neighbors to connect
            
        Returns:
            Self for method chaining
            
        This method is created to see how PyG knn graph works.
        This uses whole features vectors instead of just spatial coordinates(player xy-position).
        """
        if isinstance(features, np.ndarray):
            x = torch.tensor(features, dtype=torch.float)
        else:
            x = features
        
        edge_index = knn_graph(x, k=k, batch=None, loop=False)
        self.data = Data(x=x, edge_index=edge_index)
        
        return self

    def build_radius_pyg(self, features, radius=20.0):
        """
        Build radius graph using PyTorch Geometric's radius_graph function.
        Nodes within specified radius are connected.
        
        Args:
            features: Features for the nodes
            radius: Radius for connection
            
        Returns:
            Self for method chaining
            
        This method is created to see how PyG radius graph works.
        This uses whole features vectors instead of just spatial coordinates(player xy-position).
        """
        if isinstance(features, np.ndarray):
            x = torch.tensor(features, dtype=torch.float)
        else:
            x = features
        
        edge_index = radius_graph(x, r=radius, batch=None, loop=False)
        self.data = Data(x=x, edge_index=edge_index)
        
        return self

    def get_node_index(self, obj):
        """Get the node index for a given object."""
        return self.node_mapping.get(obj)

    def get_edge_list(self):
        """Get list of edges as (source_object, target_object) tuples."""
        if not self.data:
            return []
            
        edges = []
        nodes = list(self.node_mapping.keys())
        
        for i, j in self.data.edge_index.t().tolist():
            if i < len(nodes) and j < len(nodes):
                edges.append((nodes[i], nodes[j]))
        
        return edges

    def to_networkx(self):
        """Convert to NetworkX graph for analysis."""
        if not self.data:
            raise ValueError("No graph data available. Call build_graph first.")
        return to_networkx(self.data, to_undirected=False)
