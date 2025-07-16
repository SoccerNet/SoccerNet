"""
Main visualizer module for Soccer Visualizer.

This module provides the main SoccerVisualizer class that orchestrates
the entire visualization process, from loading data to creating animations.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import yaml
from pathlib import Path
from datetime import datetime

from .pitch import Field
from .objects import Player, Ball
from .graph import SoccerGraph
from .data_reader import DataReaderFactory

class SoccerNetVisualizer:
    """
    Main class for soccer event visualization with graph analysis.
    
    This class handles the complete visualization pipeline:
    - Loading tracking data from pickle files
    - Processing events and creating player/ball objects
    - Visualizing frames with optional graph overlay
    - Creating animations
    - Batch processing multiple events
    
    Attributes:
        config (Config): Configuration handler
        data_dir (str): Directory containing tracking data
        output_dir (str): Directory for saving visualizations
        splits (Dict): Loaded data splits
        current_event (Optional[Dict]): Currently selected event
        ball (Optional[Ball]): Ball object
        home_team (List[Player]): Home team players
        away_team (List[Player]): Away team players
        pitch (Pitch): Pitch rendering object
        
    Example:
        >>> # Using configuration file
        >>> viz = SoccerVisualizer(config_path="config.yaml")
        >>> viz.run()
        
        >>> # Programmatic usage
        >>> viz = SoccerVisualizer()
        >>> viz.load_data("visualization_tools/tracking_data/data/train_windows.pkl").select_event("Pass").create_animation()
    """
    
    def __init__(self, config_path="visualization_tools/config.yaml"):
       with open(config_path, 'r') as f:
           self.config = yaml.safe_load(f)
           
       self.output_dir = Path(self.config['data']['output_dir'])
       self.output_dir.mkdir(exist_ok=True)
       
       # Initialize components
       self.field = Field(**self.config['visualization']['field'])
       self.data_reader = None
       self.data = {}
       self.current_event = None
       
       # Object tracking 
       self.ball = None
       self.players = {}
            
    def load_data(self, file_path, **kwargs):
        """Load data using appropriate reader."""
        self.data_reader = DataReaderFactory.create_reader(file_path, **kwargs)
        self.data = self.data_reader.load_data(file_path, **kwargs)
        return self
            
    def select_event(self, event_type, split='default', index=0):
        """
        Select an event for visualization.
        
        Args:
            event_type: Type of event (e.g., "Pass", "Shot")
            split: Data split name
            index: Index of event within the type
        """
        if split not in self.data:
            available_splits = list(self.data.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available_splits}")
        
        events = [e for e in self.data[split] if e.label == event_type]
        
        if not events:
            available_types = list(set(e.label for e in self.data[split]))
            raise ValueError(f"No '{event_type}' events found. Available: {available_types}")
        
        if index >= len(events):
            raise ValueError(f"Index {index} out of range. {len(events)} events available.")
        
        self.current_event = events[index]
        self._num_frames = len(self.current_event.frames)
        self._process_event()
        
        print(f"Selected {event_type} event at {self.current_event.game_time}")
        return self
    
    def _process_event(self):
        """Process current event to create visualization objects."""
        if not self.current_event:
            return
            
        # Get configuration
        player_config = self.config['visualization']['players']
        ball_config = self.config['visualization']['ball']
        
        # Reset objects
        self.ball = Ball(
            color=ball_config['color'],
            radius=ball_config['radius']
        )
        self.players = {}
        
        # Process each frame
        for frame_num, frame_data in self.current_event.frames.items():
            for obj in frame_data.objects:
                if obj['type'] == 'ball':
                    self.ball.add_position(frame_num, obj['x'], obj['y'])
                elif obj['type'] in ['home', 'away']:
                    # Create player if not exists
                    if obj['id'] not in self.players:
                        color = player_config['home_color'] if obj['type'] == 'home' else player_config['away_color']
                        jersey = obj.get('jersey', len([p for p in self.players.values() if p.team_id == obj['type']]) + 1)
                        
                        self.players[obj['id']] = Player(
                            player_id=obj['id'],
                            team_id=obj['type'],
                            jersey_number=jersey,
                            color=color,
                            radius=player_config['radius'],
                            show_jersey=player_config['show_jersey_numbers'],
                            jersey_font_size=player_config['jersey_font_size']
                        )
                    
                    self.players[obj['id']].add_position(frame_num, obj['x'], obj['y'])
    
    def visualize_frame(self, frame, graph_type=None, graph_params=None, show=True):
        """
        Visualize a single frame with optional graph overlay.
        
        Args:
            frame: Frame number to visualize
            graph_type: Override default graph type
            graph_params: Override default graph parameters
            show: Whether to display the plot
        """
        if not self.current_event:
            raise ValueError("No event selected")
            
        if frame < 0 or frame >= self._num_frames:
            raise ValueError(f"Frame {frame} out of range (0-{self._num_frames-1})")
        
        # Use defaults from config if not specified
        if graph_type is None and self.config['graph']['enabled']:
            graph_type = self.config['graph']['type']
        if graph_params is None and graph_type:
            graph_params = self.config['graph'].get(graph_type, {})
            
        # Create figure
        fig, ax = self.field.draw()
        
        # Draw graph if enabled
        if graph_type and graph_type != 'none':
            self._draw_graph(ax, frame, graph_type, graph_params)
        
        # Draw objects
        self._draw_objects(ax, frame)
        
        # Add title and legend
        self._add_title(ax, frame, graph_type)
        if graph_type and graph_type != 'none' and self.config['graph']['show_legend']:
            self._add_legend(ax)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig, ax
    
    def create_animation(self, graph_type=None, graph_params=None, save_path=None):
        """Create animation of the current event."""
        if not self.current_event:
            raise ValueError("No event selected")
        
        # Use defaults from config
        if graph_type is None and self.config['graph']['enabled']:
            graph_type = self.config['graph']['type']
        if graph_params is None and graph_type:
            graph_params = self.config['graph'].get(graph_type, {})
        
        # Get frame range
        frame_config = self.config['visualization']['frame']
        start_frame = frame_config['start']
        end_frame = frame_config['end'] if frame_config['end'] != -1 else self._num_frames
        frames = range(start_frame, min(end_frame, self._num_frames))
        
        # Create figure
        fig, ax = self.field.draw()
        
        def update(frame):
            """Update function for animation."""
            ax.clear()
            self.field.draw(ax)
            
            # Draw graph
            if graph_type and graph_type != 'none':
                self._draw_graph(ax, frame, graph_type, graph_params)
            
            # Draw objects
            self._draw_objects(ax, frame)
            
            # Add title and legend
            self._add_title(ax, frame, graph_type)
            if graph_type and graph_type != 'none' and self.config['graph']['show_legend']:
                self._add_legend(ax)
            
            return ax.patches + ax.lines + ax.texts
        
        # Create animation
        anim_config = self.config['animation']
        ani = FuncAnimation(
            fig, update, frames=frames,
            blit=False, repeat=True,
            interval=anim_config['interval']
        )
        
        # Save if requested
        if anim_config['create']:
            filepath = save_path or self._generate_filepath(graph_type)
            print(f"Saving animation to: {filepath}")
            
            if anim_config['format'] == 'gif':
                ani.save(filepath, writer='pillow', fps=anim_config['fps'], 
                        dpi=anim_config['dpi'])
            else:
                raise ValueError(f"Unsupported format: {anim_config['format']}")
            
            print("Animation saved successfully!")
        
        plt.close(fig)
        return ani
    
    def _draw_graph(self, ax, frame, graph_type, params):
        """Draw graph overlay based on type and parameters."""
        # Get valid objects for this frame
        valid_players = [p for p in self.players.values() if p.get_position(frame)]
        ball_pos = self.ball.get_position(frame) if self.ball else None
        
        if not valid_players:
            return
            
        graph = SoccerGraph()
        
        # Handle ball-centric graphs
        if graph_type in ['ball_knn', 'ball_radius'] and ball_pos:
            if graph_type == 'ball_knn':
                graph.build_ball_knn_graph(valid_players, self.ball, frame, 
                                         k=params.get('k', 8))
            else:
                graph.build_distance_based_graph(valid_players, self.ball, frame,
                                               distance_threshold=params.get('radius', 20.0))
            
            # Draw edges from object mapping
            self._draw_object_edges(ax, graph, frame)
            
        # Handle feature-based graphs
        elif hasattr(self.current_event, 'metadata') and 'features' in self.current_event.metadata:
            features = self.current_event.metadata['features']
            if frame >= len(features):
                return
                
            frame_features = features[frame]
            
            # Build appropriate graph
            if graph_type == 'knn_spatial':
                spatial_features = frame_features[:, :2]
                graph.build_knn_pyg(spatial_features, k=params.get('k', 8))
            elif graph_type == 'radius_spatial':
                spatial_features = frame_features[:, :2]
                graph.build_radius_pyg(spatial_features, radius=params.get('radius', 20.0))
            elif graph_type == 'knn_full':
                graph.build_knn_pyg(frame_features, k=params.get('k', 8))
            elif graph_type == 'radius_full':
                graph.build_radius_pyg(frame_features, radius=params.get('radius', 20.0))
            elif graph_type == 'fc':
                graph.build_fc_graph(frame_features)
            elif graph_type == 'tactical':
                graph.build_tactical_graph(frame_features)
            else:
                return
                
            # Draw edges from features
            self._draw_feature_edges(ax, graph, frame_features)
    
    def _draw_object_edges(self, ax, graph, frame):
        """Draw edges for object-based graphs."""
        edges = graph.get_edge_list()
        edge_config = self.config['graph']['edges']
        
        for start_obj, end_obj in edges:
            start_pos = start_obj.get_position(frame)
            end_pos = end_obj.get_position(frame)
            
            if start_pos and end_pos:
                # Determine edge color
                if hasattr(start_obj, 'team_id') and hasattr(end_obj, 'team_id'):
                    if start_obj.team_id == end_obj.team_id:
                        color = edge_config['home_team_color'] if start_obj.team_id == 'home' else edge_config['away_team_color']
                    else:
                        color = 'gray'
                else:
                    color = edge_config['ball_connection_color']
                
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                       color=color, linewidth=edge_config['line_width'],
                       alpha=edge_config['alpha'], zorder=2)
    
    def _draw_feature_edges(self, ax, graph, features):
        """Draw edges for feature-based graphs."""
        if not graph.data:
            return
            
        edge_index = graph.data.edge_index
        edge_config = self.config['graph']['edges']
        
        for i in range(edge_index.shape[1]):
            start_idx = edge_index[0, i].item()
            end_idx = edge_index[1, i].item()
            
            if start_idx >= len(features) or end_idx >= len(features):
                continue
            
            # Get positions (assuming coordinate transformation already done in data reader)
            start_x, start_y = features[start_idx, 0] + 52.5, features[start_idx, 1] + 34.0
            end_x, end_y = features[end_idx, 0] + 52.5, features[end_idx, 1] + 34.0
            
            # Determine color based on object type
            is_ball_start = features[start_idx, 2] == 1.0
            is_ball_end = features[end_idx, 2] == 1.0
            is_home_start = features[start_idx, 3] == 1.0
            is_home_end = features[end_idx, 3] == 1.0
            
            if is_ball_start or is_ball_end:
                color = edge_config['ball_connection_color']
            elif is_home_start and is_home_end:
                color = edge_config['home_team_color']
            elif not is_home_start and not is_home_end and not is_ball_start and not is_ball_end:
                color = edge_config['away_team_color']
            else:
                color = 'gray'
            
            ax.plot([start_x, end_x], [start_y, end_y],
                   color=color, linewidth=edge_config['line_width'],
                   alpha=edge_config['alpha'], zorder=2)
    
    def _draw_objects(self, ax, frame):
        """Draw ball and players on the field."""
        trail_config = self.config['visualization']['trails']
        
        # Draw ball
        if self.config['visualization']['ball']['show'] and self.ball:
            pos = self.ball.get_position(frame)
            if pos:
                self.ball.draw(ax, frame,
                             show_trail=trail_config['show'],
                             trail_length=trail_config['length'],
                             trail_alpha=trail_config['ball_alpha'],
                             trail_width=trail_config['ball_width'])
        
        # Draw players
        if self.config['visualization']['players']['show']:
            for player in self.players.values():
                if player.get_position(frame):
                    player.draw(ax, frame,
                              show_trail=trail_config['show'],
                              trail_length=trail_config['length'],
                              trail_alpha=trail_config['player_alpha'],
                              trail_width=trail_config['player_width'])
    
    def _add_title(self, ax, frame, graph_type):
        """Add title to the plot."""
        event_type = self.current_event.label if self.current_event else "Unknown"
        game_time = self.current_event.game_time if self.current_event else "Unknown"
        title = f"{event_type} at {game_time} - Frame {frame}/{self._num_frames-1}"
        if graph_type:
            title += f" ({graph_type})"
        ax.set_title(title, fontsize=14, pad=10)
    
    def _add_legend(self, ax):
        """Add legend for graph connections."""
        edge_config = self.config['graph']['edges']
        position = self.config['graph']['legend_position']
        
        legend_elements = [
            Line2D([0], [0], color=edge_config['home_team_color'],
                   lw=2, alpha=0.5, label='Home team'),
            Line2D([0], [0], color=edge_config['away_team_color'],
                   lw=2, alpha=0.5, label='Away team'),
            Line2D([0], [0], color=edge_config['ball_connection_color'],
                   lw=2, alpha=0.5, label='Ball connection')
        ]
        
        ax.legend(handles=legend_elements, loc=position, fontsize=8)
    
    def _generate_filepath(self, graph_type=None):
        """Generate filepath for saving animations."""
        pattern = self.config['output']['filename_pattern']
        
        event_type = self.current_event.label.replace(" ", "_")
        game_time = self.current_event.game_time.replace(" - ", "_").replace(":", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_text = graph_type if graph_type else "no_graph"
        
        filename = pattern.format(
            event_type=event_type,
            game_time=game_time,
            timestamp=timestamp,
            graph_type=graph_text
        )
        
        ext = self.config['animation']['format']
        return self.output_dir / f"{filename}.{ext}"
    
    def get_available_events(self):
        """Get all available event types per split."""
        if not self.data_reader:
            return {}
        return self.data_reader.get_available_events(self.data)
    
    def run(self):
        """Run visualization based on configuration."""
        # Load data
        input_dir = Path(self.config['data']['input_dir'])
        split = self.config['data']['split']
        file_path = input_dir / f"{split}_windows.pkl"
        
        self.load_data(str(file_path))
        
        # Select event
        event_type = self.config['event']['type']
        event_index = self.config['event']['index']
        
        try:
            self.select_event(event_type, index=event_index)
        except ValueError as e:
            print(f"Error: {e}")
            events = self.get_available_events()
            print(f"Available events: {events}")
            return
        
        # Check what to create
        specific_frame = self.config['visualization']['frame']['specific']
        
        if specific_frame is not None:
            # Single frame visualization
            self.visualize_frame(specific_frame, show=True)
        
        if self.config['animation']['create']:
            # Create animation
            self.create_animation()
            