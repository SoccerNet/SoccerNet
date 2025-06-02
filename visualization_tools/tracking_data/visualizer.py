"""
Main visualizer module for Soccer Visualizer.

This module provides the main SoccerVisualizer class that orchestrates
the entire visualization process, from loading data to creating animations.
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from typing import List, Optional, Union, Dict, Tuple, Any
from pathlib import Path
from datetime import datetime

from pitch import Field
from objects import Player, Ball
from graph import SoccerGraph
from config import Config


class SoccerVisualizer:
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
        >>> viz.load_data("train").select_event("Pass").create_animation()
    """
    
    def __init__(
        self, 
        config_path: Optional[Union[str, Path]] = None, 
        data_dir: Optional[str] = None, 
        output_dir: Optional[str] = None
    ):
        """
        Initialize visualizer with optional configuration.
        
        Args:
            config_path: Path to YAML configuration file
            data_dir: Override data directory from config
            output_dir: Override output directory from config
        """
        # Load configuration
        self.config = Config(config_path)
        
        # Override directories if provided
        if data_dir:
            self.config.set("data.input_dir", data_dir)
        if output_dir:
            self.config.set("data.output_dir", output_dir)
            
        # Set instance variables from config
        self.data_dir = self.config.get("data.input_dir")
        self.output_dir = self.config.get("data.output_dir")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(self.config.get("advanced.random_seed", 42))
        
        # Initialize data containers
        self.splits: Dict[str, Dict] = {}
        self.current_event: Optional[Dict] = None
        self.ball: Optional[Ball] = None
        self.home_team: List[Player] = []
        self.away_team: List[Player] = []
        self.all_players: List[Player] = []
        
        # Initialize pitch with config
        self.pitch = self._create_pitch()
        
        # Track frame count for current event
        self._num_frames = 0
        
        if self.config.get("advanced.verbose"):
            print(f"SoccerVisualizer initialized")
            if config_path:
                print(f"  Config: {config_path}")
            print(f"  Data directory: {self.data_dir}")
            print(f"  Output directory: {self.output_dir}")
            
    def _create_pitch(self) -> Field:
        """Create pitch object from configuration."""
        pitch_config = self.config.get("visualization.pitch", {})
        
        return Field(
            length=pitch_config.get("length", 105.0),
            width=pitch_config.get("width", 68.0),
            line_color=pitch_config.get("line_color", "white"),
            line_width=pitch_config.get("line_width", 2),
            padding=pitch_config.get("padding", 2.0),
            num_stripes=pitch_config.get("num_stripes", 20),
            light_color=pitch_config.get("light_color", "#6da942"),
            dark_color=pitch_config.get("dark_color", "#507d2a"),
            noise_alpha=pitch_config.get("texture_alpha", 0.03) if pitch_config.get("grass_texture", True) else 0
        )
        
    def run(self) -> None:
        """
        Run visualization based on configuration file.
        
        This is the main entry point when using config-based approach.
        """
        # Load data
        split = self.config.get("data.split", "train")
        self.load_data(split)
        
        # Check if batch processing is enabled
        if self.config.get("batch.enabled", False):
            self._run_batch()
        else:
            self._run_single()
            
    def _run_single(self) -> None:
        """Run single event visualization based on config."""
        # Select event
        event_type = self.config.get("event.type", "Pass")
        event_index = self.config.get("event.index", 0)
        
        try:
            self.select_event(event_type, index=event_index)
            self.process_event()
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        # Check what to create
        specific_frame = self.config.get("visualization.frame.specific")
        
        if specific_frame is not None:
            # Visualize specific frame
            if self.config.get("advanced.verbose"):
                print(f"Visualizing frame {specific_frame}")
            
            self.visualize_frame(
                frame=specific_frame,
                with_graph=self.config.get("graph.enabled", True),
                show=True
            )
        
        if self.config.get("animation.create", True):
            # Create animation
            if self.config.get("advanced.verbose"):
                print("Creating animation...")
            
            self.create_animation()
            
    def _run_batch(self) -> None:
        """Run batch processing of multiple events."""
        event_types = self.config.get("batch.event_types", ["Pass"])
        events_per_type = self.config.get("batch.events_per_type", 5)
        
        for event_type in event_types:
            if self.config.get("advanced.verbose"):
                print(f"\nProcessing {event_type} events...")
                
            for i in range(events_per_type):
                try:
                    self.select_event(event_type, index=i)
                    self.process_event()
                    self.create_animation()
                    
                    if self.config.get("advanced.verbose"):
                        print(f"  ✓ Processed {event_type} event {i+1}/{events_per_type}")
                        
                except (ValueError, IndexError) as e:
                    if self.config.get("advanced.verbose"):
                        print(f"  - No more {event_type} events available (got {i})")
                    break
                    
    def load_data(self, split: str = "train") -> 'SoccerVisualizer':
        """
        Load data from a specific split.
        
        Args:
            split: Data split to load ("train", "valid", "test")
            
        Returns:
            Self for method chaining
            
        Raises:
            FileNotFoundError: If data file not found
        """
        file_path = os.path.join(self.data_dir, f"{split}_windows.pkl")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        with open(file_path, 'rb') as f:
            self.splits[split] = pickle.load(f)
            
        if self.config.get("advanced.verbose"):
            num_windows = len(self.splits[split].get("windows", []))
            print(f"Loaded {num_windows} windows from {split} split")
            
        return self
        
    def load_all_splits(self) -> 'SoccerVisualizer':
        """Load data from all available splits."""
        for split in ["train", "valid", "test"]:
            try:
                self.load_data(split)
            except FileNotFoundError:
                if self.config.get("advanced.verbose"):
                    print(f"Split '{split}' not found, skipping...")
        return self
        
    def select_event(
        self, 
        event_type: str, 
        split: Optional[str] = None, 
        index: int = 0
    ) -> 'SoccerVisualizer':
        """
        Select an event by type from a specific split.
        
        Args:
            event_type: Type of event (e.g., "Pass", "Shot", "Corner")
            split: Data split (if None, uses config default)
            index: Index of event within the type
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If no events of specified type found
        """
        if split is None:
            split = self.config.get("data.split", "train")
            
        if split not in self.splits:
            self.load_data(split)
            
        # Filter events by type
        events = [
            w for w in self.splits[split].get("windows", [])
            if w.get("label") == event_type
        ]
        
        if not events:
            available_types = set(w.get("label") for w in self.splits[split].get("windows", []))
            raise ValueError(
                f"No events of type '{event_type}' found in split '{split}'. "
                f"Available types: {sorted(available_types)}"
            )
            
        if index >= len(events):
            raise ValueError(
                f"Event index {index} out of range. "
                f"Only {len(events)} '{event_type}' events available."
            )
            
        self.current_event = events[index]
        self._num_frames = self.current_event["features"].shape[0]
        
        if self.config.get("advanced.verbose"):
            game_time = self.current_event.get("game_time", "Unknown")
            print(f"Selected {event_type} event at {game_time} ({self._num_frames} frames)")
            
        return self
    
    def process_event(self) -> 'SoccerVisualizer':
        """
        Process the current event to create ball and player objects.
        
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If no event selected
        """
        if not self.current_event:
            raise ValueError("No event selected. Call select_event first.")
            
        features = self.current_event["features"]
        num_frames, num_objects, num_features = features.shape
        
        # Get configuration
        player_config = self.config.get("visualization.players", {})
        ball_config = self.config.get("visualization.ball", {})
        
        # Reset objects
        self.ball = Ball(
            color=ball_config.get("color", "black"),
            radius=ball_config.get("radius", 0.7)
        )
        
        home_team_objects = {}
        away_team_objects = {}
        
        # Process each frame
        for frame in range(num_frames):
            for obj_idx in range(num_objects):
                # Skip if no position data
                if np.sum(np.abs(features[frame, obj_idx, 0:2])) <= 0:
                    continue
                
                # Check object type
                is_ball = features[frame, obj_idx, 2] == 1.0
                is_home = features[frame, obj_idx, 3] == 1.0
                is_away = features[frame, obj_idx, 4] == 1.0
                
                # Convert coordinates (assuming centered at origin)
                x = features[frame, obj_idx, 0] + 52.5  # Half pitch length
                y = features[frame, obj_idx, 1] + 34    # Half pitch width
                
                if is_ball:
                    self.ball.add_position(frame, x, y)
                    
                elif is_home:
                    if obj_idx not in home_team_objects:
                        jersey_number = len(home_team_objects) + 1
                        home_team_objects[obj_idx] = Player(
                            player_id=f"home_{obj_idx}",
                            team_id="home",
                            jersey_number=jersey_number,
                            color=player_config.get("home_color", "blue"),
                            radius=player_config.get("radius", 1.0),
                            show_jersey=player_config.get("show_jersey_numbers", True),
                            jersey_font_size=player_config.get("jersey_font_size", 8)
                        )
                    home_team_objects[obj_idx].add_position(frame, x, y)
                
                elif is_away:
                    if obj_idx not in away_team_objects:
                        jersey_number = len(away_team_objects) + 1
                        away_team_objects[obj_idx] = Player(
                            player_id=f"away_{obj_idx}",
                            team_id="away",
                            jersey_number=jersey_number,
                            color=player_config.get("away_color", "red"),
                            radius=player_config.get("radius", 1.0),
                            show_jersey=player_config.get("show_jersey_numbers", True),
                            jersey_font_size=player_config.get("jersey_font_size", 8)
                        )
                    away_team_objects[obj_idx].add_position(frame, x, y)
        
        # Convert to lists
        self.home_team = list(home_team_objects.values())
        self.away_team = list(away_team_objects.values())
        self.all_players = self.home_team + self.away_team
        
        if self.config.get("advanced.verbose"):
            print(f"Processed: {len(self.home_team)} home players, "
                  f"{len(self.away_team)} away players")
        
        return self
    
    def visualize_frame(
        self, 
        frame: int, 
        with_graph: Optional[bool] = None, 
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize a single frame from the current event.
        
        Args:
            frame: Frame number to visualize
            with_graph: Override graph setting from config
            show: Whether to display the plot
            save: Whether to save the frame
            filename: Custom filename for saving
            
        Returns:
            Tuple of (figure, axes)
        """
        if not self.ball or not self.current_event:
            self.process_event()
            
        # Use config if with_graph not specified
        if with_graph is None:
            with_graph = self.config.get("graph.enabled", True)
            
        # Validate frame number
        if frame < 0 or frame >= self._num_frames:
            raise ValueError(f"Frame {frame} out of range (0-{self._num_frames-1})")
            
        # Create figure
        fig_size = tuple(self.config.get("advanced.figure_size", [12, 8]))
        fig, ax = self.pitch.draw(figsize=fig_size)
        
        # Draw graph if enabled
        if with_graph:
            self._draw_graph(ax, frame)
        
        # Draw ball
        if self.config.get("visualization.ball.show", True):
            ball_pos = self.ball.get_position(frame)
            if ball_pos:
                trail_config = self.config.get("visualization.trails", {})
                self.ball.draw(
                    ax, frame,
                    show_trail=trail_config.get("show", True),
                    trail_length=trail_config.get("length", 5),
                    trail_alpha=trail_config.get("ball_alpha", 0.5),
                    trail_width=trail_config.get("ball_width", 1.0)
                )
        
        # Draw players
        if self.config.get("visualization.players.show", True):
            trail_config = self.config.get("visualization.trails", {})
            for player in self.all_players:
                if player.get_position(frame):
                    player.draw(
                        ax, frame,
                        show_trail=trail_config.get("show", True),
                        trail_length=trail_config.get("length", 5),
                        trail_alpha=trail_config.get("player_alpha", 0.5),
                        trail_width=trail_config.get("player_width", 1.5)
                    )
        
        # Add legend for graph
        if with_graph and self.config.get("graph.show_legend", True):
            self._add_graph_legend(ax)
        
        # Set title
        event_type = self.current_event.get("label", "Unknown")
        game_time = self.current_event.get("game_time", "Unknown")
        title = f"{event_type} at {game_time} - Frame {frame}/{self._num_frames-1}"
        ax.set_title(title, fontsize=14, pad=10)
        
        # Save frame if requested
        if save or self.config.get("output.save_frames", False):
            self._save_frame(fig, frame, filename)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig, ax
    
    def create_animation(
        self, 
        with_graph: Optional[bool] = None,
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """
        Create an animation of the current event.
        
        Args:
            with_graph: Override graph setting from config
            save_path: Custom path for saving animation
            
        Returns:
            FuncAnimation object
        """
        if not self.ball or not self.current_event:
            self.process_event()
            
        # Use config if with_graph not specified
        if with_graph is None:
            with_graph = self.config.get("graph.enabled", True)
            
        # Get animation settings
        anim_config = self.config.get("animation", {})
        frame_config = self.config.get("visualization.frame", {})
        
        # Determine frame range
        start_frame = frame_config.get("start", 0)
        end_frame = frame_config.get("end", -1)
        if end_frame == -1:
            end_frame = self._num_frames
        
        frames = range(start_frame, min(end_frame, self._num_frames))
        
        # Create figure
        fig_size = tuple(self.config.get("advanced.figure_size", [12, 8]))
        fig, ax = self.pitch.draw(figsize=fig_size)
        
        # Animation update function
        def update(frame):
            ax.clear()
            self.pitch.draw(ax)
            
            # Draw graph
            if with_graph:
                self._draw_graph(ax, frame)
            
            # Draw ball
            if self.config.get("visualization.ball.show", True) and self.ball.get_position(frame):
                trail_config = self.config.get("visualization.trails", {})
                self.ball.draw(
                    ax, frame,
                    show_trail=trail_config.get("show", True),
                    trail_length=trail_config.get("length", 5),
                    trail_alpha=trail_config.get("ball_alpha", 0.5),
                    trail_width=trail_config.get("ball_width", 1.0)
                )
            
            # Draw players
            if self.config.get("visualization.players.show", True):
                trail_config = self.config.get("visualization.trails", {})
                for player in self.all_players:
                    if player.get_position(frame):
                        player.draw(
                            ax, frame,
                            show_trail=trail_config.get("show", True),
                            trail_length=trail_config.get("length", 5),
                            trail_alpha=trail_config.get("player_alpha", 0.5),
                            trail_width=trail_config.get("player_width", 1.5)
                        )
            
            # Add legend
            if with_graph and self.config.get("graph.show_legend", True):
                self._add_graph_legend(ax)
            
            # Set title
            event_type = self.current_event.get("label", "Unknown")
            game_time = self.current_event.get("game_time", "Unknown")
            title = f"{event_type} at {game_time} - Frame {frame}/{self._num_frames-1}"
            ax.set_title(title, fontsize=14, pad=10)
            
            return ax.patches + ax.lines + ax.texts
        
        # Create animation
        ani = FuncAnimation(
            fig, update, frames=frames,
            blit=False, repeat=True,
            interval=anim_config.get("interval", 200)
        )
        
        # Save animation
        if anim_config.get("create", True):
            filepath = save_path or self._generate_filepath()
            
            if self.config.get("advanced.verbose"):
                print(f"Saving animation to: {filepath}")
            
            # Determine writer and save
            format_type = anim_config.get("format", "gif")
            fps = anim_config.get("fps", 5)
            dpi = anim_config.get("dpi", 100)
            
            if format_type == "gif":
                ani.save(filepath, writer='pillow', fps=fps, dpi=dpi)
            elif format_type == "mp4":
                ani.save(filepath, writer='ffmpeg', fps=fps, dpi=dpi)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            if self.config.get("advanced.verbose"):
                print(f"Animation saved successfully!")
        
        plt.close(fig)
        return ani
    
    def _draw_graph(self, ax: plt.Axes, frame: int) -> None:
        """Draw graph overlay on the pitch."""
        graph_config = self.config.get("graph", {})
        
        # Create graph
        graph = SoccerGraph()
        valid_players = [p for p in self.all_players if p.get_position(frame)]
        
        if not self.ball.get_position(frame) or not valid_players:
            return
        
        # Build graph based on type
        graph_type = graph_config.get("type", "distance")
        
        try:
            if graph_type == "distance":
                dist_config = graph_config.get("distance", {})
                graph.build_distance_based_graph(
                    valid_players, self.ball,
                    frame=frame,
                    distance_threshold=dist_config.get("threshold", 20.0),
                    connect_same_team=dist_config.get("connect_same_team", True),
                    connect_ball_to_players=dist_config.get("connect_ball_to_players", True)
                )
            elif graph_type == "knn":
                knn_config = graph_config.get("knn", {})
                graph.build_ball_knn_graph(
                    valid_players, self.ball,
                    frame=frame,
                    k=knn_config.get("k", 6),
                    connect_players=True,
                    team_aware=knn_config.get("connect_same_team_only", False)
                )
            else:
                raise ValueError(f"Unknown graph type: {graph_type}")
        except Exception as e:
            if self.config.get("advanced.verbose"):
                print(f"Error building graph: {e}")
            return
        
        # Draw edges
        edges = graph.get_edge_list()
        edge_config = graph_config.get("edges", {})
        
        for start_obj, end_obj in edges:
            start_pos = start_obj.get_position(frame)
            end_pos = end_obj.get_position(frame)
            
            if start_pos and end_pos:
                # Determine edge color
                if hasattr(start_obj, 'team_id') and hasattr(end_obj, 'team_id'):
                    if start_obj.team_id == end_obj.team_id:
                        if start_obj.team_id == 'home':
                            line_color = edge_config.get("home_team_color", "blue")
                        else:
                            line_color = edge_config.get("away_team_color", "red")
                    else:
                        # Different teams (shouldn't happen in current implementation)
                        line_color = "gray"
                else:
                    # Ball connection
                    line_color = edge_config.get("ball_connection_color", "black")
                
                # Draw edge
                ax.plot(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    color=line_color,
                    linewidth=edge_config.get("line_width", 1.0),
                    alpha=edge_config.get("alpha", 0.5),
                    zorder=2
                )
    
    def _add_graph_legend(self, ax: plt.Axes) -> None:
        """Add legend for graph connections."""
        edge_config = self.config.get("graph.edges", {})
        position = self.config.get("graph.legend_position", "upper left")
        
        # Create legend elements
        legend_elements = []
        
        # Home team edges
        legend_elements.append(
            Line2D([0], [0], color=edge_config.get("home_team_color", "blue"),
                   lw=2, alpha=0.5, label='Home team')
        )
        
        # Away team edges
        legend_elements.append(
            Line2D([0], [0], color=edge_config.get("away_team_color", "red"),
                   lw=2, alpha=0.5, label='Away team')
        )
        
        # Ball connections
        legend_elements.append(
            Line2D([0], [0], color=edge_config.get("ball_connection_color", "black"),
                   lw=2, alpha=0.5, label='Ball connection')
        )
        
        ax.legend(handles=legend_elements, loc=position, fontsize=8)
    
    def _generate_filepath(self) -> str:
        """Generate filepath for saving animations."""
        pattern = self.config.get("output.filename_pattern", "{event_type}_{game_time}_{timestamp}")
        
        # Prepare substitution values
        event_type = self.current_event.get("label", "Unknown").replace(" ", "_")
        game_time = self.current_event.get("game_time", "Unknown").replace(" - ", "_").replace(":", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_type = self.config.get("graph.type", "none") if self.config.get("graph.enabled") else "no_graph"
        
        # Create filename
        filename = pattern.format(
            event_type=event_type,
            game_time=game_time,
            timestamp=timestamp,
            graph_type=graph_type
        )
        
        # Add extension
        ext = self.config.get("animation.format", "gif")
        filename = f"{filename}.{ext}"
        
        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        
        return filepath
    
    def _save_frame(self, fig: plt.Figure, frame: int, filename: Optional[str] = None) -> None:
        """Save individual frame to file."""
        if filename is None:
            frame_format = self.config.get("output.frame_format", "png")
            event_type = self.current_event.get("label", "Unknown").replace(" ", "_")
            filename = f"{event_type}_frame_{frame:04d}.{frame_format}"
        
        # Create frames subdirectory
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        filepath = os.path.join(frames_dir, filename)
        fig.savefig(filepath, dpi=self.config.get("animation.dpi", 100), bbox_inches='tight')
        
        if self.config.get("advanced.verbose"):
            print(f"Saved frame to: {filepath}")
    
    def get_event_info(self) -> Dict[str, Any]:
        """Get information about the current event."""
        if not self.current_event:
            return {}
        
        return {
            "type": self.current_event.get("label", "Unknown"),
            "game_time": self.current_event.get("game_time", "Unknown"),
            "num_frames": self._num_frames,
            "num_home_players": len(self.home_team),
            "num_away_players": len(self.away_team),
            "has_ball": self.ball is not None
        }
    
    def __repr__(self) -> str:
        """String representation of visualizer."""
        info = self.get_event_info()
        if info:
            return f"SoccerVisualizer(event={info['type']}, frames={info['num_frames']})"
        return "SoccerVisualizer(no event selected)"
    