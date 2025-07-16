"""
Moving objects module for Soccer Visualizer.

This module defines classes for all moving objects on the soccer field,
including players and the ball. Each object can store positions across
multiple frames and render itself with optional motion trails.
"""

from typing import Dict, List, Optional, Tuple, Union
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.text import Text


class MovingObject:
    """
    Base class for moving objects on the pitch.
    
    This class provides common functionality for all moving objects including
    position tracking, rendering, and motion trails.
    
    Attributes:
        positions (Dict[int, Tuple[float, float]]): Frame-indexed positions
        color (str): Object color
        radius (float): Object radius for rendering
        zorder (int): Z-order for layering in matplotlib
        marker (Optional[Circle]): Current matplotlib circle patch
    
    Example:
        >>> obj = MovingObject(color="red", radius=1.0)
        >>> obj.add_position(frame=1, x=50.0, y=25.0)
        >>> obj.draw(ax, frame=1)
    """
    
    def __init__(self, color, radius=1, zorder=4):
        """
        Initialize a moving object.
        
        Args:
            color: Color for rendering the object
            radius: Radius of the object circle
            zorder: Z-order for rendering (higher values appear on top)
        """
        self.positions = {}
        self.color = color
        self.radius = radius
        self.zorder = zorder
        self.marker = None
    
    def add_position(self, frame, x, y):
        """
        Add position for a specific frame.
        
        Args:
            frame: Frame number
            x: X coordinate on the field
            y: Y coordinate on the field
        """
        self.positions[frame] = (x, y)
    
    def get_position(self, frame):
        """
        Get position at a specific frame.
        
        Args:
            frame: Frame number
            
        Returns:
            Tuple of (x, y) coordinates or None if no position exists
        """
        return self.positions.get(frame)
    
    def get_positions_range(self, start_frame, end_frame):
        """
        Get positions for a range of frames.
        
        Args:
            start_frame: Starting frame (inclusive)
            end_frame: Ending frame (inclusive)
            
        Returns:
            List of (x, y) positions
        """
        positions = []
        for frame in range(start_frame, end_frame + 1):
            pos = self.get_position(frame)
            if pos:
                positions.append(pos)
        return positions
        
    def draw(self, ax, frame, show_trail=True, 
             trail_length=5, trail_alpha=0.5,
             trail_width=1.5):
        """
        Draw the object on the field.
        
        Args:
            ax: Matplotlib axes to draw on
            frame: Current frame number
            show_trail: Whether to show motion trail
            trail_length: Number of previous frames to include in trail
            trail_alpha: Transparency of trail (0-1)
            trail_width: Width of trail line
            
        Returns:
            The matplotlib Circle patch or None if no position
        """
        position = self.get_position(frame)
        
        if position is None:
            self._clear_visual_elements(ax)
            return None
        
        x, y = position
        
        # Clear previous visual elements
        self._clear_visual_elements(ax)
        
        # Create new marker
        self._create_marker(ax, x, y)
        
        # Draw any additional elements (overridden in subclasses)
        self._draw_additional_elements(ax, x, y)
        
        # Draw trail if requested
        if show_trail:
            self._draw_trail(ax, frame, trail_length, trail_alpha, trail_width)
        
        return self.marker
    
    def _clear_visual_elements(self, ax):
        """Remove previous visual elements from the axes."""
        if self.marker and self.marker in ax.patches:
            self.marker.remove()
        self.marker = None
    
    def _create_marker(self, ax, x, y):
        """Create the circular marker for the object."""
        self.marker = Circle(
            (x, y), 
            radius=self.radius, 
            facecolor=self.color, 
            edgecolor='black', 
            linewidth=0.5,
            zorder=self.zorder
        )
        ax.add_patch(self.marker)
    
    def _draw_additional_elements(self, ax, x, y):
        """
        Draw any additional elements (overridden in subclasses).
        
        Args:
            ax: Matplotlib axes
            x: X coordinate
            y: Y coordinate
        """
        pass
    
    def _draw_trail(self, ax, frame, trail_length, 
                    trail_alpha, trail_width):
        """Draw motion trail for the object."""
        position = self.get_position(frame)
        if position is None:
            return
            
        x, y = position
        trail_positions = []
        
        # Collect previous positions
        for prev_frame in range(max(1, frame - trail_length), frame):
            prev_pos = self.get_position(prev_frame)
            if prev_pos is not None:
                trail_positions.append(prev_pos)
        
        # Draw trail if we have positions
        if trail_positions:
            trail_x: List[float] = [pos[0] for pos in trail_positions] + [x]
            trail_y: List[float] = [pos[1] for pos in trail_positions] + [y]
            
            self._draw_trail_line(ax, trail_x, trail_y, trail_alpha, trail_width)
    
    def _draw_trail_line(self, ax, trail_x, trail_y,
                         trail_alpha, trail_width):
        """Draw the trail line with custom styling."""
        ax.plot(
            trail_x, trail_y, '-', 
            color=self.color,
            alpha=trail_alpha, 
            linewidth=trail_width, 
            zorder=self.zorder - 1
        )


class Player(MovingObject):
    """
    Player object on the pitch.
    
    Extends MovingObject with player-specific features like jersey numbers
    and team identification.
    
    Attributes:
        id (Union[int, str]): Unique player identifier
        team_id (str): Team identifier (e.g., "home", "away")
        jersey (int): Jersey number
        text (Optional[Text]): Matplotlib text object for jersey number
    
    Example:
        >>> player = Player("player1", "home", 10, color="blue")
        >>> player.add_position(1, 50.0, 25.0)
        >>> player.draw(ax, frame=1)
    """
    
    def __init__(self, player_id, team_id, 
                 jersey_number, color="blue", 
                 radius=1.0, 
                 show_jersey=True,
                 jersey_font_size=8):
        """
        Initialize a player object.
        
        Args:
            player_id: Unique identifier for the player
            team_id: Team identifier (e.g., "home", "away")
            jersey_number: Player's jersey number
            color: Team color
            radius: Player circle radius
            show_jersey: Whether to display jersey number
            jersey_font_size: Font size for jersey number
        """
        super().__init__(color=color, radius=radius, zorder=4)
        self.id = player_id
        self.team_id = team_id
        self.jersey = jersey_number
        self.text = None
        self.show_jersey: bool = show_jersey
        self.jersey_font_size: int = jersey_font_size
    
    def _clear_visual_elements(self, ax: Axes) -> None:
        """Clear both marker and jersey number text."""
        super()._clear_visual_elements(ax)
        if self.text and self.text in ax.texts:
            self.text.remove()
        self.text = None
    
    def _draw_additional_elements(self, ax: Axes, x: float, y: float) -> None:
        """Draw jersey number on the player."""
        if self.show_jersey:
            self.text = ax.text(
                x, y, str(self.jersey), 
                color='white',
                ha='center', 
                va='center', 
                fontsize=self.jersey_font_size,
                fontweight='bold', 
                zorder=self.zorder + 1
            )
    
    def __repr__(self) -> str:
        """String representation of player."""
        return f"Player(id={self.id}, team={self.team_id}, jersey={self.jersey})"


class Ball(MovingObject):
    """
    Ball object on the pitch.
    
    Specialized moving object for the ball with distinct visual properties.
    
    Example:
        >>> ball = Ball()
        >>> ball.add_position(1, 52.5, 34.0)
        >>> ball.draw(ax, frame=1)
    """
    
    def __init__(self, color="black", radius=0.7):
        """
        Initialize the ball object.
        
        Args:
            color: Ball color (default: black)
            radius: Ball radius (default: 0.7, smaller than players)
        """
        # Ball has higher z-order to appear above players
        super().__init__(color=color, radius=radius, zorder=6)
    
    def _create_marker(self, ax, x, y):
        """Create ball marker with custom edge styling."""
        self.marker = Circle(
            (x, y), 
            radius=self.radius, 
            facecolor=self.color, 
            edgecolor='white',  # White edge for better visibility
            linewidth=0.5, 
            zorder=self.zorder
        )
        ax.add_patch(self.marker)
    
    def _draw_trail_line(self, ax, trail_x, trail_y,
                         trail_alpha, trail_width):
        """Draw ball trail with custom styling (gray color)."""
        ax.plot(
            trail_x, trail_y, '-', 
            color='gray',  # Gray trail for ball
            alpha=trail_alpha, 
            linewidth=trail_width * 0.7,  # Thinner trail for ball
            zorder=self.zorder - 1
        )
    
    def __repr__(self) -> str:
        """String representation of ball."""
        return "Ball()"


# Optional: Additional object types can be added here
class Referee(Player):
    """
    Referee object on the pitch.
    
    Special type of player with distinct visual properties.
    """
    
    def __init__(self, referee_id, 
                 color="yellow", radius=0.9):
        """Initialize referee with yellow color by default."""
        super().__init__(
            player_id=referee_id,
            team_id="referee",
            jersey_number=0,  # Refs don't have jersey numbers
            color=color,
            radius=radius,
            show_jersey=False  # Don't show jersey number
        )
    
    def __repr__(self):
        """String representation of referee."""
        return f"Referee(id={self.id})"