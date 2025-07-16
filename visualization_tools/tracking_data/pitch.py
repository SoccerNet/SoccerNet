"""
Soccer field rendering module.

This module provides the Field class for rendering a soccer field with
customizable appearance including stripes, grass texture, and all standard
field markings.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, Circle
import numpy as np
from scipy.ndimage import gaussian_filter


class Field:
    """
    A class for rendering a soccer field with realistic appearance.
    
    This class creates a soccer field visualization with alternating grass stripes,
    texture, and all standard field markings including penalty areas, goal areas,
    center circle, and corner arcs.
    
    Attributes:
        length (float): Field length in meters (default: 105.0)
        width (float): Field width in meters (default: 68.0)
        line_color (str): Color of field lines (default: "white")
        line_width (int): Width of field lines (default: 2)
        padding (float): Padding around the field (default: 2.0)
        num_stripes (int): Number of grass stripes (default: 20)
        light_color (str): Light grass color (default: "#6da942")
        dark_color (str): Dark grass color (default: "#507d2a")
        noise_alpha (float): Transparency of grass texture (default: 0.03)
    
    Example:
        >>> field = Field(length=100, width=64)
        >>> fig, ax = field.draw()
        >>> plt.show()
    """
    
    def __init__(
        self, 
        length = 105.0, 
        width = 68.0,
        line_color = "white",
        line_width = 2,
        padding = 2.0,
        num_stripes = 20,
        light_color = "#6da942",
        dark_color = "#507d2a",
        noise_alpha = 0.03
    ):
        """
        Initialize a Field object.
        
        Args:
            length: Field length in meters
            width: Field width in meters
            line_color: Color of field lines
            line_width: Width of field lines
            padding: Padding around the field
            num_stripes: Number of alternating grass stripes
            light_color: Hex color for light grass stripes
            dark_color: Hex color for dark grass stripes
            noise_alpha: Alpha value for grass texture overlay
        """
        self.length = length
        self.width = width
        self.line_color = line_color
        self.line_width = line_width
        self.padding = padding
        self.num_stripes = num_stripes
        self.stripe_width = length / num_stripes
        self.light_color = light_color
        self.dark_color = dark_color
        self.noise_alpha = noise_alpha
        
        # Generate grass texture
        np.random.seed(42) 
        noise = np.random.rand(200, 200)
        self.texture = gaussian_filter(noise, sigma=0.5)
    
    def draw(self, ax = None, figsize = (12, 8)):
        """
        Draw the soccer field.
        
        Args:
            ax: Matplotlib axes to draw on. If None, creates new figure
            figsize: Figure size if creating new figure
            
        Returns:
            Tuple of (figure, axes) objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
            
        self._draw_background(ax)
        self._draw_field_lines(ax)
        
        # Set axis properties
        ax.set_xlim(-self.padding, self.length + self.padding)
        ax.set_ylim(-self.padding, self.width + self.padding)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig, ax
    
    def _draw_background(self, ax):
        """Draw the field background with stripes and texture."""
        # Background with padding
        background = Rectangle(
            (-self.padding, -self.padding),
            self.length + (2 * self.padding),
            self.width + (2 * self.padding),
            color=self.light_color,
            zorder=0
        )
        ax.add_patch(background)
        
        # Draw alternating stripes
        for i in range(self.num_stripes):
            x_start = i * self.stripe_width
            color = self.light_color if i % 2 == 0 else self.dark_color
            
            stripe = Rectangle(
                (x_start, 0),
                self.stripe_width,
                self.width,
                color=color,
                zorder=0
            )
            ax.add_patch(stripe)
        
        # Add grass texture
        ax.imshow(
            self.texture,
            extent=(0, self.length, 0, self.width),
            alpha=self.noise_alpha,
            zorder=1,
            cmap='gray'
        )
    
    def _draw_field_lines(self, ax):
        """Draw all field lines and markings."""
        # Outer boundary
        ax.plot(
            [0, 0, self.length, self.length, 0],
            [0, self.width, self.width, 0, 0],
            color=self.line_color,
            linewidth=self.line_width
        )
        
        # Halfway line
        ax.plot(
            [self.length / 2, self.length / 2],
            [0, self.width],
            color=self.line_color,
            linewidth=self.line_width
        )
        
        # Center circle
        center_circle = Circle(
            (self.length / 2, self.width / 2),
            9.15,
            color=self.line_color,
            fill=False,
            linewidth=self.line_width
        )
        ax.add_patch(center_circle)
        
        # Center spot
        ax.plot(
            self.length / 2,
            self.width / 2,
            'o',
            color=self.line_color,
            markersize=5
        )
        
        # Draw areas
        self._draw_penalty_areas(ax)
        self._draw_goal_areas(ax)
        self._draw_corner_arcs(ax)
    
    def _draw_penalty_areas(self, ax):
        """Draw penalty areas and penalty spots."""
        # Left penalty area
        left_box = Rectangle(
            (0, (self.width / 2) - 20.15),
            16.5,
            40.3,
            edgecolor=self.line_color,
            fill=False,
            linewidth=self.line_width
        )
        ax.add_patch(left_box)
        
        # Right penalty area
        right_box = Rectangle(
            (self.length - 16.5, (self.width / 2) - 20.15),
            16.5,
            40.3,
            edgecolor=self.line_color,
            fill=False,
            linewidth=self.line_width
        )
        ax.add_patch(right_box)
        
        # Left penalty arc
        left_arc = Arc(
            (11, self.width / 2),
            height=18.3,
            width=18.3,
            theta1=308,
            theta2=52,
            color=self.line_color,
            linewidth=self.line_width
        )
        ax.add_patch(left_arc)
        
        # Right penalty arc
        right_arc = Arc(
            (self.length - 11, self.width / 2),
            height=18.3,
            width=18.3,
            theta1=127,
            theta2=233,
            color=self.line_color,
            linewidth=self.line_width
        )
        ax.add_patch(right_arc)
        
        # Penalty spots
        ax.plot(11, self.width / 2, 'o', color=self.line_color, markersize=5)
        ax.plot(self.length - 11, self.width / 2, 'o', color=self.line_color, markersize=5)
    
    def _draw_goal_areas(self, ax):
        """Draw goal areas and goalposts."""
        # Left goal area
        left_goal_area = Rectangle(
            (0, (self.width / 2) - 8.5),
            5.5,
            17,
            edgecolor=self.line_color,
            fill=False,
            linewidth=self.line_width
        )
        ax.add_patch(left_goal_area)
        
        # Right goal area
        right_goal_area = Rectangle(
            (self.length, (self.width / 2) - 8.5),
            -5.5,
            17,
            edgecolor=self.line_color,
            fill=False,
            linewidth=self.line_width
        )
        ax.add_patch(right_goal_area)
        
        # Goalposts
        ax.plot(
            [0, 0],
            [self.width / 2 - 3.66, self.width / 2 + 3.66],
            color=self.line_color,
            linewidth=self.line_width + 2
        )
        
        ax.plot(
            [self.length, self.length],
            [self.width / 2 - 3.66, self.width / 2 + 3.66],
            color=self.line_color,
            linewidth=self.line_width + 2
        )
    
    def _draw_corner_arcs(self, ax):
        """Draw corner arc markings."""
        corners = [
            (0, 0, 0, 90),          # Bottom left
            (0, self.width, 270, 360),  # Top left
            (self.length, 0, 90, 180),  # Bottom right
            (self.length, self.width, 180, 270)  # Top right
        ]
        
        for x, y, theta1, theta2 in corners:
            corner_arc = Arc(
                (x, y),
                1.8,
                1.8,
                angle=0,
                theta1=theta1,
                theta2=theta2,
                color=self.line_color,
                linewidth=self.line_width
            )
            ax.add_patch(corner_arc)
