"""
Simple test script to run the soccer visualizer.

This script will create an animation of a soccer game based on the config.yaml file.
"""

import sys
import os

# Add tracking_data to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tracking_data'))

from tracking_data.visualizer import SoccerVisualizer

def main():
    """Run the visualizer."""
    print("Soccer Visualizer")
    print("=" * 50)
    
    # Create visualizer using config.yaml
    viz = SoccerVisualizer(config_path="config.yaml")
    
    # This will create the animation based on config
    viz.run()
    
    print("\nDone! Check the 'output_dir' directory for output.")

if __name__ == "__main__":
    # Change to visualization_tools directory
    os.chdir(os.path.dirname(__file__))
    main()
    