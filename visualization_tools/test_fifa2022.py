import sys
from pathlib import Path
sys.path.append('.')

from tracking_data.visualizer import SoccerNetVisualizer

def visualize_all_labels():
    # Create visualizer
    viz = SoccerNetVisualizer(config_path="fifa_config.yaml")
    
    # Build paths
    base_dir = Path("/home/karkid/temporal_localization/soccernet_ball_spotting_fifa2022")
    game_id = "10502"
    
    tracking_file = base_dir / "train/videos" / f"{game_id}.jsonl.bz2"
    annotations_file = base_dir / "train/train.json"
    
    print(f"Loading: {tracking_file}")
    print(f"Annotations: {annotations_file}")
    
    # Load data
    viz.load_data(str(tracking_file), annotations_path=str(annotations_file))
    
    # Get available events
    events = viz.get_available_events()
    labels = events['default']
    
    print(f"\nFound {len(labels)} unique labels: {labels}")
    
    # Create visualization for each label
    for label in labels:
        try:
            print(f"\n{'='*50}")
            print(f"Creating visualization for: {label}")
            
            # Select first event of this type
            viz.select_event(label, index=1)
            
            # Create animation with label-specific filename
            save_path = f"smoothed/fifa_{game_id}_{label.replace(' ', '_')}.gif"
            viz.create_animation(save_path=save_path)
            
            print(f"Saved: {save_path}")
            
        except Exception as e:
            print(f"Error with {label}: {e}")
    
    print("\nAll done!")

if __name__ == "__main__":
    visualize_all_labels()
    