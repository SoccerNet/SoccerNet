from tracking_data.visualizer import SoccerNetVisualizer

def quick_test():
    print("Running visualization from config.yaml...")
    viz = SoccerNetVisualizer(config_path="config.yaml")
    viz.run()
    
    print("\nQuick test complete! Check 'visualizations' folder for output.")

if __name__ == "__main__":
    quick_test()
    