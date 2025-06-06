from abc import ABC, abstractmethod
import pickle

class FrameData:
    """Standardized frame data structure."""
    def __init__(self):
        self.objects = []
        # Each object has: {
        #   'id': unique identifier,
        #   'x': float,
        #   'y': float,
        #   'type': 'ball' | 'home' | 'away',
        #   'features': optional additional features,
        #   'jersey': optional jersey number,
        # }
        

class EventData:
    """Standardized event data structure."""
    def __init__(self, event_id, label, game_time):
        self.event_id = event_id
        self.label = label
        self.game_time = game_time
        self.frames = {}
        self.metadata = {}
        
        
class DataReader(ABC):
    """Abstract base class for data readers."""
    
    @abstractmethod
    def load_data(self, file_path):
        """Load data and return standardized format."""
        pass
    
    @abstractmethod
    def get_available_events(self, data):
        """Get available event types per split."""
        pass
    

class PKLDataReader(DataReader):
    """
    Reader for pickle files - Example implementation for extending to new formats.
    
    This reader demonstrates how to create a custom data reader for any file format.
    The key requirement is to extract:
    1. Object positions (x, y coordinates) for each frame
    2. Object types (ball, home team or away team)
    3. Optional: additional features for advanced graph algorithms
    
    To create your own reader:
    1. Inherit from DataReader base class
    2. Implement load_data() to return EventData objects
    3. Map your cooridnates to standard pitch dimensions (0-105 for x, 0-68 for y)
    4. Register your reader in DataReaderFactory
    
    Example coordinate transformation:
    - Our PKL data: x ∈ [-52.5, 52.5], y ∈ [-34, 34]
    - Standard pitch: x ∈ [0, 105], y ∈ [0, 68]
    - Transform: x_pitch = x_data + 52.5, y_pitch = y_data + 34.0
    """
    
    def load_data(self, file_path):
        """Load PKL data and convert to standardized format."""
        with open(file_path, 'rb') as f:
            raw_data = pickle.load(f)
            
        events = []
        for window in raw_data['windows']:
            event = EventData(
                event_id=window['event_id'],
                label=window['label'],
                game_time=window['game_time']
            )
            
            features = window['features']
            num_frames, num_objects, _ = features.shape
            
            for frame_idx in range(num_frames):
                frame_data = FrameData()
                
                for obj_idx in range(num_objects):
                    obj_data = {
                        'id': f"obj_{obj_idx}",
                        # NOTE: Our data has x-coordinates in the range of -52.5 to 52.5
                        # and y-coordinates in the range of -34.0 to 34.0.
                        # So we need to convert them to the range of 0 to 105 and 0 to 68.
                        # Make sure to see your own data to understand this.
                        'x': features[frame_idx, obj_idx, 0] + 52.5,
                        'y': features[frame_idx, obj_idx, 1] + 34.0,
                        'features': features[frame_idx, obj_idx, :].tolist(),
                    }
                    
                    if features[frame_idx, obj_idx, 2] == 1.0:
                        obj_data['type'] = 'ball'
                    elif features[frame_idx, obj_idx, 3] == 1.0:
                        obj_data['type'] = 'home'
                    elif features[frame_idx, obj_idx, 4] == 1.0:
                        obj_data['type'] = 'away'
                    else:
                        continue
                        
                    frame_data.objects.append(obj_data)
                    
                event.frames[frame_idx] = frame_data
                
            # Store original features for graph building
            event.metadata['features'] = features
            events.append(event)
            
        return {'default': events}
    
    def get_available_events(self, data):
        """Get unique event types."""
        
        event_types = {}
        for split, events in data.items():
            event_types[split] = list(set(e.label for e in events))
            
        return event_types
    
class DataReaderFactory:
    """
    Factory class for creating data readers.
    Extend this class to add support for new file types.
    """
    
    @staticmethod
    def create_reader(file_path):
        """Create a data reader based on the file extension."""
        if file_path.endswith('.pkl'):
            return PKLDataReader()
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        