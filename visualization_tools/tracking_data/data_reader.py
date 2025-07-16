from abc import ABC, abstractmethod
import pickle
import json
import bz2
from pathlib import Path

class FrameData:
    """Standardized frame data structure."""
    def __init__(self):
        self.objects = []
        

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
    def load_data(self, file_path, **kwargs):
        """Load data and return standardized format."""
        pass
    
    @abstractmethod
    def get_available_events(self, data):
        """Get available event types per split."""
        pass
    

class PKLDataReader(DataReader):
    """Reader for pickle files."""
    
    def load_data(self, file_path, **kwargs):
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
                
            event.metadata['features'] = features
            events.append(event)
            
        return {'default': events}
    
    def get_available_events(self, data):
        """Get unique event types."""
        event_types = {}
        for split, events in data.items():
            event_types[split] = list(set(e.label for e in events))
        return event_types
    
class JsonlBz2DataReader(DataReader):
    """Reader for FIFA World Cup 2022 tracking data (.jsonl.bz2 files)."""
    
    def __init__(self):
        self.annotations = None
        self.annotations_by_game = {}
    
    def load_data(self, file_path, **kwargs):
        """Load tracking data and create events based on annotations."""
        annotations_path = kwargs.get('annotations_path')
        
        if annotations_path and Path(annotations_path).exists():
            with open(annotations_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'videos' in data:
                    for video_info in data['videos']:
                        game_id = video_info['gameId']
                        self.annotations_by_game[game_id] = video_info['annotations']
        
        game_id = Path(file_path).stem.split('.')[0]
        
        frames_by_time = {}
        frame_count = 0
        
        print(f"Loading tracking data for game {game_id}...")
        with bz2.open(file_path, 'rt') as f:
            for line in f:
                frame = json.loads(line)
                time_ms = frame['videoTimeMs']
                frames_by_time[time_ms] = frame
                frame_count += 1
                
        print(f"Loaded {frame_count} frames")
        
        events = []
        if game_id in self.annotations_by_game:
            annotations = self.annotations_by_game[game_id]
            print(f"Found {len(annotations)} annotations for game {game_id}")
            
            for ann in annotations:
                event = self._create_event_from_annotation(ann, frames_by_time)
                if event:
                    events.append(event)
        else:
            print(f"No annotations found for game {game_id}, creating from tracking data")
            events = self._create_events_from_tracking(frames_by_time)
        
        return {'default': events}
    
    def _create_event_from_annotation(self, annotation, frames_by_time):
        """Create EventData from annotation and tracking frames."""
        event_id = str(annotation.get('gameEventId', annotation.get('possessionEventId', 'unknown')))
        label = annotation['label']
        time_ms = annotation['position']  
        
        event = EventData(
            event_id=event_id,
            label=label,
            game_time=annotation['gameTime']
        )
        
        window_ms = 2000
        start_time = max(0, time_ms - window_ms)
        end_time = time_ms + window_ms
        
        frame_idx = 0
        for frame_time in sorted(frames_by_time.keys()):
            if start_time <= frame_time <= end_time:
                frame = frames_by_time[frame_time]
                frame_data = self._extract_frame_data(frame)
                if frame_data:
                    event.frames[frame_idx] = frame_data
                    frame_idx += 1
        
        event.metadata['window_start_ms'] = start_time
        event.metadata['window_end_ms'] = end_time
        event.metadata['event_time_ms'] = time_ms
        event.metadata['num_frames'] = len(event.frames)
        
        return event if event.frames else None
    
    def _create_events_from_tracking(self, frames_by_time):
        """Create events from tracking data when no annotations available."""
        events = []
        event_frames = {}
        
        for time_ms, frame in frames_by_time.items():
            event_id = frame.get('game_event_id')
            if event_id:
                if event_id not in event_frames:
                    event_frames[event_id] = []
                event_frames[event_id].append((time_ms, frame))
        
        for event_id, frame_list in event_frames.items():
            if len(frame_list) < 10:
                continue
                
            frame_list.sort(key=lambda x: x[0])
            
            event = EventData(
                event_id=event_id,
                label="Unknown",
                game_time=f"{frame_list[0][0]}ms"
            )
            
            for idx, (time_ms, frame) in enumerate(frame_list):
                frame_data = self._extract_frame_data(frame)
                if frame_data:
                    event.frames[idx] = frame_data
            
            if event.frames:
                events.append(event)
        
        return events
    
    def _extract_frame_data(self, frame):
        """Extract standardized frame data from FIFA tracking frame."""
        frame_data = FrameData()
        
        ball = frame.get('ballsSmoothed', [])
        if ball:
            if ball.get('x') is not None and ball.get('y') is not None:
                ball_x = ball['x'] + 52.5
                ball_y = ball['y'] + 34.0
                
                frame_data.objects.append({
                    'id': 'ball',
                    'x': ball_x,
                    'y': ball_y,
                    'type': 'ball',
                    'features': [ball_x, ball_y, ball.get('z', 0)]
                })
        
        home_players = frame.get('homePlayersSmoothed', [])
        for i, player in enumerate(home_players):
            if player.get('x') is not None and player.get('y') is not None:
                player_x = player['x'] + 52.5
                player_y = player['y'] + 34.0
                
                frame_data.objects.append({
                    'id': f"home_{i}",
                    'x': player_x,
                    'y': player_y,
                    'type': 'home',
                    'jersey': int(player.get('jerseyNum', i + 1)),
                    'features': [player_x, player_y, player.get('z', 0)]
                })
        
        away_players = frame.get('awayPlayersSmoothed', [])
        for i, player in enumerate(away_players):
            if player.get('x') is not None and player.get('y') is not None:
                player_x = player['x'] + 52.5
                player_y = player['y'] + 34.0
                
                frame_data.objects.append({
                    'id': f"away_{i}",
                    'x': player_x,
                    'y': player_y,
                    'type': 'away',
                    'jersey': int(player.get('jerseyNum', i + 1)),
                    'features': [player_x, player_y, player.get('z', 0)]
                })
        
        return frame_data if frame_data.objects else None
    
    def get_available_events(self, data):
        """Get unique event types per split."""
        event_types = {}
        for split, events in data.items():
            event_types[split] = list(set(e.label for e in events))
        return event_types


class DataReaderFactory:
    """Factory class for creating data readers."""
    
    @staticmethod
    def create_reader(file_path, **kwargs):
        """Create a data reader based on the file extension."""
        if file_path.endswith('.pkl'):
            return PKLDataReader()
        elif file_path.endswith('.jsonl.bz2'):
            return JsonlBz2DataReader()
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        