# core/video_processor.py

import cv2
from typing import List, Tuple
import numpy as np
from pathlib import Path

class VideoProcessor:
    """
    Video processing extension for similarity search and duplicate detection
    """
    
    def __init__(self, keyframe_interval: int = 30):
        self.keyframe_interval = keyframe_interval
        self.feature_extractor = None
    
    def extract_keyframes(self, video_path: str, 
                         method: str = 'uniform') -> List[np.ndarray]:
        """
        Extract keyframes from video
        
        Methods:
        - 'uniform': Extract frames at regular intervals
        - 'scene': Extract frames at scene changes
        - 'motion': Extract frames with significant motion
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        if method == 'uniform':
            keyframes = self._extract_uniform_keyframes(cap)
        elif method == 'scene':
            keyframes = self._extract_scene_keyframes(cap)
        elif method == 'motion':
            keyframes = self._extract_motion_keyframes(cap)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        cap.release()
        return keyframes
    
    def _extract_uniform_keyframes(self, cap: cv2.VideoCapture) -> List[np.ndarray]:
        """Extract frames at uniform intervals"""
        keyframes = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.keyframe_interval == 0:
                keyframes.append(frame)
            
            frame_idx += 1
        
        return keyframes
    
    def _extract_scene_keyframes(self, cap: cv2.VideoCapture) -> List[np.ndarray]:
        """
        Extract keyframes at scene changes using histogram difference
        """
        keyframes = []
        prev_hist = None
        threshold = 0.7  # Scene change threshold
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Compute color histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, 
                               [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                # Compare histograms
                correlation = cv2.compareHist(prev_hist, hist, 
                                            cv2.HISTCMP_CORREL)
                
                # If correlation is low, it's likely a scene change
                if correlation < threshold:
                    keyframes.append(frame)
            else:
                # Always add first frame
                keyframes.append(frame)
            
            prev_hist = hist
        
        return keyframes
    
    def _extract_motion_keyframes(self, cap: cv2.VideoCapture) -> List[np.ndarray]:
        """
        Extract frames with significant motion using optical flow
        """
        keyframes = []
        prev_gray = None
        motion_threshold = 5.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Calculate motion magnitude
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion = np.mean(mag)
                
                # If significant motion detected
                if motion > motion_threshold:
                    keyframes.append(frame)
            else:
                keyframes.append(frame)
            
            prev_gray = gray
        
        return keyframes
    
    def compute_video_embedding(self, video_path: str) -> np.ndarray:
        """
        Compute aggregated embedding for entire video
        """
        from core.feature_extractors import CLIPFeatureExtractor
        
        if self.feature_extractor is None:
            self.feature_extractor = CLIPFeatureExtractor()
        
        # Extract keyframes
        keyframes = self.extract_keyframes(video_path, method='scene')
        
        # Extract features from each keyframe
        frame_embeddings = []
        for frame in keyframes:
            embedding = self.feature_extractor.extract(frame)
            frame_embeddings.append(embedding)
        
        # Aggregate embeddings (mean pooling)
        video_embedding = np.mean(frame_embeddings, axis=0)
        
        return video_embedding
    
    def find_similar_videos(self, query_video: str, 
                           video_database: List[str],
                           top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find videos similar to query video
        """
        # Extract query video embedding
        query_embedding = self.compute_video_embedding(query_video)
        
        # Compare with database videos
        similarities = []
        
        for db_video in video_database:
            try:
                db_embedding = self.compute_video_embedding(db_video)
                
                # Cosine similarity
                similarity = np.dot(query_embedding, db_embedding) / \
                           (np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))
                
                similarities.append((db_video, similarity))
            except Exception as e:
                print(f"Error processing {db_video}: {e}")
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def detect_duplicate_videos(self, video_paths: List[str]) -> Dict[str, List[str]]:
        """
        Detect duplicate or near-duplicate videos
        """
        # Extract embeddings for all videos
        video_embeddings = {}
        
        for video_path in video_paths:
            try:
                embedding = self.compute_video_embedding(video_path)
                video_embeddings[video_path] = embedding
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
        
        # Find duplicates using similarity threshold
        duplicates = {}
        processed = set()
        threshold = 0.95
        
        for video1, emb1 in video_embeddings.items():
            if video1 in processed:
                continue
            
            group = []
            processed.add(video1)
            
            for video2, emb2 in video_embeddings.items():
                if video2 in processed:
                    continue
                
                # Calculate similarity
                similarity = np.dot(emb1, emb2) / \
                           (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                if similarity >= threshold:
                    group.append(video2)
                    processed.add(video2)
            
            if group:
                duplicates[video1] = group
        
        return duplicates
    
    def extract_video_thumbnail(self, video_path: str, 
                               timestamp: float = None) -> np.ndarray:
        """
        Extract thumbnail from video at specific timestamp
        """
        cap = cv2.VideoCapture(video_path)
        
        if timestamp is not None:
            # Seek to timestamp
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        else:
            # Get middle frame
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None


class VideoSceneSegmentation:
    """
    Advanced scene segmentation for videos
    """
    
    def __init__(self):
        self.threshold = 30.0  # Scene change threshold
    
    def segment_scenes(self, video_path: str) -> List[Tuple[int, int]]:
        """
        Segment video into scenes
        
        Returns:
            List of (start_frame, end_frame) tuples for each scene
        """
        cap = cv2.VideoCapture(video_path)
        
        scene_boundaries = [0]  # Start with first frame
        prev_frame = None
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                # Compute frame difference
                diff = self._compute_frame_difference(prev_frame, frame)
                
                if diff > self.threshold:
                    scene_boundaries.append(frame_idx)
            
            prev_frame = frame
            frame_idx += 1
        
        # Add final frame
        scene_boundaries.append(frame_idx)
        
        cap.release()
        
        # Create scene segments
        scenes = []
        for i in range(len(scene_boundaries) - 1):
            scenes.append((scene_boundaries[i], scene_boundaries[i + 1]))
        
        return scenes
    
    def _compute_frame_difference(self, frame1: np.ndarray, 
                                 frame2: np.ndarray) -> float:
        """Compute difference between consecutive frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Return mean difference
        return np.mean(diff)