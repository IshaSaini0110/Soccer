import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import os
from collections import defaultdict
import json

class PlayerTracker:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.players = {}  # Store player information
        self.next_player_id = 1
        self.frame_count = 0
        self.similarity_threshold = 0.7  # Adjust based on testing
        
    def extract_features(self, frame, bbox):
        """Extract features from player bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        player_crop = frame[y1:y2, x1:x2]
        
        # Simple feature extraction - you can improve this
        # Using color histogram and basic shape features
        features = []
        
        # Color histogram (BGR)
        for i in range(3):
            hist = cv2.calcHist([player_crop], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # Add basic shape features
        height, width = player_crop.shape[:2]
        aspect_ratio = width / height if height > 0 else 0
        features.append(aspect_ratio)
        
        return np.array(features)
    
    def find_matching_player(self, features, bbox):
        """Find the best matching existing player"""
        best_match_id = None
        best_similarity = 0
        
        current_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        for player_id, player_info in self.players.items():
            if player_info['active']:  # Only match with recently seen players
                # Calculate feature similarity
                similarity = cosine_similarity([features], [player_info['features']])[0][0]
                
                # Calculate position similarity (closer players more likely to match)
                last_center = player_info['last_center']
                distance = np.sqrt((current_center[0] - last_center[0])**2 + 
                                 (current_center[1] - last_center[1])**2)
                
                # Combine feature and position similarity
                position_weight = max(0, 1 - distance / 200)  # Adjust 200 based on frame size
                combined_score = similarity * 0.7 + position_weight * 0.3
                
                if combined_score > best_similarity and combined_score > self.similarity_threshold:
                    best_similarity = combined_score
                    best_match_id = player_id
        
        return best_match_id, best_similarity
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        
        # Run detection
        results = self.model(frame)
        
        # Mark all players as inactive initially
        for player_id in self.players:
            self.players[player_id]['active'] = False
        
        current_detections = []
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence > 0.5:  # Confidence threshold
                        # Extract features
                        features = self.extract_features(frame, bbox)
                        
                        # Try to match with existing player
                        matched_id, similarity = self.find_matching_player(features, bbox)
                        
                        if matched_id is not None:
                            # Update existing player
                            self.players[matched_id].update({
                                'features': features,
                                'bbox': bbox,
                                'last_center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                                'active': True,
                                'last_seen_frame': self.frame_count,
                                'confidence': confidence
                            })
                            current_detections.append((matched_id, bbox, confidence))
                        else:
                            # Create new player
                            new_id = self.next_player_id
                            self.next_player_id += 1
                            
                            self.players[new_id] = {
                                'features': features,
                                'bbox': bbox,
                                'last_center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                                'active': True,
                                'first_seen_frame': self.frame_count,
                                'last_seen_frame': self.frame_count,
                                'confidence': confidence
                            }
                            current_detections.append((new_id, bbox, confidence))
        
        # Remove players not seen for too long (optional)
        frames_to_keep = 30  # Adjust based on your needs
        players_to_remove = []
        for player_id, player_info in self.players.items():
            if self.frame_count - player_info['last_seen_frame'] > frames_to_keep:
                players_to_remove.append(player_id)
        
        for player_id in players_to_remove:
            del self.players[player_id]
        
        return current_detections
    
    def draw_tracking_results(self, frame, detections):
        """Draw bounding boxes and IDs on frame"""
        for player_id, bbox, confidence in detections:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw player ID
            label = f"Player {player_id}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw confidence
            conf_label = f"{confidence:.2f}"
            cv2.putText(frame, conf_label, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_tracking(self, output_path):
        """Main tracking loop"""
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {fps} FPS, {width}x{height}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            detections = self.process_frame(frame)
            
            # Draw results
            output_frame = self.draw_tracking_results(frame.copy(), detections)
            
            # Write frame
            out.write(output_frame)
            
            # Optional: Display frame (comment out for faster processing)
            cv2.imshow('Player Tracking', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            print(f"Frame {self.frame_count}: {len(detections)} players detected")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Tracking completed! Output saved to: {output_path}")
        print(f"Total unique players identified: {len(self.players)}")