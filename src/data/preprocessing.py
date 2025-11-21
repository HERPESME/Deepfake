"""
Data preprocessing pipeline for deepfake detection datasets.
Supports FaceForensics++, Celeb-DF, DFDC, and WildDeepfake datasets.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from tqdm import tqdm
import logging

# Face detection libraries
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: MTCNN not available. Install with: pip install mtcnn")

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Install with: pip install dlib")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection and cropping utility."""
    
    def __init__(self, method: str = "mtcnn", confidence_threshold: float = 0.9):
        """
        Initialize face detector.
        
        Args:
            method: Detection method ("mtcnn" or "dlib")
            confidence_threshold: Minimum confidence for face detection
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.detector = None
        
        if method == "mtcnn" and MTCNN_AVAILABLE:
            self.detector = MTCNN()
        elif method == "dlib" and DLIB_AVAILABLE:
            # Download dlib's face landmark predictor if not present
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(predictor_path):
                logger.warning("dlib predictor not found. Please download from: "
                             "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            self.detector = dlib.get_frontal_face_detector()
        else:
            logger.warning(f"Face detection method {method} not available. Using OpenCV.")
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        if self.method == "mtcnn" and MTCNN_AVAILABLE:
            detections = self.detector.detect_faces(image)
            faces = []
            for detection in detections:
                if detection['confidence'] >= self.confidence_threshold:
                    x, y, w, h = detection['box']
                    faces.append((x, y, w, h))
            return faces
        
        elif self.method == "dlib" and DLIB_AVAILABLE:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.detector(gray)
            faces = []
            for detection in detections:
                x, y, w, h = detection.left(), detection.top(), \
                            detection.width(), detection.height()
                faces.append((x, y, w, h))
            return faces
        
        else:  # OpenCV fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.1, 4)
            return [(x, y, w, h) for x, y, w, h in faces]
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                  padding: float = 0.2) -> np.ndarray:
        """
        Crop face from image with padding.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            padding: Padding factor around face
            
        Returns:
            Cropped face image
        """
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w_img, x + w + pad_w)
        y2 = min(h_img, y + h + pad_h)
        
        return image[y1:y2, x1:x2]


class VideoProcessor:
    """Video processing utilities."""
    
    @staticmethod
    def extract_frames(video_path: str, output_dir: str, 
                      frame_interval: int = 1, max_frames: Optional[int] = None) -> List[str]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            frame_interval: Extract every Nth frame
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frame paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return frame_paths


class DatasetPreprocessor:
    """Main preprocessing class for deepfake datasets."""
    
    def __init__(self, data_root: str, output_root: str, 
                 face_detector: Optional[FaceDetector] = None,
                 max_videos_per_class: Optional[int] = None,
                 frame_stride: int = 30,
                 max_frames_per_video: Optional[int] = 10):
        """
        Initialize preprocessor.
        
        Args:
            data_root: Root directory of raw datasets
            output_root: Root directory for processed data
            face_detector: Face detector instance
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.face_detector = face_detector or FaceDetector()
        # Subset controls to reduce disk usage
        self.max_videos_per_class = max_videos_per_class
        self.frame_stride = frame_stride
        self.max_frames_per_video = max_frames_per_video
        
        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "real").mkdir(exist_ok=True)
        (self.output_root / "fake").mkdir(exist_ok=True)
    
    def preprocess_faceforensics(self, dataset_path: str, 
                                compression: str = "c23") -> Dict[str, int]:
        """
        Preprocess FaceForensics++ dataset.
        
        Supports both standard structure (original_sequences/manipulated_sequences) 
        and flat structure (original/, Deepfakes/, Face2Face/, etc.)
        
        Args:
            dataset_path: Path to FaceForensics++ dataset
            compression: Compression level ("c23" or "c40") - ignored for flat structure
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"Preprocessing FaceForensics++ dataset")
        
        dataset_path = Path(dataset_path)
        stats = {"real": 0, "fake": 0, "total_frames": 0}
        
        # Check for flat structure first (Kaggle format)
        original_path_flat = dataset_path / "original"
        if original_path_flat.exists():
            logger.info("Detected flat folder structure (Kaggle format)")
            # Process original videos
            stats["real"] += self._process_video_folder(original_path_flat, "real")
            
            # Process all fake method folders
            fake_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", 
                          "FaceShifter", "DeepFakeDetection"]
            for method in fake_methods:
                method_path = dataset_path / method
                if method_path.exists():
                    logger.info(f"Processing {method} videos...")
                    stats["fake"] += self._process_video_folder(method_path, "fake")
        
        else:
            # Try standard structure
            logger.info("Trying standard folder structure")
            original_path = dataset_path / "original_sequences" / "youtube" / compression
            if original_path.exists():
                stats["real"] += self._process_video_folder(original_path, "real")
            
            # Process manipulated videos
            methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
            for method in methods:
                method_path = dataset_path / "manipulated_sequences" / method / compression
                if method_path.exists():
                    stats["fake"] += self._process_video_folder(method_path, "fake")
        
        stats["total_frames"] = stats["real"] + stats["fake"]
        logger.info(f"FaceForensics++ preprocessing complete: {stats}")
        return stats
    
    def preprocess_celebd(self, dataset_path: str) -> Dict[str, int]:
        """
        Preprocess Celeb-DF dataset.
        
        Args:
            dataset_path: Path to Celeb-DF dataset
            
        Returns:
            Statistics dictionary
        """
        logger.info("Preprocessing Celeb-DF dataset")
        
        dataset_path = Path(dataset_path)
        stats = {"real": 0, "fake": 0, "total_frames": 0}
        
        # Process real videos
        real_path = dataset_path / "Celeb-real"
        if real_path.exists():
            stats["real"] += self._process_video_folder(real_path, "real")
        
        # Process fake videos
        fake_path = dataset_path / "Celeb-synthesis"
        if fake_path.exists():
            stats["fake"] += self._process_video_folder(fake_path, "fake")
        
        stats["total_frames"] = stats["real"] + stats["fake"]
        logger.info(f"Celeb-DF preprocessing complete: {stats}")
        return stats
    
    def preprocess_dfdc(self, dataset_path: str, metadata_path: str) -> Dict[str, int]:
        """
        Preprocess DFDC dataset.
        
        Args:
            dataset_path: Path to DFDC dataset
            metadata_path: Path to metadata JSON file
            
        Returns:
            Statistics dictionary
        """
        logger.info("Preprocessing DFDC dataset")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        dataset_path = Path(dataset_path)
        stats = {"real": 0, "fake": 0, "total_frames": 0}
        
        for video_id, info in tqdm(metadata.items(), desc="Processing DFDC videos"):
            video_path = dataset_path / f"{video_id}.mp4"
            if not video_path.exists():
                continue
            
            label = "fake" if info.get("label") == "FAKE" else "real"
            frames_extracted = self._process_single_video(video_path, label, video_id)
            stats[label] += frames_extracted
        
        stats["total_frames"] = stats["real"] + stats["fake"]
        logger.info(f"DFDC preprocessing complete: {stats}")
        return stats
    
    def preprocess_wilddeepfake(self, dataset_path: str) -> Dict[str, int]:
        """
        Preprocess WildDeepfake dataset.
        
        Args:
            dataset_path: Path to WildDeepfake dataset
            
        Returns:
            Statistics dictionary
        """
        logger.info("Preprocessing WildDeepfake dataset")
        
        dataset_path = Path(dataset_path)
        stats = {"real": 0, "fake": 0, "total_frames": 0}
        
        # WildDeepfake is already preprocessed as face images
        real_path = dataset_path / "real"
        fake_path = dataset_path / "fake"
        
        if real_path.exists():
            stats["real"] = self._process_image_folder(real_path, "real")
        
        if fake_path.exists():
            stats["fake"] = self._process_image_folder(fake_path, "fake")
        
        stats["total_frames"] = stats["real"] + stats["fake"]
        logger.info(f"WildDeepfake preprocessing complete: {stats}")
        return stats
    
    def _process_video_folder(self, folder_path: Path, label: str) -> int:
        """Process all videos in a folder."""
        total_frames = 0
        video_files = list(folder_path.glob("*.mp4")) + list(folder_path.glob("*.avi"))
        # Apply subset limit per class if requested
        if self.max_videos_per_class is not None:
            video_files = video_files[: max(0, int(self.max_videos_per_class))]
        
        for video_path in tqdm(video_files, desc=f"Processing {label} videos"):
            frames_extracted = self._process_single_video(video_path, label)
            total_frames += frames_extracted
        
        return total_frames
    
    def _process_single_video(self, video_path: Path, label: str, 
                            video_id: Optional[str] = None) -> int:
        """Process a single video file."""
        # Use video stem as video_id if not provided
        if video_id is None:
            video_id = video_path.stem
        
        # Extract frames
        temp_dir = self.output_root / "temp" / str(video_path.stem)
        frame_paths = VideoProcessor.extract_frames(
            str(video_path), str(temp_dir), 
            frame_interval=int(self.frame_stride), 
            max_frames=self.max_frames_per_video
        )
        
        frames_extracted = 0
        for idx, frame_path in enumerate(frame_paths):
            if self._process_single_frame(frame_path, label, video_id, frame_idx=idx):
                frames_extracted += 1
        
        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return frames_extracted
    
    def _process_single_frame(self, frame_path: str, label: str, 
                            video_id: Optional[str] = None, frame_idx: int = 0) -> bool:
        """Process a single frame."""
        try:
            image = cv2.imread(frame_path)
            if image is None:
                return False
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            if not faces:
                return False
            
            # Use the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            cropped_face = self.face_detector.crop_face(image, largest_face)
            
            # Resize to standard size
            resized_face = cv2.resize(cropped_face, (224, 224))
            
            # Save processed face with unique filename using video_id and frame_idx
            video_name = str(video_id) if video_id else Path(frame_path).parent.name
            frame_base = Path(frame_path).stem
            filename = f"{video_name}_frame{frame_idx:06d}_{frame_base}.jpg"
            output_path = self.output_root / label / filename
            cv2.imwrite(str(output_path), resized_face)
            
            return True
            
        except Exception as e:
            logger.warning(f"Error processing frame {frame_path}: {e}")
            return False
    
    def _process_image_folder(self, folder_path: Path, label: str) -> int:
        """Process preprocessed face images."""
        image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"Processing {label} images"):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Resize to standard size
                resized_image = cv2.resize(image, (224, 224))
                
                # Save to output directory
                output_path = self.output_root / label / img_path.name
                cv2.imwrite(str(output_path), resized_image)
                
            except Exception as e:
                logger.warning(f"Error processing image {img_path}: {e}")
        
        return len(image_files)
    
    def create_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                     test_ratio: float = 0.1, random_seed: int = 42) -> Dict[str, List[str]]:
        """
        Create train/validation/test splits.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with split file lists
        """
        np.random.seed(random_seed)
        
        splits = {"train": [], "val": [], "test": []}
        
        for label in ["real", "fake"]:
            label_dir = self.output_root / label
            if not label_dir.exists():
                continue
            
            files = list(label_dir.glob("*.jpg"))
            np.random.shuffle(files)
            
            n_files = len(files)
            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)
            
            splits["train"].extend(files[:n_train])
            splits["val"].extend(files[n_train:n_train + n_val])
            splits["test"].extend(files[n_train + n_val:])
        
        # Save splits in the dataset-specific directory
        splits_dir = self.output_root / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        for split_name, files in splits.items():
            split_file = splits_dir / f"{split_name}.txt"
            with open(split_file, 'w') as f:
                for file_path in files:
                    # Write path relative to output_root (dataset directory)
                    rel_path = file_path.relative_to(self.output_root)
                    f.write(f"{rel_path}\n")
        
        logger.info(f"Created splits: {len(splits['train'])} train, "
                   f"{len(splits['val'])} val, {len(splits['test'])} test")
        
        return splits


def main():
    """Main preprocessing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess deepfake datasets")
    parser.add_argument("--dataset", required=True, 
                       choices=["faceforensics", "celebd", "dfdc", "wilddeepfake"],
                       help="Dataset to preprocess")
    parser.add_argument("--data_path", required=True, help="Path to raw dataset")
    parser.add_argument("--output_path", required=True, help="Output path for processed data")
    parser.add_argument("--metadata_path", help="Path to metadata file (for DFDC)")
    parser.add_argument("--compression", default="c23", help="Compression level (for FaceForensics++)")
    parser.add_argument("--face_detector", default="mtcnn", choices=["mtcnn", "dlib", "opencv"])
    # Subset controls to reduce disk usage
    parser.add_argument("--max_videos_per_class", type=int, default=None, help="Limit videos per class (real/fake)")
    parser.add_argument("--frame_stride", type=int, default=30, help="Extract every Nth frame from video")
    parser.add_argument("--max_frames_per_video", type=int, default=10, help="Cap frames extracted per video")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    face_detector = FaceDetector(method=args.face_detector)
    preprocessor = DatasetPreprocessor(
        args.data_path, args.output_path, face_detector,
        max_videos_per_class=args.max_videos_per_class,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
    )
    
    # Process dataset
    if args.dataset == "faceforensics":
        stats = preprocessor.preprocess_faceforensics(args.data_path, args.compression)
    elif args.dataset == "celebd":
        stats = preprocessor.preprocess_celebd(args.data_path)
    elif args.dataset == "dfdc":
        if not args.metadata_path:
            raise ValueError("Metadata path required for DFDC dataset")
        stats = preprocessor.preprocess_dfdc(args.data_path, args.metadata_path)
    elif args.dataset == "wilddeepfake":
        stats = preprocessor.preprocess_wilddeepfake(args.data_path)
    
    # Create splits
    preprocessor.create_splits()
    
    print(f"Preprocessing complete! Statistics: {stats}")


if __name__ == "__main__":
    main()
