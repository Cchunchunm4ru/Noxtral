"""
Robust Real-time Speaker Verifier Module

This module provides real-time speaker verification capabilities using the same
architecture and methods as comp.py but designed for real-time processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
import glob
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configuration constants
FS = 16000
EMBEDDING_DIM = 128
MODEL_CONFIDENCE_THRESHOLD = 0.65  # Lowered from 0.70 to be more permissive
FEATURE_SIMILARITY_THRESHOLD = 0.75  # Lowered from 0.90 to be more realistic

logger = logging.getLogger(__name__)

# --- FEATURE EXTRACTION (same as comp.py) ---

def normalize_features(features):
    """Normalizes a feature vector to have zero mean and unit variance."""
    return (features - np.mean(features)) / (np.std(features) + 1e-8)

def extract_combined_features(waveform, sr=FS):
    """
    Extracts a combination of voice biometric features to create a rich voiceprint.
    Features: MFCC, Spectral Contrast, Zero-Crossing Rate, and Pitch (F0).
    """
    if waveform.ndim > 1:
        waveform = waveform.flatten()

    # 1. MFCCs
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs, axis=1)

    # 2. Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr)
    contrast_processed = np.mean(spectral_contrast, axis=1)

    # 3. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=waveform)
    zcr_processed = np.mean(zcr, axis=1)

    # 4. Pitch (Fundamental Frequency - F0)
    pitch, _, _ = librosa.pyin(y=waveform, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    # Replace NaN (unvoiced frames) with 0
    pitch[np.isnan(pitch)] = 0
    pitch_processed = np.mean(pitch)
    # Ensure it's an array for concatenation
    pitch_processed = np.array([pitch_processed])

    # Combine all features into a single vector
    combined_features = np.concatenate((
        mfccs_processed,
        contrast_processed,
        zcr_processed,
        pitch_processed
    ))

    return combined_features

def extract_mfcc_for_model(waveform, sr=FS, n_mfcc=40):
    """Extracts ONLY MFCCs, specifically for the pre-trained model's input."""
    if waveform.ndim > 1:
        waveform = waveform.flatten()
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=128)
    return mfcc

def prepare_mfcc_tensor_for_model(waveform, device='cpu'):
    """Prepares a single waveform into a tensor suitable for the ResNet model."""
    mfcc = extract_mfcc_for_model(waveform)
    mfcc_normalized = normalize_features(mfcc)

    # Fixed length for your model
    fixed_len = 32  # Adjust based on your model's expected input
    if mfcc_normalized.shape[1] < fixed_len:
        mfcc_normalized = np.pad(mfcc_normalized, ((0, 0), (0, fixed_len - mfcc_normalized.shape[1])), mode='constant')
    else:
        mfcc_normalized = mfcc_normalized[:, :fixed_len]

    mfcc_tensor = torch.tensor(mfcc_normalized).unsqueeze(0).unsqueeze(0).float().to(device)
    return mfcc_tensor

# --- NEURAL NETWORK MODEL (same as comp.py) ---

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = torch.relu(out)
        return out

class AudioResNet(nn.Module):
    def __init__(self, num_classes):
        super(AudioResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Corrected ResNet-18 architecture with [2, 2, 2, 2] blocks
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# --- ROBUST REALTIME VERIFIER CLASS ---

class RobustRealtimeVerifier:
    """
    Robust real-time speaker verification system.
    
    This class provides real-time speaker verification using the same methodology
    as comp.py but designed for continuous audio processing.
    """
    
    def __init__(self, target_speaker_id, model_path, ref_path, device='cpu', 
                 model_confidence_threshold=MODEL_CONFIDENCE_THRESHOLD,
                 feature_similarity_threshold=FEATURE_SIMILARITY_THRESHOLD,
                 compatibility_mode=None):
        """
        Initialize the robust real-time verifier.
        
        Args:
            target_speaker_id: The ID of the target speaker to verify
            model_path: Path to the trained model file
            ref_path: Path to reference recordings directory
            device: Device to run inference on ('cpu' or 'cuda')
            model_confidence_threshold: Threshold for model confidence
            feature_similarity_threshold: Threshold for feature similarity
            compatibility_mode: Optional compatibility mode (e.g., 'comp_py')
        """
        self.target_speaker_id = target_speaker_id
        self.model_path = model_path
        self.ref_path = ref_path
        self.device = torch.device(device)
        self.model_confidence_threshold = max(model_confidence_threshold, 0.60)  # Minimum 60% for more permissive matching
        self.feature_similarity_threshold = max(feature_similarity_threshold, 0.70)  # Minimum 70% for realistic matching
        self.compatibility_mode = compatibility_mode
        
        # State tracking
        self.is_initialized = False
        self.model = None
        self.num_classes = None
        self.reference_features = {}
        
        # Initialize the verifier
        self._initialize()
    
    def _initialize(self):
        """Initialize the model and reference features."""
        try:
            # Load the model
            self.model, self.num_classes = self._load_model()
            if self.model is None:
                logger.error("[ROBUST_VERIFIER] Failed to load model")
                return
            
            # Load reference features
            self.reference_features = self._load_reference_features()
            
            if not self.reference_features:
                logger.warning("[ROBUST_VERIFIER] No reference features loaded - using model-only verification")
            
            self.is_initialized = True
            logger.info(f"[ROBUST_VERIFIER] ✅ Initialized for target: {self.target_speaker_id}")
            
        except Exception as e:
            logger.error(f"[ROBUST_VERIFIER] Initialization failed: {e}")
            self.is_initialized = False
    
    def _load_model(self):
        """Load the trained AudioWAV model."""
        try:
            logger.info(f"[ROBUST_VERIFIER] Loading model from {self.model_path}")
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # ── support nested 'state_dict' key ──
            raw_state = checkpoint.get('state_dict', checkpoint)
            
            new_state_dict = OrderedDict()
            for k, v in raw_state.items():
                name = k[7:] if k.startswith('resnet.') else k
                new_state_dict[name] = v
            
            # infer class count dynamically
            num_classes = new_state_dict.get('fc.weight').shape[0]
            logger.info(f"[ROBUST_VERIFIER] Model expects {num_classes} classes")
            
            model = AudioResNet(num_classes=num_classes)
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            if missing or unexpected:
                logger.warning(f"[ROBUST_VERIFIER] Loaded with missing keys: {missing}, unexpected keys: {unexpected}")
            model.to(self.device)
            model.eval()
            
            return model, num_classes
            
        except Exception as e:
            logger.error(f"[ROBUST_VERIFIER] Model loading failed: {e}")
            return None, None
    
    def _load_reference_features(self):
        """Load reference voiceprints from the recordings directory."""
        logger.info(f"[ROBUST_VERIFIER] Loading reference features from {self.ref_path}")
        reference_features = {}
        
        if not os.path.exists(self.ref_path):
            logger.error(f"[ROBUST_VERIFIER] Reference path not found: {self.ref_path}")
            return reference_features
        
        for wav_file in glob.glob(f"{self.ref_path}/*.wav"):
            try:
                filename = os.path.basename(wav_file)
                speaker_name = filename.replace('.wav', '')
                waveform, sr = librosa.load(wav_file, sr=FS)
                
                features = extract_combined_features(waveform, sr)
                features_normalized = normalize_features(features)
                
                reference_features[speaker_name] = features_normalized
                logger.info(f"[ROBUST_VERIFIER] Loaded reference for: {speaker_name}")
                
            except Exception as e:
                logger.warning(f"[ROBUST_VERIFIER] Failed to process {wav_file}: {e}")
        
        return reference_features
    
    def verify_speaker(self, waveform, sr=FS):
        """
        Verify if the given waveform matches the target speaker.
        
        Args:
            waveform: Audio waveform as numpy array
            sr: Sample rate (default: FS)
            
        Returns:
            dict: Verification result containing:
                - is_verified: bool
                - confidence: float
                - similarity: float
                - reason: str
                - best_match: str
        """
        if not self.is_initialized:
            return {
                'is_verified': False,
                'confidence': 0.0,
                'similarity': 0.0,
                'reason': 'NOT_INITIALIZED',
                'best_match': None
            }
        
        try:
            # Get model prediction
            predicted_class, model_confidence = self._identify_speaker_with_model(waveform)

            # Extract features for similarity comparison
            input_features = extract_combined_features(waveform, sr)
            input_features_normalized = normalize_features(input_features)
            
            # Find best matching reference
            best_match, best_similarity, all_similarities = self._find_best_reference_match(
                input_features_normalized
            )
            
            # Apply verification logic
            is_verified, similarity, reason = self._apply_verification_filters(
                best_match, model_confidence, input_features_normalized
            )
            
            return {
                'is_verified': is_verified,
                'confidence': model_confidence,
                'similarity': similarity,
                'reason': reason,
                'best_match': best_match,
                'all_similarities': all_similarities
            }
            
        except Exception as e:
            logger.error(f"[ROBUST_VERIFIER] Verification failed: {e}")
            return {
                'is_verified': False,
                'confidence': 0.0,
                'similarity': 0.0,
                'reason': 'ERROR',
                'best_match': None
            }
    
    def _identify_speaker_with_model(self, waveform):
        """Use the trained model to get speaker prediction with overfitting detection."""
        mfcc_tensor = prepare_mfcc_tensor_for_model(waveform, self.device)
        mfcc_tensor = F.interpolate(mfcc_tensor, size=(64, 64), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            logits = self.model(mfcc_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # STRICT OVERFITTING DETECTION
            # Get top predictions to check confidence distribution
            top5_values, top5_indices = torch.topk(probabilities, min(5, probabilities.shape[1]), dim=1)
            second_confidence = top5_values[0][1].item() if len(top5_values[0]) > 1 else 0.0
            confidence_gap = confidence.item() - second_confidence
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1).item()
            
            # Check for overfitting signs
            is_overfitted = False
            if confidence.item() > 0.98 or confidence.item() >= 0.999:  # More realistic threshold
                is_overfitted = True
                logger.warning(f"[ROBUST_VERIFIER] Overfitting detected: extremely high confidence ({confidence.item():.3f})")
            
            if confidence_gap > 0.70:  # More permissive gap threshold
                is_overfitted = True
                logger.warning(f"[ROBUST_VERIFIER] Overfitting detected: excessive confidence gap ({confidence_gap:.3f})")
            
            if entropy < 0.3:  # More realistic entropy threshold
                is_overfitted = True
                logger.warning(f"[ROBUST_VERIFIER] Overfitting detected: low entropy ({entropy:.3f})")
            
            # Apply moderate confidence reduction for overfitted predictions
            final_confidence = confidence.item()
            if is_overfitted:
                final_confidence = min(confidence.item() * 0.8, 0.85)  # Less aggressive reduction
                logger.warning(f"[ROBUST_VERIFIER] Reduced confidence: {confidence.item():.3f} → {final_confidence:.3f}")
        
        return predicted_class.item(), final_confidence
    
    def _find_best_reference_match(self, input_features):
        """Find the best matching reference speaker."""
        best_match = None
        best_similarity = -1.0
        similarities = {}
        
        # If no reference features, return None values
        if not self.reference_features:
            return None, 0.0, {}
        
        for speaker_name, ref_features in self.reference_features.items():
            similarity = cosine_similarity(
                input_features.reshape(1, -1), 
                ref_features.reshape(1, -1)
            )[0][0]
            similarities[speaker_name] = similarity
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_name
        
        return best_match, best_similarity, similarities
    
    def _apply_verification_filters(
        self,
        best_match_speaker,
        model_confidence,
        input_features
    ):
        # 1) reject if best‐reference isn’t the target
        if best_match_speaker != self.target_speaker_id:
            return False, 0.0, "PREDICTION_MISMATCH"

        # Check model confidence
        if model_confidence < self.model_confidence_threshold:
            return False, 0.0, "LOW_CONFIDENCE"
        
        # If no reference features available, accept based on model confidence alone
        if not self.reference_features:
            logger.info("[ROBUST_VERIFIER] No reference features - accepting based on model confidence alone")
            return True, model_confidence, "MODEL_ONLY_ACCEPTED"
        
        # Check if we have reference for the best match
        if best_match_speaker not in self.reference_features:
            return False, 0.0, "NO_REFERENCE"
        
        # Check feature similarity
        ref_features = self.reference_features[best_match_speaker]
        similarity = cosine_similarity(
            input_features.reshape(1, -1), 
            ref_features.reshape(1, -1)
        )[0][0]
        
        if similarity >= self.feature_similarity_threshold:
            return True, similarity, "ACCEPTED"
        else:
            return False, similarity, "LOW_SIMILARITY"
    
    def get_status(self):
        """Get the current status of the verifier."""
        return {
            'is_initialized': self.is_initialized,
            'target_speaker_id': self.target_speaker_id,
            'model_loaded': self.model is not None,
            'num_references': len(self.reference_features),
            'reference_speakers': list(self.reference_features.keys()),
            'model_confidence_threshold': self.model_confidence_threshold,
            'feature_similarity_threshold': self.feature_similarity_threshold
        }

    # --- Add this method to satisfy external `_report_status` calls ---
    def _report_status(self):
        """Alias for get_status to support external status reporting."""
        return self.get_status()
    
    def verify_frame_realtime(self, waveform):
        """
        Real-time frame verification method compatible with speaker_system.py.
        
        Args:
            waveform: Audio waveform as numpy array
            
        Returns:
            A verification result object with the following attributes:
                - is_verified: bool
                - identified_speaker: str
                - model_confidence: float
                - feature_similarity: float
                - verification_reason: str
                - filtered_audio: numpy array
        """
        # Use the main verify_speaker method
        result = self.verify_speaker(waveform)
        
        # Create a result object compatible with speaker_system expectations
        class VerificationResult:
            def __init__(self, result_dict, waveform):
                self.is_verified = result_dict['is_verified']
                self.identified_speaker = result_dict.get('best_match', 'unknown')
                self.model_confidence = result_dict['confidence']
                self.feature_similarity = result_dict['similarity']
                self.verification_reason = result_dict['reason']
                # If verified, return original audio; if not, return zeros
                if self.is_verified:
                    self.filtered_audio = waveform
                else:
                    self.filtered_audio = np.zeros_like(waveform)
        
        return VerificationResult(result, waveform)