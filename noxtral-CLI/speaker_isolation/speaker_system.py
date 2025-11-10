#!/usr/bin/env python3
"""
Advanced Speaker Identification and Audio Filtering System

This module provides comprehensive speaker verification, identification, and audio filtering
capabilities for real-time voice applications. It includes:

1. Speaker Recognition and Verification using an ECAPA-TDNN model
2. Audio Quality Assessment and Filtering
3. Real-time Audio Processing
4. Feature Extraction and Matching
5. Neural Network Model Integration
6. Robust Audio Diarization
"""
import os
import uuid
import torch 
import librosa 
import numpy as np
import glob
import logging
import asyncio
import time

try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        def mock_list_audio_backends():
            return ['soundfile']
        torchaudio.list_audio_backends = mock_list_audio_backends
        
    if not hasattr(torchaudio, 'get_audio_backend'):
        def mock_get_audio_backend():
            return 'soundfile'
        torchaudio.get_audio_backend = mock_get_audio_backend
except ImportError:
    pass

from typing import List, Dict, Any, Optional, Tuple
from scipy.ndimage import uniform_filter, gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity

try:
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.inference.diarization import SpeakerDiarization as Diarization
except ImportError:
    try:
        from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
        SpeakerRecognition = None
    Diarization = None

import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import requests  # if you use HTTP for Gemini
from collections import defaultdict  # Add this missing import
from F_def.capture import Frames, FrameCapture,AudioFrame,AudioRawFrame
from F_def.frame import FrameProcessor,Frame
from STT.frame import TranscriptionFrame,InterruptionFrame
f = Frame(type="system")

try:
    from robust_realtime_verifier import RobustRealtimeVerifier
except ImportError:
    RobustRealtimeVerifier = None

# ─── Ensure AudioRawFrame always has id & transport_destination ───
_orig_ar_init = AudioRawFrame.__init__
def _safe_audioraw_init(self, *args, **kwargs):
    _orig_ar_init(self, *args, **kwargs)
    if not hasattr(self, 'id') or self.id is None:
        self.id = str(uuid.uuid4())
    if not hasattr(self, 'transport_destination'):
        self.transport_destination = None
AudioRawFrame.__init__ = _safe_audioraw_init

logger = logging.getLogger(__name__)

# Audio processing constants (from cap.py)
fs = 16000
n_fft = 512
hop_length = 256
min_db = -80
max_db = -20
RECORDINGS_REF_PATH = r"C:\Users\Admin\Desktop\noxtral\captured_audio"
MODEL_CONFIDENCE_THRESHOLD = 0.50  
FEATURE_SIMILARITY_THRESHOLD = 0.50
EMBEDDING_SIMILARITY_THRESHOLD = 0.50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RECOGNITION_THRESHOLD = 0.75
EMBEDDING_DIM = 128
MARGIN = 0.4

_global_verifier = None  
_speaker_system_initialized = False

CAPTURE_MIN_DB = -75
CAPTURE_MAX_DB = -5
DB_TOLERANCE = 5

ENABLE_TEST_TIME_AUGMENTATION = False  
ENABLE_ENSEMBLE_VERIFICATION = False   
USE_PREPROCESSING_VARIANTS = False   

AUGMENTATION_SPEED_FACTORS = [0.9, 1.0, 1.1]  
AUGMENTATION_VOLUME_FACTORS = [0.8, 1.0, 1.2]  
AUGMENTATION_NOISE_LEVELS = [0.001, 0.003, 0.005]  
AUGMENTATION_WEIGHTS = [0.4, 1.0, 0.4]  

try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.warning("[AUGMENTATION] torchaudio not available, using basic augmentation only")

def extract_mfcc(waveform, sr=16000, n_mfcc=40, pad_frames=20):
    """Legacy function name compatibility - calls extract_mfcc_from_waveform"""
    return extract_mfcc_from_waveform(waveform, sr, n_mfcc, pad_frames)

def extract_mfcc_from_waveform(waveform, sr=16000, n_mfcc=40, pad_frames=20):
    """Enhanced MFCC extraction with padding"""
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    if mfcc.shape[1] < pad_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_frames - mfcc.shape[1])), mode='constant')
    return mfcc

def wiener_custom(y, mysize=3, noise_factor=5.0):
    """Enhanced Wiener filter from cap.py"""
    if len(y) == 0:
        return y
    try:
        mean_val = np.mean(y)
        variance = np.var(y)
        if variance == 0:
            return y - mean_val
        noise_variance = variance / noise_factor
        wiener_gain = variance / (variance + noise_variance)
        return (y - mean_val) * wiener_gain + mean_val
    except Exception as e:
        logger.warning(f"Wiener filter failed: {e}, returning original signal")
        return y

def calculate_audio_db(waveform):
    """Calculate dB level of audio waveform"""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    rms = np.sqrt(np.mean(waveform**2))
    if rms > 0:
        db_level = 20 * np.log10(rms)
    else:
        db_level = min_db

    return max(db_level, min_db)

def should_capture_audio(db_level):
    """Determine if audio should be captured based on dB level (with tolerance)."""
    lower = CAPTURE_MIN_DB - DB_TOLERANCE
    upper = CAPTURE_MAX_DB + DB_TOLERANCE
    return lower <= db_level <= upper

def augment_input_audio_for_inference(waveform, sample_rate=fs):
    """
    Generate multiple augmented versions of input audio for test-time augmentation.
    Returns a list of (augmented_waveform, weight) tuples.
    """
    if not ENABLE_TEST_TIME_AUGMENTATION:
        return [(waveform, 1.0)]
    
    augmented_samples = []
    
    try:
        augmented_samples.append((waveform.copy(), 1.0))

        for i, speed_factor in enumerate(AUGMENTATION_SPEED_FACTORS):
            if speed_factor != 1.0:  
                try:
                    if TORCHAUDIO_AVAILABLE:
                        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
                        speed_perturb = T.SpeedPerturbation(sample_rate, [speed_factor])
                        augmented_tensor, _ = speed_perturb(waveform_tensor.unsqueeze(0))
                        augmented_waveform = augmented_tensor.squeeze(0).numpy()
                    else:
                        target_length = int(len(waveform) / speed_factor)
                        augmented_waveform = np.interp(
                            np.linspace(0, len(waveform), target_length),
                            np.arange(len(waveform)),
                            waveform
                        )
                    
                    weight = AUGMENTATION_WEIGHTS[i] if i < len(AUGMENTATION_WEIGHTS) else 0.3
                    augmented_samples.append((augmented_waveform, weight))
                except Exception as e:
                    logger.warning(f"[AUGMENTATION] Speed perturbation {speed_factor} failed: {e}")
        
        for volume_factor in AUGMENTATION_VOLUME_FACTORS:
            if volume_factor != 1.0: 
                try:
                    augmented_waveform = waveform * volume_factor
                    augmented_waveform = np.clip(augmented_waveform, -1.0, 1.0)
                    augmented_samples.append((augmented_waveform, 0.4))
                except Exception as e:
                    logger.warning(f"[AUGMENTATION] Volume perturbation {volume_factor} failed: {e}")
                    
        for noise_level in AUGMENTATION_NOISE_LEVELS:
            try:
                noise = np.random.normal(0, noise_level, size=waveform.shape)
                augmented_waveform = waveform + noise
                augmented_waveform = np.clip(augmented_waveform, -1.0, 1.0)
                augmented_samples.append((augmented_waveform, 0.3))
            except Exception as e:
                logger.warning(f"[AUGMENTATION] Noise addition {noise_level} failed: {e}")
        if TORCHAUDIO_AVAILABLE:
            try:
                waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
                for pitch_shift in [-2, 2]:  # semitones
                    try:
                        pitch_shifter = T.PitchShift(sample_rate, n_steps=pitch_shift)
                        augmented_tensor = pitch_shifter(waveform_tensor)
                        augmented_waveform = augmented_tensor.squeeze(0).numpy()
                        augmented_samples.append((augmented_waveform, 0.3))
                    except Exception as e:
                        logger.warning(f"[AUGMENTATION] Pitch shift {pitch_shift} failed: {e}")
                
                for stretch_factor in [0.95, 1.05]:
                    try:
                        time_stretch = T.TimeStretch(hop_length=hop_length, n_freq=n_fft//2 + 1)
                        spec_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
                        ispec_transform = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
                        
                        spec = spec_transform(waveform_tensor)
                        stretched_spec = time_stretch(spec, stretch_factor)
                        augmented_tensor = ispec_transform(stretched_spec)
                        augmented_waveform = augmented_tensor.squeeze(0).numpy()
                        augmented_samples.append((augmented_waveform, 0.2))
                    except Exception as e:
                        logger.warning(f"[AUGMENTATION] Time stretch {stretch_factor} failed: {e}")
                        
            except Exception as e:
                logger.warning(f"[AUGMENTATION] Advanced torchaudio augmentations failed: {e}")
        
       
        if USE_PREPROCESSING_VARIANTS:
            try:
                for noise_factor in [3.0, 5.0, 10.0]:
                    try:
                        augmented_waveform = wiener_custom(waveform, noise_factor=noise_factor)
                        augmented_samples.append((augmented_waveform, 0.3))
                    except Exception as e:
                        logger.warning(f"[AUGMENTATION] Wiener filtering {noise_factor} failed: {e}")
                
                for sigma in [0.5, 1.0, 2.0]:
                    try:
                        augmented_waveform = gaussian_filter1d(waveform, sigma=sigma)
                        augmented_samples.append((augmented_waveform, 0.2))
                    except Exception as e:
                        logger.warning(f"[AUGMENTATION] Gaussian smoothing {sigma} failed: {e}")
               
                for target_rms in [0.1, 0.2, 0.3]:
                    try:
                        current_rms = np.sqrt(np.mean(waveform**2))
                        if current_rms > 0:
                            scale_factor = target_rms / current_rms
                            augmented_waveform = waveform * scale_factor
                            augmented_waveform = np.clip(augmented_waveform, -1.0, 1.0)
                            augmented_samples.append((augmented_waveform, 0.2))
                    except Exception as e:
                        logger.warning(f"[AUGMENTATION] RMS normalization {target_rms} failed: {e}")
                        
            except Exception as e:
                logger.warning(f"[AUGMENTATION] Preprocessing variants failed: {e}")
        
        logger.info(f"[AUGMENTATION] Generated {len(augmented_samples)} augmented samples")
        return augmented_samples
        
    except Exception as e:
        logger.error(f"[AUGMENTATION] Augmentation failed, returning original: {e}")
        return [(waveform, 1.0)]

def extract_augmented_speaker_embedding(waveform, model, sample_rate=fs):
    """
    Extract speaker embedding using test-time augmentation.
    Returns weighted average of embeddings from multiple augmented versions.
    """
    if not ENABLE_TEST_TIME_AUGMENTATION:
        return get_embedding_from_waveform(waveform, model)
    
    try:
        augmented_samples = augment_input_audio_for_inference(waveform, sample_rate)
        
        embeddings = []
        weights = []
        
        for aug_waveform, weight in augmented_samples:
            try:
                embedding = get_embedding_from_waveform(aug_waveform, model)
                if embedding is not None:
                    embeddings.append(embedding)
                    weights.append(weight)
            except Exception as e:
                logger.warning(f"[AUGMENTATION] Failed to extract embedding from augmented sample: {e}")
        
        if not embeddings:
            logger.warning("[AUGMENTATION] No embeddings extracted, falling back to standard method")
            return get_embedding_from_waveform(waveform, model)
        
        embeddings = np.array(embeddings)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        weighted_embedding = np.average(embeddings, axis=0, weights=weights)
        norm = np.linalg.norm(weighted_embedding)
        if norm > 0:
            weighted_embedding = weighted_embedding / norm
        
        logger.info(f"[AUGMENTATION] Combined {len(embeddings)} embeddings with TTA")
        return weighted_embedding
        
    except Exception as e:
        logger.error(f"[AUGMENTATION] Augmented embedding extraction failed: {e}")
        return get_embedding_from_waveform(waveform, model)

def get_embedding_from_waveform_augmented(waveform, model):
    """
    Main function for augmented embedding extraction with fallback to standard extraction.
    """
    try:
        if ENABLE_TEST_TIME_AUGMENTATION:
            return extract_augmented_speaker_embedding(waveform, model)
        else:
            return get_embedding_from_waveform(waveform, model)
    except Exception as e:
        logger.error(f"[AUGMENTATION] Augmented extraction failed, using standard: {e}")
        return get_embedding_from_waveform(waveform, model)

def ensemble_speaker_verification(waveform, reference_embeddings, model, target_speaker_id=None):
    """
    Advanced verification using multiple prediction strategies and ensemble methods.
    """
    if not ENABLE_ENSEMBLE_VERIFICATION:
        # Fallback to standard verification
        embedding = get_embedding_from_waveform(waveform, model)
        if embedding is None:
            return {"verified": False, "confidence": 0.0, "method": "standard_failed"}
        
        # Compare against references
        if target_speaker_id and target_speaker_id in reference_embeddings:
            similarity = cosine_similarity([embedding], [reference_embeddings[target_speaker_id]])[0][0]
            return {
                "verified": similarity >= EMBEDDING_SIMILARITY_THRESHOLD,
                "confidence": similarity,
                "method": "standard"
            }
        
        return {"verified": False, "confidence": 0.0, "method": "standard_no_reference"}
    
    try:
        verification_results = []

        try:
            embedding_standard = get_embedding_from_waveform(waveform, model)
            if embedding_standard is not None and target_speaker_id in reference_embeddings:
                similarity = cosine_similarity([embedding_standard], [reference_embeddings[target_speaker_id]])[0][0]
                verification_results.append({
                    "method": "standard",
                    "verified": similarity >= EMBEDDING_SIMILARITY_THRESHOLD,
                    "confidence": similarity,
                    "weight": 0.3
                })
        except Exception as e:
            logger.warning(f"[ENSEMBLE] Standard method failed: {e}")
        
        try:
            embedding_augmented = extract_augmented_speaker_embedding(waveform, model)
            if embedding_augmented is not None and target_speaker_id in reference_embeddings:
                similarity = cosine_similarity([embedding_augmented], [reference_embeddings[target_speaker_id]])[0][0]
                verification_results.append({
                    "method": "augmented",
                    "verified": similarity >= EMBEDDING_SIMILARITY_THRESHOLD,
                    "confidence": similarity,
                    "weight": 0.5 
                })
        except Exception as e:
            logger.warning(f"[ENSEMBLE] Augmented method failed: {e}")
        
        if USE_PREPROCESSING_VARIANTS:
            preprocessing_methods = [
                ("wiener", lambda w: wiener_custom(w, noise_factor=5.0)),
                ("gaussian", lambda w: gaussian_filter1d(w, sigma=1.0)),
                ("volume_norm", lambda w: w * (0.2 / max(np.sqrt(np.mean(w**2)), 1e-8)))
            ]
            
            for method_name, preprocessor in preprocessing_methods:
                try:
                    preprocessed_waveform = preprocessor(waveform.copy())
                    embedding_preprocessed = get_embedding_from_waveform(preprocessed_waveform, model)
                    if embedding_preprocessed is not None and target_speaker_id in reference_embeddings:
                        similarity = cosine_similarity([embedding_preprocessed], [reference_embeddings[target_speaker_id]])[0][0]
                        verification_results.append({
                            "method": f"preprocessed_{method_name}",
                            "verified": similarity >= EMBEDDING_SIMILARITY_THRESHOLD,
                            "confidence": similarity,
                            "weight": 0.2
                        })
                except Exception as e:
                    logger.warning(f"[ENSEMBLE] Preprocessing method {method_name} failed: {e}")
        
        if not verification_results:
            logger.warning("[ENSEMBLE] No verification methods succeeded")
            return {"verified": False, "confidence": 0.0, "method": "ensemble_failed"}
       
        total_weight = sum(result["weight"] for result in verification_results)
        verified_weight = sum(result["weight"] for result in verification_results if result["verified"])
        weighted_confidence = sum(result["confidence"] * result["weight"] for result in verification_results) / total_weight
        
        verification_ratio = verified_weight / total_weight
        final_verified = verification_ratio >= 0.5  #
        
        logger.info(f"[ENSEMBLE] Used {len(verification_results)} methods, ratio: {verification_ratio:.3f}, confidence: {weighted_confidence:.3f}")
        
        return {
            "verified": final_verified,
            "confidence": weighted_confidence,
            "method": "ensemble",
            "details": {
                "methods_used": len(verification_results),
                "verification_ratio": verification_ratio,
                "individual_results": verification_results
            }
        }
        
    except Exception as e:
        logger.error(f"[ENSEMBLE] Ensemble verification failed: {e}")
        # Final fallback to standard method
        try:
            embedding = get_embedding_from_waveform(waveform, model)
            if embedding is not None and target_speaker_id in reference_embeddings:
                similarity = cosine_similarity([embedding], [reference_embeddings[target_speaker_id]])[0][0]
                return {
                    "verified": similarity >= EMBEDDING_SIMILARITY_THRESHOLD,
                    "confidence": similarity,
                    "method": "fallback_standard"
                }
        except:
            pass
        
        return {"verified": False, "confidence": 0.0, "method": "ensemble_total_failure"}


class TDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TDNNLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class AttentiveStatisticalPooling(nn.Module):
    """Attentive Statistical Pooling as in ECAPA-TDNN."""
    def __init__(self, in_dim, bottleneck_dim):
        super(AttentiveStatisticalPooling, self).__init__()
        self.tdnn = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.attention = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        t = self.relu(self.tdnn(x))
        e = self.attention(t)
        alpha = self.softmax(e)

        mean = torch.sum(alpha * x, dim=2, keepdim=True)
        std = torch.sqrt(torch.sum(alpha * (x - mean)**2, dim=2, keepdim=True) + 1e-6)
        
        stat_pool = torch.cat((mean, std), dim=1)
        return stat_pool.squeeze(dim=2)

class ECAPA_TDNN(nn.Module):
    """
    Enhanced Channel-dependent Attention and Parameter-Averaged TDNN model.
    This model is used to extract fixed-length speaker embeddings.
    """
    def __init__(self, input_dim=40, output_dim=192):
        super(ECAPA_TDNN, self).__init__()
        
        self.layer1 = TDNNLayer(input_dim, 512, kernel_size=5, dilation=1)
        
        self.layer2 = nn.Sequential(
            TDNNLayer(512, 512, kernel_size=3, dilation=2),
            SELayer(512),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        self.layer3 = nn.Sequential(
            TDNNLayer(512, 512, kernel_size=3, dilation=3),
            SELayer(512),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        self.layer4 = nn.Sequential(
            TDNNLayer(512, 512, kernel_size=3, dilation=4),
            SELayer(512),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        self.layer5 = nn.Sequential(
            TDNNLayer(1536, 1536, kernel_size=1, dilation=1)
        )
        
        self.pooling = AttentiveStatisticalPooling(1536, 128)
        
        self.bn_before_fc = nn.BatchNorm1d(1536 * 2)
        
        self.fc = nn.Linear(1536 * 2, output_dim)
        
    def forward(self, x):
        x = x.squeeze(1)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x1)
        x4 = self.layer4(x1)
        
        x = torch.cat((x2, x3, x4), dim=1)
        x = self.layer5(x)
        
        x = self.pooling(x)
        
        x = self.bn_before_fc(x)
        
        embedding = self.fc(x)
        
        return F.normalize(embedding, p=2, dim=1)

def normalize_features(features):
    """Normalize features to zero mean and unit variance."""
    if features.ndim == 1:
        return (features - np.mean(features)) / (np.std(features) + 1e-8)
    else:
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        return (features - mean) / (std + 1e-8)


def extract_combined_features(waveform, sr=fs):
    """Extract combined voice biometric features."""
    if len(waveform) == 0:
        return np.zeros(40)  

    try:
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception as e:
        logger.warning(f"Feature extraction failed: {e}")
        return np.zeros(40)


def prepare_mfcc_tensor_for_model(waveform, device='cpu'):
    """Prepares a single waveform into a tensor suitable for the ECAPA-TDNN model."""
    try:
        if len(waveform) == 0:
            waveform = np.zeros(fs)
        waveform_filtered = wiener_custom(waveform)
        waveform_final = gaussian_filter1d(waveform_filtered, sigma=1.0)
        mfcc = librosa.feature.mfcc(
            y=waveform_final, 
            sr=fs, 
            n_mfcc=40,
            n_fft=n_fft, 
            hop_length=hop_length
        )

        mfcc_normalized = normalize_features(mfcc)
        mfcc_tensor = torch.tensor(mfcc_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        logger.info(f"[MFCC_TENSOR] Input shape: {mfcc_tensor.shape}, dtype: {mfcc_tensor.dtype}")
        return mfcc_tensor
        
    except Exception as e:
        logger.error(f"[MFCC_TENSOR] Preparation failed: {e}")
        fallback = torch.zeros(1, 40, 100, dtype=torch.float32).to(device)
        return fallback

def load_recordings_reference_features():
    """Load reference voiceprints from the recordings directory with comprehensive augmentation."""
    logger.info(f"[SPEAKER_SYSTEM] Loading reference features from {RECORDINGS_REF_PATH}")
    reference_features = {}
    if not os.path.exists(RECORDINGS_REF_PATH):
        logger.warning(f"[SPEAKER_SYSTEM] Recordings directory not found: {RECORDINGS_REF_PATH}")
        return reference_features

    for wav_file in glob.glob(f"{RECORDINGS_REF_PATH}/*.wav"):
        try:
            speaker_name = os.path.splitext(os.path.basename(wav_file))[0]
            waveform, sr = librosa.load(wav_file, sr=fs)
            if len(waveform) > 0:
                logger.info(f"[SPEAKER_SYSTEM] Applying comprehensive augmentation to reference '{speaker_name}'")
                
                if ENABLE_TEST_TIME_AUGMENTATION:
                    augmented_samples = augment_input_audio_for_inference(waveform, sr)
                    augmented_features = []
                    for aug_waveform, weight in augmented_samples:
                        try:
                            proc = wiener_custom(aug_waveform)
                            noise = np.random.normal(0, 0.001, size=proc.shape)
                            proc = proc + noise
                            features = extract_combined_features(proc, sr)
                            augmented_features.append((features, weight))
                        except Exception as e:
                            logger.warning(f"[SPEAKER_SYSTEM] Failed to extract features from augmented sample: {e}")
                    
                    if augmented_features:
                        features_arrays = [f[0] for f in augmented_features]
                        weights = [f[1] for f in augmented_features]
                        weights = np.array(weights)
                        weights = weights / np.sum(weights)
                        combined_features = np.average(features_arrays, axis=0, weights=weights)
                        reference_features[speaker_name] = combined_features
                        
                        logger.info(f"[SPEAKER_SYSTEM] Loaded & augmented reference for '{speaker_name}': "
                                  f"{len(waveform)} samples, {len(augmented_features)} augmented variants")
                    else:
                        proc = wiener_custom(waveform)
                        proc = gaussian_filter1d(proc, sigma=1.0)
                        noise = np.random.normal(0, 0.001, size=proc.shape)
                        proc = proc + noise
                        features = extract_combined_features(proc, sr)
                        reference_features[speaker_name] = features
                        logger.info(f"[SPEAKER_SYSTEM] Loaded reference for '{speaker_name}' with basic augmentation")
                else:
                    proc = wiener_custom(waveform)
                    proc = gaussian_filter1d(proc, sigma=1.0)
                    noise = np.random.normal(0, 0.001, size=proc.shape)
                    proc = proc + noise
                    features = extract_combined_features(proc, sr)
                    reference_features[speaker_name] = features
                    logger.info(f"[SPEAKER_SYSTEM] Loaded reference for '{speaker_name}' with basic augmentation")
                    
        except Exception as e:
            logger.error(f"[SPEAKER_SYSTEM] Failed to load {wav_file}: {e}")

    logger.info(f"[SPEAKER_SYSTEM] Successfully loaded {len(reference_features)} reference speakers with augmentation")
    return reference_features


def identify_speaker_with_model(waveform, model):
    """
    This function is no longer a classification task. It's now for internal analysis.
    The ECAPA-TDNN model is for embedding extraction, not classification.
    We return a placeholder result to maintain compatibility with the calling code.
    """
    if model is None:
        logger.warning("[MODEL_INFERENCE] Model is None")
        return -1, 0.0
        
    try:
        model.eval()
        with torch.no_grad():
            confidence = 0.5  
            predicted_idx = -1 # No classification index
            
            return predicted_idx, confidence
            
    except Exception as e:
        logger.error(f"[MODEL_INFERENCE] Identification failed: {e}", exc_info=True)
        return -1, 0.0


def compute_feature_similarity(vec1, vec2):
    """Compute cosine similarity between two feature vectors."""
    try:
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    except:
        return 0.0


def extract_speaker_embedding(waveform, model):
    """Extracts speaker embedding from waveform using the trained ECAPA-TDNN model."""
    try:
        mfcc_tensor = prepare_mfcc_tensor_for_model(waveform, DEVICE)
        with torch.no_grad():
            embedding = model(mfcc_tensor)
            embedding = embedding.cpu().numpy().flatten()
            return embedding
    except Exception as e:
        logger.error(f"[EMBEDDING] Extraction failed: {e}")
        return None


def find_best_reference_match(input_features, reference_features):
    """Find the best matching reference speaker based on combined feature similarity."""
    best_match = None
    best_similarity = -1.0
    similarities = {}

    try:
        for speaker_name, ref_features in reference_features.items():
            similarity = compute_feature_similarity(input_features, ref_features)
            similarities[speaker_name] = similarity
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_name

        return best_match, best_similarity, similarities

    except Exception as e:
        logger.error(f"[SPEAKER_SYSTEM] Reference matching failed: {e}")
        return None, 0.0, {}

def set_rms_to_zero(waveform):
    """Zeros out the waveform to silence it (diarization effect)."""
    return np.zeros_like(waveform)

def filter_audio_by_confidence_and_similarity(waveform, best_match_speaker=None, reference_features=None, model_confidence=0.0, model=None, mfcc=None):
    """Filter audio by running the simple verifier and silencing non-verified speaker."""
    global _global_verifier
    if not _speaker_system_initialized or _global_verifier is None:
        logger.warning("[FILTER] Verifier not ready, silencing audio")
        return set_rms_to_zero(waveform)
    try:
        result = _global_verifier.verify_speaker(waveform)
        if result.get('is_verified', False):
            logger.info(f"[FILTER] ✅ Audio ACCEPTED: {result.get('identified_speaker')} "
                       f"(conf: {result.get('model_confidence', 0):.2f}, sim: {result.get('feature_similarity', 0):.2f})")
            return waveform
        else:
            logger.info(f"[FILTER] REJECTED: {result.get('identified_speaker', 'unknown')} ")
            return set_rms_to_zero(waveform)
            
    except Exception as e:
        logger.error(f"[FILTER] Verification failed: {e}")
        return set_rms_to_zero(waveform)


def compute_euclidean_distance(embedding1, embedding2):
    """Compute Euclidean distance between two speaker embeddings."""
    if embedding1 is None or embedding2 is None:
        return float('inf')
    try:
        return np.linalg.norm(embedding1 - embedding2)
    except Exception as e:
        logger.error(f"[DISTANCE] Computation failed: {e}")
        return float('inf')


def load_reference_embeddings():
    """Load reference speaker embeddings using the trained model with comprehensive augmentation."""
    logger.info(f"[SPEAKER_SYSTEM] Loading reference embeddings from {RECORDINGS_REF_PATH}")
    
    try:
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        model.to(DEVICE)
    except Exception as e:
        logger.error(f"[SPEAKER_SYSTEM] Failed to load SpeechBrain model: {e}")
        return {}

    reference_embeddings = {}
    if not os.path.exists(RECORDINGS_REF_PATH):
        logger.warning(f"[SPEAKER_SYSTEM] Recordings directory not found: {RECORDINGS_REF_PATH}")
        return reference_embeddings

    for wav_file in glob.glob(f"{RECORDINGS_REF_PATH}/*.wav"):
        try:
            speaker_name = os.path.splitext(os.path.basename(wav_file))[0]
            waveform, sr = librosa.load(wav_file, sr=16000)  
            
            if len(waveform) > 0:
                logger.info(f"[SPEAKER_SYSTEM] Extracting augmented embedding for reference '{speaker_name}'")
                
                if ENABLE_TEST_TIME_AUGMENTATION:
                    augmented_samples = augment_input_audio_for_inference(waveform, sr)
                    
                    embeddings = []
                    weights = []
                    
                    for aug_waveform, weight in augmented_samples:
                        try:
                            waveform_tensor = torch.tensor(aug_waveform, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                            embedding = model.encode_batch(waveform_tensor).squeeze().cpu().numpy()
                            
                            if embedding is not None:
                                embeddings.append(embedding)
                                weights.append(weight)
                        except Exception as e:
                            logger.warning(f"[SPEAKER_SYSTEM] Failed to extract embedding from augmented sample: {e}")
                    
                    if embeddings:
                        embeddings = np.array(embeddings)
                        weights = np.array(weights)
                        weights = weights / np.sum(weights) 
                        final_embedding = np.average(embeddings, axis=0, weights=weights)
                        norm = np.linalg.norm(final_embedding)
                        if norm > 0:
                            final_embedding = final_embedding / norm
                        
                        reference_embeddings[speaker_name] = final_embedding
                        logger.info(f"[SPEAKER_SYSTEM] Extracted augmented embedding for '{speaker_name}': "
                                  f"shape {final_embedding.shape}, from {len(embeddings)} variants")
                    else:
                        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        embedding = model.encode_batch(waveform_tensor).squeeze().cpu().numpy()
                        if embedding is not None:
                            reference_embeddings[speaker_name] = embedding
                            logger.info(f"[SPEAKER_SYSTEM] Extracted standard embedding for '{speaker_name}' (fallback)")
                else:
                    waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    embedding = model.encode_batch(waveform_tensor).squeeze().cpu().numpy()
                    
                    if embedding is not None:
                        reference_embeddings[speaker_name] = embedding
                        logger.info(f"[SPEAKER_SYSTEM] Extracted standard embedding for '{speaker_name}': shape {embedding.shape}")
                    else:
                        logger.warning(f"[SPEAKER_SYSTEM] Failed to extract embedding for '{speaker_name}'")
                        
        except Exception as e:
            logger.error(f"[SPEAKER_SYSTEM] Failed to process {wav_file}: {e}")

    logger.info(f"[SPEAKER_SYSTEM] Successfully loaded {len(reference_embeddings)} reference embeddings with augmentation")
    return reference_embeddings


def get_embedding_from_waveform(waveform_np, model, sr=16000):
    """Get speaker embedding from numpy waveform using encode_batch."""
    try:
        min_samples = int(1.0 * sr)  
        if len(waveform_np) < min_samples:
            waveform_np = np.pad(waveform_np, (0, min_samples - len(waveform_np)), mode='constant')
        waveform_tensor = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model.encode_batch(waveform_tensor).squeeze().cpu().numpy()
        
        return embedding
    except Exception as e:
        logger.error(f"[EMBEDDING] Failed to extract embedding: {e}")
        return None


def verify_speaker_with_embeddings(input_waveform, reference_embeddings, model, target_speaker_id, threshold=EMBEDDING_SIMILARITY_THRESHOLD):
    """
    Verify speaker using embedding comparison with ensemble verification and comprehensive augmentation.
    """
    try:
        logger.info(f"[EMBEDDING_VERIFICATION] Augmentation status - TTA: {ENABLE_TEST_TIME_AUGMENTATION}, Ensemble: {ENABLE_ENSEMBLE_VERIFICATION}, Preprocessing: {USE_PREPROCESSING_VARIANTS}")
        if ENABLE_ENSEMBLE_VERIFICATION:
            logger.info(f"[EMBEDDING_VERIFICATION] Using ensemble verification for speaker '{target_speaker_id}'")
            ensemble_result = ensemble_speaker_verification(input_waveform, reference_embeddings, model, target_speaker_id)
   
            return {
                'is_verified': ensemble_result.get('verified', False),
                'identified_speaker': target_speaker_id if ensemble_result.get('verified', False) else None,
                'feature_similarity': ensemble_result.get('confidence', 0.0),
                'threshold': threshold,
                'method': ensemble_result.get('method', 'ensemble'),
                'ensemble_details': ensemble_result.get('details', {})
            }
        else:
            logger.info(f"[EMBEDDING_VERIFICATION] Using standard verification with augmentation for speaker '{target_speaker_id}'")
            
            if ENABLE_TEST_TIME_AUGMENTATION:
                logger.info("[EMBEDDING_VERIFICATION] ✅ Test-time augmentation ENABLED")
                input_embedding = get_embedding_from_waveform_augmented(input_waveform, model)
            else:
                logger.info("[EMBEDDING_VERIFICATION] ⚠️ Test-time augmentation DISABLED")
                input_embedding = get_embedding_from_waveform(input_waveform, model, sr=16000)
            
            if input_embedding is None:
                return {
                    'is_verified': False,
                    'error': 'Failed to extract input embedding',
                    'distance': float('inf'),
                    'threshold': threshold,
                    'method': 'standard_failed'
                }
            
            best_match = None
            best_similarity = -1.0
            all_similarities = {}
            
            for speaker_name, ref_embedding in reference_embeddings.items():
                similarity = cosine_similarity(
                    input_embedding.reshape(1, -1),
                    ref_embedding.reshape(1, -1)
                )[0][0]
                all_similarities[speaker_name] = similarity
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker_name
            
            logger.info(f"[EMBEDDING_VERIFICATION] All similarities: {all_similarities}")
            
            if best_similarity > EMBEDDING_SIMILARITY_THRESHOLD:
                is_verified = best_match == target_speaker_id
                logger.info(f"[EMBEDDING_VERIFICATION] Best match: '{best_match}' with similarity: {best_similarity:.3f}")
                logger.info(f"[EMBEDDING_VERIFICATION] Verification result: {is_verified}")
            else:
                is_verified = False
                logger.info(f"[EMBEDDING_VERIFICATION] Similarity {best_similarity:.3f} below threshold {EMBEDDING_SIMILARITY_THRESHOLD}")
            
            method = 'augmented' if ENABLE_TEST_TIME_AUGMENTATION else 'standard'
            return {
                'is_verified': is_verified,
                'identified_speaker': best_match,
                'feature_similarity': best_similarity,
                'threshold': threshold,
                'all_similarities': all_similarities,
                'method': method
            }
            
    except Exception as e:
        logger.error(f"[EMBEDDING_VERIFICATION] ❌ Verification failed with error: {e}")
        return {
            'is_verified': False,
            'error': str(e),
            'distance': float('inf'),
            'threshold': threshold,
            'method': 'error'
        }


def initialize_speaker_system(target_speaker_id: str = "merchant0") -> bool:
    """Initialize the global speaker verification system using SpeechBrain."""
    global _global_verifier, _speaker_system_initialized

    if _speaker_system_initialized and _global_verifier:
        return True

    logger.info("[SPEAKER_SYSTEM] Initializing SpeechBrain-based speaker system...")
    
    try:
        if Diarization is None:
            logger.warning("[SPEAKER_SYSTEM] Diarization not available, using speaker recognition only")
            
            try:
                speaker_rec_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                speaker_rec_model.to(DEVICE)
                logger.info("[SPEAKER_SYSTEM] ✅ EncoderClassifier model loaded successfully")
            except Exception as e:
                logger.error(f"[SPEAKER_SYSTEM] ❌ Failed to load EncoderClassifier model: {e}")
                return False
            reference_embeddings = {}
            if not os.path.exists(RECORDINGS_REF_PATH):
                logger.warning(f"[SPEAKER_SYSTEM] Recordings directory not found: {RECORDINGS_REF_PATH}")
                return False

            for wav_file in glob.glob(f"{RECORDINGS_REF_PATH}/*.wav"):
                try:
                    speaker_name = os.path.splitext(os.path.basename(wav_file))[0]
                    waveform, sr = librosa.load(wav_file, sr=16000)  # Ensure 16kHz
                    
                    if len(waveform) > 0:
                        embedding = get_embedding_from_waveform(waveform, speaker_rec_model, sr=16000)
                        
                        if embedding is not None:
                            reference_embeddings[speaker_name] = embedding
                            logger.info(f"[SPEAKER_SYSTEM] Extracted embedding for '{speaker_name}': shape {embedding.shape}")
                        else:
                            logger.warning(f"[SPEAKER_SYSTEM] Failed to extract embedding for '{speaker_name}'")
                except Exception as e:
                    logger.error(f"[SPEAKER_SYSTEM] Failed to process {wav_file}: {e}")

            if not reference_embeddings:
                logger.error("[SPEAKER_SYSTEM] ❌ No reference embeddings loaded")
                return False
            
            class SpeechBrainVerifierWithoutDiarization:
                def __init__(self, recognizer, ref_embeddings, target_speaker):
                    self.recognizer = recognizer
                    self.reference_embeddings = ref_embeddings
                    self.target_speaker_id = target_speaker
                    self.ref_path = RECORDINGS_REF_PATH

                def verify_and_diarize(self, waveform):
                    """Performs recognition without diarization - treats entire audio as single speaker."""
                    try:
                        enrollment_files = glob.glob(f"{self.ref_path.replace('/recordings', '/enrollments')}/*.wav")
                        if enrollment_files:
                            logger.info("[NOXTRAL] Reloading embeddings from enrollment files")
                            current_embeddings = {}
                            for wav_file in enrollment_files:
                                try:
                                    speaker_name = os.path.splitext(os.path.basename(wav_file))[0]
                                    waveform_ref, sr = librosa.load(wav_file, sr=16000)
                                    if len(waveform_ref) > 0:
                                        embedding = get_embedding_from_waveform(waveform_ref, self.recognizer, sr=16000)
                                        if embedding is not None:
                                            current_embeddings[speaker_name] = embedding
                                            logger.info(f"[NOXTRAL] Loaded enrollment embedding: {speaker_name}")
                                except Exception as e:
                                    logger.error(f"[NOXTRAL] Failed to load enrollment {wav_file}: {e}")
                            
                            reference_embeddings_to_use = current_embeddings if current_embeddings else self.reference_embeddings
                        else:
                            reference_embeddings_to_use = self.reference_embeddings
                        embedding = get_embedding_from_waveform(waveform, self.recognizer, sr=16000)
                        
                        if embedding is None:
                            logger.error("[NOXTRAL] Failed to extract embedding")
                            return []
                        
                        best_match = None
                        best_similarity = -1.0
                        
                        for ref_name, ref_emb in reference_embeddings_to_use.items():
                            similarity = cosine_similarity(
                                embedding.reshape(1, -1),
                                ref_emb.reshape(1, -1)
                            )[0][0]
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = ref_name
                        
                        logger.info(f"[NOXTRAL] Single speaker match: '{best_match}' with similarity {best_similarity:.3f}")

                        recognized_segments = []
                        if best_similarity > EMBEDDING_SIMILARITY_THRESHOLD:
                            recognized_segments.append({
                                'speaker': best_match,
                                'start_s': 0.0,
                                'end_s': len(waveform) / fs,
                                'audio': waveform
                            })
                            logger.info(f"[NOXTRAL] ✅ Recognized speaker: {best_match} (threshold: {EMBEDDING_SIMILARITY_THRESHOLD})")
                        else:
                            logger.info(f"[NOXTRAL] Similarity {best_similarity:.3f} below threshold {EMBEDDING_SIMILARITY_THRESHOLD}")
                
                        return recognized_segments
                        
                    except Exception as e:
                        logger.error(f"[NOXTRAL] Recognition failed: {e}")
                        return []

            _global_verifier = SpeechBrainVerifierWithoutDiarization(speaker_rec_model, reference_embeddings, target_speaker_id)
            
        else:
            diarization_model = Diarization.from_hparams(
                source="speechbrain/speaker-diarization", 
                savedir="pretrained_models/speaker-diarization"
            )
            speaker_rec_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            reference_embeddings = load_reference_embeddings()
            if not reference_embeddings:
                logger.error("[SPEAKER_SYSTEM] ❌ No reference embeddings loaded")
                return False
            
            class SpeechBrainVerifier:
                def __init__(self, diarizer, recognizer, ref_embeddings):
                    self.diarizer = diarizer
                    self.recognizer = recognizer
                    self.reference_embeddings = ref_embeddings
                    self.ref_path = RECORDINGS_REF_PATH

                def verify_and_diarize(self, waveform):
                    """Performs diarization and then recognition."""
                    diarization_result = self.diarizer.diarize_from_waveform(
                        torch.tensor(waveform, dtype=torch.float32).unsqueeze(0), fs
                    )
                    
                    num_speakers = len(diarization_result)
                    logger.info(f"[NOXTRAL] Detected {num_speakers} speakers.")
                    
                    recognized_segments = []
                    for start, end, speaker_id in diarization_result:
                        start_sample = int(start * fs)
                        end_sample = int(end * fs)
                        segment_audio = waveform[start_sample:end_sample]
                        
                        if len(segment_audio) > 0:
                            embedding = get_embedding_from_waveform(segment_audio, self.recognizer, sr=16000)
                            
                            if embedding is None:
                                continue
                            
                            best_match = None
                            best_similarity = -1.0
                            
                            for ref_name, ref_emb in self.reference_embeddings.items():
                                similarity = cosine_similarity(
                                    embedding.reshape(1, -1),
                                    ref_emb.reshape(1, -1)
                                )[0][0]
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = ref_name
                            
                            logger.info(f"[NOXTRAL] Segment {speaker_id} match: '{best_match}' with similarity {best_similarity:.3f}")

                            if best_similarity > EMBEDDING_SIMILARITY_THRESHOLD:
                                recognized_segments.append({
                                    'speaker': best_match,
                                    'start_s': start,
                                    'end_s': end,
                                    'audio': segment_audio
                                })
                                logger.info(f"[NOXTRAL] ✅ Recognized speaker: {best_match}")
                    
                    return recognized_segments

            _global_verifier = SpeechBrainVerifier(diarization_model, speaker_rec_model, reference_embeddings)
        
        _speaker_system_initialized = True
        
        if Diarization is None:
            logger.info(f"[SPEAKER_SYSTEM] ✅ SpeechBrain-based system ready (EncoderClassifier only, no diarization).")
        else:
            logger.info(f"[SPEAKER_SYSTEM] ✅ SpeechBrain-based system ready (with diarization).")
        
        return True
        
    except Exception as e:
        logger.error(f"[SPEAKER_SYSTEM] ❌ SpeechBrain system initialization failed: {e}")
        _speaker_system_initialized = False
        return False

class LiveAudioFilter(FrameProcessor):
    """
    Applies a sophisticated filtering pipeline to live audio frames, with buffering.
    """
    def __init__(self):
        super().__init__()
        self._filter_enabled = True
        self._audio_buffer = []
        self._min_samples = 1024 
        self._frame_count = 0         
        self.in_utterance = False
        self.utterance_buffer = bytearray()
        self.silence_duration_ms = 0
        self.silence_timeout_ms = 50

    async def process_frame(self, frame: Frame,):
        await super().process_frame(frame)
        self._frame_count += 1

        await self.push_frame(frame)

        if not isinstance(frame, AudioRawFrame):
            return

        is_speech = self.vad.is_speech(frame.audio, sample_rate=fs)
        chunk_duration_ms = (len(frame.audio) / 2) / fs * 1000

        if is_speech:
            if not self.in_utterance:
                self.in_utterance = True
                self.utterance_buffer.clear()
            self.utterance_buffer.extend(frame.audio)
            self.silence_duration_ms = 0
            return

        if self.in_utterance:
            self.silence_duration_ms += chunk_duration_ms
            if self.silence_duration_ms < self.silence_timeout_ms:
                self.utterance_buffer.extend(frame.audio)
                return
            self.in_utterance = False
            utterance_data = bytes(self.utterance_buffer)
            self.utterance_buffer.clear()

            waveform_np = np.frombuffer(utterance_data, dtype=np.int16).astype(np.float32) / 32768.0
            xs_wiener = wiener_custom(waveform_np)
            xs_final = gaussian_filter1d(xs_wiener, sigma=1.0)
            processed_bytes = (xs_final * 32768.0).astype(np.int16).tobytes()

            processed_frame = AudioRawFrame(
                audio=processed_bytes,
                sample_rate=fs,
                num_channels=1
            )
            processed_frame.id = getattr(frame, 'id', str(uuid.uuid4()))
            processed_frame.transport_destination = frame.transport_destination
            if hasattr(frame, 'pts'):
                processed_frame.pts = frame.pts
            await self.push_frame(processed_frame)
            return

        return


def send_transcript_to_llm(transcript: str):
    """Send final transcription to the Gemini LLM service."""
    try:
        logger.info(f"[LLM] ▶ Sending to Gemini LLM: “{transcript}”")
        response = "<stubbed response>"
        logger.info(f"[LLM] ✅ Received from Gemini LLM: {response}")
    except Exception as e:
        logger.error(f"[LLM] ❌ Error communicating with Gemini LLM: {e}")

class SpeakerIsolationSystem:
    """
    Simple wrapper class for speaker isolation functionality.
    Provides a clean interface to the speaker verification and processing system.
    """
    
    def __init__(self, target_speaker_id: str = "merchant0"):
        """
        Initialize the speaker isolation system.
        
        Args:
            target_speaker_id: ID of the target speaker to verify against
        """
        self.target_speaker_id = target_speaker_id
        self.processor = None
        self.initialized = False
        logger.info(f"[SPEAKER_ISOLATION] Initializing system for target speaker: {target_speaker_id}")
        
    def initialize(self):
        """Initialize the speaker isolation system."""
        try:
            # Initialize the global speaker system
            if initialize_speaker_system(self.target_speaker_id):
                self.initialized = True
                logger.info(f"[SPEAKER_ISOLATION] System initialized for speaker: {self.target_speaker_id}")
                return True
            else:
                logger.error("[SPEAKER_ISOLATION] Failed to initialize speaker system")
                return False
        except Exception as e:
            logger.error(f"[SPEAKER_ISOLATION] Initialization failed: {e}")
            return False
    
    def isolate_speaker(self, audio_chunk: bytes) -> bytes:
        """
        Process audio chunk and isolate the target speaker.
        
        Args:
            audio_chunk: Raw audio data as bytes
            
        Returns:
            Processed audio data or silenced audio if speaker not verified
        """
        if not self.initialized:
            logger.warning("[SPEAKER_ISOLATION] System not initialized, attempting auto-initialization")
            if not self.initialize():
                logger.error("[SPEAKER_ISOLATION] Auto-initialization failed, returning silenced audio")
                return b'\x00' * len(audio_chunk)  # Return silence
        
        try:
            # Convert bytes to numpy array for processing
            waveform = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Use the global verifier for speaker isolation
            if _global_verifier is not None:
                recognized_segments = _global_verifier.verify_and_diarize(waveform)
                
                if recognized_segments:
                    # Speaker verification passed - return original audio
                    logger.debug(f"[SPEAKER_ISOLATION] ✅ Speaker verified, passing through audio")
                    return audio_chunk
                else:
                    # Speaker verification failed - return silenced audio
                    logger.debug(f"[SPEAKER_ISOLATION] ❌ Speaker verification failed, silencing audio")
                    return b'\x00' * len(audio_chunk)
            else:
                logger.warning("[SPEAKER_ISOLATION] No verifier available, returning silenced audio")
                return b'\x00' * len(audio_chunk)
            
        except Exception as e:
            logger.error(f"[SPEAKER_ISOLATION] Audio processing failed: {e}")
            return b'\x00' * len(audio_chunk)  # Return silence on error
    
    def process_audio(self, audio_data):
        """
        Legacy method for compatibility.
        
        Args:
            audio_data: Raw audio data to process
            
        Returns:
            Processed audio data or None if speaker not verified
        """
        if isinstance(audio_data, bytes):
            return self.isolate_speaker(audio_data)
        else:
            logger.error("[SPEAKER_ISOLATION] Unsupported audio data format")
            return None
    
    def get_status(self):
        """Get the current status of the speaker isolation system."""
        return {
            'initialized': self.initialized,
            'target_speaker': self.target_speaker_id,
            'verification_ready': _speaker_system_initialized,
            'global_verifier_ready': _global_verifier is not None
        }

class RobustSpeakerAudioProcessor(FrameProcessor):
    """
    ENHANCED: Ultra-robust real-time speaker verification and diarization processor.

    This processor provides:
    1. Utterance-based real-time processing
    2. Advanced multi-layered speaker verification
    3. Adaptive threshold management
    4. Rolling confidence tracking
    5. Comprehensive state management
    6. Real-time audio filtering and diarization
    """

    def __init__(self, target_speaker_id: str, start_event: asyncio.Event = None, advanced_verifier=None, bypass_vad: bool = False):
        super().__init__()
        self.target_speaker_id = target_speaker_id
        self._start_event = start_event
        self.advanced_verifier = advanced_verifier
        self.verifier = None
        self.verification_ready = False
        self.is_collecting = False
        self.current_speech_frames: List[AudioRawFrame] = []
        self._segment_count = 0
        self._consecutive_rejections = 0
        self._speaker_rejected = False
        self._rejection_cooldown_seconds = 0.0  # Shortened cooldown
        self._rejection_cooldown_until = 0
        self._max_consecutive_rejections = 10  # Increased threshold
        self._permanent_block_duration = 2.0  # Shortened permanent block
        self._permanent_block_until = 0
        self.min_verification_frames = 1
        self.confidence_accumulator = []
        self.similarity_accumulator = []
        self.accumulator_size = 3  # Reduced for faster response
        self.required_average_confidence = 0.60  # Lowered significantly
        self.required_average_similarity = 0.60  
        self.verified_frame_count = 0
        self._last_verification_success_time = 0  # Track when verification last succeeded
        self._last_verification_attempt_time = 0  
        self.min_volume_threshold = 1e-3
        self.quality_check_enabled = True
        self.strict_quality_mode = False  
        self.max_segment_duration_seconds = 10.0
        
        # NEW: VAD bypass configuration
        self.bypass_vad = bypass_vad
        self._frame_buffer_timer = None
        self._buffer_timeout_seconds = 2.0  # Process buffer every 2 seconds when bypassing VAD
        self._last_buffer_process_time = 0
        
        self.stats = {
            'total_frames': 0,
            'verified_frames': 0,
            'rejected_frames': 0,
            'processing_errors': 0
        }
        self.enrollment_enabled = True
        self.enrollment_steps = 3  # First 3 queries for enrollment
        self.current_step = 0  # Track current step (0 = not started, 1-3 = enrollment steps, 4+ = verification)
        self.enrollment_dir = "enrollments"
        os.makedirs(self.enrollment_dir, exist_ok=True)
        logger.info(f"[ENROLLMENT] Enrollments directory ready: {os.path.abspath(self.enrollment_dir)}")
        self._initialize_verifier()
        self._started = False
        self._pre_start_buffer = []

        if self.enrollment_enabled:
            logger.info(f"[ENROLLMENT] 📚 Three-step enrollment enabled: Steps 1-{self.enrollment_steps} will record enrollment audio")
            logger.info(f"[ENROLLMENT] 🔐 Verification will start from step {self.enrollment_steps + 1}")
        
        # Log VAD bypass status
        if self.bypass_vad:
            logger.info(f"[ROBUST_PROCESSOR] ⚠️ VAD BYPASS ENABLED - Processing audio continuously every {self._buffer_timeout_seconds}s")
        else:
            logger.info(f"[ROBUST_PROCESSOR] 🎤 VAD ENABLED - Processing audio on speech start/stop events")
        
        logger.info(f"[ROBUST_PROCESSOR] Initialized for target speaker: '{self.target_speaker_id}'")

    def _initialize_verifier(self):
        """Initialize the verification system using our improved simple verifier."""
        try:
            if initialize_speaker_system(self.target_speaker_id):
                self.verifier = _global_verifier
                self.verification_ready = True
                logger.info(f"[ROBUST_PROCESSOR] ✅ Simple verifier initialized for '{self.target_speaker_id}'")
            else:
                logger.error("[ROBUST_PROCESSOR] ❌ Simple verifier failed to initialize")
                self.verification_ready = False

        except Exception as e:
            logger.error(f"[ROBUST_PROCESSOR] ❌ Verifier initialization failed: {e}")
            self.verification_ready = False

    def _should_process_buffer_by_timeout(self) -> bool:
        """Check if buffer should be processed based on timeout when bypassing VAD."""
        if not self.bypass_vad:
            return False
        
        current_time = time.time()
        return (current_time - self._last_buffer_process_time) >= self._buffer_timeout_seconds

    async def process_frame(self, frame: Frame):
        """Enhanced frame processing with VAD bypass logic"""
        if self._start_event:
            await self._start_event.wait()
        if not self._started:
            if isinstance(frame, AudioRawFrame):
                self._started = True
                for buffered_frame in self._pre_start_buffer:
                    await self.process_frame(buffered_frame)
                self._pre_start_buffer = []
            else:
                self._pre_start_buffer.append(frame)
                return

        await super().process_frame(frame)

        if isinstance(frame, AudioRawFrame):
            if self._speaker_rejected and time.time() < self._rejection_cooldown_until:
                logger.debug("[ROBUST_PROCESSOR] 🔇 Audio blocked during rejection cooldown")
                return
            if self._speaker_rejected and time.time() >= self._rejection_cooldown_until:
                self._speaker_rejected = False
                self._consecutive_rejections = 0 
                logger.info("[ROBUST_PROCESSOR] Cooldown expired - resuming verification")
            
            # Always start collecting for audio frames
            if not self.is_collecting:
                await self._start_collecting()
            self.current_speech_frames.append(frame)
            
            # NEW: Process buffer by timeout when bypassing VAD
            if self.bypass_vad and self._should_process_buffer_by_timeout():
                logger.info("[ROBUST_PROCESSOR] ⏰ VAD BYPASS: Processing buffer due to timeout")
                await self._stop_collecting()

                
            if self.is_collecting:
                await self._stop_collecting()
            logger.info("[ROBUST_PROCESSOR] 🛑 VAD speech end detected - triggering strict verification")
            await self.push_frame(frame)

        elif isinstance(frame, TranscriptionFrame):
            if frame.text and getattr(frame, 'is_final', False):
                logger.debug(f"[ROBUST_PROCESSOR] 📝 Final transcription received: '{frame.text}' - audio already processed by STT")
            await self.push_frame(frame)
        else:
            await self.push_frame(frame)

    async def _start_collecting(self):
        """Start collecting audio frames."""
        if not self.is_collecting:
            self.is_collecting = True
            self.current_speech_frames = []
            self._last_buffer_process_time = time.time()  # Reset timeout timer
            if self.bypass_vad:
                logger.info("[ROBUST_PROCESSOR] ✅▶️ START COLLECTING utterance (VAD BYPASS MODE).")
            else:
                logger.info("[ROBUST_PROCESSOR] ✅▶️ START COLLECTING utterance.")

    async def _stop_collecting(self):
        """Stop collecting and process the collected audio."""
        if self.is_collecting:
            self.is_collecting = False
            frames_to_process = self.current_speech_frames
            self.current_speech_frames = []
            self._last_buffer_process_time = time.time()  # Update last process time
            
            if self.bypass_vad:
                logger.info(f"[ROBUST_PROCESSOR] ⏹️ STOP COLLECTING (VAD BYPASS). Buffer has {len(frames_to_process)} frames.")
            else:
                logger.info(f"[ROBUST_PROCESSOR] ⏹️ STOP COLLECTING. Buffer has {len(frames_to_process)} frames.")
                
            if frames_to_process:
                await self._process_and_filter_audio(frames_to_process)

    async def _process_and_filter_audio(self, frames: List[AudioRawFrame]):
        """
        Process audio with proper async handling and fix interruption logic.
        """
        self._segment_count += 1
        self._last_verification_attempt_time = time.time()  # Track verification attempt
        
        try:
            if not frames:
                return

            waveforms = [np.frombuffer(f.audio, dtype=np.int16).astype(np.float32) / 32768.0 for f in frames]
            full_waveform_np = np.concatenate(waveforms)
            
            min_duration_seconds = 0.5
            actual_duration = len(full_waveform_np) / fs
            
            if actual_duration < min_duration_seconds:
                logger.info(f"[ROBUST_PROCESSOR] 🔇 Utterance too short ({actual_duration:.2f}s), skipping analysis.")
                return

            logger.info(f"[ROBUST_PROCESSOR] ⚙️ Processing segment of {actual_duration:.2f}s...")
            
            self.current_step += 1
            
            if self.enrollment_enabled and self.current_step <= self.enrollment_steps:
                logger.info(f"[ENROLLMENT] �️ ENROLLMENT STEP {self.current_step}/{self.enrollment_steps} - Recording enrollment audio")
    
                await self._save_enrollment_audio_step(full_waveform_np, actual_duration, self.current_step)
                
                logger.info(f"[ENROLLMENT] ✅ ENROLLMENT STEP {self.current_step} COMPLETE - {self.enrollment_steps - self.current_step} steps remaining")
                
                await self._send_verified_audio(full_waveform_np, frames, original_frame=frames[-1])
                return
            
            if self.enrollment_enabled and self.current_step == self.enrollment_steps + 1:
                logger.info(f"[ENROLLMENT] 🔓 ENROLLMENT COMPLETE! Starting verification from step {self.current_step}")
                logger.info(f"[VERIFICATION] 🛡️ All future audio will be verified against enrollment samples")
            
            if not self.verification_ready or not self.verifier:
                logger.warning("[ROBUST_PROCESSOR] ⚠️ Verifier not ready, rejecting audio for safety.")
                silenced_waveform = set_rms_to_zero(full_waveform_np)
                await self._send_verified_audio(silenced_waveform, frames, original_frame=frames[-1])
                await self._send_interruption_frame()
                self._handle_rejection("verifier_not_ready")
                return

            logger.info(f"[VERIFICATION] 🔍 Step {self.current_step}: Running speaker verification...")
            recognized_segments = self.verifier.verify_and_diarize(full_waveform_np)

            if recognized_segments:
                self.stats['verified_frames'] += 1
                self._consecutive_rejections = 0
                
                recognized_audio_list = [seg['audio'] for seg in recognized_segments]
                verified_waveform = np.concatenate(recognized_audio_list)
                
                logger.info(f"✅ [ROBUST_PROCESSOR] VERIFICATION PASSED. Found {len(recognized_segments)} recognized speakers.")
                await self._send_verified_audio(verified_waveform, frames, original_frame=frames[-1])
                
            else:
                self.stats['rejected_frames'] += 1
                logger.info("🔇 [ROBUST_PROCESSOR] VERIFICATION FAILED. No known speakers found.")
                silenced_waveform = set_rms_to_zero(full_waveform_np)
                await self._send_verified_audio(silenced_waveform, frames, original_frame=frames[-1])
                self._handle_rejection("no_known_speaker")

        except Exception as e:
            self.stats['processing_errors'] += 1
            logger.error(f"[ROBUST_PROCESSOR] ❌ Error processing audio segment: {e}", exc_info=True)
            
            try:
                silenced_waveform = set_rms_to_zero(np.zeros(1000))  # Small silent buffer
                await self._send_verified_audio(silenced_waveform, frames, original_frame=frames[-1] if frames else None)
                await self._send_interruption_frame()
                self._handle_rejection("processing_error")
            except:
                logger.error("[ROBUST_PROCESSOR] ❌ Failed to handle processing error gracefully")

    async def _save_enrollment_audio_step(self, waveform: np.ndarray, duration: float, step: int):
        """Save enrollment audio for a specific step (1, 2, or 3)."""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            enrollment_filename = f"{self.target_speaker_id}_enrollment_step{step}_{timestamp}.wav"
            enrollment_path = os.path.join(self.enrollment_dir, enrollment_filename)
            
            logger.info(f"[ENROLLMENT] 🎙️ Recording enrollment step {step}/{self.enrollment_steps}")
            
            waveform_int16 = (waveform * 32767).astype(np.int16)
            sf.write(enrollment_path, waveform_int16, fs)
            
            logger.info(f"[ENROLLMENT] ✅ Step {step} audio saved: {enrollment_path}")
            logger.info(f"[ENROLLMENT] Audio duration: {duration:.2f}s, samples: {len(waveform)}")
            
            logger.info(f"[ENROLLMENT] 📁 Only audio file saved, no JSON metadata created")
            
            remaining_steps = self.enrollment_steps - step
            if remaining_steps > 0:
                logger.info(f"[ENROLLMENT] 📋 {remaining_steps} more enrollment step(s) needed before verification starts")
            else:
                logger.info(f"[ENROLLMENT] 🎉 All {self.enrollment_steps} enrollment steps completed! Next audio will trigger verification.")
            
        except Exception as e:
            logger.error(f"[ENROLLMENT] ❌ Failed to save enrollment step {step} audio: {e}")

    async def _save_enrollment_audio(self, waveform: np.ndarray, duration: float):
        """Legacy enrollment function - now calls step-based enrollment."""
        await self._save_enrollment_audio_step(waveform, duration, 1)

    def _prevent_overfitting(self, result):
        if result.model_confidence > 0.995:
            result.model_confidence = 0.90
        return result

    def _passes_quality_checks(self, audio_data: np.ndarray) -> bool:
        duration = len(audio_data) / fs
        if duration < 0.3:
            return False
        return True

    def _make_ultra_robust_decision(self, verification_result) -> bool:
        """FIX: Simplified verification decision"""
        return verification_result.get('is_verified', False)

    def _check_accumulated_metrics(self, confidence: float, similarity: float) -> bool:
        """FIX: Simplified metrics check"""
        return True

    def _update_accumulators(self, confidence: float, similarity: float):
        self.confidence_accumulator.append(confidence)
        self.similarity_accumulator.append(similarity)
        if len(self.confidence_accumulator) > self.accumulator_size:
            self.confidence_accumulator.pop(0)
        if len(self.similarity_accumulator) > self.accumulator_size:
            self.similarity_accumulator.pop(0)

    def _handle_rejection(self, reason: str):
        """Handle rejection without sending interruption here (already sent above)."""
        self._consecutive_rejections += 1
        self.verified_frame_count = 0
        
        self._speaker_rejected = True
        self._rejection_cooldown_until = time.time() + self._rejection_cooldown_seconds
        
        logger.info(f"[ROBUST_PROCESSOR] 🔒 Rejection cooldown activated for {self._rejection_cooldown_seconds}s (reason: {reason})")
        # Remove the interruption frame sending from here to avoid duplicates

    async def _send_verified_audio(self, audio_data: np.ndarray, frames: List[AudioRawFrame], original_frame: AudioRawFrame):
        """Forward verified audio to STT and set verification status."""
        if np.sqrt(np.mean(audio_data**2)) > 1e-5:
            # VERIFICATION PASSED: Forward original frames to STT for transcription
            self._last_verification_success_time = time.time()
            logger.info(f"[ROBUST_PROCESSOR] ✅ Verified {len(audio_data)} samples - FORWARDING {len(frames)} frames to STT for transcription.")
            
            # Forward all collected frames to STT when verification passes
            for frame in frames:
                await self.push_frame(frame)
                
        else:
            # VERIFICATION FAILED: Audio was silenced, don't forward to STT
            logger.info(f"[ROBUST_PROCESSOR] 🔇 Silenced audio due to failed verification - BLOCKING {len(frames)} frames from STT.")

    async def _send_interruption_frame(self):
        """Send interruption frame with proper async handling."""
        try:
            interruption_frame = InterruptionFrame()
            await self.push_frame(interruption_frame)
            logger.info("[ROBUST_PROCESSOR] 🛑 Sent interruption frame due to verification failure")
        except Exception as e:
            logger.error(f"[ROBUST_PROCESSOR] ❌ Failed to send interruption frame: {e}")

    def get_enrollment_status(self) -> Dict[str, Any]:
        """Get the current enrollment status."""
        enrollment_status = "completed" if self.current_step > self.enrollment_steps else "in_progress"
        if self.current_step == 0:
            enrollment_status = "not_started"
        
        return {
            'enabled': self.enrollment_enabled,
            'current_step': self.current_step,
            'total_steps': self.enrollment_steps,
            'status': enrollment_status,
            'remaining_steps': max(0, self.enrollment_steps - self.current_step),
            'verification_active': self.current_step > self.enrollment_steps
        }

    def _report_status(self) -> Dict[str, Any]:
        """Return status dict with enrollment progress and VAD bypass status."""
        enrollment_status = "completed" if self.current_step > self.enrollment_steps else "in_progress"
        if self.current_step == 0:
            enrollment_status = "not_started"
        
        return {
            'target_speaker': self.target_speaker_id,
            'verification_ready': self.verification_ready,
            'is_collecting': self.is_collecting,
            'vad_bypass': {
                'enabled': self.bypass_vad,
                'buffer_timeout_seconds': self._buffer_timeout_seconds,
                'time_since_last_process': time.time() - self._last_buffer_process_time if self._last_buffer_process_time > 0 else 0
            },
            'enrollment': {
                'enabled': self.enrollment_enabled,
                'current_step': self.current_step,
                'total_steps': self.enrollment_steps,
                'status': enrollment_status,
                'remaining_steps': max(0, self.enrollment_steps - self.current_step)
            },
            'segment_count': self._segment_count,
            'consecutive_rejections': self._consecutive_rejections,

            'speaker_rejected': self._speaker_rejected,
            'verified_frame_count': self.verified_frame_count,
            'stats': self.stats.copy(),
            'thresholds': {
                'min_verification_frames': self.min_verification_frames,
                'required_average_confidence': self.required_average_confidence,
                'required_average_similarity': self.required_average_similarity,
                'max_segment_duration': self.max_segment_duration_seconds
            },
            'accumulators': {
                'confidence_size': len(self.confidence_accumulator),
                'similarity_size': len(self.similarity_accumulator),
                'avg_confidence': np.mean(self.confidence_accumulator) if self.confidence_accumulator else 0.0,
                'avg_similarity': np.mean(self.similarity_accumulator) if self.similarity_accumulator else 0.0
            }
        }