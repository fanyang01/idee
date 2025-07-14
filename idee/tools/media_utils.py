"""
Utility functions for media file handling and type detection.

This module provides helpers for:
- Detecting file types based on content/extension
- Reading files and encoding as base64
- Media type categorization
"""

import mimetypes
import os
from pathlib import Path
from typing import Optional, Dict, Any, Literal, Union
import base64
import logging

logger = logging.getLogger(__name__)

# Common media type mappings
MEDIA_TYPE_CATEGORIES = {
    # Images
    "image/png": "image",
    "image/jpeg": "image", 
    "image/jpg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/bmp": "image",
    "image/svg+xml": "image",
    
    # Audio
    "audio/wav": "audio",
    "audio/mp3": "audio",
    "audio/mpeg": "audio",
    "audio/ogg": "audio",
    "audio/flac": "audio",
    "audio/m4a": "audio",
    
    # Video
    "video/mp4": "video",
    "video/mpeg": "video",
    "video/quicktime": "video",
    "video/webm": "video",
    "video/avi": "video",
    "video/mov": "video",
    
    # Documents
    "application/pdf": "document",
    "text/plain": "document",
    "text/csv": "document",
    "application/json": "document",
    "application/xml": "document",
    "text/html": "document",
    "application/msword": "document",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
    "application/vnd.ms-excel": "document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "document",
}

def detect_media_type(file_path: Union[str, Path]) -> str:
    """
    Detect the MIME type of a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The detected MIME type, or 'application/octet-stream' if unknown
    """
    file_path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or 'application/octet-stream'

def get_media_category(media_type: str) -> Literal["image", "audio", "video", "document"]:
    """
    Get the media category for a given MIME type.
    
    Args:
        media_type: The MIME type
        
    Returns:
        The media category (image, audio, video, document)
    """
    return MEDIA_TYPE_CATEGORIES.get(media_type, "document")

def read_file_as_base64(file_path: Union[str, Path]) -> str:
    """
    Read a file and encode it as base64.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Base64 encoded file content
    """
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')