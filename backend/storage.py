from typing import Protocol, Optional, Union
from abc import ABC, abstractmethod
import os
import shutil

class StorageInterface(Protocol):
    def put(self, uri: str, data: bytes) -> None:
        ...

    def get(self, uri: str) -> bytes:
        ...
        
    def exists(self, uri: str) -> bool:
        ...

class LocalFSStorage:
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _resolve(self, uri: str) -> str:
        # Simple mock URI resolver: fs://path/to/file -> base_path/path/to/file
        # Remove scheme
        if uri.startswith("fs://"):
            rel_path = uri[5:]
        else:
            rel_path = uri
            
        full_path = os.path.join(self.base_path, rel_path)
        return full_path

    def put(self, uri: str, data: bytes) -> None:
        path = self._resolve(uri)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def get(self, uri: str) -> bytes:
        path = self._resolve(uri)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {uri}")
        with open(path, 'rb') as f:
            return f.read()
            
    def exists(self, uri: str) -> bool:
        path = self._resolve(uri)
        return os.path.exists(path)

# Mock DB Interfaces would go here (e.g., using a dict or sqlite for metadata)
