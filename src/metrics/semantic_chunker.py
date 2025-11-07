import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class SemanticChunker:
    """Semantic chunking for more granular document retrieval"""

    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 separator: str = "\n"):
        """
        Initialize the semantic chunker

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            separator: Preferred separator for chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into semantic chunks

        Args:
            document: Document dict with 'text' and 'metadata' keys

        Returns:
            List of document chunks with updated metadata
        """
        text = document.get('text', '')
        metadata = document.get('metadata', {})

        if not text or len(text) <= self.chunk_size:
            return [document]  # Document is already small enough

        # Split text into semantic chunks
        chunks = self._split_text(text)

        # Create document chunks with updated metadata
        doc_chunks = []
        for i, chunk in enumerate(chunks):
            # Create a copy of metadata and add chunk info
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_count': len(chunks),
                'is_chunk': True,
                'original_text_length': len(text)
            })

            doc_chunks.append({
                'text': chunk,
                'metadata': chunk_metadata
            })

        return doc_chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into semantic chunks

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # First try to split by separator
        if self.separator in text:
            segments = text.split(self.separator)
            return self._merge_segments(segments)

        # If no separator found, split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return self._merge_segments(sentences)

    def _merge_segments(self, segments: List[str]) -> List[str]:
        """
        Merge segments into chunks of appropriate size

        Args:
            segments: List of text segments

        Returns:
            List of merged chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for segment in segments:
            segment_length = len(segment)

            # If adding this segment exceeds chunk size and we already have content,
            # finalize the current chunk and start a new one
            if current_length + segment_length > self.chunk_size and current_chunk:
                chunks.append(self.separator.join(current_chunk))

                # Start new chunk with overlap by keeping some segments
                overlap_length = 0
                overlap_segments = []

                # Add segments from the end until we reach desired overlap
                for seg in reversed(current_chunk):
                    if overlap_length + len(seg) <= self.chunk_overlap:
                        overlap_segments.insert(0, seg)
                        overlap_length += len(seg)
                    else:
                        break

                current_chunk = overlap_segments
                current_length = overlap_length

            # Add segment to current chunk
            current_chunk.append(segment)
            current_length += segment_length

            # If a single segment is larger than chunk size, split it
            if segment_length > self.chunk_size and len(current_chunk) == 1:
                # Split large segment into smaller pieces
                words = segment.split()
                current_chunk = []
                current_length = 0
                current_piece = []
                piece_length = 0

                for word in words:
                    word_length = len(word) + 1  # +1 for space

                    if piece_length + word_length > self.chunk_size and current_piece:
                        chunks.append(' '.join(current_piece))
                        current_piece = []
                        piece_length = 0

                    current_piece.append(word)
                    piece_length += word_length

                if current_piece:
                    current_chunk = [' '.join(current_piece)]
                    current_length = piece_length

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))

        return chunks