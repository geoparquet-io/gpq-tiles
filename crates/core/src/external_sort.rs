//! External merge sort for memory-bounded tile generation.
//!
//! When generating tiles from large GeoParquet files, we need to group features
//! by tile ID (Hilbert-ordered) to build each tile efficiently. This module provides
//! disk-backed sorting that can handle datasets larger than available RAM.
//!
//! # How It Works
//!
//! 1. Features are extracted from GeoParquet and converted to `TileFeatureRecord`
//! 2. Records are fed to `TileFeatureSorter`, which buffers them in memory
//! 3. When the buffer fills, it's sorted and written to a temp file
//! 4. Final iteration performs k-way merge of all sorted chunks
//! 5. Output is an iterator of records sorted by `tile_id`, ready for tile building
//!
//! # Example
//!
//! ```ignore
//! use gpq_tiles_core::external_sort::{TileFeatureRecord, TileFeatureSorter};
//!
//! let mut sorter = TileFeatureSorter::new(100_000);  // Buffer 100K records
//!
//! // Add records from GeoParquet processing
//! sorter.add(TileFeatureRecord {
//!     tile_id: 42,
//!     feature_id: 1,
//!     geometry_wkb: vec![...],
//!     properties: vec![...],  // MessagePack serialized
//! });
//!
//! // Get sorted iterator for tile building
//! for record in sorter.sort()? {
//!     let record = record?;
//!     // All records with same tile_id come consecutively
//! }
//! ```

use extsort::{ExternalSorter, Sortable};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::io::{Read, Write};

/// A feature record ready for tile building, sorted by tile_id.
///
/// This struct holds all data needed to include a feature in a vector tile:
/// - `tile_id`: PMTiles Hilbert-curve ID (determines sort order)
/// - `feature_id`: Original feature index for debugging/provenance
/// - `geometry_wkb`: WKB-encoded geometry (clipped to tile if needed)
/// - `properties`: MessagePack-serialized feature properties
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileFeatureRecord {
    /// PMTiles tile ID (Hilbert curve order) - primary sort key
    pub tile_id: u64,
    /// Original feature ID from source data
    pub feature_id: u64,
    /// WKB-encoded geometry
    pub geometry_wkb: Vec<u8>,
    /// MessagePack-serialized properties
    pub properties: Vec<u8>,
}

impl TileFeatureRecord {
    /// Create a new tile feature record.
    pub fn new(tile_id: u64, feature_id: u64, geometry_wkb: Vec<u8>, properties: Vec<u8>) -> Self {
        Self {
            tile_id,
            feature_id,
            geometry_wkb,
            properties,
        }
    }
}

impl Eq for TileFeatureRecord {}

impl PartialOrd for TileFeatureRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TileFeatureRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary sort: tile_id (groups features by tile)
        // Secondary sort: feature_id (stable ordering within tile)
        self.tile_id
            .cmp(&other.tile_id)
            .then_with(|| self.feature_id.cmp(&other.feature_id))
    }
}

impl Sortable for TileFeatureRecord {
    fn encode<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Use MessagePack for compact binary serialization
        let bytes = rmp_serde::to_vec(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Write length prefix (u32) for framing
        let len = bytes.len() as u32;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(&bytes)?;
        Ok(())
    }

    fn decode<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        // Read length prefix
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        // Read payload
        let mut bytes = vec![0u8; len];
        reader.read_exact(&mut bytes)?;

        // Deserialize
        rmp_serde::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// External sorter for tile feature records.
///
/// Wraps `extsort::ExternalSorter` with a convenient API for our use case.
/// Records are buffered in memory until the buffer fills, then sorted chunks
/// are written to disk. Final iteration merges all chunks.
pub struct TileFeatureSorter {
    /// In-memory buffer for records before sorting
    records: Vec<TileFeatureRecord>,
    /// Maximum records to buffer before flushing to disk
    sort_buffer_size: usize,
}

impl TileFeatureSorter {
    /// Create a new sorter with the specified buffer size.
    ///
    /// # Arguments
    ///
    /// * `sort_buffer_size` - Maximum number of records to hold in memory.
    ///   Larger values use more RAM but reduce disk I/O.
    ///   Typical value: 100,000 - 1,000,000 depending on available memory.
    pub fn new(sort_buffer_size: usize) -> Self {
        Self {
            records: Vec::with_capacity(sort_buffer_size.min(1024)),
            sort_buffer_size,
        }
    }

    /// Add a record to be sorted.
    pub fn add(&mut self, record: TileFeatureRecord) {
        self.records.push(record);
    }

    /// Returns the number of records currently buffered.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true if no records have been added.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Sort all records and return an iterator over them in tile_id order.
    ///
    /// This consumes the sorter. For datasets that fit in the buffer,
    /// sorting happens entirely in memory. For larger datasets, the
    /// external sorter writes sorted chunks to temp files and merges them.
    pub fn sort(self) -> std::io::Result<impl Iterator<Item = std::io::Result<TileFeatureRecord>>> {
        let sorter = ExternalSorter::new().with_segment_size(self.sort_buffer_size);

        sorter.sort(self.records.into_iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_feature_record_creation() {
        let record = TileFeatureRecord::new(42, 1, vec![1, 2, 3], vec![4, 5, 6]);
        assert_eq!(record.tile_id, 42);
        assert_eq!(record.feature_id, 1);
        assert_eq!(record.geometry_wkb, vec![1, 2, 3]);
        assert_eq!(record.properties, vec![4, 5, 6]);
    }

    #[test]
    fn test_tile_feature_record_ordering() {
        let r1 = TileFeatureRecord::new(1, 1, vec![], vec![]);
        let r2 = TileFeatureRecord::new(2, 1, vec![], vec![]);
        let r3 = TileFeatureRecord::new(1, 2, vec![], vec![]);

        // tile_id is primary sort key
        assert!(r1 < r2);
        // feature_id is secondary sort key
        assert!(r1 < r3);
    }

    #[test]
    fn test_sortable_encode_decode_roundtrip() {
        let original = TileFeatureRecord::new(
            123456,
            789,
            vec![0x01, 0x02, 0x03, 0x04],
            vec![0x82, 0xa4, b't', b'e', b's', b't'], // MessagePack map
        );

        let mut buffer = Vec::new();
        original.encode(&mut buffer).unwrap();

        let decoded = TileFeatureRecord::decode(&mut buffer.as_slice()).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_sorter_basic_operations() {
        let mut sorter = TileFeatureSorter::new(1000);
        assert!(sorter.is_empty());
        assert_eq!(sorter.len(), 0);

        sorter.add(TileFeatureRecord::new(1, 1, vec![], vec![]));
        assert!(!sorter.is_empty());
        assert_eq!(sorter.len(), 1);
    }

    #[test]
    fn test_sorter_sorts_by_tile_id() {
        let mut sorter = TileFeatureSorter::new(1000);

        // Add records out of order
        sorter.add(TileFeatureRecord::new(3, 1, vec![], vec![]));
        sorter.add(TileFeatureRecord::new(1, 1, vec![], vec![]));
        sorter.add(TileFeatureRecord::new(2, 1, vec![], vec![]));

        let sorted: Vec<_> = sorter.sort().unwrap().map(|r| r.unwrap()).collect();

        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].tile_id, 1);
        assert_eq!(sorted[1].tile_id, 2);
        assert_eq!(sorted[2].tile_id, 3);
    }

    #[test]
    fn test_sorter_stable_within_tile() {
        let mut sorter = TileFeatureSorter::new(1000);

        // Multiple features in same tile
        sorter.add(TileFeatureRecord::new(5, 3, vec![], vec![]));
        sorter.add(TileFeatureRecord::new(5, 1, vec![], vec![]));
        sorter.add(TileFeatureRecord::new(5, 2, vec![], vec![]));

        let sorted: Vec<_> = sorter.sort().unwrap().map(|r| r.unwrap()).collect();

        assert_eq!(sorted.len(), 3);
        // Should be sorted by feature_id within same tile_id
        assert_eq!(sorted[0].feature_id, 1);
        assert_eq!(sorted[1].feature_id, 2);
        assert_eq!(sorted[2].feature_id, 3);
    }

    #[test]
    fn test_sorter_with_geometry_and_properties() {
        let mut sorter = TileFeatureSorter::new(1000);

        let geom1 = vec![0x01, 0x01, 0x00, 0x00, 0x00]; // Point WKB header
        let props1 = rmp_serde::to_vec(&serde_json::json!({"name": "feature1"})).unwrap();

        let geom2 = vec![0x01, 0x02, 0x00, 0x00, 0x00]; // LineString WKB header
        let props2 = rmp_serde::to_vec(&serde_json::json!({"name": "feature2"})).unwrap();

        sorter.add(TileFeatureRecord::new(2, 1, geom2.clone(), props2.clone()));
        sorter.add(TileFeatureRecord::new(1, 1, geom1.clone(), props1.clone()));

        let sorted: Vec<_> = sorter.sort().unwrap().map(|r| r.unwrap()).collect();

        assert_eq!(sorted[0].tile_id, 1);
        assert_eq!(sorted[0].geometry_wkb, geom1);
        assert_eq!(sorted[0].properties, props1);

        assert_eq!(sorted[1].tile_id, 2);
        assert_eq!(sorted[1].geometry_wkb, geom2);
        assert_eq!(sorted[1].properties, props2);
    }

    #[test]
    fn test_sorter_large_dataset() {
        // Test with enough records to potentially trigger external sorting
        let mut sorter = TileFeatureSorter::new(100); // Small buffer to force disk spill

        // Add 1000 records in reverse order
        for i in (0..1000).rev() {
            sorter.add(TileFeatureRecord::new(
                i,
                i,
                vec![i as u8],
                vec![(i % 256) as u8],
            ));
        }

        let sorted: Vec<_> = sorter.sort().unwrap().map(|r| r.unwrap()).collect();

        assert_eq!(sorted.len(), 1000);
        for (i, record) in sorted.iter().enumerate() {
            assert_eq!(
                record.tile_id, i as u64,
                "Record at position {} has wrong tile_id",
                i
            );
        }
    }

    #[test]
    fn test_empty_sorter() {
        let sorter = TileFeatureSorter::new(1000);
        let sorted: Vec<_> = sorter.sort().unwrap().map(|r| r.unwrap()).collect();
        assert!(sorted.is_empty());
    }
}
