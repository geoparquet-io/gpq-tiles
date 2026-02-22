//! PMTiles v3 writer implementation.
//!
//! Implements the PMTiles v3 spec: https://github.com/protomaps/PMTiles/blob/main/spec/v3/spec.md
//!
//! Key design decisions:
//! - Uses Hilbert curve ordering for tile IDs (spatial locality)
//! - Delta-encoded directories for better compression
//! - Configurable compression (gzip, brotli, zstd) for both directories and tiles
//! - Clustered mode for efficient sequential reads

use crate::compression::{self, Compression};
use crate::tile::TileBounds;
use crate::{Error, Result};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// PMTiles v3 magic number
const PMTILES_MAGIC: &[u8; 7] = b"PMTiles";
const PMTILES_VERSION: u8 = 3;

/// Tile type enumeration (byte 99 in header)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TileType {
    Unknown = 0,
    Mvt = 1,
    Png = 2,
    Jpeg = 3,
    Webp = 4,
    Avif = 5,
}

// Compression enum is now imported from crate::compression

/// PMTiles v3 header (127 bytes)
///
/// Layout follows the spec exactly:
/// - Bytes 0-6: Magic "PMTiles"
/// - Byte 7: Version (3)
/// - Bytes 8-95: Offsets and lengths (8 u64s)
/// - Bytes 96-99: Flags (clustered, compression, type)
/// - Bytes 100-101: Zoom levels
/// - Bytes 102-117: Bounds (min_lon, min_lat, max_lon, max_lat as i32 * 10_000_000)
/// - Bytes 118-126: Center (zoom, lon, lat)
#[derive(Debug, Clone)]
pub struct Header {
    pub root_dir_offset: u64,
    pub root_dir_length: u64,
    pub json_metadata_offset: u64,
    pub json_metadata_length: u64,
    pub leaf_dirs_offset: u64,
    pub leaf_dirs_length: u64,
    pub tile_data_offset: u64,
    pub tile_data_length: u64,
    pub addressed_tiles_count: u64,
    pub tile_entries_count: u64,
    pub tile_contents_count: u64,
    pub clustered: bool,
    pub internal_compression: Compression,
    pub tile_compression: Compression,
    pub tile_type: TileType,
    pub min_zoom: u8,
    pub max_zoom: u8,
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
    pub center_zoom: u8,
    pub center_lon: f64,
    pub center_lat: f64,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            root_dir_offset: 127, // Immediately after header
            root_dir_length: 0,
            json_metadata_offset: 0,
            json_metadata_length: 0,
            leaf_dirs_offset: 0,
            leaf_dirs_length: 0,
            tile_data_offset: 0,
            tile_data_length: 0,
            addressed_tiles_count: 0,
            tile_entries_count: 0,
            tile_contents_count: 0,
            clustered: true,
            internal_compression: Compression::Gzip,
            tile_compression: Compression::Gzip,
            tile_type: TileType::Mvt,
            min_zoom: 0,
            max_zoom: 14,
            min_lon: -180.0,
            min_lat: -85.0,
            max_lon: 180.0,
            max_lat: 85.0,
            center_zoom: 0,
            center_lon: 0.0,
            center_lat: 0.0,
        }
    }
}

impl Header {
    /// Serialize header to exactly 127 bytes
    ///
    /// Position encoding follows the spec: multiply by 10,000,000 and store as i32 LE
    pub fn to_bytes(&self) -> [u8; 127] {
        let mut buf = [0u8; 127];

        // Magic (7 bytes) + Version (1 byte)
        buf[0..7].copy_from_slice(PMTILES_MAGIC);
        buf[7] = PMTILES_VERSION;

        // Offsets and lengths (8 bytes each, little-endian)
        buf[8..16].copy_from_slice(&self.root_dir_offset.to_le_bytes());
        buf[16..24].copy_from_slice(&self.root_dir_length.to_le_bytes());
        buf[24..32].copy_from_slice(&self.json_metadata_offset.to_le_bytes());
        buf[32..40].copy_from_slice(&self.json_metadata_length.to_le_bytes());
        buf[40..48].copy_from_slice(&self.leaf_dirs_offset.to_le_bytes());
        buf[48..56].copy_from_slice(&self.leaf_dirs_length.to_le_bytes());
        buf[56..64].copy_from_slice(&self.tile_data_offset.to_le_bytes());
        buf[64..72].copy_from_slice(&self.tile_data_length.to_le_bytes());

        // Tile counts
        buf[72..80].copy_from_slice(&self.addressed_tiles_count.to_le_bytes());
        buf[80..88].copy_from_slice(&self.tile_entries_count.to_le_bytes());
        buf[88..96].copy_from_slice(&self.tile_contents_count.to_le_bytes());

        // Clustered flag
        buf[96] = if self.clustered { 1 } else { 0 };

        // Compression and type
        buf[97] = self.internal_compression as u8;
        buf[98] = self.tile_compression as u8;
        buf[99] = self.tile_type as u8;

        // Zoom levels
        buf[100] = self.min_zoom;
        buf[101] = self.max_zoom;

        // Bounds: lon/lat as i32 * 10,000,000 (spec-compliant encoding)
        let encode_coord = |v: f64| -> [u8; 4] { ((v * 10_000_000.0) as i32).to_le_bytes() };

        buf[102..106].copy_from_slice(&encode_coord(self.min_lon));
        buf[106..110].copy_from_slice(&encode_coord(self.min_lat));
        buf[110..114].copy_from_slice(&encode_coord(self.max_lon));
        buf[114..118].copy_from_slice(&encode_coord(self.max_lat));

        // Center: zoom + lon/lat
        buf[118] = self.center_zoom;
        buf[119..123].copy_from_slice(&encode_coord(self.center_lon));
        buf[123..127].copy_from_slice(&encode_coord(self.center_lat));

        buf
    }
}

/// Convert tile coordinates (z, x, y) to a TileID for PMTiles
///
/// Uses Hilbert curve ordering for spatial locality. The tile ID is a cumulative
/// position on the series of Hilbert curves starting at zoom level 0.
///
/// Examples from spec:
/// - Z=0, X=0, Y=0 → TileID=0
/// - Z=1, X=0, Y=0 → TileID=1
/// - Z=1, X=0, Y=1 → TileID=2
/// - Z=1, X=1, Y=1 → TileID=3
/// - Z=1, X=1, Y=0 → TileID=4
/// - Z=2, X=0, Y=0 → TileID=5
pub fn tile_id(z: u8, x: u32, y: u32) -> u64 {
    if z == 0 {
        return 0;
    }

    // Calculate base ID: sum of all tiles in previous zoom levels
    // At zoom z, there are 4^z tiles. Base for zoom z is sum of 4^i for i in 1..z
    let base_id: u64 = (1..z as u64).map(|i| 4u64.pow(i as u32)).sum();
    let hilbert_idx = xy_to_hilbert(z, x, y);
    base_id + hilbert_idx + 1
}

/// Convert x,y coordinates to Hilbert curve index at zoom level z
///
/// Implementation follows the standard Hilbert curve algorithm:
/// https://en.wikipedia.org/wiki/Hilbert_curve
fn xy_to_hilbert(z: u8, x: u32, y: u32) -> u64 {
    let n = 1u32 << z;
    let mut rx: u32;
    let mut ry: u32;
    let mut s: u32;
    let mut d: u64 = 0;
    let mut x = x;
    let mut y = y;

    s = n / 2;
    while s > 0 {
        rx = if (x & s) > 0 { 1 } else { 0 };
        ry = if (y & s) > 0 { 1 } else { 0 };
        d += (s as u64) * (s as u64) * ((3 * rx) ^ ry) as u64;

        // Rotate quadrant - use n-1 (full grid size - 1) not s-1
        if ry == 0 {
            if rx == 1 {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

// ============================================================================
// Task 8: Directory Encoding
// ============================================================================

/// A directory entry pointing to tile data
///
/// In PMTiles, directories are columnar: all tile_ids are stored together,
/// then all run_lengths, then all lengths, then all offsets.
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub tile_id: u64,
    pub offset: u64,
    pub length: u32,
    pub run_length: u32, // Number of consecutive tiles with same data (0 = leaf directory)
}

/// Encode a u64 as a varint (protobuf-style, little-endian)
///
/// Each byte uses 7 bits for data, MSB indicates continuation.
pub fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

/// Decode a varint from bytes
///
/// Returns (value, bytes_consumed) or None if invalid/incomplete.
pub fn decode_varint(data: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0;
    for (i, &byte) in data.iter().enumerate() {
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return None; // Overflow
        }
    }
    None
}

/// Encode directory entries in PMTiles columnar format with delta encoding
///
/// Format: count, delta_tile_ids[], run_lengths[], lengths[], offsets[]
/// All values are varints. Tile IDs use simple delta encoding.
///
/// Offset encoding follows the PMTiles v3 spec:
/// - If offset equals expected position (contiguous), encode as 0
/// - Otherwise, encode as offset + 1
///
/// This allows efficient representation of contiguous tile data (common case).
pub fn encode_directory(entries: &[DirEntry]) -> Vec<u8> {
    let mut buf = Vec::new();

    // Number of entries
    encode_varint(entries.len() as u64, &mut buf);

    if entries.is_empty() {
        return buf;
    }

    // Delta-encoded tile IDs
    let mut last_id = 0u64;
    for entry in entries {
        encode_varint(entry.tile_id - last_id, &mut buf);
        last_id = entry.tile_id;
    }

    // Run lengths
    for entry in entries {
        encode_varint(entry.run_length as u64, &mut buf);
    }

    // Lengths
    for entry in entries {
        encode_varint(entry.length as u64, &mut buf);
    }

    // Offset encoding per PMTiles v3 spec:
    // - For contiguous entries (offset == expected_offset): encode 0
    // - Otherwise: encode offset + 1
    let mut expected_offset = 0u64;
    for (i, entry) in entries.iter().enumerate() {
        let is_contiguous = i > 0 && entry.offset == expected_offset;
        if is_contiguous {
            encode_varint(0, &mut buf);
        } else {
            encode_varint(entry.offset + 1, &mut buf);
        }

        // Update expected offset for next entry (only if this entry has data)
        if entry.run_length > 0 {
            expected_offset = entry.offset + entry.length as u64;
        }
    }

    buf
}

/// Compress data with gzip (backward compatibility wrapper)
pub fn gzip_compress(data: &[u8]) -> std::io::Result<Vec<u8>> {
    compression::compress(data, Compression::Gzip)
}

// ============================================================================
// Task 9: Full PMTiles Writer
// ============================================================================

/// PMTiles v3 writer
///
/// Accumulates tiles in memory (sorted by tile_id via BTreeMap),
/// then writes the complete archive on finalize.
pub struct PmtilesWriter {
    tiles: BTreeMap<u64, Vec<u8>>, // tile_id -> compressed tile data
    min_zoom: u8,
    max_zoom: u8,
    bounds: TileBounds,
    layer_name: String,
    /// Field metadata: field name -> MVT type ("String", "Number", "Boolean")
    fields: HashMap<String, String>,
    /// Total feature count across all tiles
    total_features: u64,
    /// Feature count per zoom level
    features_per_zoom: HashMap<u8, u64>,
    /// Compression algorithm for tile data
    tile_compression: Compression,
    /// Compression algorithm for internal data (directories, metadata)
    internal_compression: Compression,
}

impl PmtilesWriter {
    /// Create a new PMTiles writer with default gzip compression
    pub fn new() -> Self {
        Self {
            tiles: BTreeMap::new(),
            min_zoom: 255,
            max_zoom: 0,
            bounds: TileBounds::empty(),
            layer_name: "layer".to_string(),
            fields: HashMap::new(),
            total_features: 0,
            features_per_zoom: HashMap::new(),
            tile_compression: Compression::Gzip,
            internal_compression: Compression::Gzip,
        }
    }

    /// Create a new PMTiles writer with specified compression
    ///
    /// Both tile data and internal data (directories, metadata) will use
    /// the same compression algorithm.
    pub fn with_compression(compression: Compression) -> Self {
        Self {
            tiles: BTreeMap::new(),
            min_zoom: 255,
            max_zoom: 0,
            bounds: TileBounds::empty(),
            layer_name: "layer".to_string(),
            fields: HashMap::new(),
            total_features: 0,
            features_per_zoom: HashMap::new(),
            tile_compression: compression,
            internal_compression: compression,
        }
    }

    /// Set the compression algorithm for tile data
    pub fn set_tile_compression(&mut self, compression: Compression) {
        self.tile_compression = compression;
    }

    /// Set the compression algorithm for internal data (directories, metadata)
    pub fn set_internal_compression(&mut self, compression: Compression) {
        self.internal_compression = compression;
    }

    /// Get the current tile compression setting
    pub fn tile_compression(&self) -> Compression {
        self.tile_compression
    }

    /// Get the current internal compression setting
    pub fn internal_compression(&self) -> Compression {
        self.internal_compression
    }

    /// Set the layer name for vector_layers metadata
    pub fn set_layer_name(&mut self, name: &str) {
        self.layer_name = name.to_string();
    }

    /// Set field metadata for vector_layers.fields
    ///
    /// Field types should be MVT-style: "String", "Number", or "Boolean"
    pub fn set_fields(&mut self, fields: HashMap<String, String>) {
        self.fields = fields;
    }

    /// Build the fields JSON object string
    fn build_fields_json(&self) -> String {
        if self.fields.is_empty() {
            return "{}".to_string();
        }

        // Sort field names for deterministic output
        let mut field_pairs: Vec<_> = self.fields.iter().collect();
        field_pairs.sort_by_key(|(k, _)| *k);

        let field_strings: Vec<String> = field_pairs
            .iter()
            .map(|(name, type_str)| format!(r#""{}":"{}""#, name, type_str))
            .collect();

        format!("{{{}}}", field_strings.join(","))
    }

    /// Build the tilestats JSON fragment
    fn build_tilestats_json(&self) -> String {
        if self.total_features == 0 {
            return String::new();
        }

        format!(
            r#""tilestats":{{"layerCount":1,"layers":[{{"layer":"{}","count":{},"attributeCount":{}}}]}},"#,
            self.layer_name,
            self.total_features,
            self.fields.len()
        )
    }

    /// Add a tile (will be gzip compressed)
    ///
    /// The tile data should be uncompressed MVT bytes.
    /// Use `add_tile_with_count` if you have feature count available.
    pub fn add_tile(&mut self, z: u8, x: u32, y: u32, data: &[u8]) -> std::io::Result<()> {
        self.add_tile_with_count(z, x, y, data, 0)
    }

    /// Add a tile with feature count for tilestats
    ///
    /// The tile data should be uncompressed MVT bytes.
    pub fn add_tile_with_count(
        &mut self,
        z: u8,
        x: u32,
        y: u32,
        data: &[u8],
        feature_count: usize,
    ) -> std::io::Result<()> {
        let compressed = compression::compress(data, self.tile_compression)?;
        let id = tile_id(z, x, y);
        self.tiles.insert(id, compressed);

        // Track zoom range
        self.min_zoom = self.min_zoom.min(z);
        self.max_zoom = self.max_zoom.max(z);

        // Track feature counts for tilestats
        self.total_features += feature_count as u64;
        *self.features_per_zoom.entry(z).or_insert(0) += feature_count as u64;

        Ok(())
    }

    /// Add a pre-compressed tile
    ///
    /// Use this if the tile data is already gzip compressed.
    pub fn add_tile_compressed(
        &mut self,
        z: u8,
        x: u32,
        y: u32,
        compressed_data: Vec<u8>,
    ) -> std::io::Result<()> {
        let id = tile_id(z, x, y);
        self.tiles.insert(id, compressed_data);

        self.min_zoom = self.min_zoom.min(z);
        self.max_zoom = self.max_zoom.max(z);

        Ok(())
    }

    /// Set geographic bounds for the tileset
    pub fn set_bounds(&mut self, bounds: &TileBounds) {
        self.bounds = *bounds;
    }

    /// Get the number of tiles added
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Write the PMTiles archive to a file
    ///
    /// Layout: [Header (127)] [Root Directory] [Metadata] [Tile Data]
    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| Error::PMTilesWrite(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        // Build tile data buffer and directory entries
        let mut tile_data_buf = Vec::new();
        let mut entries = Vec::new();

        for (&id, data) in &self.tiles {
            entries.push(DirEntry {
                tile_id: id,
                offset: tile_data_buf.len() as u64,
                length: data.len() as u32,
                run_length: 1, // Each tile is unique (no deduplication yet)
            });
            tile_data_buf.extend_from_slice(data);
        }

        // Encode and compress directory using configured internal compression
        let dir_bytes = encode_directory(&entries);
        let compressed_dir = compression::compress(&dir_bytes, self.internal_compression)
            .map_err(|e| Error::PMTilesWrite(format!("Failed to compress directory: {}", e)))?;

        // JSON metadata with vector_layers and tilestats
        let min_z = if self.min_zoom == 255 {
            0
        } else {
            self.min_zoom
        };
        let max_z = if self.max_zoom == 0 && self.tiles.is_empty() {
            0
        } else {
            self.max_zoom
        };
        let fields_json = self.build_fields_json();
        let tilestats_json = self.build_tilestats_json();
        let metadata = format!(
            r#"{{"vector_layers":[{{"id":"{}","minzoom":{},"maxzoom":{},"fields":{}}}],{}"format":"pbf","generator":"gpq-tiles"}}"#,
            self.layer_name, min_z, max_z, fields_json, tilestats_json
        );
        let compressed_metadata =
            compression::compress(metadata.as_bytes(), self.internal_compression)
                .map_err(|e| Error::PMTilesWrite(format!("Failed to compress metadata: {}", e)))?;

        // Calculate section offsets
        let root_dir_offset = 127u64;
        let root_dir_length = compressed_dir.len() as u64;
        let metadata_offset = root_dir_offset + root_dir_length;
        let metadata_length = compressed_metadata.len() as u64;
        let tile_data_offset = metadata_offset + metadata_length;
        let tile_data_length = tile_data_buf.len() as u64;

        // Build header
        let header = Header {
            root_dir_offset,
            root_dir_length,
            json_metadata_offset: metadata_offset,
            json_metadata_length: metadata_length,
            leaf_dirs_offset: 0, // No leaf directories (simple archive)
            leaf_dirs_length: 0,
            tile_data_offset,
            tile_data_length,
            addressed_tiles_count: self.tiles.len() as u64,
            tile_entries_count: entries.len() as u64,
            tile_contents_count: self.tiles.len() as u64,
            clustered: true,
            internal_compression: self.internal_compression,
            tile_compression: self.tile_compression,
            tile_type: TileType::Mvt,
            min_zoom: if self.min_zoom == 255 {
                0
            } else {
                self.min_zoom
            },
            max_zoom: if self.max_zoom == 0 && self.tiles.is_empty() {
                0
            } else {
                self.max_zoom
            },
            min_lon: self.bounds.lng_min,
            min_lat: self.bounds.lat_min,
            max_lon: self.bounds.lng_max,
            max_lat: self.bounds.lat_max,
            center_zoom: if self.tiles.is_empty() {
                0
            } else {
                (self.min_zoom + self.max_zoom) / 2
            },
            center_lon: (self.bounds.lng_min + self.bounds.lng_max) / 2.0,
            center_lat: (self.bounds.lat_min + self.bounds.lat_max) / 2.0,
        };

        // Write all sections
        writer
            .write_all(&header.to_bytes())
            .map_err(|e| Error::PMTilesWrite(format!("Failed to write header: {}", e)))?;
        writer
            .write_all(&compressed_dir)
            .map_err(|e| Error::PMTilesWrite(format!("Failed to write directory: {}", e)))?;
        writer
            .write_all(&compressed_metadata)
            .map_err(|e| Error::PMTilesWrite(format!("Failed to write metadata: {}", e)))?;
        writer
            .write_all(&tile_data_buf)
            .map_err(|e| Error::PMTilesWrite(format!("Failed to write tile data: {}", e)))?;

        writer
            .flush()
            .map_err(|e| Error::PMTilesWrite(format!("Failed to flush: {}", e)))?;

        Ok(())
    }
}

impl Default for PmtilesWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests (TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // -------------------------------------------------------------------------
    // Task 7: Header and Structures Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_header_size_is_127_bytes() {
        let header = Header::default();
        let bytes = header.to_bytes();
        assert_eq!(
            bytes.len(),
            127,
            "PMTiles v3 header must be exactly 127 bytes"
        );
    }

    #[test]
    fn test_header_magic_and_version() {
        let header = Header::default();
        let bytes = header.to_bytes();
        assert_eq!(&bytes[0..7], b"PMTiles", "Magic number must be 'PMTiles'");
        assert_eq!(bytes[7], 3, "Version must be 3");
    }

    #[test]
    fn test_header_default_offsets() {
        let header = Header::default();
        let bytes = header.to_bytes();

        // Root directory offset should be 127 (immediately after header)
        let root_offset = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(root_offset, 127);
    }

    #[test]
    fn test_header_bounds_encoding() {
        let header = Header {
            min_lon: -122.4194, // San Francisco
            min_lat: 37.7749,
            max_lon: -122.3894,
            max_lat: 37.8049,
            ..Default::default()
        };

        let bytes = header.to_bytes();

        // Decode min_lon (bytes 102-105)
        let min_lon_encoded = i32::from_le_bytes(bytes[102..106].try_into().unwrap());
        let min_lon_decoded = min_lon_encoded as f64 / 10_000_000.0;
        assert!(
            (min_lon_decoded - header.min_lon).abs() < 0.0001,
            "Lon encoding should preserve precision to ~0.0001 degrees"
        );
    }

    #[test]
    fn test_tile_id_zoom_0() {
        // At zoom 0, there's only one tile (0,0,0) with ID 0
        assert_eq!(tile_id(0, 0, 0), 0);
    }

    #[test]
    fn test_tile_id_zoom_1_matches_spec() {
        // From PMTiles spec examples:
        // Z=1, X=0, Y=0 → TileID=1
        // Z=1, X=0, Y=1 → TileID=2
        // Z=1, X=1, Y=1 → TileID=3
        // Z=1, X=1, Y=0 → TileID=4
        assert_eq!(tile_id(1, 0, 0), 1);
        assert_eq!(tile_id(1, 0, 1), 2);
        assert_eq!(tile_id(1, 1, 1), 3);
        assert_eq!(tile_id(1, 1, 0), 4);
    }

    #[test]
    fn test_tile_id_zoom_2_base() {
        // Z=2, X=0, Y=0 → TileID=5 (base for zoom 2)
        assert_eq!(tile_id(2, 0, 0), 5);
    }

    #[test]
    fn test_tile_id_unique_at_each_zoom() {
        // All tiles at a given zoom should have unique IDs
        for z in 0..=4u8 {
            let mut ids = Vec::new();
            let n = 1u32 << z;
            for y in 0..n {
                for x in 0..n {
                    ids.push(tile_id(z, x, y));
                }
            }
            let original_len = ids.len();
            ids.sort();
            ids.dedup();
            assert_eq!(
                ids.len(),
                original_len,
                "All tile IDs at zoom {} should be unique",
                z
            );
        }
    }

    #[test]
    fn test_tile_id_increasing_with_zoom() {
        // Max ID at zoom z should be less than min ID at zoom z+1
        for z in 0..4u8 {
            let n = 1u32 << z;
            let max_id_at_z = (0..n)
                .flat_map(|y| (0..n).map(move |x| tile_id(z, x, y)))
                .max()
                .unwrap();

            let min_id_at_z_plus_1 = tile_id(z + 1, 0, 0);

            assert!(
                max_id_at_z < min_id_at_z_plus_1,
                "Max ID at zoom {} ({}) should be < min ID at zoom {} ({})",
                z,
                max_id_at_z,
                z + 1,
                min_id_at_z_plus_1
            );
        }
    }

    // -------------------------------------------------------------------------
    // Task 8: Directory Encoding Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encode_varint_small_values() {
        // Values < 128 encode to single byte
        let mut buf = Vec::new();
        encode_varint(0, &mut buf);
        assert_eq!(buf, vec![0]);

        buf.clear();
        encode_varint(1, &mut buf);
        assert_eq!(buf, vec![1]);

        buf.clear();
        encode_varint(127, &mut buf);
        assert_eq!(buf, vec![127]);
    }

    #[test]
    fn test_encode_varint_128() {
        // 128 = 0x80 needs 2 bytes: [0x80, 0x01]
        let mut buf = Vec::new();
        encode_varint(128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x01]);
    }

    #[test]
    fn test_encode_varint_300() {
        // 300 = 0x12C = 0b1_0010_1100
        // Low 7 bits: 0010_1100 = 0x2C, with continuation: 0xAC
        // High bits: 0000_0010 = 0x02
        let mut buf = Vec::new();
        encode_varint(300, &mut buf);
        assert_eq!(buf, vec![0xAC, 0x02]);
    }

    #[test]
    fn test_varint_roundtrip() {
        let test_values = [0u64, 1, 127, 128, 255, 256, 300, 16383, 16384, u64::MAX];

        for &value in &test_values {
            let mut buf = Vec::new();
            encode_varint(value, &mut buf);
            let (decoded, bytes_consumed) = decode_varint(&buf).expect("Should decode");
            assert_eq!(decoded, value, "Roundtrip failed for {}", value);
            assert_eq!(bytes_consumed, buf.len());
        }
    }

    #[test]
    fn test_encode_directory_empty() {
        let entries: Vec<DirEntry> = vec![];
        let encoded = encode_directory(&entries);
        // Should just be count = 0
        assert_eq!(encoded, vec![0]);
    }

    #[test]
    fn test_encode_directory_single_entry() {
        let entries = vec![DirEntry {
            tile_id: 1,
            offset: 0,
            length: 100,
            run_length: 1,
        }];
        let encoded = encode_directory(&entries);

        // Should start with count = 1
        assert!(!encoded.is_empty());
        assert_eq!(encoded[0], 1);
    }

    #[test]
    fn test_encode_directory_multiple_entries() {
        let entries = vec![
            DirEntry {
                tile_id: 5,
                offset: 0,
                length: 100,
                run_length: 1,
            },
            DirEntry {
                tile_id: 42,
                offset: 100,
                length: 200,
                run_length: 1,
            },
            DirEntry {
                tile_id: 69,
                offset: 300,
                length: 50,
                run_length: 1,
            },
        ];
        let encoded = encode_directory(&entries);

        // Should start with count = 3
        assert_eq!(encoded[0], 3);

        // The encoding should be smaller than naive (due to delta encoding)
        // Each entry would be ~24 bytes naive, but delta should compress
        assert!(encoded.len() < entries.len() * 24);
    }

    #[test]
    fn test_gzip_compress_roundtrip() {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let original = b"Hello, PMTiles! This is test data.";
        let compressed = gzip_compress(original).expect("Should compress");

        // Should be shorter than original (for non-trivial data)
        // Note: very small inputs might expand

        // Decompress and verify
        let mut decoder = GzDecoder::new(&compressed[..]);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .expect("Should decompress");

        assert_eq!(decompressed, original);
    }

    // -------------------------------------------------------------------------
    // Task 9: Full Writer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_writer_creation() {
        let writer = PmtilesWriter::new();
        assert_eq!(writer.tile_count(), 0);
    }

    #[test]
    fn test_writer_add_single_tile() {
        let mut writer = PmtilesWriter::new();
        let mvt_data = vec![0x1a, 0x00]; // Minimal MVT-like data

        writer.add_tile(0, 0, 0, &mvt_data).unwrap();
        assert_eq!(writer.tile_count(), 1);
    }

    #[test]
    fn test_writer_creates_valid_pmtiles_file() {
        let mut writer = PmtilesWriter::new();

        // Add a minimal tile
        let mvt_data = vec![0x1a, 0x00];
        writer.add_tile(0, 0, 0, &mvt_data).unwrap();
        writer.set_bounds(&TileBounds::new(-180.0, -85.0, 180.0, 85.0));

        let path = Path::new("/tmp/test-pmtiles-writer.pmtiles");
        let _ = fs::remove_file(path);

        writer.write_to_file(path).expect("Should write file");

        // Verify file exists and has correct structure
        assert!(path.exists(), "File should exist");

        let data = fs::read(path).unwrap();

        // Check magic number and version
        assert_eq!(&data[0..7], b"PMTiles");
        assert_eq!(data[7], 3);

        // Check file is at least header size + some data
        assert!(data.len() > 127);

        // Check root directory offset points to position 127
        let root_offset = u64::from_le_bytes(data[8..16].try_into().unwrap());
        assert_eq!(root_offset, 127);

        // Clean up
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_writer_multiple_tiles_multiple_zooms() {
        let mut writer = PmtilesWriter::new();

        // Add tiles at zooms 0, 1, 2
        for z in 0..3u8 {
            let n = 1u32 << z;
            for x in 0..n {
                for y in 0..n {
                    let mvt_data = vec![0x1a, z, x as u8, y as u8];
                    writer.add_tile(z, x, y, &mvt_data).unwrap();
                }
            }
        }

        // Should have 1 + 4 + 16 = 21 tiles
        assert_eq!(writer.tile_count(), 21);

        writer.set_bounds(&TileBounds::new(-180.0, -85.0, 180.0, 85.0));

        let path = Path::new("/tmp/test-pmtiles-multi.pmtiles");
        let _ = fs::remove_file(path);

        writer.write_to_file(path).expect("Should write file");

        // Verify basic structure
        let data = fs::read(path).unwrap();
        assert_eq!(&data[0..7], b"PMTiles");
        assert_eq!(data[7], 3);

        // Check tile counts in header
        let addressed_count = u64::from_le_bytes(data[72..80].try_into().unwrap());
        assert_eq!(addressed_count, 21);

        // Check zoom range
        assert_eq!(data[100], 0); // min_zoom
        assert_eq!(data[101], 2); // max_zoom

        // Clean up
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_writer_empty_tileset() {
        let writer = PmtilesWriter::new();

        let path = Path::new("/tmp/test-pmtiles-empty.pmtiles");
        let _ = fs::remove_file(path);

        writer.write_to_file(path).expect("Should write empty file");

        let data = fs::read(path).unwrap();
        assert_eq!(&data[0..7], b"PMTiles");

        // Clean up
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_writer_tile_ordering() {
        let mut writer = PmtilesWriter::new();

        // Add tiles in random order
        writer.add_tile(2, 3, 3, &[1, 2, 3]).unwrap();
        writer.add_tile(0, 0, 0, &[4, 5, 6]).unwrap();
        writer.add_tile(1, 1, 0, &[7, 8, 9]).unwrap();

        // BTreeMap should maintain Hilbert curve order
        assert_eq!(writer.tile_count(), 3);

        let path = Path::new("/tmp/test-pmtiles-ordering.pmtiles");
        let _ = fs::remove_file(path);

        writer.write_to_file(path).expect("Should write file");

        // Should succeed (clustered mode requires sorted tiles)
        assert!(path.exists());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_writer_bounds_preserved() {
        let mut writer = PmtilesWriter::new();
        writer.add_tile(0, 0, 0, &[1, 2, 3]).unwrap();

        let bounds = TileBounds::new(-122.5, 37.7, -122.3, 37.9);
        writer.set_bounds(&bounds);

        let path = Path::new("/tmp/test-pmtiles-bounds.pmtiles");
        let _ = fs::remove_file(path);

        writer.write_to_file(path).expect("Should write file");

        let data = fs::read(path).unwrap();

        // Decode bounds from header
        let decode_coord = |offset: usize| -> f64 {
            let val = i32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            val as f64 / 10_000_000.0
        };

        let min_lon = decode_coord(102);
        let min_lat = decode_coord(106);
        let max_lon = decode_coord(110);
        let max_lat = decode_coord(114);

        assert!((min_lon - bounds.lng_min).abs() < 0.0001);
        assert!((min_lat - bounds.lat_min).abs() < 0.0001);
        assert!((max_lon - bounds.lng_max).abs() < 0.0001);
        assert!((max_lat - bounds.lat_max).abs() < 0.0001);

        let _ = fs::remove_file(path);
    }

    // -------------------------------------------------------------------------
    // Field Metadata Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_fields_json_empty() {
        let writer = PmtilesWriter::new();
        assert_eq!(writer.build_fields_json(), "{}");
    }

    #[test]
    fn test_build_fields_json_with_fields() {
        let mut writer = PmtilesWriter::new();
        let mut fields = HashMap::new();
        fields.insert("name".to_string(), "String".to_string());
        fields.insert("area".to_string(), "Number".to_string());
        writer.set_fields(fields);

        let json = writer.build_fields_json();
        // Fields are sorted alphabetically
        assert_eq!(json, r#"{"area":"Number","name":"String"}"#);
    }

    #[test]
    fn test_writer_field_metadata_in_output() {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut writer = PmtilesWriter::new();
        writer.add_tile(0, 0, 0, &[1, 2, 3]).unwrap();
        writer.set_layer_name("buildings");

        let mut fields = HashMap::new();
        fields.insert("name".to_string(), "String".to_string());
        fields.insert("height".to_string(), "Number".to_string());
        writer.set_fields(fields);

        let path = Path::new("/tmp/test-pmtiles-fields.pmtiles");
        let _ = fs::remove_file(path);

        writer.write_to_file(path).expect("Should write file");

        let data = fs::read(path).unwrap();

        // Extract metadata offset and length from header
        let metadata_offset = u64::from_le_bytes(data[24..32].try_into().unwrap()) as usize;
        let metadata_length = u64::from_le_bytes(data[32..40].try_into().unwrap()) as usize;

        // Decompress the metadata
        let compressed_metadata = &data[metadata_offset..metadata_offset + metadata_length];
        let mut decoder = GzDecoder::new(compressed_metadata);
        let mut metadata_json = String::new();
        decoder
            .read_to_string(&mut metadata_json)
            .expect("Should decompress metadata");

        // Verify fields are present
        assert!(metadata_json.contains(r#""height":"Number""#));
        assert!(metadata_json.contains(r#""name":"String""#));
        assert!(metadata_json.contains(r#""id":"buildings""#));

        let _ = fs::remove_file(path);
    }

    // -------------------------------------------------------------------------
    // Compression Configuration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_writer_with_compression_constructor() {
        let writer = PmtilesWriter::with_compression(Compression::Brotli);
        assert_eq!(writer.tile_compression(), Compression::Brotli);
        assert_eq!(writer.internal_compression(), Compression::Brotli);
    }

    #[test]
    fn test_writer_set_compression() {
        let mut writer = PmtilesWriter::new();
        assert_eq!(writer.tile_compression(), Compression::Gzip); // default

        writer.set_tile_compression(Compression::Zstd);
        assert_eq!(writer.tile_compression(), Compression::Zstd);

        writer.set_internal_compression(Compression::Brotli);
        assert_eq!(writer.internal_compression(), Compression::Brotli);
    }

    #[test]
    fn test_writer_brotli_compression() {
        let mut writer = PmtilesWriter::with_compression(Compression::Brotli);
        let mvt_data = vec![0x1a; 100]; // Compressible data

        writer.add_tile(0, 0, 0, &mvt_data).unwrap();
        writer.set_bounds(&TileBounds::new(-180.0, -85.0, 180.0, 85.0));

        let path = Path::new("/tmp/test-pmtiles-brotli.pmtiles");
        let _ = fs::remove_file(path);

        writer
            .write_to_file(path)
            .expect("Should write file with brotli");

        let data = fs::read(path).unwrap();

        // Verify header
        assert_eq!(&data[0..7], b"PMTiles");
        assert_eq!(data[7], 3);

        // Check compression bytes in header (97 = internal, 98 = tile)
        assert_eq!(data[97], Compression::Brotli as u8);
        assert_eq!(data[98], Compression::Brotli as u8);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_writer_zstd_compression() {
        let mut writer = PmtilesWriter::with_compression(Compression::Zstd);
        let mvt_data = vec![0x1a; 100];

        writer.add_tile(0, 0, 0, &mvt_data).unwrap();
        writer.set_bounds(&TileBounds::new(-180.0, -85.0, 180.0, 85.0));

        let path = Path::new("/tmp/test-pmtiles-zstd.pmtiles");
        let _ = fs::remove_file(path);

        writer
            .write_to_file(path)
            .expect("Should write file with zstd");

        let data = fs::read(path).unwrap();

        // Check compression bytes in header
        assert_eq!(data[97], Compression::Zstd as u8);
        assert_eq!(data[98], Compression::Zstd as u8);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_writer_no_compression() {
        let mut writer = PmtilesWriter::with_compression(Compression::None);
        let mvt_data = vec![0x1a, 0x00];

        writer.add_tile(0, 0, 0, &mvt_data).unwrap();
        writer.set_bounds(&TileBounds::new(-180.0, -85.0, 180.0, 85.0));

        let path = Path::new("/tmp/test-pmtiles-none.pmtiles");
        let _ = fs::remove_file(path);

        writer
            .write_to_file(path)
            .expect("Should write file without compression");

        let data = fs::read(path).unwrap();

        // Check compression bytes in header
        assert_eq!(data[97], Compression::None as u8);
        assert_eq!(data[98], Compression::None as u8);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_writer_mixed_compression() {
        // Test different compression for internal vs tile data
        let mut writer = PmtilesWriter::new();
        writer.set_internal_compression(Compression::Gzip);
        writer.set_tile_compression(Compression::Zstd);

        let mvt_data = vec![0x1a; 100];
        writer.add_tile(0, 0, 0, &mvt_data).unwrap();
        writer.set_bounds(&TileBounds::new(-180.0, -85.0, 180.0, 85.0));

        let path = Path::new("/tmp/test-pmtiles-mixed.pmtiles");
        let _ = fs::remove_file(path);

        writer
            .write_to_file(path)
            .expect("Should write file with mixed compression");

        let data = fs::read(path).unwrap();

        // Check compression bytes in header
        assert_eq!(data[97], Compression::Gzip as u8); // internal
        assert_eq!(data[98], Compression::Zstd as u8); // tile

        let _ = fs::remove_file(path);
    }
}
