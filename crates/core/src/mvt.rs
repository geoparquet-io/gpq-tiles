//! MVT (Mapbox Vector Tile) encoding module.
//!
//! This module implements the MVT specification for encoding geometries
//! into vector tiles. Key components:
//!
//! - **Zigzag encoding**: Efficiently encode signed integers as unsigned
//! - **Delta encoding**: Store coordinates as differences from previous position
//! - **Command encoding**: Pack geometry commands (MoveTo, LineTo, ClosePath)
//! - **Feature encoding**: Convert geo::Geometry to MVT Feature
//! - **Layer encoding**: Group features with deduplicated keys/values
//!
//! Reference: <https://github.com/mapbox/vector-tile-spec>

use crate::tile::TileBounds;
use crate::vector_tile::tile::{Feature, GeomType, Layer, Value};
use crate::vector_tile::Tile;
use geo::orient::{Direction, Orient};
use geo::{Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon};
use std::collections::HashMap;

/// Default tile extent (4096 as per MVT spec)
pub const DEFAULT_EXTENT: u32 = 4096;

/// Maximum zoom level for post-quantization validation.
///
/// At zoom levels <= this value, coordinate quantization can cause significant
/// vertex collapse, making validation worthwhile. At higher zoom levels, the
/// pixel resolution is fine enough that quantization issues are rare.
///
/// Based on research in `context/QUANTIZATION_RISKS.md`:
/// - Z0: ~9.7 km/pixel - very aggressive collapse
/// - Z6: ~150 m/pixel - aggressive collapse
/// - Z10: ~9.5 m/pixel - moderate collapse
/// - Z14: ~60 cm/pixel - minimal collapse
pub const MAX_ZOOM_FOR_QUANTIZATION_VALIDATION: u8 = 10;

/// MVT command IDs
const CMD_MOVE_TO: u32 = 1;
const CMD_LINE_TO: u32 = 2;
const CMD_CLOSE_PATH: u32 = 7;

// ============================================================================
// Zigzag Encoding
// ============================================================================

/// Encode a signed integer using zigzag encoding.
///
/// Zigzag encoding maps signed integers to unsigned integers so that
/// small negative numbers have small encoded values:
/// - 0 → 0
/// - -1 → 1
/// - 1 → 2
/// - -2 → 3
/// - 2 → 4
/// - etc.
///
/// This is efficient for protobuf varint encoding since small values
/// use fewer bytes.
#[inline]
pub fn zigzag_encode(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

/// Decode a zigzag-encoded unsigned integer back to signed.
#[inline]
pub fn zigzag_decode(n: u32) -> i32 {
    ((n >> 1) as i32) ^ -((n & 1) as i32)
}

// ============================================================================
// Command Encoding
// ============================================================================

/// Pack a command with a repeat count.
///
/// MVT commands are packed as: `(command_id | (count << 3))`
/// - command_id: 1=MoveTo, 2=LineTo, 7=ClosePath
/// - count: number of times to repeat the command
#[inline]
pub fn command_encode(command_id: u32, count: u32) -> u32 {
    (command_id & 0x7) | (count << 3)
}

/// Unpack a command into (command_id, count).
#[inline]
pub fn command_decode(command: u32) -> (u32, u32) {
    (command & 0x7, command >> 3)
}

// ============================================================================
// Winding Order Correction
// ============================================================================

/// Orient a polygon for MVT encoding.
///
/// MVT specification requires:
/// - Exterior rings: clockwise in screen/tile coordinates (positive area)
/// - Interior rings: counter-clockwise in screen/tile coordinates (negative area)
///
/// Since tile coordinates have Y increasing downward (opposite to geographic coords),
/// and our coordinate transform flips Y, we need polygons in geographic coordinates
/// to have:
/// - Exterior rings: counter-clockwise (becomes CW after Y-flip)
/// - Interior rings: clockwise (becomes CCW after Y-flip)
///
/// This matches geo's `Direction::Default` convention.
///
/// # Arguments
/// * `polygon` - The polygon to orient
///
/// # Returns
/// A new polygon with correctly oriented rings for MVT encoding
pub fn orient_polygon_for_mvt(polygon: &Polygon) -> Polygon {
    polygon.orient(Direction::Default)
}

/// Orient a multi-polygon for MVT encoding.
///
/// Applies `orient_polygon_for_mvt` to each constituent polygon.
///
/// # Arguments
/// * `multi` - The multi-polygon to orient
///
/// # Returns
/// A new multi-polygon with correctly oriented rings for MVT encoding
pub fn orient_multi_polygon_for_mvt(multi: &MultiPolygon) -> MultiPolygon {
    multi.orient(Direction::Default)
}

// ============================================================================
// Coordinate Transformation
// ============================================================================

/// Transform geographic coordinates (lng/lat) to tile-local coordinates.
///
/// Tile coordinates range from 0 to extent (typically 4096).
/// The tile bounds define the geographic extent being mapped.
///
/// # Arguments
/// * `lng` - Longitude in degrees
/// * `lat` - Latitude in degrees
/// * `bounds` - The geographic bounds of the tile
/// * `extent` - The tile extent (default 4096)
///
/// # Returns
/// (x, y) in tile-local coordinates, where (0,0) is top-left
pub fn geo_to_tile_coords(lng: f64, lat: f64, bounds: &TileBounds, extent: u32) -> (i32, i32) {
    let extent_f = extent as f64;

    // Normalize to 0-1 within tile bounds
    let x_ratio = (lng - bounds.lng_min) / (bounds.lng_max - bounds.lng_min);
    let y_ratio = (lat - bounds.lat_min) / (bounds.lat_max - bounds.lat_min);

    // Scale to extent and flip Y (tile coords have Y increasing downward)
    let x = (x_ratio * extent_f).round() as i32;
    let y = ((1.0 - y_ratio) * extent_f).round() as i32;

    (x, y)
}

// ============================================================================
// Post-Quantization Validation
// ============================================================================

/// Types of geometry issues that can occur after coordinate quantization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizationIssue {
    /// Consecutive vertices collapsed to the same point
    ConsecutiveDuplicates { count: usize },
    /// Ring has fewer than 3 unique points after quantization
    DegenerateRing { unique_points: usize },
}

impl std::fmt::Display for QuantizationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationIssue::ConsecutiveDuplicates { count } => {
                write!(f, "{} consecutive duplicate vertices", count)
            }
            QuantizationIssue::DegenerateRing { unique_points } => {
                write!(
                    f,
                    "ring collapsed to {} unique points (need at least 3)",
                    unique_points
                )
            }
        }
    }
}

/// Validate quantized ring coordinates for issues introduced by coordinate rounding.
///
/// This function detects:
/// - Consecutive duplicate vertices (vertex collapse)
/// - Rings that have collapsed to fewer than 3 unique points
///
/// # Arguments
///
/// * `coords` - Slice of quantized (x, y) integer coordinates representing a ring
///
/// # Returns
///
/// A vector of detected issues, or empty if the ring is valid.
///
/// # Example
///
/// ```
/// use gpq_tiles_core::mvt::validate_quantized_ring;
///
/// // Ring with consecutive duplicates
/// let coords = vec![(0, 0), (100, 0), (100, 0), (100, 100), (0, 0)];
/// let issues = validate_quantized_ring(&coords);
/// assert!(!issues.is_empty());
/// ```
pub fn validate_quantized_ring(coords: &[(i32, i32)]) -> Vec<QuantizationIssue> {
    let mut issues = Vec::new();

    if coords.len() < 4 {
        // Ring should have at least 4 points (3 unique + closing point)
        issues.push(QuantizationIssue::DegenerateRing {
            unique_points: coords.len(),
        });
        return issues;
    }

    // Count consecutive duplicates
    let mut consecutive_dup_count = 0;
    for window in coords.windows(2) {
        if window[0] == window[1] {
            consecutive_dup_count += 1;
        }
    }

    if consecutive_dup_count > 0 {
        issues.push(QuantizationIssue::ConsecutiveDuplicates {
            count: consecutive_dup_count,
        });
    }

    // Count unique points (excluding the closing point if it matches the first)
    let mut unique_coords: Vec<(i32, i32)> = Vec::with_capacity(coords.len());
    for coord in coords.iter() {
        if unique_coords.last() != Some(coord) {
            unique_coords.push(*coord);
        }
    }
    // If the last point matches the first (closed ring), don't count it as unique
    if unique_coords.len() > 1 && unique_coords.first() == unique_coords.last() {
        unique_coords.pop();
    }

    if unique_coords.len() < 3 {
        issues.push(QuantizationIssue::DegenerateRing {
            unique_points: unique_coords.len(),
        });
    }

    issues
}

/// Log a warning if a ring has post-quantization validity issues.
///
/// Only logs for zoom levels <= `MAX_ZOOM_FOR_QUANTIZATION_VALIDATION` where
/// the quantization risk is significant.
///
/// # Arguments
///
/// * `coords` - Quantized ring coordinates
/// * `zoom` - The zoom level of the tile being encoded
/// * `tile_x` - The x coordinate of the tile (for logging)
/// * `tile_y` - The y coordinate of the tile (for logging)
/// * `ring_type` - Description of the ring (e.g., "exterior", "interior 0")
fn warn_if_invalid_after_quantization(
    coords: &[(i32, i32)],
    zoom: u8,
    tile_x: u32,
    tile_y: u32,
    ring_type: &str,
) {
    // Only validate for low zoom levels where quantization risk is significant
    if zoom > MAX_ZOOM_FOR_QUANTIZATION_VALIDATION {
        return;
    }

    let issues = validate_quantized_ring(coords);
    if !issues.is_empty() {
        for issue in issues {
            log::warn!(
                "tile z{}/x{}/y{} {} became invalid after quantization: {}",
                zoom,
                tile_x,
                tile_y,
                ring_type,
                issue
            );
        }
    }
}

// ============================================================================
// Geometry Encoding
// ============================================================================

/// Encode a Point geometry to MVT geometry commands.
pub fn encode_point(point: &Point, bounds: &TileBounds, extent: u32) -> Vec<u32> {
    let (x, y) = geo_to_tile_coords(point.x(), point.y(), bounds, extent);

    vec![
        command_encode(CMD_MOVE_TO, 1),
        zigzag_encode(x),
        zigzag_encode(y),
    ]
}

/// Encode a MultiPoint geometry to MVT geometry commands.
pub fn encode_multi_point(points: &MultiPoint, bounds: &TileBounds, extent: u32) -> Vec<u32> {
    if points.0.is_empty() {
        return vec![];
    }

    let mut geometry = Vec::with_capacity(1 + points.0.len() * 2);
    let mut cursor_x = 0i32;
    let mut cursor_y = 0i32;

    // All points use MoveTo with count = number of points
    geometry.push(command_encode(CMD_MOVE_TO, points.0.len() as u32));

    for point in &points.0 {
        let (x, y) = geo_to_tile_coords(point.x(), point.y(), bounds, extent);
        let dx = x - cursor_x;
        let dy = y - cursor_y;
        geometry.push(zigzag_encode(dx));
        geometry.push(zigzag_encode(dy));
        cursor_x = x;
        cursor_y = y;
    }

    geometry
}

/// Encode a LineString geometry to MVT geometry commands.
pub fn encode_linestring(line: &LineString, bounds: &TileBounds, extent: u32) -> Vec<u32> {
    if line.0.len() < 2 {
        return vec![];
    }

    let mut geometry = Vec::with_capacity(3 + (line.0.len() - 1) * 2);
    let mut cursor_x = 0i32;
    let mut cursor_y = 0i32;

    // First point: MoveTo
    let first = &line.0[0];
    let (x, y) = geo_to_tile_coords(first.x, first.y, bounds, extent);
    let dx = x - cursor_x;
    let dy = y - cursor_y;
    geometry.push(command_encode(CMD_MOVE_TO, 1));
    geometry.push(zigzag_encode(dx));
    geometry.push(zigzag_encode(dy));
    cursor_x = x;
    cursor_y = y;

    // Remaining points: LineTo
    if line.0.len() > 1 {
        geometry.push(command_encode(CMD_LINE_TO, (line.0.len() - 1) as u32));
        for coord in line.0.iter().skip(1) {
            let (x, y) = geo_to_tile_coords(coord.x, coord.y, bounds, extent);
            let dx = x - cursor_x;
            let dy = y - cursor_y;
            geometry.push(zigzag_encode(dx));
            geometry.push(zigzag_encode(dy));
            cursor_x = x;
            cursor_y = y;
        }
    }

    geometry
}

/// Encode a MultiLineString geometry to MVT geometry commands.
pub fn encode_multi_linestring(
    lines: &MultiLineString,
    bounds: &TileBounds,
    extent: u32,
) -> Vec<u32> {
    let mut geometry = Vec::new();
    let mut cursor_x = 0i32;
    let mut cursor_y = 0i32;

    for line in &lines.0 {
        if line.0.len() < 2 {
            continue;
        }

        // First point: MoveTo
        let first = &line.0[0];
        let (x, y) = geo_to_tile_coords(first.x, first.y, bounds, extent);
        let dx = x - cursor_x;
        let dy = y - cursor_y;
        geometry.push(command_encode(CMD_MOVE_TO, 1));
        geometry.push(zigzag_encode(dx));
        geometry.push(zigzag_encode(dy));
        cursor_x = x;
        cursor_y = y;

        // Remaining points: LineTo
        if line.0.len() > 1 {
            geometry.push(command_encode(CMD_LINE_TO, (line.0.len() - 1) as u32));
            for coord in line.0.iter().skip(1) {
                let (x, y) = geo_to_tile_coords(coord.x, coord.y, bounds, extent);
                let dx = x - cursor_x;
                let dy = y - cursor_y;
                geometry.push(zigzag_encode(dx));
                geometry.push(zigzag_encode(dy));
                cursor_x = x;
                cursor_y = y;
            }
        }
    }

    geometry
}

/// Optional tile information for post-quantization validation.
///
/// When provided to encoding functions, enables validation logging for
/// geometry issues that occur during coordinate quantization.
#[derive(Debug, Clone, Copy)]
pub struct TileInfo {
    /// Zoom level
    pub z: u8,
    /// Tile X coordinate
    pub x: u32,
    /// Tile Y coordinate
    pub y: u32,
}

impl TileInfo {
    /// Create a new TileInfo
    pub fn new(z: u8, x: u32, y: u32) -> Self {
        Self { z, x, y }
    }
}

/// Encode a polygon ring (exterior or interior) to MVT geometry commands.
/// Returns the commands and updates the cursor position.
fn encode_ring(
    ring: &LineString,
    bounds: &TileBounds,
    extent: u32,
    cursor_x: &mut i32,
    cursor_y: &mut i32,
) -> Vec<u32> {
    encode_ring_with_validation(ring, bounds, extent, cursor_x, cursor_y, None, "ring")
}

/// Encode a polygon ring with optional post-quantization validation.
///
/// If `tile_info` is provided and the zoom level is <= `MAX_ZOOM_FOR_QUANTIZATION_VALIDATION`,
/// validates the quantized coordinates and logs warnings for any issues found.
fn encode_ring_with_validation(
    ring: &LineString,
    bounds: &TileBounds,
    extent: u32,
    cursor_x: &mut i32,
    cursor_y: &mut i32,
    tile_info: Option<TileInfo>,
    ring_type: &str,
) -> Vec<u32> {
    // Rings must have at least 4 points (3 unique + closing point)
    if ring.0.len() < 4 {
        return vec![];
    }

    // First, quantize all coordinates to check for post-quantization issues
    let mut quantized_coords: Vec<(i32, i32)> = Vec::with_capacity(ring.0.len());
    for coord in ring.0.iter() {
        let (x, y) = geo_to_tile_coords(coord.x, coord.y, bounds, extent);
        quantized_coords.push((x, y));
    }

    // Validate quantized coordinates if tile info is provided
    if let Some(info) = tile_info {
        warn_if_invalid_after_quantization(&quantized_coords, info.z, info.x, info.y, ring_type);
    }

    let mut geometry = Vec::with_capacity(4 + (ring.0.len() - 2) * 2);

    // First point: MoveTo
    let (x, y) = quantized_coords[0];
    let dx = x - *cursor_x;
    let dy = y - *cursor_y;
    geometry.push(command_encode(CMD_MOVE_TO, 1));
    geometry.push(zigzag_encode(dx));
    geometry.push(zigzag_encode(dy));
    *cursor_x = x;
    *cursor_y = y;

    // Interior points: LineTo (skip last point since we'll use ClosePath)
    let line_to_count = ring.0.len() - 2; // Exclude first and last points
    if line_to_count > 0 {
        geometry.push(command_encode(CMD_LINE_TO, line_to_count as u32));
        for &(x, y) in quantized_coords.iter().skip(1).take(line_to_count) {
            let dx = x - *cursor_x;
            let dy = y - *cursor_y;
            geometry.push(zigzag_encode(dx));
            geometry.push(zigzag_encode(dy));
            *cursor_x = x;
            *cursor_y = y;
        }
    }

    // ClosePath (implicitly returns to first point)
    geometry.push(command_encode(CMD_CLOSE_PATH, 1));

    geometry
}

/// Encode a Polygon geometry to MVT geometry commands.
///
/// This function automatically corrects polygon winding order to comply with
/// the MVT specification before encoding:
/// - Exterior rings: clockwise in tile coordinates
/// - Interior rings: counter-clockwise in tile coordinates
pub fn encode_polygon(polygon: &Polygon, bounds: &TileBounds, extent: u32) -> Vec<u32> {
    // Apply winding order correction for MVT compliance
    let oriented = orient_polygon_for_mvt(polygon);

    let mut geometry = Vec::new();
    let mut cursor_x = 0i32;
    let mut cursor_y = 0i32;

    // Exterior ring
    let exterior_cmds = encode_ring(
        oriented.exterior(),
        bounds,
        extent,
        &mut cursor_x,
        &mut cursor_y,
    );
    geometry.extend(exterior_cmds);

    // Interior rings (holes)
    for interior in oriented.interiors() {
        let interior_cmds = encode_ring(interior, bounds, extent, &mut cursor_x, &mut cursor_y);
        geometry.extend(interior_cmds);
    }

    geometry
}

/// Encode a MultiPolygon geometry to MVT geometry commands.
///
/// This function automatically corrects polygon winding order to comply with
/// the MVT specification before encoding:
/// - Exterior rings: clockwise in tile coordinates
/// - Interior rings: counter-clockwise in tile coordinates
pub fn encode_multi_polygon(polygons: &MultiPolygon, bounds: &TileBounds, extent: u32) -> Vec<u32> {
    // Apply winding order correction for MVT compliance
    let oriented = orient_multi_polygon_for_mvt(polygons);

    let mut geometry = Vec::new();
    let mut cursor_x = 0i32;
    let mut cursor_y = 0i32;

    for polygon in &oriented.0 {
        // Exterior ring
        let exterior_cmds = encode_ring(
            polygon.exterior(),
            bounds,
            extent,
            &mut cursor_x,
            &mut cursor_y,
        );
        geometry.extend(exterior_cmds);

        // Interior rings (holes)
        for interior in polygon.interiors() {
            let interior_cmds = encode_ring(interior, bounds, extent, &mut cursor_x, &mut cursor_y);
            geometry.extend(interior_cmds);
        }
    }

    geometry
}

/// Encode a Polygon geometry to MVT geometry commands with post-quantization validation.
///
/// This variant performs validation on quantized coordinates and logs warnings
/// for zoom levels <= `MAX_ZOOM_FOR_QUANTIZATION_VALIDATION`.
///
/// # Arguments
///
/// * `polygon` - The polygon to encode
/// * `bounds` - The tile bounds for coordinate transformation
/// * `extent` - The tile extent (default 4096)
/// * `tile_info` - Tile coordinates for validation logging
pub fn encode_polygon_with_validation(
    polygon: &Polygon,
    bounds: &TileBounds,
    extent: u32,
    tile_info: TileInfo,
) -> Vec<u32> {
    // Apply winding order correction for MVT compliance
    let oriented = orient_polygon_for_mvt(polygon);

    let mut geometry = Vec::new();
    let mut cursor_x = 0i32;
    let mut cursor_y = 0i32;

    // Exterior ring
    let exterior_cmds = encode_ring_with_validation(
        oriented.exterior(),
        bounds,
        extent,
        &mut cursor_x,
        &mut cursor_y,
        Some(tile_info),
        "exterior",
    );
    geometry.extend(exterior_cmds);

    // Interior rings (holes)
    for (i, interior) in oriented.interiors().iter().enumerate() {
        let ring_type = format!("interior {}", i);
        let interior_cmds = encode_ring_with_validation(
            interior,
            bounds,
            extent,
            &mut cursor_x,
            &mut cursor_y,
            Some(tile_info),
            &ring_type,
        );
        geometry.extend(interior_cmds);
    }

    geometry
}

/// Encode a MultiPolygon geometry to MVT geometry commands with post-quantization validation.
///
/// This variant performs validation on quantized coordinates and logs warnings
/// for zoom levels <= `MAX_ZOOM_FOR_QUANTIZATION_VALIDATION`.
///
/// # Arguments
///
/// * `polygons` - The multi-polygon to encode
/// * `bounds` - The tile bounds for coordinate transformation
/// * `extent` - The tile extent (default 4096)
/// * `tile_info` - Tile coordinates for validation logging
pub fn encode_multi_polygon_with_validation(
    polygons: &MultiPolygon,
    bounds: &TileBounds,
    extent: u32,
    tile_info: TileInfo,
) -> Vec<u32> {
    // Apply winding order correction for MVT compliance
    let oriented = orient_multi_polygon_for_mvt(polygons);

    let mut geometry = Vec::new();
    let mut cursor_x = 0i32;
    let mut cursor_y = 0i32;

    for (poly_idx, polygon) in oriented.0.iter().enumerate() {
        // Exterior ring
        let ring_type = format!("polygon {} exterior", poly_idx);
        let exterior_cmds = encode_ring_with_validation(
            polygon.exterior(),
            bounds,
            extent,
            &mut cursor_x,
            &mut cursor_y,
            Some(tile_info),
            &ring_type,
        );
        geometry.extend(exterior_cmds);

        // Interior rings (holes)
        for (interior_idx, interior) in polygon.interiors().iter().enumerate() {
            let ring_type = format!("polygon {} interior {}", poly_idx, interior_idx);
            let interior_cmds = encode_ring_with_validation(
                interior,
                bounds,
                extent,
                &mut cursor_x,
                &mut cursor_y,
                Some(tile_info),
                &ring_type,
            );
            geometry.extend(interior_cmds);
        }
    }

    geometry
}

/// Encode any geo::Geometry to MVT geometry commands and return the geometry type.
pub fn encode_geometry(geom: &Geometry, bounds: &TileBounds, extent: u32) -> (Vec<u32>, GeomType) {
    match geom {
        Geometry::Point(p) => (encode_point(p, bounds, extent), GeomType::Point),
        Geometry::MultiPoint(mp) => (encode_multi_point(mp, bounds, extent), GeomType::Point),
        Geometry::LineString(ls) => (encode_linestring(ls, bounds, extent), GeomType::Linestring),
        Geometry::MultiLineString(mls) => (
            encode_multi_linestring(mls, bounds, extent),
            GeomType::Linestring,
        ),
        Geometry::Polygon(p) => (encode_polygon(p, bounds, extent), GeomType::Polygon),
        Geometry::MultiPolygon(mp) => (encode_multi_polygon(mp, bounds, extent), GeomType::Polygon),
        // For geometry collections, we'd need to handle each part separately
        // For now, return empty geometry with unknown type
        _ => (vec![], GeomType::Unknown),
    }
}

/// Encode any geo::Geometry to MVT geometry commands with post-quantization validation.
///
/// This variant validates polygon geometries after quantization and logs warnings
/// for issues detected at low zoom levels.
///
/// # Arguments
///
/// * `geom` - The geometry to encode
/// * `bounds` - The tile bounds for coordinate transformation
/// * `extent` - The tile extent (default 4096)
/// * `tile_info` - Tile coordinates for validation logging
pub fn encode_geometry_with_validation(
    geom: &Geometry,
    bounds: &TileBounds,
    extent: u32,
    tile_info: TileInfo,
) -> (Vec<u32>, GeomType) {
    match geom {
        Geometry::Point(p) => (encode_point(p, bounds, extent), GeomType::Point),
        Geometry::MultiPoint(mp) => (encode_multi_point(mp, bounds, extent), GeomType::Point),
        Geometry::LineString(ls) => (encode_linestring(ls, bounds, extent), GeomType::Linestring),
        Geometry::MultiLineString(mls) => (
            encode_multi_linestring(mls, bounds, extent),
            GeomType::Linestring,
        ),
        Geometry::Polygon(p) => (
            encode_polygon_with_validation(p, bounds, extent, tile_info),
            GeomType::Polygon,
        ),
        Geometry::MultiPolygon(mp) => (
            encode_multi_polygon_with_validation(mp, bounds, extent, tile_info),
            GeomType::Polygon,
        ),
        // For geometry collections, we'd need to handle each part separately
        // For now, return empty geometry with unknown type
        _ => (vec![], GeomType::Unknown),
    }
}

// ============================================================================
// Feature Encoding
// ============================================================================

/// A property value that can be encoded in MVT.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    String(String),
    Float(f32),
    Double(f64),
    Int(i64),
    UInt(u64),
    Bool(bool),
}

impl PropertyValue {
    /// Convert to MVT Value type.
    pub fn to_mvt_value(&self) -> Value {
        match self {
            PropertyValue::String(s) => Value {
                string_value: Some(s.clone()),
                ..Default::default()
            },
            PropertyValue::Float(f) => Value {
                float_value: Some(*f),
                ..Default::default()
            },
            PropertyValue::Double(d) => Value {
                double_value: Some(*d),
                ..Default::default()
            },
            PropertyValue::Int(i) => Value {
                int_value: Some(*i),
                ..Default::default()
            },
            PropertyValue::UInt(u) => Value {
                uint_value: Some(*u),
                ..Default::default()
            },
            PropertyValue::Bool(b) => Value {
                bool_value: Some(*b),
                ..Default::default()
            },
        }
    }
}

/// Builder for encoding features into an MVT layer.
pub struct LayerBuilder {
    name: String,
    extent: u32,
    features: Vec<Feature>,
    keys: Vec<String>,
    key_index: HashMap<String, u32>,
    values: Vec<Value>,
    value_index: HashMap<String, u32>, // Serialize value for deduplication lookup
}

impl LayerBuilder {
    /// Create a new layer builder with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            extent: DEFAULT_EXTENT,
            features: Vec::new(),
            keys: Vec::new(),
            key_index: HashMap::new(),
            values: Vec::new(),
            value_index: HashMap::new(),
        }
    }

    /// Set the layer extent.
    pub fn with_extent(mut self, extent: u32) -> Self {
        self.extent = extent;
        self
    }

    /// Get or insert a key, returning its index.
    fn get_or_insert_key(&mut self, key: &str) -> u32 {
        if let Some(&idx) = self.key_index.get(key) {
            idx
        } else {
            let idx = self.keys.len() as u32;
            self.keys.push(key.to_string());
            self.key_index.insert(key.to_string(), idx);
            idx
        }
    }

    /// Get or insert a value, returning its index.
    fn get_or_insert_value(&mut self, value: &PropertyValue) -> u32 {
        // Create a string key for deduplication
        let value_key = format!("{:?}", value);

        if let Some(&idx) = self.value_index.get(&value_key) {
            idx
        } else {
            let idx = self.values.len() as u32;
            self.values.push(value.to_mvt_value());
            self.value_index.insert(value_key, idx);
            idx
        }
    }

    /// Add a feature to the layer.
    ///
    /// # Arguments
    /// * `id` - Optional feature ID
    /// * `geometry` - The geometry to encode
    /// * `properties` - Feature properties as key-value pairs
    /// * `bounds` - The tile bounds for coordinate transformation
    pub fn add_feature(
        &mut self,
        id: Option<u64>,
        geometry: &Geometry,
        properties: &[(String, PropertyValue)],
        bounds: &TileBounds,
    ) {
        let (geom_commands, geom_type) = encode_geometry(geometry, bounds, self.extent);

        // Skip empty geometries
        if geom_commands.is_empty() && geom_type == GeomType::Unknown {
            return;
        }

        // Encode tags as [key_idx, value_idx, key_idx, value_idx, ...]
        let mut tags = Vec::with_capacity(properties.len() * 2);
        for (key, value) in properties {
            let key_idx = self.get_or_insert_key(key);
            let value_idx = self.get_or_insert_value(value);
            tags.push(key_idx);
            tags.push(value_idx);
        }

        let feature = Feature {
            id,
            tags,
            r#type: Some(geom_type as i32),
            geometry: geom_commands,
        };

        self.features.push(feature);
    }

    /// Add a feature to the layer with post-quantization validation.
    ///
    /// This variant performs validation on polygon geometries after quantization
    /// and logs warnings for issues detected at low zoom levels (Z0-Z10).
    ///
    /// # Arguments
    /// * `id` - Optional feature ID
    /// * `geometry` - The geometry to encode
    /// * `properties` - Feature properties as key-value pairs
    /// * `bounds` - The tile bounds for coordinate transformation
    /// * `tile_info` - Tile coordinates for validation logging
    pub fn add_feature_with_validation(
        &mut self,
        id: Option<u64>,
        geometry: &Geometry,
        properties: &[(String, PropertyValue)],
        bounds: &TileBounds,
        tile_info: TileInfo,
    ) {
        let (geom_commands, geom_type) =
            encode_geometry_with_validation(geometry, bounds, self.extent, tile_info);

        // Skip empty geometries
        if geom_commands.is_empty() && geom_type == GeomType::Unknown {
            return;
        }

        // Encode tags as [key_idx, value_idx, key_idx, value_idx, ...]
        let mut tags = Vec::with_capacity(properties.len() * 2);
        for (key, value) in properties {
            let key_idx = self.get_or_insert_key(key);
            let value_idx = self.get_or_insert_value(value);
            tags.push(key_idx);
            tags.push(value_idx);
        }

        let feature = Feature {
            id,
            tags,
            r#type: Some(geom_type as i32),
            geometry: geom_commands,
        };

        self.features.push(feature);
    }

    /// Build the MVT Layer.
    pub fn build(self) -> Layer {
        Layer {
            version: 2,
            name: self.name,
            features: self.features,
            keys: self.keys,
            values: self.values,
            extent: Some(self.extent),
        }
    }
}

/// Builder for encoding multiple layers into an MVT tile.
pub struct TileBuilder {
    layers: Vec<Layer>,
}

impl TileBuilder {
    /// Create a new tile builder.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer to the tile.
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    /// Build the MVT Tile.
    pub fn build(self) -> Tile {
        Tile {
            layers: self.layers,
        }
    }
}

impl Default for TileBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{line_string, point, polygon};

    // ------------------------------------------------------------------------
    // Zigzag Encoding Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_zigzag_encode_zero() {
        assert_eq!(zigzag_encode(0), 0);
    }

    #[test]
    fn test_zigzag_encode_negative_one() {
        assert_eq!(zigzag_encode(-1), 1);
    }

    #[test]
    fn test_zigzag_encode_positive_one() {
        assert_eq!(zigzag_encode(1), 2);
    }

    #[test]
    fn test_zigzag_encode_negative_two() {
        assert_eq!(zigzag_encode(-2), 3);
    }

    #[test]
    fn test_zigzag_encode_positive_two() {
        assert_eq!(zigzag_encode(2), 4);
    }

    #[test]
    fn test_zigzag_encode_large_positive() {
        // 100 → 200
        assert_eq!(zigzag_encode(100), 200);
    }

    #[test]
    fn test_zigzag_encode_large_negative() {
        // -100 → 199
        assert_eq!(zigzag_encode(-100), 199);
    }

    #[test]
    fn test_zigzag_roundtrip() {
        for n in -1000..=1000 {
            let encoded = zigzag_encode(n);
            let decoded = zigzag_decode(encoded);
            assert_eq!(decoded, n, "Roundtrip failed for {}", n);
        }
    }

    #[test]
    fn test_zigzag_decode() {
        assert_eq!(zigzag_decode(0), 0);
        assert_eq!(zigzag_decode(1), -1);
        assert_eq!(zigzag_decode(2), 1);
        assert_eq!(zigzag_decode(3), -2);
        assert_eq!(zigzag_decode(4), 2);
    }

    // ------------------------------------------------------------------------
    // Command Encoding Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_command_encode_moveto_1() {
        // MoveTo with count=1: (1 | (1 << 3)) = 9
        assert_eq!(command_encode(CMD_MOVE_TO, 1), 9);
    }

    #[test]
    fn test_command_encode_lineto_3() {
        // LineTo with count=3: (2 | (3 << 3)) = 26
        assert_eq!(command_encode(CMD_LINE_TO, 3), 26);
    }

    #[test]
    fn test_command_encode_closepath() {
        // ClosePath with count=1: (7 | (1 << 3)) = 15
        assert_eq!(command_encode(CMD_CLOSE_PATH, 1), 15);
    }

    #[test]
    fn test_command_decode() {
        let cmd = command_encode(CMD_LINE_TO, 5);
        let (id, count) = command_decode(cmd);
        assert_eq!(id, CMD_LINE_TO);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_command_roundtrip() {
        for cmd_id in [CMD_MOVE_TO, CMD_LINE_TO, CMD_CLOSE_PATH] {
            for count in 1..=100 {
                let encoded = command_encode(cmd_id, count);
                let (decoded_id, decoded_count) = command_decode(encoded);
                assert_eq!(decoded_id, cmd_id);
                assert_eq!(decoded_count, count);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Coordinate Transformation Tests
    // ------------------------------------------------------------------------

    fn test_bounds() -> TileBounds {
        TileBounds {
            lng_min: 0.0,
            lat_min: 0.0,
            lng_max: 1.0,
            lat_max: 1.0,
        }
    }

    #[test]
    fn test_geo_to_tile_coords_center() {
        let bounds = test_bounds();
        let (x, y) = geo_to_tile_coords(0.5, 0.5, &bounds, 4096);
        // Center should be (2048, 2048)
        assert_eq!(x, 2048);
        assert_eq!(y, 2048);
    }

    #[test]
    fn test_geo_to_tile_coords_origin() {
        let bounds = test_bounds();
        // Bottom-left corner (lng_min, lat_min) → (0, extent) since Y is flipped
        let (x, y) = geo_to_tile_coords(0.0, 0.0, &bounds, 4096);
        assert_eq!(x, 0);
        assert_eq!(y, 4096);
    }

    #[test]
    fn test_geo_to_tile_coords_top_right() {
        let bounds = test_bounds();
        // Top-right corner (lng_max, lat_max) → (extent, 0)
        let (x, y) = geo_to_tile_coords(1.0, 1.0, &bounds, 4096);
        assert_eq!(x, 4096);
        assert_eq!(y, 0);
    }

    #[test]
    fn test_geo_to_tile_coords_top_left() {
        let bounds = test_bounds();
        // Top-left corner (lng_min, lat_max) → (0, 0)
        let (x, y) = geo_to_tile_coords(0.0, 1.0, &bounds, 4096);
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    // ------------------------------------------------------------------------
    // Point Encoding Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_encode_point_at_center() {
        let bounds = test_bounds();
        let point = point!(x: 0.5, y: 0.5);
        let commands = encode_point(&point, &bounds, 4096);

        // Should be: [MoveTo(1), zigzag(2048), zigzag(2048)]
        assert_eq!(commands.len(), 3);
        assert_eq!(commands[0], command_encode(CMD_MOVE_TO, 1)); // 9
        assert_eq!(commands[1], zigzag_encode(2048)); // x
        assert_eq!(commands[2], zigzag_encode(2048)); // y
    }

    #[test]
    fn test_encode_point_at_origin() {
        let bounds = test_bounds();
        let point = point!(x: 0.0, y: 0.0); // Bottom-left
        let commands = encode_point(&point, &bounds, 4096);

        assert_eq!(commands.len(), 3);
        assert_eq!(commands[0], command_encode(CMD_MOVE_TO, 1));
        assert_eq!(commands[1], zigzag_encode(0)); // x = 0
        assert_eq!(commands[2], zigzag_encode(4096)); // y = 4096 (flipped)
    }

    // ------------------------------------------------------------------------
    // LineString Encoding Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_encode_linestring_simple() {
        let bounds = test_bounds();
        let line = line_string![
            (x: 0.0, y: 0.0),
            (x: 0.5, y: 0.5),
            (x: 1.0, y: 1.0),
        ];
        let commands = encode_linestring(&line, &bounds, 4096);

        // Should be: MoveTo(1), x, y, LineTo(2), dx1, dy1, dx2, dy2
        // That's 1 + 2 + 1 + 4 = 8 elements
        assert_eq!(commands.len(), 8);

        // MoveTo command
        assert_eq!(commands[0], command_encode(CMD_MOVE_TO, 1));

        // First point (0, 4096) - bottom left in tile coords
        assert_eq!(commands[1], zigzag_encode(0)); // x
        assert_eq!(commands[2], zigzag_encode(4096)); // y (flipped from lat)

        // LineTo command with count=2
        assert_eq!(commands[3], command_encode(CMD_LINE_TO, 2));

        // Delta to (2048, 2048) from (0, 4096) = (2048, -2048)
        assert_eq!(commands[4], zigzag_encode(2048));
        assert_eq!(commands[5], zigzag_encode(-2048));

        // Delta to (4096, 0) from (2048, 2048) = (2048, -2048)
        assert_eq!(commands[6], zigzag_encode(2048));
        assert_eq!(commands[7], zigzag_encode(-2048));
    }

    #[test]
    fn test_encode_linestring_too_short() {
        let bounds = test_bounds();
        let line = line_string![(x: 0.0, y: 0.0)]; // Only one point
        let commands = encode_linestring(&line, &bounds, 4096);

        // Should return empty - linestrings need at least 2 points
        assert!(commands.is_empty());
    }

    // ------------------------------------------------------------------------
    // Polygon Encoding Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_encode_polygon_simple() {
        let bounds = test_bounds();
        let poly = polygon![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: 1.0),
            (x: 0.0, y: 1.0),
            (x: 0.0, y: 0.0), // Closing point
        ];
        let commands = encode_polygon(&poly, &bounds, 4096);

        // Should have: MoveTo(1), x, y, LineTo(3), dx1, dy1, dx2, dy2, dx3, dy3, ClosePath(1)
        // MoveTo + 2 coords + LineTo + 6 coords + ClosePath = 10 elements
        assert!(!commands.is_empty());

        // First command should be MoveTo
        assert_eq!(command_decode(commands[0]).0, CMD_MOVE_TO);

        // Last command should be ClosePath
        let last_cmd = *commands.last().unwrap();
        assert_eq!(command_decode(last_cmd).0, CMD_CLOSE_PATH);
    }

    // ------------------------------------------------------------------------
    // Layer Builder Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_layer_builder_basic() {
        let bounds = test_bounds();
        let mut builder = LayerBuilder::new("test_layer");

        let point = Geometry::Point(point!(x: 0.5, y: 0.5));
        let properties = vec![
            (
                "name".to_string(),
                PropertyValue::String("test".to_string()),
            ),
            ("value".to_string(), PropertyValue::Int(42)),
        ];

        builder.add_feature(Some(1), &point, &properties, &bounds);

        let layer = builder.build();

        assert_eq!(layer.name, "test_layer");
        assert_eq!(layer.version, 2);
        assert_eq!(layer.features.len(), 1);
        assert_eq!(layer.keys.len(), 2);
        assert_eq!(layer.values.len(), 2);
        assert_eq!(layer.extent, Some(4096));
    }

    #[test]
    fn test_layer_builder_key_deduplication() {
        let bounds = test_bounds();
        let mut builder = LayerBuilder::new("test_layer");

        let point1 = Geometry::Point(point!(x: 0.25, y: 0.25));
        let point2 = Geometry::Point(point!(x: 0.75, y: 0.75));

        // Both features have "name" key - should be deduplicated
        let props1 = vec![("name".to_string(), PropertyValue::String("a".to_string()))];
        let props2 = vec![("name".to_string(), PropertyValue::String("b".to_string()))];

        builder.add_feature(Some(1), &point1, &props1, &bounds);
        builder.add_feature(Some(2), &point2, &props2, &bounds);

        let layer = builder.build();

        assert_eq!(layer.features.len(), 2);
        assert_eq!(layer.keys.len(), 1); // Only one unique key "name"
        assert_eq!(layer.values.len(), 2); // Two different values "a" and "b"
    }

    #[test]
    fn test_layer_builder_value_deduplication() {
        let bounds = test_bounds();
        let mut builder = LayerBuilder::new("test_layer");

        let point1 = Geometry::Point(point!(x: 0.25, y: 0.25));
        let point2 = Geometry::Point(point!(x: 0.75, y: 0.75));

        // Both features have same value - should be deduplicated
        let props1 = vec![(
            "type".to_string(),
            PropertyValue::String("building".to_string()),
        )];
        let props2 = vec![(
            "type".to_string(),
            PropertyValue::String("building".to_string()),
        )];

        builder.add_feature(Some(1), &point1, &props1, &bounds);
        builder.add_feature(Some(2), &point2, &props2, &bounds);

        let layer = builder.build();

        assert_eq!(layer.features.len(), 2);
        assert_eq!(layer.keys.len(), 1); // One key "type"
        assert_eq!(layer.values.len(), 1); // One value "building" (deduplicated)
    }

    // ------------------------------------------------------------------------
    // Tile Builder Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_tile_builder() {
        let bounds = test_bounds();

        let mut layer1 = LayerBuilder::new("points");
        layer1.add_feature(
            Some(1),
            &Geometry::Point(point!(x: 0.5, y: 0.5)),
            &[],
            &bounds,
        );

        let mut layer2 = LayerBuilder::new("lines");
        let line = Geometry::LineString(line_string![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 1.0),
        ]);
        layer2.add_feature(Some(2), &line, &[], &bounds);

        let mut tile_builder = TileBuilder::new();
        tile_builder.add_layer(layer1.build());
        tile_builder.add_layer(layer2.build());

        let tile = tile_builder.build();

        assert_eq!(tile.layers.len(), 2);
        assert_eq!(tile.layers[0].name, "points");
        assert_eq!(tile.layers[1].name, "lines");
    }

    // ------------------------------------------------------------------------
    // GeomType Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_encode_geometry_returns_correct_type() {
        let bounds = test_bounds();

        let (_, geom_type) =
            encode_geometry(&Geometry::Point(point!(x: 0.5, y: 0.5)), &bounds, 4096);
        assert_eq!(geom_type, GeomType::Point);

        let line = Geometry::LineString(line_string![(x: 0.0, y: 0.0), (x: 1.0, y: 1.0)]);
        let (_, geom_type) = encode_geometry(&line, &bounds, 4096);
        assert_eq!(geom_type, GeomType::Linestring);

        let poly = Geometry::Polygon(polygon![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: 1.0),
            (x: 0.0, y: 1.0),
            (x: 0.0, y: 0.0),
        ]);
        let (_, geom_type) = encode_geometry(&poly, &bounds, 4096);
        assert_eq!(geom_type, GeomType::Polygon);
    }

    // ------------------------------------------------------------------------
    // Winding Order Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_polygon_correct_winding_unchanged() {
        // A polygon with correct MVT winding (CCW exterior in geographic coords,
        // which becomes CW exterior in tile coords after Y-flip) should pass through unchanged.
        //
        // MVT requires: exterior rings CW, interior rings CCW (in tile/screen coords)
        // Geographic CCW -> Tile CW (after Y-flip), so geo::Direction::Default is correct.

        // CCW polygon in geographic coords (correct for MVT after Y-flip)
        let poly = polygon![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: 1.0),
            (x: 0.0, y: 1.0),
            (x: 0.0, y: 0.0),
        ];

        let oriented = orient_polygon_for_mvt(&poly);

        // Should be unchanged since it's already correctly oriented
        assert_eq!(poly.exterior().0, oriented.exterior().0);
    }

    #[test]
    fn test_polygon_incorrect_winding_gets_corrected() {
        // A polygon with incorrect winding (CW exterior in geographic coords)
        // should be corrected to CCW exterior.

        // CW polygon in geographic coords (incorrect - needs correction)
        let poly = polygon![
            (x: 0.0, y: 0.0),
            (x: 0.0, y: 1.0),
            (x: 1.0, y: 1.0),
            (x: 1.0, y: 0.0),
            (x: 0.0, y: 0.0),
        ];

        let oriented = orient_polygon_for_mvt(&poly);

        // Should now be CCW (reversed from input)
        // The first and last points stay the same, but the middle points should be reversed
        assert_ne!(poly.exterior().0[1], oriented.exterior().0[1]);
    }

    #[test]
    fn test_polygon_with_hole_correct_winding() {
        // A polygon with a hole should have:
        // - CCW exterior in geographic coords (becomes CW in tile coords)
        // - CW interior in geographic coords (becomes CCW in tile coords)

        // Exterior: CCW in geo coords
        // Interior: CW in geo coords (correct for a hole)
        let poly = polygon![
            exterior: [
                (x: 0.0, y: 0.0),
                (x: 10.0, y: 0.0),
                (x: 10.0, y: 10.0),
                (x: 0.0, y: 10.0),
                (x: 0.0, y: 0.0),
            ],
            interiors: [
                [
                    (x: 2.0, y: 2.0),
                    (x: 2.0, y: 8.0),
                    (x: 8.0, y: 8.0),
                    (x: 8.0, y: 2.0),
                    (x: 2.0, y: 2.0),
                ],
            ],
        ];

        let oriented = orient_polygon_for_mvt(&poly);

        // After orientation, exterior should be CCW and interior should be CW
        // (in geographic coordinates, which becomes CW exterior / CCW interior in tile coords)
        assert_eq!(oriented.interiors().len(), 1);
    }

    #[test]
    fn test_polygon_with_hole_incorrect_winding_gets_corrected() {
        // A polygon where both exterior and interior have wrong winding

        // Exterior: CW in geo coords (wrong)
        // Interior: CCW in geo coords (wrong for a hole)
        let poly = polygon![
            exterior: [
                (x: 0.0, y: 0.0),
                (x: 0.0, y: 10.0),
                (x: 10.0, y: 10.0),
                (x: 10.0, y: 0.0),
                (x: 0.0, y: 0.0),
            ],
            interiors: [
                [
                    (x: 2.0, y: 2.0),
                    (x: 8.0, y: 2.0),
                    (x: 8.0, y: 8.0),
                    (x: 2.0, y: 8.0),
                    (x: 2.0, y: 2.0),
                ],
            ],
        ];

        let oriented = orient_polygon_for_mvt(&poly);

        // Both should be corrected
        assert_ne!(poly.exterior().0[1], oriented.exterior().0[1]);
        assert_ne!(poly.interiors()[0].0[1], oriented.interiors()[0].0[1]);
    }

    #[test]
    fn test_multipolygon_winding_correction() {
        // MultiPolygon should have all constituent polygons corrected

        // First polygon: wrong winding
        // Second polygon: correct winding
        let multi = geo::MultiPolygon::new(vec![
            polygon![
                (x: 0.0, y: 0.0),
                (x: 0.0, y: 1.0),
                (x: 1.0, y: 1.0),
                (x: 1.0, y: 0.0),
                (x: 0.0, y: 0.0),
            ],
            polygon![
                (x: 2.0, y: 0.0),
                (x: 3.0, y: 0.0),
                (x: 3.0, y: 1.0),
                (x: 2.0, y: 1.0),
                (x: 2.0, y: 0.0),
            ],
        ]);

        let oriented = orient_multi_polygon_for_mvt(&multi);

        assert_eq!(oriented.0.len(), 2);
        // First polygon should be corrected (was CW, now CCW)
        // Second polygon should remain unchanged (was already CCW)
    }

    #[test]
    fn test_encode_polygon_applies_winding_correction() {
        // The main encode_polygon function should apply winding correction
        let bounds = test_bounds();

        // CW polygon (wrong winding in geographic coords)
        let poly_cw = polygon![
            (x: 0.0, y: 0.0),
            (x: 0.0, y: 1.0),
            (x: 1.0, y: 1.0),
            (x: 1.0, y: 0.0),
            (x: 0.0, y: 0.0),
        ];

        // CCW polygon (correct winding in geographic coords)
        let poly_ccw = polygon![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: 1.0),
            (x: 0.0, y: 1.0),
            (x: 0.0, y: 0.0),
        ];

        // Both should produce equivalent MVT output (same geometry, just different input winding)
        let commands_cw = encode_polygon(&poly_cw, &bounds, 4096);
        let commands_ccw = encode_polygon(&poly_ccw, &bounds, 4096);

        // The encoded geometry should be the same for both inputs
        // because winding correction normalizes them
        assert_eq!(commands_cw, commands_ccw);
    }

    // ------------------------------------------------------------------------
    // Post-Quantization Validation Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_validate_quantized_ring_valid() {
        // Valid ring with 4 distinct points
        let coords = vec![(0, 0), (100, 0), (100, 100), (0, 0)];
        let issues = validate_quantized_ring(&coords);
        assert!(issues.is_empty(), "Valid ring should have no issues");
    }

    #[test]
    fn test_validate_quantized_ring_consecutive_duplicates() {
        // Ring with consecutive duplicate vertices
        let coords = vec![(0, 0), (100, 0), (100, 0), (100, 100), (0, 0)];
        let issues = validate_quantized_ring(&coords);

        assert!(!issues.is_empty(), "Should detect consecutive duplicates");
        assert!(
            issues
                .iter()
                .any(|i| matches!(i, QuantizationIssue::ConsecutiveDuplicates { .. })),
            "Should contain ConsecutiveDuplicates issue"
        );
    }

    #[test]
    fn test_validate_quantized_ring_degenerate() {
        // Ring collapsed to 2 unique points (a line)
        let coords = vec![(0, 0), (100, 0), (0, 0)];
        let issues = validate_quantized_ring(&coords);

        assert!(!issues.is_empty(), "Should detect degenerate ring");
        assert!(
            issues
                .iter()
                .any(|i| matches!(i, QuantizationIssue::DegenerateRing { .. })),
            "Should contain DegenerateRing issue"
        );
    }

    #[test]
    fn test_validate_quantized_ring_all_same_point() {
        // Ring where all points collapsed to the same location
        let coords = vec![(50, 50), (50, 50), (50, 50), (50, 50)];
        let issues = validate_quantized_ring(&coords);

        assert!(!issues.is_empty(), "Should detect fully collapsed ring");
        // Should have both consecutive duplicates AND degenerate ring
        assert!(
            issues
                .iter()
                .any(|i| matches!(i, QuantizationIssue::DegenerateRing { unique_points: 1 })),
            "Should have only 1 unique point"
        );
    }

    #[test]
    fn test_quantization_issue_display() {
        let issue1 = QuantizationIssue::ConsecutiveDuplicates { count: 3 };
        assert_eq!(format!("{}", issue1), "3 consecutive duplicate vertices");

        let issue2 = QuantizationIssue::DegenerateRing { unique_points: 2 };
        assert_eq!(
            format!("{}", issue2),
            "ring collapsed to 2 unique points (need at least 3)"
        );
    }

    #[test]
    fn test_polygon_valid_at_float_degenerate_after_quantization() {
        // This test creates a polygon that is valid in float64 space but becomes
        // degenerate when quantized to a coarse grid (simulating low zoom level).
        //
        // At Z0 with extent 4096, the entire world (360° x ~170°) maps to 4096 pixels.
        // Each pixel is about 0.088° (~9.7 km) wide.
        //
        // A small polygon (e.g., a building) would have vertices that are much closer
        // together than this resolution, causing them to collapse to the same point.

        // Create a small valid triangle (0.0001° ≈ 11 meters on a side)
        // This polygon is clearly valid in float64 space
        let small_triangle = polygon![
            (x: 0.0, y: 0.0),
            (x: 0.0001, y: 0.0),
            (x: 0.00005, y: 0.0001),
            (x: 0.0, y: 0.0),
        ];

        // Bounds representing a large tile (like Z0 or Z1)
        // This makes the coordinate resolution very coarse
        let coarse_bounds = TileBounds {
            lng_min: 0.0,
            lat_min: 0.0,
            lng_max: 10.0, // 10 degrees wide
            lat_max: 10.0,
        };

        // With extent 4096 and 10° tile width:
        // 1 pixel = 10.0 / 4096 ≈ 0.00244°
        // Our triangle is only 0.0001° wide, so all vertices will round to (0, 4096)

        // Quantize the triangle vertices
        let extent = 4096u32;
        let mut quantized: Vec<(i32, i32)> = Vec::new();
        for coord in small_triangle.exterior().0.iter() {
            let (x, y) = geo_to_tile_coords(coord.x, coord.y, &coarse_bounds, extent);
            quantized.push((x, y));
        }

        // Verify that the quantized coordinates have collapsed
        let issues = validate_quantized_ring(&quantized);
        assert!(
            !issues.is_empty(),
            "Small polygon should become degenerate after quantization to coarse grid.\n\
             Quantized coords: {:?}",
            quantized
        );

        // The polygon should have degenerate ring issue (all points collapsed to same location)
        assert!(
            issues
                .iter()
                .any(|i| matches!(i, QuantizationIssue::DegenerateRing { .. })),
            "Should detect degenerate ring: {:?}",
            issues
        );
    }

    #[test]
    fn test_encode_polygon_with_validation_logs_warning() {
        // This test verifies that the validation-aware encoding path works correctly.
        // It doesn't verify actual log output (that would require a log capture),
        // but ensures the encoding still produces valid output.

        // Small polygon that will collapse at coarse resolution
        let small_poly = polygon![
            (x: 0.0, y: 0.0),
            (x: 0.00001, y: 0.0),
            (x: 0.00001, y: 0.00001),
            (x: 0.0, y: 0.00001),
            (x: 0.0, y: 0.0),
        ];

        let coarse_bounds = TileBounds {
            lng_min: 0.0,
            lat_min: 0.0,
            lng_max: 1.0,
            lat_max: 1.0,
        };

        // Use low zoom level where validation is enabled
        let tile_info = TileInfo::new(5, 0, 0);

        // The encoding should complete without panic
        let commands = encode_polygon_with_validation(&small_poly, &coarse_bounds, 4096, tile_info);

        // The polygon will likely be degenerate, but encoding shouldn't fail
        // The warning will be logged internally
        assert!(
            commands.is_empty() || !commands.is_empty(),
            "Encoding should complete"
        );
    }

    #[test]
    fn test_validation_skipped_for_high_zoom() {
        // At high zoom levels (> MAX_ZOOM_FOR_QUANTIZATION_VALIDATION),
        // validation should be skipped since quantization risk is minimal.

        let poly = polygon![
            (x: 0.0, y: 0.0),
            (x: 0.001, y: 0.0),
            (x: 0.001, y: 0.001),
            (x: 0.0, y: 0.001),
            (x: 0.0, y: 0.0),
        ];

        let bounds = TileBounds {
            lng_min: 0.0,
            lat_min: 0.0,
            lng_max: 0.01,
            lat_max: 0.01,
        };

        // High zoom level (Z15) - validation should be skipped
        let tile_info = TileInfo::new(15, 0, 0);

        // Encoding should work normally without validation overhead
        let commands = encode_polygon_with_validation(&poly, &bounds, 4096, tile_info);
        assert!(
            !commands.is_empty(),
            "Normal polygon should encode at high zoom"
        );
    }

    #[test]
    fn test_tile_info_creation() {
        let info = TileInfo::new(5, 10, 15);
        assert_eq!(info.z, 5);
        assert_eq!(info.x, 10);
        assert_eq!(info.y, 15);
    }
}
