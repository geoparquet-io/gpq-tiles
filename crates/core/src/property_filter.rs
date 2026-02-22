//! Property filtering for controlling which attributes are included in output tiles.
//!
//! This module provides filtering capabilities matching tippecanoe's approach:
//! - `-y` / `--include`: Include only specified fields (whitelist)
//! - `-x` / `--exclude`: Exclude specified fields (blacklist)
//! - `-X` / `--exclude-all`: Exclude all attributes, keep only geometries
//!
//! The geometry column is ALWAYS preserved regardless of filter settings.

use std::collections::HashSet;

/// Filter mode for property/attribute selection.
///
/// Matches tippecanoe's property filtering behavior:
/// - `Include` corresponds to `-y` flag (whitelist)
/// - `Exclude` corresponds to `-x` flag (blacklist)
/// - `ExcludeAll` corresponds to `-X` flag (geometry only)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyFilter {
    /// Include all properties (no filtering)
    None,
    /// Include only the specified properties (whitelist)
    /// Corresponds to tippecanoe's `-y` flag
    Include(HashSet<String>),
    /// Exclude the specified properties (blacklist)
    /// Corresponds to tippecanoe's `-x` flag
    Exclude(HashSet<String>),
    /// Exclude all properties, keep only geometry
    /// Corresponds to tippecanoe's `-X` flag
    ExcludeAll,
}

impl Default for PropertyFilter {
    fn default() -> Self {
        Self::None
    }
}

impl PropertyFilter {
    /// Create an include filter (whitelist) from field names.
    pub fn include<I, S>(fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Include(fields.into_iter().map(|s| s.into()).collect())
    }

    /// Create an exclude filter (blacklist) from field names.
    pub fn exclude<I, S>(fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Exclude(fields.into_iter().map(|s| s.into()).collect())
    }

    /// Check if a field should be included in the output.
    ///
    /// Geometry columns are ALWAYS included regardless of filter.
    /// This method does NOT handle geometry columns - caller must check separately.
    pub fn should_include(&self, field_name: &str) -> bool {
        match self {
            Self::None => true,
            Self::Include(whitelist) => whitelist.contains(field_name),
            Self::Exclude(blacklist) => !blacklist.contains(field_name),
            Self::ExcludeAll => false,
        }
    }

    /// Filter a collection of field names, returning only those that should be included.
    pub fn filter_fields<'a>(&self, fields: &'a [String]) -> Vec<&'a String> {
        fields.iter().filter(|f| self.should_include(f)).collect()
    }

    /// Check if this filter is active (not None).
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== PropertyFilter Construction Tests ==========

    #[test]
    fn test_property_filter_default_is_none() {
        let filter = PropertyFilter::default();
        assert_eq!(filter, PropertyFilter::None);
    }

    #[test]
    fn test_property_filter_include_from_vec() {
        let filter = PropertyFilter::include(vec!["name", "population"]);
        match filter {
            PropertyFilter::Include(set) => {
                assert!(set.contains("name"));
                assert!(set.contains("population"));
                assert_eq!(set.len(), 2);
            }
            _ => panic!("Expected Include variant"),
        }
    }

    #[test]
    fn test_property_filter_exclude_from_vec() {
        let filter = PropertyFilter::exclude(vec!["internal_id", "temp_field"]);
        match filter {
            PropertyFilter::Exclude(set) => {
                assert!(set.contains("internal_id"));
                assert!(set.contains("temp_field"));
                assert_eq!(set.len(), 2);
            }
            _ => panic!("Expected Exclude variant"),
        }
    }

    #[test]
    fn test_property_filter_include_from_strings() {
        let filter = PropertyFilter::include(vec!["name".to_string(), "type".to_string()]);
        match filter {
            PropertyFilter::Include(set) => {
                assert!(set.contains("name"));
                assert!(set.contains("type"));
            }
            _ => panic!("Expected Include variant"),
        }
    }

    // ========== should_include Tests ==========

    #[test]
    fn test_none_filter_includes_all() {
        let filter = PropertyFilter::None;
        assert!(filter.should_include("name"));
        assert!(filter.should_include("population"));
        assert!(filter.should_include("anything"));
    }

    #[test]
    fn test_include_filter_only_includes_whitelisted() {
        let filter = PropertyFilter::include(vec!["name", "population"]);

        // Whitelisted fields should be included
        assert!(filter.should_include("name"));
        assert!(filter.should_include("population"));

        // Non-whitelisted fields should be excluded
        assert!(!filter.should_include("internal_id"));
        assert!(!filter.should_include("temp_field"));
    }

    #[test]
    fn test_exclude_filter_excludes_blacklisted() {
        let filter = PropertyFilter::exclude(vec!["internal_id", "temp_field"]);

        // Blacklisted fields should be excluded
        assert!(!filter.should_include("internal_id"));
        assert!(!filter.should_include("temp_field"));

        // Non-blacklisted fields should be included
        assert!(filter.should_include("name"));
        assert!(filter.should_include("population"));
    }

    #[test]
    fn test_exclude_all_excludes_everything() {
        let filter = PropertyFilter::ExcludeAll;

        assert!(!filter.should_include("name"));
        assert!(!filter.should_include("population"));
        assert!(!filter.should_include("anything"));
    }

    // ========== filter_fields Tests ==========

    #[test]
    fn test_filter_fields_with_none() {
        let filter = PropertyFilter::None;
        let fields = vec![
            "name".to_string(),
            "population".to_string(),
            "area".to_string(),
        ];

        let result = filter.filter_fields(&fields);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_filter_fields_with_include() {
        let filter = PropertyFilter::include(vec!["name", "area"]);
        let fields = vec![
            "name".to_string(),
            "population".to_string(),
            "area".to_string(),
        ];

        let result = filter.filter_fields(&fields);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&&"name".to_string()));
        assert!(result.contains(&&"area".to_string()));
        assert!(!result.contains(&&"population".to_string()));
    }

    #[test]
    fn test_filter_fields_with_exclude() {
        let filter = PropertyFilter::exclude(vec!["population"]);
        let fields = vec![
            "name".to_string(),
            "population".to_string(),
            "area".to_string(),
        ];

        let result = filter.filter_fields(&fields);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&&"name".to_string()));
        assert!(result.contains(&&"area".to_string()));
        assert!(!result.contains(&&"population".to_string()));
    }

    #[test]
    fn test_filter_fields_with_exclude_all() {
        let filter = PropertyFilter::ExcludeAll;
        let fields = vec![
            "name".to_string(),
            "population".to_string(),
            "area".to_string(),
        ];

        let result = filter.filter_fields(&fields);
        assert!(result.is_empty());
    }

    // ========== is_active Tests ==========

    #[test]
    fn test_is_active() {
        assert!(!PropertyFilter::None.is_active());
        assert!(PropertyFilter::include(vec!["name"]).is_active());
        assert!(PropertyFilter::exclude(vec!["temp"]).is_active());
        assert!(PropertyFilter::ExcludeAll.is_active());
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_empty_include_filter_excludes_all() {
        // An include filter with no fields should exclude everything
        let filter = PropertyFilter::include(Vec::<String>::new());
        assert!(!filter.should_include("name"));
        assert!(!filter.should_include("anything"));
    }

    #[test]
    fn test_empty_exclude_filter_includes_all() {
        // An exclude filter with no fields should include everything
        let filter = PropertyFilter::exclude(Vec::<String>::new());
        assert!(filter.should_include("name"));
        assert!(filter.should_include("anything"));
    }

    #[test]
    fn test_case_sensitive_matching() {
        // Field names should be case-sensitive
        let filter = PropertyFilter::include(vec!["Name"]);
        assert!(filter.should_include("Name"));
        assert!(!filter.should_include("name"));
        assert!(!filter.should_include("NAME"));
    }
}
