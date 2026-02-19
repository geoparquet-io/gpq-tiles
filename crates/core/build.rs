// Build script for compiling protobuf definitions

fn main() -> Result<(), Box<dyn std::error::Error>> {
    prost_build::compile_protos(&["proto/vector_tile.proto"], &["proto/"])?;
    Ok(())
}
