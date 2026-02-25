# Contributing to gpq-tiles

## Development Setup

```bash
git clone https://github.com/geoparquet-io/gpq-tiles.git
cd gpq-tiles
git config core.hooksPath .githooks
cargo build && cargo test
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for Python setup.

## Commit Convention

[Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description |
|------|-------------|
| `feat` | New feature (bumps minor) |
| `fix` | Bug fix (bumps patch) |
| `docs` | Documentation only |
| `perf` | Performance improvement |
| `refactor` | Code change (no feature/fix) |
| `test` | Tests only |
| `chore` | Maintenance |

## Pull Request Process

1. Branch from `main`
2. `cargo test && cargo fmt --all && cargo clippy`
3. Submit PR

## Releasing (Maintainers)

### Prerequisites

1. **Commitizen** installed globally: `uv tool install commitizen`
2. **GitHub secrets** configured:
   - `CARGO_REGISTRY_TOKEN` from [crates.io/settings/tokens](https://crates.io/settings/tokens)
   - PyPI trusted publishing at [pypi.org](https://pypi.org/manage/project/gpq-tiles/settings/publishing/)

### Release Workflow

```bash
# 1. Create release branch from main
git checkout main && git pull
git checkout -b release/vX.Y.Z

# 2. Bump version (from repo root, NOT crates/python)
cz bump --increment MINOR --changelog   # or PATCH/MAJOR

# 3. Verify build works
cargo check

# 4. Push and create PR
git push -u origin release/vX.Y.Z
gh pr create --title "Release vX.Y.Z" --body "Automated release"

# 5. Merge PR → release.yml auto-publishes
```

### What Commitizen Updates

The config in `crates/python/pyproject.toml` updates these files:

| File | Pattern |
|------|---------|
| `Cargo.toml` | `version = "X.Y.Z"` (workspace) |
| `crates/python/pyproject.toml` | `version = "X.Y.Z"` |
| `crates/cli/Cargo.toml` | `gpq-tiles-core = { ..., version = "X.Y.Z" }` |

### Recovery

```bash
# If release fails, delete orphan tag
git push origin :refs/tags/vX.Y.Z

# Re-trigger manually
gh workflow run release.yml --ref main
```

### Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `failed to select version for gpq-tiles-core` | CLI dependency not updated | Ensure `crates/cli/Cargo.toml` has correct version |
| `cz: command not found` | Commitizen not installed | `uv tool install commitizen` |
| CI timeout on benchmarks | Large benchmarks running | Check regex excludes `large_*` |
