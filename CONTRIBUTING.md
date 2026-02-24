# Contributing to gpq-tiles

Thank you for your interest in contributing to gpq-tiles!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/geoparquet-io/gpq-tiles.git
cd gpq-tiles

# Enable pre-commit hooks
git config core.hooksPath .githooks

# Build and test
cargo build
cargo test
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed setup instructions, including Python bindings.

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) with [Commitizen](https://commitizen-tools.github.io/commitizen/).

### Commit Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, no logic change) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |

### Examples

```bash
# Good commit messages
feat: add WKT geometry encoding support
fix: guard against degenerate linestrings in simplify
perf: parallelize geometry processing within row groups
docs: update ROADMAP with Phase 9 streaming writer

# With scope
feat(cli): add --streaming-mode flag
fix(core): prevent OOM on large polygon files
```

### Using Commitizen (optional)

If you have commitizen installed globally:

```bash
npm install -g commitizen cz-conventional-changelog
# Then use:
git cz
```

## Test-Driven Development

This project follows strict TDD. Every feature must have tests written **before** implementation:

```bash
# 1. Write a failing test
cargo test --package gpq-tiles-core <test_name> -- --nocapture  # Should fail

# 2. Implement the feature
# ... write code ...

# 3. Verify the test passes
cargo test --package gpq-tiles-core <test_name> -- --nocapture  # Should pass

# 4. Commit with descriptive message
git commit -m "feat: implement X (TDD green)"
```

## Pull Request Process

1. Create a feature branch from `main`
2. Ensure all tests pass: `cargo test`
3. Format code: `cargo fmt --all`
4. Update documentation if needed
5. Submit PR with clear description

## Releasing (Maintainers)

Releases are triggered by commits starting with `bump:`.

```bash
# 1. Update version in Cargo.toml [workspace.package]
# Edit: version = "0.2.0"

# 2. Update CHANGELOG.md with changes since last release

# 3. Commit with bump prefix
git add Cargo.toml CHANGELOG.md
git commit -m "bump: v0.2.0"
git push

# 4. Workflows auto-trigger:
#    - release.yml → creates git tag, publishes to crates.io
#    - python-release.yml → builds wheels, publishes to PyPI
#    - docs.yml → deploys documentation
```

**Prerequisites (one-time setup):**

| Secret | Source | Purpose |
|--------|--------|---------|
| `CARGO_REGISTRY_TOKEN` | [crates.io/settings/tokens](https://crates.io/settings/tokens) | Publish to crates.io |
| PyPI trusted publishing | [PyPI project settings](https://pypi.org/manage/project/gpq-tiles/settings/publishing/) | Publish to PyPI |

**Recovery if release fails:**
```bash
# Delete orphan tag if needed
git push origin :refs/tags/vX.Y.Z

# Re-trigger manually
gh workflow run release.yml --ref main
gh workflow run python-release.yml --ref main
```

## Questions?

Open an issue or start a discussion on GitHub.
