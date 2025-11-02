# Release Guide

This guide covers how to release new versions of Free Transformer to PyPI.

## Prerequisites

1. **Permissions**: You need maintainer access to the GitHub repository
2. **PyPI Account**: Access to the PyPI project (or Test PyPI for testing)
3. **Environment**: Local development environment set up

## Release Process

### 1. Prepare the Release

#### Update Documentation
```bash
# Ensure documentation is up to date
make docs-build
make docs-check
```

#### Run Full Test Suite
```bash
# Run all tests and quality checks
make ci
```

#### Update Changelog
Edit `CHANGELOG.md` to document changes in the new version:

```markdown
## [0.2.0] - 2024-11-15

### Added
- New feature X
- Enhancement Y

### Changed
- Improved Z

### Fixed
- Bug fix A
```

### 2. Version Bumping

Choose the appropriate version bump type:

- **Patch** (0.1.0 → 0.1.1): Bug fixes, small improvements
- **Minor** (0.1.0 → 0.2.0): New features, backward compatible
- **Major** (0.1.0 → 1.0.0): Breaking changes

```bash
# Bump version (choose one)
make version-bump-patch   # For bug fixes
make version-bump-minor   # For new features
make version-bump-major   # For breaking changes

# Or do it manually
python scripts/bump_version.py patch --dry-run  # Preview changes
python scripts/bump_version.py patch            # Apply changes
```

### 3. Commit and Tag

```bash
# Review changes
git diff

# Commit version bump
git add .
git commit -m "Bump version to $(grep '^version = ' pyproject.toml | cut -d'"' -f2)"

# Create and push tag
VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
git tag "v$VERSION"
git push origin main
git push origin "v$VERSION"
```

### 4. Test Release (Recommended)

Test the release process with Test PyPI:

```bash
# Build and publish to Test PyPI
make publish-test

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ free-transformer

# Test the installed package
python -c "from free_transformer import FreeTransformer; print('Import successful')"
```

### 5. Production Release

#### Option A: Automated Release (Recommended)

1. **Create GitHub Release**:
   - Go to [GitHub Releases](https://github.com/udapy/free-transformer/releases)
   - Click "Create a new release"
   - Choose the tag you created (e.g., `v0.2.0`)
   - Title: `Free Transformer v0.2.0`
   - Description: Copy relevant section from CHANGELOG.md
   - Click "Publish release"

2. **Automatic PyPI Publishing**:
   - GitHub Actions will automatically build and publish to PyPI
   - Monitor the [Actions tab](https://github.com/udapy/free-transformer/actions)

#### Option B: Manual Release

```bash
# Build and publish to PyPI
make publish

# Verify publication
pip install free-transformer
python -c "from free_transformer import FreeTransformer; print('Release successful')"
```

### 6. Post-Release

#### Update Documentation
The documentation will be automatically updated via GitHub Actions when you push to main.

#### Verify Release
1. **PyPI**: Check [PyPI page](https://pypi.org/project/free-transformer/)
2. **Documentation**: Check [documentation site](https://udapy.github.io/free-transformer/)
3. **Installation**: Test `pip install free-transformer`

#### Announce Release
- Update README badges if needed
- Post in relevant communities/forums
- Update any dependent projects

## Release Checklist

### Pre-Release
- [ ] All tests pass (`make test`)
- [ ] Code quality checks pass (`make quality`)
- [ ] Documentation builds without errors (`make docs-check`)
- [ ] CHANGELOG.md updated
- [ ] Version bumped appropriately
- [ ] Changes committed and tagged

### Release
- [ ] Test PyPI release successful (optional but recommended)
- [ ] Production PyPI release successful
- [ ] GitHub release created
- [ ] Documentation deployed

### Post-Release
- [ ] PyPI package accessible
- [ ] Documentation updated
- [ ] Installation tested
- [ ] Release announced

## Troubleshooting

### Common Issues

**Build Fails**
```bash
# Clean and rebuild
make clean-dist
make build
```

**PyPI Upload Fails**
```bash
# Check package
make check-package

# Verify credentials (if using manual upload)
twine check dist/*
```

**Documentation Not Updating**
```bash
# Manual documentation deployment
make docs-deploy
```

**Version Conflicts**
```bash
# Check current version
grep '^version = ' pyproject.toml

# Ensure tag matches version
git tag -l | grep $(grep '^version = ' pyproject.toml | cut -d'"' -f2)
```

### Emergency Procedures

**Yanking a Release**
If you need to remove a problematic release:

```bash
# Yank from PyPI (makes it unavailable for new installs)
pip install twine
twine yank free-transformer <version> -r pypi

# Delete GitHub release
# Go to GitHub releases and delete manually
```

**Hotfix Release**
For critical bug fixes:

```bash
# Create hotfix branch from tag
git checkout -b hotfix/v0.1.1 v0.1.0

# Make minimal fix
# ... edit files ...

# Bump patch version
make version-bump-patch

# Commit and tag
git commit -am "Hotfix: critical bug fix"
VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
git tag "v$VERSION"

# Merge back to main
git checkout main
git merge hotfix/v0.1.1
git push origin main
git push origin "v$VERSION"

# Release as normal
```

## Automation

The project includes several automation features:

- **GitHub Actions**: Automatic testing, building, and publishing
- **Version Bumping**: Automated version management
- **Documentation**: Auto-deployment to GitHub Pages
- **Quality Checks**: Automated code quality enforcement

## Security

- **Trusted Publishing**: Uses GitHub's OIDC for secure PyPI publishing
- **No Secrets**: No API keys stored in repository
- **Signed Releases**: All releases are signed and verifiable

For questions about the release process, open an issue or discussion on GitHub.