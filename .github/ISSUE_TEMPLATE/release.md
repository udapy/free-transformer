---
name: Release Checklist
about: Checklist for preparing a new release
title: 'Release v[VERSION]'
labels: 'release'
assignees: ''
---

## Release Checklist for v[VERSION]

### Pre-Release
- [ ] All planned features/fixes are merged
- [ ] Tests pass (`make test`)
- [ ] Quality checks pass (`make quality`)
- [ ] Documentation builds (`make docs-check`)
- [ ] CHANGELOG.md updated with new version
- [ ] Version bumped (`make version-bump-[patch|minor|major]`)
- [ ] Changes committed and tagged

### Testing
- [ ] Test PyPI release (`make publish-test`)
- [ ] Test installation from Test PyPI
- [ ] Validate package (`make validate-release`)

### Release
- [ ] GitHub release created
- [ ] PyPI release published (automatic via GitHub Actions)
- [ ] Documentation deployed
- [ ] Release announcement prepared

### Post-Release
- [ ] PyPI package verified
- [ ] Documentation site updated
- [ ] Installation tested (`pip install free-transformer`)
- [ ] Release announced in relevant channels

### Notes
<!-- Add any specific notes about this release -->

### Breaking Changes
<!-- List any breaking changes if this is a major release -->

### Migration Guide
<!-- If needed, add migration instructions for users -->