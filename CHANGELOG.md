# Changelog

All notable changes to Free Transformer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive documentation with MkDocs
- GitHub Pages documentation deployment
- PyPI packaging and release automation
- Version bumping scripts
- Enhanced README with badges and links

### Changed

- Improved project structure and organization
- Enhanced Makefile with documentation and publishing commands

## [0.1.0] - 2024-11-02

### Added

- Initial release of Free Transformer
- Core Free Transformer architecture with latent planning
- Baseline Transformer for comparison
- Conditional VAE training with free bits regularization
- Multi-GPU training support with FSDP
- Synthetic data generation for prototyping
- Docker support for easy deployment
- Comprehensive test suite
- Code quality tools (Black, Ruff, MyPy)
- Example scripts and configurations
- Basic documentation

### Features

- **Architecture**: Llama-style backbone with RMSNorm, SwiGLU, RoPE, GQA
- **Latent Planning**: Explicit binary plan variable with differentiable sampling
- **Training**: Conditional VAE loss with reconstruction and KL divergence
- **Scaling**: FSDP support for multi-GPU training
- **Development**: Full development environment with quality checks
- **Usability**: Modular API with YAML configuration

[Unreleased]: https://github.com/udapy/free-transformer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/udapy/free-transformer/releases/tag/v0.1.0
