# Contributing to Scion

Thank you for your interest in contributing to Scion! This document provides guidelines for contributing.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/Scion.git
   cd Scion
   ```
3. **Install** in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.12+
- Git

### Running Tests

```bash
pytest tests/ -x -q
```

### Linting & Type Checking

```bash
ruff check scion/
mypy scion/
```

## Pull Request Process

1. Ensure all tests pass (`pytest tests/ -x -q`)
2. Ensure linting passes (`ruff check scion/`)
3. Update documentation if your change affects public APIs
4. Write clear commit messages describing what and why
5. Open a pull request against the `main` branch

## Code Style

- Follow existing code conventions in the project
- Use type hints for all public functions
- Write docstrings for public classes and methods
- Keep functions focused and small

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include steps to reproduce for bugs
- Include Python version, OS, and relevant configuration

## Security Vulnerabilities

**Do NOT report security vulnerabilities via GitHub Issues.**
See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
