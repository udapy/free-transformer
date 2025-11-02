#!/usr/bin/env python3
"""Version bumping script for Free Transformer."""

import argparse
import re
import sys
from pathlib import Path

def get_current_version():
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    content = pyproject_path.read_text()
    
    # Find version line
    version_match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not version_match:
        raise ValueError("Version not found in pyproject.toml")
    
    return version_match.group(1)

def parse_version(version_str):
    """Parse version string into components."""
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$', version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    
    major, minor, patch, pre = match.groups()
    return int(major), int(minor), int(patch), pre

def format_version(major, minor, patch, pre=None):
    """Format version components into string."""
    version = f"{major}.{minor}.{patch}"
    if pre:
        version += f"-{pre}"
    return version

def bump_version(version_str, bump_type):
    """Bump version according to type."""
    major, minor, patch, pre = parse_version(version_str)
    
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
        pre = None
    elif bump_type == "minor":
        minor += 1
        patch = 0
        pre = None
    elif bump_type == "patch":
        patch += 1
        pre = None
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    
    return format_version(major, minor, patch, pre)

def update_version_in_file(file_path, old_version, new_version):
    """Update version in a file."""
    if not file_path.exists():
        return False
    
    content = file_path.read_text()
    
    # Update version in pyproject.toml
    if file_path.name == "pyproject.toml":
        content = re.sub(
            r'^version = "[^"]+"',
            f'version = "{new_version}"',
            content,
            flags=re.MULTILINE
        )
    
    # Update version in __init__.py
    elif file_path.name == "__init__.py":
        content = re.sub(
            r'^__version__ = "[^"]+"',
            f'__version__ = "{new_version}"',
            content,
            flags=re.MULTILINE
        )
    
    file_path.write_text(content)
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Bump version for Free Transformer")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        help="Type of version bump"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes"
    )
    
    args = parser.parse_args()
    
    try:
        # Get current version
        current_version = get_current_version()
        print(f"Current version: {current_version}")
        
        # Calculate new version
        new_version = bump_version(current_version, args.bump_type)
        print(f"New version: {new_version}")
        
        if args.dry_run:
            print("Dry run - no changes made")
            return
        
        # Update files
        files_to_update = [
            Path("pyproject.toml"),
            Path("src/free_transformer/__init__.py"),
        ]
        
        updated_files = []
        for file_path in files_to_update:
            if update_version_in_file(file_path, current_version, new_version):
                updated_files.append(str(file_path))
                print(f"Updated: {file_path}")
        
        if updated_files:
            print(f"\n✅ Version bumped from {current_version} to {new_version}")
            print("Updated files:")
            for file_path in updated_files:
                print(f"  - {file_path}")
            
            print(f"\nNext steps:")
            print(f"  1. Review changes: git diff")
            print(f"  2. Commit changes: git add . && git commit -m 'Bump version to {new_version}'")
            print(f"  3. Create tag: git tag v{new_version}")
            print(f"  4. Push changes: git push && git push --tags")
        else:
            print("No files were updated")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()