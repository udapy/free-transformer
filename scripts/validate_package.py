#!/usr/bin/env python3
"""Validate package before release."""

import subprocess
import sys
import tempfile
from pathlib import Path

def run_command(cmd, description, capture_output=True):
    """Run a command and return success status."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=capture_output,
            text=True
        )
        print(f"✅ {description} - OK")
        return True, result.stdout if capture_output else ""
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        if capture_output and e.stderr:
            print(f"   Error: {e.stderr}")
        return False, ""

def validate_package():
    """Validate the package for release."""
    print("🚀 Free Transformer Package Validation")
    print("=" * 50)
    
    checks = []
    
    # 1. Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Not in project root (pyproject.toml not found)")
        return False
    
    # 2. Run tests
    success, _ = run_command("make test", "Running tests")
    checks.append(("Tests", success))
    
    # 3. Run quality checks
    success, _ = run_command("make quality", "Quality checks")
    checks.append(("Quality", success))
    
    # 4. Check documentation builds
    success, _ = run_command("make docs-check", "Documentation build")
    checks.append(("Documentation", success))
    
    # 5. Build package
    success, _ = run_command("make build", "Building package")
    checks.append(("Package build", success))
    
    # 6. Check package with twine
    success, _ = run_command("make check-package", "Package validation")
    checks.append(("Package validation", success))
    
    # 7. Test import in clean environment
    print("🔍 Testing import in clean environment...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Install package in temporary environment
            install_cmd = f"cd {temp_dir} && python -m venv test_env && source test_env/bin/activate && pip install {Path.cwd()}/dist/*.whl"
            result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Test import
                test_cmd = f"cd {temp_dir} && source test_env/bin/activate && python -c 'from free_transformer import FreeTransformer; print(\"Import successful\")'"
                result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Import test - OK")
                    checks.append(("Import test", True))
                else:
                    print("❌ Import test - FAILED")
                    print(f"   Error: {result.stderr}")
                    checks.append(("Import test", False))
            else:
                print("❌ Package installation - FAILED")
                print(f"   Error: {result.stderr}")
                checks.append(("Import test", False))
    except Exception as e:
        print(f"❌ Import test - FAILED: {e}")
        checks.append(("Import test", False))
    
    # 8. Check version consistency
    print("🔍 Checking version consistency...")
    try:
        # Get version from pyproject.toml
        with open("pyproject.toml", "r") as f:
            content = f.read()
            import re
            version_match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
            if version_match:
                pyproject_version = version_match.group(1)
            else:
                raise ValueError("Version not found in pyproject.toml")
        
        # Get version from __init__.py
        with open("src/free_transformer/__init__.py", "r") as f:
            content = f.read()
            version_match = re.search(r'^__version__ = "([^"]+)"', content, re.MULTILINE)
            if version_match:
                init_version = version_match.group(1)
            else:
                raise ValueError("Version not found in __init__.py")
        
        if pyproject_version == init_version:
            print("✅ Version consistency - OK")
            checks.append(("Version consistency", True))
        else:
            print(f"❌ Version mismatch: pyproject.toml={pyproject_version}, __init__.py={init_version}")
            checks.append(("Version consistency", False))
            
    except Exception as e:
        print(f"❌ Version check failed: {e}")
        checks.append(("Version consistency", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Ready for release!")
        print("\nNext steps:")
        print("  1. make publish-test  # Test on Test PyPI")
        print("  2. make publish       # Release to PyPI")
        return True
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before release")
        return False

def main():
    """Main function."""
    success = validate_package()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()