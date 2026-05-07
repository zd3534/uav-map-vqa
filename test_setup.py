#!/usr/bin/env python3
"""
Test that the environment is set up correctly.

Usage:
    python test_setup.py
"""

import sys


def test_imports():
    """Test that required packages are installed."""
    print("Testing package imports...")

    try:
        import openai
        print("  ✓ openai")
    except ImportError:
        print("  ✗ openai - run: pip install openai")
        return False

    try:
        from PIL import Image
        print("  ✓ PIL (Pillow)")
    except ImportError:
        print("  ✗ PIL - run: pip install pillow")
        return False

    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - run: pip install numpy")
        return False

    return True


def test_api_key():
    """Test that API key is set."""
    import os

    print("\nTesting API key...")
    api_key = os.environ.get('OPENROUTER_API_KEY')

    if api_key:
        print(f"  ✓ OPENROUTER_API_KEY is set ({api_key[:10]}...)")
        return True
    else:
        print("  ✗ OPENROUTER_API_KEY not set")
        print("    Set it with: export OPENROUTER_API_KEY='sk-or-v1-...'")
        return False


def test_data_dir():
    """Test that data directory exists."""
    from pathlib import Path

    print("\nTesting data directory...")
    data_dir = Path('./data')

    if data_dir.exists():
        datasets = ['AirLock', 'ALTO', 'UAV_VisLoc', 'VETRA']
        found = [d for d in datasets if (data_dir / d).exists()]

        if found:
            print(f"  ✓ Data directory exists with datasets: {', '.join(found)}")
            return True
        else:
            print("  ⚠ Data directory exists but no datasets found")
            print("    Expected: AirLock, ALTO, UAV_VisLoc, VETRA")
            return False
    else:
        print("  ✗ Data directory not found")
        print("    Expected: ./data/")
        return False


def main():
    print("=" * 60)
    print("UAV-Map-VQA Setup Test")
    print("=" * 60)
    print()

    results = []

    results.append(("Imports", test_imports()))
    results.append(("API Key", test_api_key()))
    results.append(("Data Directory", test_data_dir()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = all(r[1] for r in results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")

    print()

    if all_passed:
        print("✓ All tests passed! Ready to run evaluation.")
        print("\nNext steps:")
        print("  1. Verify dataset: python verify_data.py --data_dir ./data")
        print("  2. Inspect sample: python inspect_sample.py --data_dir ./data")
        print("  3. Run evaluation: python evaluate.py --data_dir ./data --models \"qwen/qwen-vl-max\"")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
