#!/usr/bin/env python3
"""
Validation script to verify GNN pipeline setup and functionality
Run: python3 validate_setup.py
"""

import os
import sys
import subprocess
from pathlib import Path


def check_file_exists(path, description):
    """Check if a required file exists"""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} NOT FOUND")
        return False


def check_python_syntax(filepath):
    """Check Python file syntax"""
    try:
        compile(open(filepath).read(), filepath, 'exec')
        print(f"✅ Python syntax OK: {filepath}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {filepath}: {e}")
        return False


def check_imports(module_name, package_name=None):
    """Check if a Python package is importable"""
    pkg = package_name or module_name
    try:
        __import__(module_name)
        print(f"✅ {pkg} installed")
        return True
    except ImportError:
        print(f"❌ {pkg} NOT installed - run: pip install {pkg}")
        return False


def main():
    print("=" * 60)
    print("GNN Kinase Affinity Predictor - Setup Validation")
    print("=" * 60)
    
    results = []
    
    # 1. Check directory structure
    print("\n[1] Directory Structure")
    dirs = [
        ('models', 'Models directory'),
        ('data', 'Data directory'),
        ('data/processed', 'Processed data directory'),
        ('results', 'Results directory'),
    ]
    for dir_path, desc in dirs:
        exists = Path(dir_path).exists()
        print(f"{'✅' if exists else '❌'} {desc}: {dir_path}")
        results.append(exists)
    
    # 2. Check required files
    print("\n[2] Required Python Files")
    files = [
        ('train.py', 'Training pipeline'),
        ('web_server.py', 'Web server'),
        ('evaluate.py', 'Evaluation script'),
        ('models/gnn_model.py', 'GNN model architecture'),
        ('index.html', 'Web UI'),
        ('requirements.txt', 'Dependencies'),
    ]
    for file_path, desc in files:
        exists = check_file_exists(file_path, desc)
        results.append(exists)
    
    # 3. Check Python syntax
    print("\n[3] Python Syntax Check")
    py_files = ['train.py', 'web_server.py', 'evaluate.py', 'models/gnn_model.py']
    for py_file in py_files:
        if Path(py_file).exists():
            results.append(check_python_syntax(py_file))
    
    # 4. Check Python dependencies
    print("\n[4] Python Dependencies")
    packages = [
        ('torch', 'PyTorch'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
    ]
    for module, name in packages:
        results.append(check_imports(module, name))
    
    # 5. Check data files
    print("\n[5] Data Files")
    csv_file = 'data/processed/real_data_merged_5200.csv'
    if Path(csv_file).exists():
        size_mb = Path(csv_file).stat().st_size / (1024 * 1024)
        print(f"✅ Training data: {csv_file} ({size_mb:.1f} MB)")
        results.append(True)
    else:
        print(f"⚠️  Training data not found: {csv_file}")
        print("   (Will generate synthetic data during training if missing)")
        results.append(True)  # Not critical
    
    # 6. Check HTML validity
    print("\n[6] Web UI")
    if Path('index.html').exists():
        content = open('index.html').read()
        if '<div class="card">' in content and '/predict_full' in content:
            print("✅ index.html contains GNN UI elements (cards, /predict_full)")
            results.append(True)
        else:
            print("❌ index.html missing GNN UI elements")
            results.append(False)
    
    # 7. Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Setup Validation: {passed}/{total} checks passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✅ All checks passed! You're ready to:")
        print("   1. python3 train.py --num_samples 10  # Quick test")
        print("   2. python3 web_server.py               # Start server")
        print("   3. Open http://localhost:8000         # Web UI")
        return 0
    else:
        print(f"\n❌ {total - passed} check(s) failed. Please fix and retry.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
