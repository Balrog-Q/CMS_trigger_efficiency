#!/usr/bin/env python3
"""
Script to extract all libraries used in the project and their versions.
This script scans all Python files (.py and .ipynb) in the project directory
and extracts import statements, then attempts to get version information.
"""

import os
import re
import ast
import json
import importlib
import pkg_resources
from pathlib import Path
from collections import defaultdict
import subprocess
import sys

def extract_imports_from_python_file(file_path):
    """Extract import statements from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file to extract imports
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except SyntaxError:
            # If AST parsing fails, try regex as fallback
            import_patterns = [
                r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)',
                r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import',
            ]
            for pattern in import_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    imports.add(match.split('.')[0])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return imports

def extract_imports_from_notebook(file_path):
    """Extract import statements from a Jupyter notebook."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    source = ''.join(source)
                
                # Extract imports using regex
                import_patterns = [
                    r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)',
                    r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import',
                ]
                for pattern in import_patterns:
                    matches = re.findall(pattern, source, re.MULTILINE)
                    for match in matches:
                        imports.add(match.split('.')[0])
    except Exception as e:
        print(f"Error reading notebook {file_path}: {e}")
    
    return imports

def get_package_version(package_name):
    """Get the version of a package."""
    try:
        # First try pkg_resources (works with pip installed packages)
        return pkg_resources.get_distribution(package_name).version
    except:
        try:
            # Try importing and checking __version__
            module = importlib.import_module(package_name)
            if hasattr(module, '__version__'):
                return module.__version__
            elif hasattr(module, 'version'):
                return module.version
            elif hasattr(module, 'VERSION'):
                return module.VERSION
        except:
            pass
    
    # Special cases for common packages
    special_cases = {
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
    }
    
    if package_name in special_cases:
        try:
            return pkg_resources.get_distribution(special_cases[package_name]).version
        except:
            pass
    
    # Try conda list if available (for ROOT, etc.)
    try:
        result = subprocess.run(['conda', 'list', package_name], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith(package_name) and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
    except:
        pass
    
    return "Version not found"

def scan_directory(directory):
    """Scan directory for Python files and extract imports."""
    all_imports = defaultdict(set)
    file_count = 0
    
    # Patterns to match Python files
    python_extensions = ['.py', '.ipynb']
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'build', 'dist', 'egg-info']]
        
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1]
            
            if file_ext in python_extensions:
                file_count += 1
                rel_path = os.path.relpath(file_path, directory)
                print(f"Scanning: {rel_path}")
                
                if file_ext == '.py':
                    imports = extract_imports_from_python_file(file_path)
                elif file_ext == '.ipynb':
                    imports = extract_imports_from_notebook(file_path)
                
                for imp in imports:
                    all_imports[imp].add(rel_path)
    
    print(f"\nScanned {file_count} files")
    return all_imports

def generate_requirements_format(imports_with_versions):
    """Generate requirements.txt format output."""
    requirements = []
    for package, version in imports_with_versions.items():
        if version != "Version not found" and version != "Built-in module":
            # Clean version string and add to requirements
            version_clean = version.split()[0]  # Take first part if multiple
            requirements.append(f"{package}>={version_clean}")
        elif version == "Built-in module":
            requirements.append(f"# {package} - Built-in Python module")
        else:
            requirements.append(f"# {package} - Version not found")
    
    return sorted(requirements)

def main():
    """Main function to extract and report library usage."""
    project_dir = Path.cwd()
    print(f"Scanning project directory: {project_dir}")
    print("=" * 70)
    
    # Extract all imports
    all_imports = scan_directory(project_dir)
    
    # Get versions and file usage
    print("\nGetting package versions...")
    print("=" * 70)
    
    imports_with_versions = {}
    builtin_modules = set([
        'os', 'sys', 'time', 'datetime', 'json', 're', 'math', 'random',
        'collections', 'itertools', 'functools', 'operator', 'pathlib',
        'subprocess', 'threading', 'multiprocessing', 'pickle', 'array',
        'typing', 'warnings', 'logging', 'argparse', 'configparser',
        'tempfile', 'shutil', 'glob', 'csv', 'sqlite3', 'urllib',
        'http', 'email', 'xml', 'html', 'gzip', 'zipfile', 'tarfile'
    ])
    
    for package in sorted(all_imports.keys()):
        if package in builtin_modules:
            version = "Built-in module"
        else:
            version = get_package_version(package)
        imports_with_versions[package] = version
        print(f"{package:<25} : {version}")
    
    # Generate comprehensive report
    report_file = project_dir / "library_usage_report.txt"
    requirements_file = project_dir / "extracted_requirements.txt"
    
    print(f"\nGenerating reports...")
    print(f"Full report: {report_file}")
    print(f"Requirements file: {requirements_file}")
    
    # Write full report
    with open(report_file, 'w') as f:
        f.write("CMS Trigger Efficiency Project - Library Usage Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated on: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}\n")
        f.write(f"Project directory: {project_dir}\n")
        f.write(f"Total packages found: {len(all_imports)}\n\n")
        
        f.write("PACKAGE VERSIONS\n")
        f.write("-" * 70 + "\n")
        for package, version in sorted(imports_with_versions.items()):
            f.write(f"{package:<30} : {version}\n")
        
        f.write("\n\nFILE USAGE DETAILS\n")
        f.write("-" * 70 + "\n")
        for package in sorted(all_imports.keys()):
            f.write(f"\n{package} ({imports_with_versions[package]}):\n")
            for file_path in sorted(all_imports[package]):
                f.write(f"  - {file_path}\n")
        
        f.write("\n\nSUMMARY BY CATEGORY\n")
        f.write("-" * 70 + "\n")
        
        categories = {
            'Machine Learning': ['sklearn', 'xgboost', 'lightgbm', 'catboost', 'tensorflow', 'keras', 'torch', 'optuna'],
            'Data Analysis': ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly'],
            'Physics/HEP': ['ROOT', 'uproot', 'awkward', 'hist', 'mplhep', 'coffea'],
            'Jupyter/Interactive': ['IPython', 'ipywidgets', 'jupyter'],
            'Utilities': ['tqdm', 'pathlib', 'pickle', 'joblib'],
            'Built-in': list(builtin_modules)
        }
        
        for category, packages in categories.items():
            found_packages = [p for p in packages if p in all_imports]
            if found_packages:
                f.write(f"\n{category}:\n")
                for pkg in found_packages:
                    f.write(f"  - {pkg}: {imports_with_versions[pkg]}\n")
    
    # Write requirements.txt format
    requirements = generate_requirements_format(imports_with_versions)
    with open(requirements_file, 'w') as f:
        f.write("# CMS Trigger Efficiency Project - Extracted Requirements\n")
        f.write(f"# Generated automatically from project files\n")
        f.write(f"# Date: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}\n\n")
        
        f.write("# Core Dependencies\n")
        for req in requirements:
            f.write(req + "\n")
    
    print("\nReport generation complete!")
    print(f"Found {len([v for v in imports_with_versions.values() if v not in ['Version not found', 'Built-in module']])} packages with versions")
    print(f"Found {len([v for v in imports_with_versions.values() if v == 'Built-in module'])} built-in modules")
    print(f"Could not determine version for {len([v for v in imports_with_versions.values() if v == 'Version not found'])} packages")

if __name__ == "__main__":
    main()
