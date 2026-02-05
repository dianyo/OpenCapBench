#!/usr/bin/env python3
"""
Test script to verify all dependencies are correctly installed for OpenCapBench.

Run this after completing the setup to ensure your environment is ready.
"""

import sys
from typing import Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test importing a module and return success status with version if available."""
    try:
        if package_name:
            exec(f"import {module_name}")
            module = sys.modules[module_name]
        else:
            module = __import__(module_name)
            package_name = module_name
        
        # Try to get version
        version = "unknown"
        for attr in ['__version__', 'VERSION', 'version']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                break
        
        return True, f"{package_name} (v{version})"
    except ImportError as e:
        return False, f"{package_name}: {str(e)}"
    except Exception as e:
        return False, f"{package_name}: Unexpected error - {str(e)}"

def main():
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}OpenCapBench Dependency Verification{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    # Track results
    all_passed = True
    results = []
    
    # Core dependencies
    print(f"{Colors.BLUE}[1/5] Testing Core Dependencies...{Colors.END}")
    core_deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
    ]
    
    for module, name in core_deps:
        success, msg = test_import(module, name)
        results.append((name, success, msg))
        if success:
            print(f"  {Colors.GREEN}✓{Colors.END} {msg}")
        else:
            print(f"  {Colors.RED}✗{Colors.END} {msg}")
            all_passed = False
    
    # Computer Vision dependencies
    print(f"\n{Colors.BLUE}[2/5] Testing Computer Vision Dependencies...{Colors.END}")
    cv_deps = [
        ("cv2", "OpenCV"),
        ("mmcv", "MMCV"),
        ("mmengine", "MMEngine"),
        ("mmdet", "MMDetection"),
        ("mmpose", "MMPose"),
    ]
    
    for module, name in cv_deps:
        success, msg = test_import(module, name)
        results.append((name, success, msg))
        if success:
            print(f"  {Colors.GREEN}✓{Colors.END} {msg}")
        else:
            print(f"  {Colors.RED}✗{Colors.END} {msg}")
            all_passed = False
    
    # Biomechanics dependencies
    print(f"\n{Colors.BLUE}[3/5] Testing Biomechanics Dependencies...{Colors.END}")
    bio_deps = [
        ("opensim", "OpenSim"),
        ("smplx", "SMPL-X"),
    ]
    
    for module, name in bio_deps:
        success, msg = test_import(module, name)
        results.append((name, success, msg))
        if success:
            print(f"  {Colors.GREEN}✓{Colors.END} {msg}")
        else:
            print(f"  {Colors.RED}✗{Colors.END} {msg}")
            all_passed = False
    
    # Utility dependencies
    print(f"\n{Colors.BLUE}[4/5] Testing Utility Dependencies...{Colors.END}")
    util_deps = [
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("huggingface_hub", "Hugging Face Hub"),
    ]
    
    for module, name in util_deps:
        success, msg = test_import(module, name)
        results.append((name, success, msg))
        if success:
            print(f"  {Colors.GREEN}✓{Colors.END} {msg}")
        else:
            print(f"  {Colors.RED}✗{Colors.END} {msg}")
            all_passed = False
    
    # GPU Check
    print(f"\n{Colors.BLUE}[5/5] Testing GPU Availability...{Colors.END}")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  {Colors.GREEN}✓{Colors.END} CUDA is available")
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.END} CUDA is NOT available (CPU mode)")
            print(f"    This will significantly slow down processing.")
    except Exception as e:
        print(f"  {Colors.RED}✗{Colors.END} Error checking CUDA: {e}")
        all_passed = False
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}Summary{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All dependencies verified successfully! ({passed}/{total}){Colors.END}")
        print(f"\n{Colors.GREEN}Your environment is ready to run OpenCapBench.{Colors.END}")
        print(f"\nNext steps:")
        print(f"  1. Download the OpenCap dataset (see docs/DATA_DOWNLOAD.md)")
        print(f"  2. Download model weights (see docs/MODEL_DOWNLOAD.md)")
        print(f"  3. Configure benchmarking/constants.py")
        print(f"  4. Run: python benchmarking/benchmark.py --help")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some dependencies are missing ({passed}/{total} passed){Colors.END}")
        print(f"\n{Colors.YELLOW}Failed dependencies:{Colors.END}")
        for name, success, msg in results:
            if not success:
                print(f"  - {msg}")
        
        print(f"\n{Colors.YELLOW}To fix:{Colors.END}")
        print(f"  1. Make sure you activated the virtual environment:")
        print(f"     source .venv/bin/activate")
        print(f"  2. Run the setup script:")
        print(f"     ./setup.sh")
        print(f"  3. If issues persist, install missing packages manually:")
        print(f"     uv pip install <package-name>")
        return 1

if __name__ == "__main__":
    sys.exit(main())
