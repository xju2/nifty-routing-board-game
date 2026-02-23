#!/usr/bin/env python
import torch
import subprocess
import sys
import os
import argparse
import shutil
import multiprocessing

def get_cuda_tag():
    """
    Returns the CUDA tag (e.g., 'cu118', 'cu121', 'cpu') based on the
    current PyTorch installation.
    """
    cuda_info = torch.version.cuda  # type: ignore
    if not torch.cuda.is_available():
        # Check if it's explicitly a CPU build or just no GPU found
        if cuda_info is None:
            return 'cpu'

    # If CUDA is available or torch was built with CUDA
    if cuda_info:
        # Remove dots from version (e.g., '12.1' -> '121')
        version_clean = cuda_info.replace('.', '')
        return f'cu{version_clean}'

    return 'cpu'

def get_torch_version():
    """
    Returns the simplified PyTorch version (e.g., '2.3.0') ignoring
    the local suffix (like '+cu121').
    """
    version = torch.__version__
    return version.split('+')[0]

def check_nvcc():
    """Checks if nvcc is available for source compilation."""
    if shutil.which("nvcc") is None:
        print("\n[WARNING] 'nvcc' not found in PATH.")
        print("If you are compiling for GPU, please load the CUDA module.")
        print("Example: module load cudatoolkit")
        print("Proceeding, but compilation may fail or fallback to CPU-only...\n")
        return False
    return True

def configure_compiler():
    """Configures CC and CXX environment variables to use g++ explicitly."""
    gpp_path = shutil.which("g++")
    gcc_path = shutil.which("gcc")

    if gpp_path and gcc_path:
        print(f"\n[INFO] Found compiler at: {gpp_path}")
        print("[INFO] Forcing build to use 'gcc' and 'g++' instead of generic 'c++'.")
        os.environ["CC"] = gcc_path
        os.environ["CXX"] = gpp_path
    else:
        print("\n[WARNING] 'g++' or 'gcc' not found in PATH. Using system default 'c++' (this may fail).")

def optimize_compilation(num_cores=None):
    """
    Sets environment variables to speed up compilation:
    1. MAX_JOBS: Uses specified or all CPU cores.
    2. TORCH_CUDA_ARCH_LIST: Compiles ONLY for the current GPU arch.
    """
    # 1. Parallelize compilation
    if num_cores:
        cores = int(num_cores)
        print(f"[SPEEDUP] Using user-specified MAX_JOBS={cores}")
    else:
        cores = multiprocessing.cpu_count()
        print(f"[SPEEDUP] Using detected MAX_JOBS={cores}")

    os.environ["MAX_JOBS"] = str(cores)

    # 2. Target specific GPU architecture (Major compilation speedup)
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability()
            arch = f"{capability[0]}.{capability[1]}"
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            print(f"[SPEEDUP] Limiting compilation to local GPU Arch: {arch}")
        except Exception as e:
            print(f"[WARNING] Could not detect GPU architecture: {e}")
    else:
        print("[INFO] No GPU detected, skipping architecture optimization.")

def get_install_cmd_prefix():
    """
    Returns the installation command prefix, preferring 'uv' if available.
    """
    uv_path = shutil.which("uv")
    if uv_path:
        print("[INFO] Using 'uv' for faster installation.")
        # uv pip install requires targeting the specific python environment
        return [uv_path, "pip", "install", "--python", sys.executable]
    else:
        return [sys.executable, "-m", "pip", "install"]

def install_pyg_dependencies(force_source=False, num_cores=None):
    print("--- Diagnosing Environment ---")
    print(f"Python: {sys.version.split()[0]}")

    try:
        torch_ver = get_torch_version()
        cuda_tag = get_cuda_tag()
        print(f"PyTorch Version: {torch_ver}")
        print(f"CUDA Tag: {cuda_tag}")
    except Exception as e:
        print(f"Error checking PyTorch version: {e}")
        print("Please ensure PyTorch is installed first.")
        return

    # Calculate wheel URL (needed for both source and binary modes for pyg_lib)
    whl_url = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"
    print("\n--- Target Wheel URL ---")
    print(f"{whl_url}")

    # Define package groups
    source_pkgs = [
        "pyg_lib",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv"
    ]

    # Check if the existing installation is broken
    print("\n--- Checking existing installations ---")
    broken_install = False
    for pkg in source_pkgs:
        try:
            mod = __import__(pkg)
            print(f"[FOUND] {pkg} version: {mod.__version__}")
        except ImportError:
            print(f"[MISSING] {pkg} not found.")
            broken_install = True
        except Exception as e:
            print(f"[ERROR] Could not import {pkg}: {e}")
            broken_install = True

    if not broken_install:
        print("\nAll PyG dependencies are already installed and functional.")
        return

    # Determine base install command (uv vs pip)
    install_prefix = get_install_cmd_prefix()
    uninstall_prefix = [sys.executable, "-m", "pip", "uninstall", "-y"] # pip uninstall is reliable

    # 1. Uninstall existing broken versions
    print("\n--- Cleaning up broken installations ---")
    subprocess.run(uninstall_prefix + source_pkgs)

    if force_source:

        # Step B: Install others from Source
        print(f"\n[Step 1/1] Compiling {', '.join(source_pkgs)} from source (GLIBC Fix)...")
        print("This will take several minutes per package, but we have optimized settings.")

        # Configure Compiler & Optimizations
        configure_compiler()
        optimize_compilation(num_cores=num_cores)

        if cuda_tag != 'cpu':
            check_nvcc()
            os.environ['FORCE_CUDA'] = '1'

        # Command to install from source
        # Swapped --no-binary :all: for --no-build-isolation as requested.
        # This prevents pip from creating a fresh build environment, ensuring it uses
        # the currently installed PyTorch/CUDA for compilation.
        install_src_cmd = install_prefix + ["--no-build-isolation"] + source_pkgs + ["-f", whl_url]

        print(f"Running: {' '.join(install_src_cmd)}")
        result = subprocess.run(install_src_cmd)

    else:
        # 2. Install everything from binary wheels (Standard)
        print("\n--- Installing all compatible wheels ---")

        install_cmd = install_prefix + source_pkgs + ["-f", whl_url]

        print(f"Running: {' '.join(install_cmd)}")
        result = subprocess.run(install_cmd)

    if result.returncode == 0:
        print("\nSUCCESS: PyG dependencies installed successfully.")
        print("You can verify by running: python -c 'import torch_scatter; print(torch_scatter.__version__)'")
    else:
        print("\nFAILURE: Installation failed.")
        if not force_source:
            print("Tip: If you saw 'GLIBC not found' errors, try running with: python fix_pyg.py --source")
        else:
            print("Tip: Ensure 'nvcc' matches your PyTorch CUDA version and gcc is up to date.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install PyG dependencies.")
    parser.add_argument("--source", action="store_true", help="Force install from source (fixes GLIBC errors)")
    parser.add_argument("--num-cores", type=int, help="Number of CPU cores to use for compilation", default=None)
    args = parser.parse_args()

    install_pyg_dependencies(force_source=args.source, num_cores=args.num_cores)
