#!/usr/bin/env python3
"""
setup_dataset.py

Self-contained setup script:
1. Requires ZERO host dependencies or API keys (uses only Python 3 standard library).
2. Automatically creates a virtual environment (.venv) if one does not exist.
3. Installs 'torch' and 'kagglehub' into the virtual environment.
4. Uses 'kagglehub' to download 'angelikasita/waste-images' completely anonymously
   (no Kaggle API key, account, or kaggle.json required).
5. Places the extracted dataset directly into the target 'dataset' directory.
"""

import os
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path


def get_venv_executables(venv_dir: Path) -> tuple[Path, Path]:
    """Returns paths to (python, pip) within the virtual environment."""
    if sys.platform == "win32":
        python_bin = venv_dir / "Scripts" / "python.exe"
        pip_bin = venv_dir / "Scripts" / "pip.exe"
    else:
        python_bin = venv_dir / "bin" / "python"
        pip_bin = venv_dir / "bin" / "pip"
    return python_bin, pip_bin


def create_virtual_environment(venv_dir: Path) -> Path:
    """Creates a .venv virtual environment if it does not already exist."""
    python_bin, _ = get_venv_executables(venv_dir)
    if python_bin.exists():
        print(f"[*] Virtual environment already exists at: {venv_dir}")
        return python_bin

    print(f"[*] Creating virtual environment at: {venv_dir}...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    print("[+] Virtual environment created successfully.")
    return python_bin


def install_packages(python_bin: Path, packages: list[str]) -> None:
    """Installs required packages inside the virtual environment."""
    print("[*] Ensuring pip is up-to-date...")
    subprocess.run([str(python_bin), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)

    print(f"[*] Installing packages ({', '.join(packages)}) into .venv...")
    subprocess.run([str(python_bin), "-m", "pip", "install"] + packages, check=True)
    print("[+] Dependencies installed successfully inside .venv.")


def download_dataset_anonymously(python_bin: Path, dataset_slug: str, target_dir: Path) -> Path:
    """
    Downloads the Kaggle dataset anonymously using kagglehub (no API key needed)
    and copies/moves it into target_dir.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading '{dataset_slug}' anonymously via kagglehub (no keys required)...")

    # Python one-liner run inside the venv to trigger unauthenticated kagglehub download
    code = (
        f"import kagglehub\n"
        f"path = kagglehub.dataset_download('{dataset_slug}')\n"
        f"print('KAGGLEHUB_PATH=' + str(path))\n"
    )

    proc = subprocess.run(
        [str(python_bin), "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    downloaded_path_str = None
    for line in proc.stdout.splitlines():
        if line.startswith("KAGGLEHUB_PATH="):
            downloaded_path_str = line.split("=", 1)[1].strip()
            break

    if not downloaded_path_str or not Path(downloaded_path_str).exists():
        raise RuntimeError(f"Failed to locate downloaded dataset path. Output:\n{proc.stdout}\n{proc.stderr}")

    source_path = Path(downloaded_path_str)
    print(f"[+] Downloaded and extracted by kagglehub to: {source_path}")

    # Copy / sync files into target_dir
    print(f"[*] Syncing dataset files to target destination: {target_dir}...")
    for item in source_path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print(f"[+] Dataset ready at: {target_dir}")
    return target_dir


def extract_and_cleanup_zips(target_dir: Path) -> None:
    """Extracts any lingering .zip files in target_dir and removes them."""
    zip_files = list(target_dir.glob("*.zip"))
    if not zip_files:
        return

    for zip_path in zip_files:
        print(f"[*] Extracting '{zip_path.name}' to '{target_dir}'...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        zip_path.unlink()
        print(f"[+] Extracted and removed '{zip_path.name}'.")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    venv_dir = base_dir / ".venv"
    dataset_dir = base_dir / "dataset"
    dataset_slug = "angelikasita/waste-images"

    print("=" * 60)
    print(" Waste Images Dataset Setup (Zero Keys / Zero Setup Needed)")
    print("=" * 60)

    # 1. Create .venv (standard library, zero host dependencies)
    python_bin = create_virtual_environment(venv_dir)

    # 2. Install torch and kagglehub inside .venv
    # kagglehub allows 100% anonymous, keyless downloads of public Kaggle datasets
    install_packages(python_bin, ["torch", "kagglehub"])

    # 3. Download dataset without credentials & sync to dataset/
    download_dataset_anonymously(python_bin, dataset_slug, dataset_dir)

    # 4. Clean up any zip files if present
    extract_and_cleanup_zips(dataset_dir)

    print("\n[✓] All done! Virtual environment ready & dataset downloaded.")
    print(f"    - Virtual environment: {venv_dir}")
    print(f"    - Dataset folder:      {dataset_dir}")


if __name__ == "__main__":
    main()
