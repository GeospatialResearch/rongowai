import importlib
import sys
import platform
import subprocess

def check_python_environment(required_packages, target_python):
    """
    Checks the current Python version and validates if required packages are installed.
    Offers automated installation via conda-forge if packages are missing.
    """
    # 1. Check Python Version
    current_version = platform.python_version()
    print(f"Current Environment Python: {current_version}")

    if not current_version.startswith(target_python):
        print(f"⚠️  WARNING: You are running {current_version}, but your target is {target_python}.")
        cont = input("Continue anyway? (y/n): ").lower()
        if cont != 'y':
            sys.exit("Execution stopped. Please switch your VS Code kernel.")

    # 2. Check Package Status
    missing = []
    print(f"\n{'Package':<15} | {'Status':<12}")
    print("-" * 30)

    for lib, pkg in required_packages.items():
        try:
            importlib.import_module(lib)
            status = "✅ Ready"
        except ImportError:
            status = "❌ Missing"
            missing.append(pkg)
        print(f"{pkg:<15} | {status:<12}")

    # 3. Automated Install Prompt
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        response = input(f"Install these {len(missing)} packages via conda-forge? (y/n): ").lower()
        
        if response == 'y':
            packages_str = " ".join(missing)
            command = f"conda install {packages_str} -c conda-forge -y -v"
            
            print(f"Starting installation: {command}\n")
            
            # Stream the output live
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="")
            process.wait()
            print("\n✨ Installation complete! RESTART YOUR KERNEL now.")
        else:
            print("\nInstallation skipped.")
    else:
        print("\nAll systems go! Environment is fully configured.")
        