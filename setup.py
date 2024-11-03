import os
import subprocess
import sys
import venv
from pathlib import Path

def create_and_setup_venv():
    # Define paths
    venv_dir = '.venv'
    requirements_file = 'requirements.txt'
    
    # Check if requirements.txt exists
    if not os.path.exists(requirements_file):
        print("Error: requirements.txt not found!")
        print("Please create a requirements.txt file with your dependencies.")
        sys.exit(1)
    
    # Create virtual environment
    print("Creating virtual environment...")
    venv.create(venv_dir, with_pip=True)
    
    # Determine the pip path based on operating system
    if sys.platform == 'win32':
        pip_path = os.path.join(venv_dir, 'Scripts', 'pip')
        python_path = os.path.join(venv_dir, 'Scripts', 'python')
    else:  # Unix-based systems
        pip_path = os.path.join(venv_dir, 'bin', 'pip')
        python_path = os.path.join(venv_dir, 'bin', 'python')
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.check_call([python_path, '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # Install requirements
    print("Installing requirements...")
    subprocess.check_call([pip_path, 'install', '-r', requirements_file])
    
    print("\nSetup completed successfully!")
    print("\nTo activate the virtual environment:")
    if sys.platform == 'win32':
        print(f".\\{venv_dir}\\Scripts\\activate")
    else:
        print(f"source {venv_dir}/bin/activate")

if __name__ == '__main__':
    try:
        create_and_setup_venv()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)