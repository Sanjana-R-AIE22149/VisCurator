import subprocess
import os
import sys
import platform

def run_command(command, shell=False):
    """Utility to run shell commands."""
    try:
        process = subprocess.Popen(
            command,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        return process.returncode
    except Exception as e:
        print(f"Error executing command: {e}")
        return 1

def main():
    # Ensure we are in the project root (where package.json lives)
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("--- Initializing CVAgent Frontend ---")

    # 1. Check for node_modules
    if not os.path.exists("node_modules"):
        print("[*] node_modules not found. Installing dependencies...")
        install_cmd = "npm install"
        if platform.system() == "Windows":
            # On Windows, sometimes you need to call 'npm.cmd' or use shell=True
            result = run_command("npm install", shell=True)
        else:
            result = run_command(["npm", "install"])
            
        if result != 0:
            print("[!] Failed to install dependencies. Please ensure Node.js and npm are installed.")
            sys.exit(1)

    # 2. Run the dev server
    print("[+] Starting Vite development server...")
    if platform.system() == "Windows":
        run_command("npm run dev", shell=True)
    else:
        run_command(["npm", "run", "dev"])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Shutting down CVAgent...")
        sys.exit(0)
