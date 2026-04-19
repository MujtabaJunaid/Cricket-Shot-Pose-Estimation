#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode == 0

def main():
    print("Cricket Pose Classification - Development Server Startup")
    print("="*60)
    
    processes = []
    
    try:
        print("\n1. Starting Backend (FastAPI)...")
        backend_cmd = [sys.executable, "-m", "uvicorn", "backend.app:app", 
                      "--host", "0.0.0.0", "--port", "8000", "--reload"]
        backend_process = subprocess.Popen(backend_cmd)
        processes.append(("Backend", backend_process))
        print("Backend started on http://localhost:8000")
        
        import time
        time.sleep(3)
        
        print("\n2. Starting Frontend (React)...")
        frontend_cmd = ["npm", "start"]
        frontend_process = subprocess.Popen(frontend_cmd, cwd="frontend")
        processes.append(("Frontend", frontend_process))
        print("Frontend starting on http://localhost:3000")
        
        print("\n" + "="*60)
        print("Services running:")
        print("  - Backend API: http://localhost:8000")
        print("  - Frontend: http://localhost:3000")
        print("  - API Docs: http://localhost:8000/docs")
        print("="*60)
        
        for name, proc in processes:
            proc.wait()
    
    except KeyboardInterrupt:
        print("\n\nShutting down services...")
        for name, proc in processes:
            proc.terminate()
            print(f"{name} terminated")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
