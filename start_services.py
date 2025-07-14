#!/usr/bin/env python3
"""
Service Orchestrator for Automated Task Manager
Manages both FastAPI backend and Streamlit frontend with proper cleanup
"""

import subprocess
import sys
import time
import os
import signal
import psutil
from pathlib import Path

def kill_process_on_port(port):
    """Kill any process running on the specified port."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections']:
                    if conn.laddr.port == port:
                        print(f"🔄 Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                        proc.terminate()
                        proc.wait(timeout=5)
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
    except Exception as e:
        print(f"⚠️ Warning: Could not check processes on port {port}: {e}")
    return False

def is_port_in_use(port):
    """Check if a port is currently in use."""
    try:
        for proc in psutil.process_iter(['connections']):
            try:
                for conn in proc.info['connections']:
                    if conn.laddr.port == port:
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    return False

def start_fastapi():
    """Start the FastAPI backend server."""
    print("🚀 Starting FastAPI backend...")
    
    # Kill any existing process on port 8000
    if is_port_in_use(8000):
        print("🔄 Port 8000 is in use, cleaning up...")
        kill_process_on_port(8000)
        time.sleep(2)
    
    try:
        # Start FastAPI server
        fastapi_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "utils.api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        
        # Wait a moment and check if it started successfully
        time.sleep(3)
        if fastapi_process.poll() is None:  # Process is still running
            print("✅ FastAPI backend started on http://localhost:8000")
            return fastapi_process
        else:
            print("❌ FastAPI backend failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start FastAPI: {e}")
        return None

def start_streamlit():
    """Start the Streamlit frontend."""
    print("🚀 Starting Streamlit frontend...")
    
    # Kill any existing process on port 8501
    if is_port_in_use(8501):
        print("🔄 Port 8501 is in use, cleaning up...")
        kill_process_on_port(8501)
        time.sleep(2)
    
    try:
        # Start Streamlit
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
        
        # Wait a moment and check if it started successfully
        time.sleep(3)
        if streamlit_process.poll() is None:  # Process is still running
            print("✅ Streamlit frontend started on http://localhost:8501")
            return streamlit_process
        else:
            print("❌ Streamlit frontend failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return None

def cleanup_processes():
    """Clean up any existing processes on our ports."""
    print("🧹 Cleaning up existing processes...")
    kill_process_on_port(8000)
    kill_process_on_port(8501)
    time.sleep(2)

def main():
    """Main startup function."""
    print("🤖 Automated Task Manager - Starting Services...")
    print("=" * 50)
    
    # Check if required files exist
    if not Path("utils/api.py").exists():
        print("❌ FastAPI backend not found (utils/api.py)")
        return
    
    if not Path("app.py").exists():
        print("❌ Streamlit app not found (app.py)")
        return
    
    # Clean up any existing processes
    cleanup_processes()
    
    # Start FastAPI backend
    fastapi_process = start_fastapi()
    if not fastapi_process:
        print("⚠️ Continuing without FastAPI backend (chat history will not be saved)")
    
    # Wait a moment for FastAPI to start
    if fastapi_process:
        time.sleep(3)
    
    # Start Streamlit frontend
    streamlit_process = start_streamlit()
    if not streamlit_process:
        print("❌ Failed to start Streamlit frontend")
        if fastapi_process:
            fastapi_process.terminate()
        return
    
    print("\n🎉 Both services started successfully!")
    print("📱 Streamlit UI: http://localhost:8501")
    print("🔧 FastAPI Backend: http://localhost:8000")
    print("\n💡 Press Ctrl+C to stop both services")
    
    try:
        # Keep both processes running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if fastapi_process and fastapi_process.poll() is not None:
                print("❌ FastAPI backend stopped unexpectedly")
                break
            
            if streamlit_process.poll() is not None:
                print("❌ Streamlit frontend stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
        # Terminate processes
        if fastapi_process:
            fastapi_process.terminate()
            print("✅ FastAPI backend stopped")
        
        if streamlit_process:
            streamlit_process.terminate()
            print("✅ Streamlit frontend stopped")
        
        # Final cleanup
        cleanup_processes()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main() 