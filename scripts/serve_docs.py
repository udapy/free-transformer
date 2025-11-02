#!/usr/bin/env python3
"""Smart documentation server that finds available ports."""

import argparse
import socket
import subprocess
import sys
import time
from contextlib import closing

def find_free_port(start_port=8000, max_attempts=10):
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_attempts}")

def kill_existing_mkdocs():
    """Kill any existing mkdocs processes."""
    try:
        # Find and kill existing mkdocs processes
        result = subprocess.run(
            ["pgrep", "-f", "mkdocs serve"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"Killing existing mkdocs process (PID: {pid})")
                    subprocess.run(["kill", pid], capture_output=True)
                    time.sleep(0.5)
    except Exception as e:
        # Ignore errors - this is best effort cleanup
        pass

def serve_docs(dev_mode=False, port=None):
    """Serve documentation with smart port selection."""
    
    # Kill any existing mkdocs processes
    kill_existing_mkdocs()
    
    # Find available port
    if port is None:
        try:
            port = find_free_port()
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Build command
    addr = f"127.0.0.1:{port}"
    cmd = ["mkdocs", "serve", "--dev-addr", addr]
    
    if dev_mode:
        cmd.append("--livereload")
    
    print(f"🚀 Starting documentation server...")
    print(f"📖 Documentation will be available at: http://{addr}")
    print(f"🔄 {'Live reload enabled' if dev_mode else 'Static serving'}")
    print(f"⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the server
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Serve MkDocs documentation with smart port selection")
    parser.add_argument("--dev", action="store_true", help="Enable development mode with live reload")
    parser.add_argument("--port", type=int, help="Specific port to use (will check if available)")
    parser.add_argument("--kill", action="store_true", help="Kill existing mkdocs processes and exit")
    
    args = parser.parse_args()
    
    if args.kill:
        print("🔍 Looking for existing mkdocs processes...")
        kill_existing_mkdocs()
        print("✅ Cleanup complete")
        return
    
    # Validate port if specified
    if args.port:
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.bind(('127.0.0.1', args.port))
                port = args.port
        except OSError:
            print(f"❌ Port {args.port} is already in use. Finding alternative...")
            port = None
    else:
        port = None
    
    serve_docs(dev_mode=args.dev, port=port)

if __name__ == "__main__":
    main()