#!/usr/bin/env python3
"""
Start Combined AI Chat + MCP Server Application
"""

import os
import sys
import threading
import time
from multiprocessing import Process

def start_main_app():
    """Start the main AI chat application"""
    print("🚀 Starting Main AI Chat Application...")
    try:
        from app import app
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Main app failed: {e}")

def start_mcp_server():
    """Start the MCP server"""
    print("🚀 Starting MCP Server...")
    try:
        from mcp_server import mcp_app
        mcp_app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ MCP server failed: {e}")

def main():
    """Start both applications"""
    print("🎯 Starting Combined AI Chat + MCP Server System...")
    print("=" * 60)
    
    # Start MCP server in a separate process
    mcp_process = Process(target=start_mcp_server)
    mcp_process.start()
    
    # Give MCP server time to start
    time.sleep(3)
    
    # Start main app in a separate process
    main_process = Process(target=start_main_app)
    main_process.start()
    
    print("\n✅ Both applications started!")
    print("📍 Main AI Chat App: http://127.0.0.1:5000")
    print("📍 MCP Server: http://127.0.0.1:5001")
    print("📍 MCP Dashboard: http://127.0.0.1:5001/admin/dashboard")
    print("📍 Health Check: http://127.0.0.1:5001/health")
    print("📊 Metrics: http://127.0.0.1:5001/metrics")
    print("\n🔄 Press Ctrl+C to stop both applications")
    
    try:
        # Wait for both processes
        main_process.join()
        mcp_process.join()
    except KeyboardInterrupt:
        print("\n🛑 Stopping applications...")
        main_process.terminate()
        mcp_process.terminate()
        main_process.join()
        mcp_process.join()
        print("✅ Applications stopped")

if __name__ == "__main__":
    main()
