#!/usr/bin/env python3
"""
Deployment script for the sentiment ML service on Modal.
Run this script to test and deploy the service.
"""
import subprocess
import sys
import os
from datetime import datetime


def run_command(cmd, description):
    """Run a shell command and handle output."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed!")
        if result.stderr:
            print(f"Error details: {result.stderr}")
        return False
    
    print(f"✅ {description} completed successfully!")
    return True


def main():
    """Main deployment process."""
    print(f"""
╔════════════════════════════════════════════════════════════╗
║     Sentiment ML Service - Modal Deployment Script        ║
║                                                            ║
║     Starting deployment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}           ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check Modal authentication
    if not run_command("modal token", "Checking Modal authentication"):
        print("\n⚠️  Please authenticate with Modal first:")
        print("   Run: modal setup")
        sys.exit(1)
    
    # Step 2: Run local tests
    print("\n" + "="*60)
    print("🧪 Running local tests...")
    print("="*60)
    
    # First, check if we're in a virtual environment
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  Warning: Not in a virtual environment.")
        print("   Consider activating your virtual environment first:")
        print("   source venv/bin/activate")
    
    # Run pytest locally
    if not run_command("python -m pytest tests/ -v --tb=short", "Running local tests"):
        print("\n❌ Local tests failed! Fix tests before deploying.")
        sys.exit(1)
    
    # Step 3: Deploy to Modal with integrated testing
    print("\n" + "="*60)
    print("🚀 Deploying to Modal (includes remote testing)...")
    print("="*60)
    
    # Run the modal_app.py script which includes testing and deployment
    if not run_command("python modal_app.py", "Deploying to Modal"):
        print("\n❌ Modal deployment failed!")
        sys.exit(1)
    
    # Step 4: Verify deployment
    print("\n" + "="*60)
    print("🔍 Verifying deployment...")
    print("="*60)
    
    # Get the deployment URL
    result = subprocess.run(
        "modal app list | grep sentiment-ml-service",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ App is deployed and running!")
        print("\n📊 Deployment Summary:")
        print("   - Service: sentiment-ml-service")
        print("   - GPU: NVIDIA T4")
        print("   - Auto-scaling: 1-10 containers")
        print("   - Endpoint: https://theresaanna--sentiment-ml-service-fastapi-app.modal.run")
        print("\n📝 Available endpoints:")
        print("   - GET  /health              - Health check")
        print("   - POST /analyze-text        - Single text analysis")
        print("   - POST /analyze-batch       - Batch text analysis")
        
        print("\n🧪 Test your deployment:")
        print('   curl https://theresaanna--sentiment-ml-service-fastapi-app.modal.run/health')
        print('   curl -X POST https://theresaanna--sentiment-ml-service-fastapi-app.modal.run/analyze-text \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{"text": "I love this service!"}\'')
    else:
        print("⚠️  Could not verify deployment status")
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║              Deployment Complete! 🎉                      ║
║                                                            ║
║     Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                      ║
╚════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)