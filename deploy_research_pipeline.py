#!/usr/bin/env python3
"""
Deploy Email Intelligence Research Pipeline to AWS
Complete infrastructure deployment with CDK
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are installed"""
    print("🔍 Checking prerequisites...")
    
    # Check AWS CLI
    try:
        result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
        print(f"✅ AWS CLI: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ AWS CLI not found. Install: https://aws.amazon.com/cli/")
        return False
    
    # Check CDK
    try:
        result = subprocess.run(['cdk', '--version'], capture_output=True, text=True)
        print(f"✅ AWS CDK: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ AWS CDK not found. Install: npm install -g aws-cdk")
        return False
    
    # Check Python dependencies
    try:
        import boto3
        print(f"✅ Boto3: {boto3.__version__}")
    except ImportError:
        print("❌ Boto3 not found. Install: pip install boto3")
        return False
    
    return True

def setup_aws_credentials():
    """Setup AWS credentials"""
    print("\n🔐 Setting up AWS credentials...")
    
    try:
        result = subprocess.run(['aws', 'sts', 'get-caller-identity'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            identity = json.loads(result.stdout)
            print(f"✅ AWS Account: {identity['Account']}")
            print(f"✅ AWS User: {identity['Arn']}")
            return identity['Account']
        else:
            print("❌ AWS credentials not configured")
            print("Run: aws configure")
            return None
    except Exception as e:
        print(f"❌ Error checking AWS credentials: {e}")
        return None

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    requirements = [
        "aws-cdk-lib>=2.0.0",
        "constructs>=10.0.0",
        "boto3>=1.26.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.0.0",
        "networkx>=3.0.0",
        "scikit-learn>=1.3.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "bertopic>=0.15.0",
        "sentence-transformers>=2.2.0",
        "neo4j>=5.0.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0"
    ]
    
    for requirement in requirements:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', requirement], 
                         check=True, capture_output=True)
            print(f"✅ Installed {requirement}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {requirement}: {e}")

def bootstrap_cdk(account_id, region="us-east-1"):
    """Bootstrap CDK in the AWS account"""
    print(f"\n🚀 Bootstrapping CDK in account {account_id}, region {region}...")
    
    try:
        result = subprocess.run([
            'cdk', 'bootstrap', 
            f'aws://{account_id}/{region}'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CDK bootstrap successful")
            return True
        else:
            print(f"❌ CDK bootstrap failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ CDK bootstrap error: {e}")
        return False

def deploy_infrastructure(account_id, region="us-east-1"):
    """Deploy the research infrastructure"""
    print(f"\n🏗️ Deploying Email Intelligence Research Infrastructure...")
    
    # Update the CDK app with correct account ID
    cdk_app_path = Path("aws_research_pipeline.py")
    if cdk_app_path.exists():
        content = cdk_app_path.read_text()
        content = content.replace("123456789012", account_id)
        content = content.replace("us-east-1", region)
        cdk_app_path.write_text(content)
        print(f"✅ Updated CDK app with account {account_id}")
    
    try:
        # Synthesize the CDK app
        result = subprocess.run(['cdk', 'synth'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ CDK synth failed: {result.stderr}")
            return False
        
        print("✅ CDK synthesis successful")
        
        # Deploy the stack
        result = subprocess.run([
            'cdk', 'deploy', 
            'EmailIntelligenceResearchStack',
            '--require-approval', 'never'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Infrastructure deployment successful")
            print("\n📊 Deployed Resources:")
            print("   🪣 S3 Buckets: Raw data, processed data, models, notebooks, results")
            print("   💻 EC2 Launch Template: Research instances with Jupyter")
            print("   🧠 SageMaker Notebook: ML research environment")
            print("   ⚡ Lambda Functions: Data preprocessing")
            print("   🔄 Batch Compute: Large-scale processing")
            print("   📊 Step Functions: Workflow orchestration")
            print("   📈 CloudWatch: Monitoring and logging")
            return True
        else:
            print(f"❌ Infrastructure deployment failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return False

def upload_notebooks():
    """Upload Jupyter notebooks to S3"""
    print("\n📓 Uploading Jupyter notebooks to S3...")
    
    try:
        import boto3
        s3_client = boto3.client('s3')
        
        notebooks_bucket = 'email-intelligence-notebooks'
        notebooks_dir = Path('notebooks')
        
        if notebooks_dir.exists():
            for notebook_file in notebooks_dir.glob('*.ipynb'):
                try:
                    s3_client.upload_file(
                        str(notebook_file),
                        notebooks_bucket,
                        notebook_file.name
                    )
                    print(f"✅ Uploaded {notebook_file.name}")
                except Exception as e:
                    print(f"⚠️ Failed to upload {notebook_file.name}: {e}")
        
        print("✅ Notebooks uploaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Notebook upload error: {e}")
        return False

def create_env_file():
    """Create .env file template for Neo4j credentials"""
    print("\n📝 Creating .env file template...")
    
    env_content = """# Neo4j Aura Credentials
# Get these from your Neo4j Aura console: https://console.neo4j.io/
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# AWS Configuration (optional - uses default profile)
AWS_REGION=us-east-1
AWS_PROFILE=default
"""
    
    env_path = Path('.env')
    if not env_path.exists():
        env_path.write_text(env_content)
        print("✅ Created .env template")
        print("📝 Please update .env with your Neo4j Aura credentials")
    else:
        print("✅ .env file already exists")

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 DEPLOYMENT COMPLETE!")
    print("=" * 50)
    
    print("\n🚀 NEXT STEPS:")
    print("1. 📝 Update .env file with Neo4j Aura credentials")
    print("2. 🖥️ Launch EC2 instance from launch template")
    print("3. 🌐 Access Jupyter at http://your-instance-ip:8888")
    print("4. 📓 Run notebooks in order:")
    print("   - 01_data_exploration.ipynb")
    print("   - 02_ai_processing.ipynb")
    print("   - (Additional notebooks as created)")
    
    print("\n📊 RESEARCH RESOURCES:")
    print("   🪣 S3 Buckets: Check AWS Console for data storage")
    print("   🧠 SageMaker: Notebook instance for ML research")
    print("   📈 CloudWatch: Monitor processing and costs")
    print("   🗄️ Neo4j Aura: Set up at https://console.neo4j.io/")
    
    print("\n💡 RESEARCH WORKFLOW:")
    print("   1. Load Enron dataset (automatically downloaded)")
    print("   2. Run data exploration notebook")
    print("   3. Process emails with complete AI suite")
    print("   4. Analyze results and generate insights")
    print("   5. Export findings for research publication")
    
    print("\n🔬 This demonstrates the REAL POWER of the AI system!")
    print("📊 Complete research pipeline with actual data processing!")

def main():
    """Main deployment function"""
    print("🧠 EMAIL INTELLIGENCE RESEARCH PIPELINE DEPLOYMENT")
    print("=" * 60)
    print("🎯 Deploying complete AWS infrastructure for research")
    print("📊 Processing real Enron dataset with advanced AI")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install required tools.")
        return 1
    
    # Setup AWS credentials
    account_id = setup_aws_credentials()
    if not account_id:
        print("\n❌ AWS credentials not configured.")
        return 1
    
    # Install dependencies
    install_dependencies()
    
    # Bootstrap CDK
    if not bootstrap_cdk(account_id):
        print("\n❌ CDK bootstrap failed.")
        return 1
    
    # Deploy infrastructure
    if not deploy_infrastructure(account_id):
        print("\n❌ Infrastructure deployment failed.")
        return 1
    
    # Upload notebooks
    upload_notebooks()
    
    # Create env file
    create_env_file()
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())