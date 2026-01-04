#!/usr/bin/env python3
"""
AWS Research Pipeline for Email Intelligence System
Complete infrastructure setup using AWS CDK for processing Enron dataset
"""

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_batch as batch,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
    aws_logs as logs,
    Duration,
    RemovalPolicy
)
from constructs import Construct
from typing import Dict, Any


class EmailIntelligenceResearchStack(Stack):
    """
    Complete AWS infrastructure for email intelligence research pipeline
    Processes Enron dataset with ML models and stores results in S3
    """
    
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Create S3 buckets for data pipeline
        self.create_s3_infrastructure()
        
        # Create compute infrastructure
        self.create_compute_infrastructure()
        
        # Create ML processing infrastructure
        self.create_ml_infrastructure()
        
        # Create monitoring and logging
        self.create_monitoring()
        
        # Create Step Functions workflow
        self.create_workflow()
    
    def create_s3_infrastructure(self):
        """Create S3 buckets for data storage"""
        
        # Raw data bucket for Enron dataset
        self.raw_data_bucket = s3.Bucket(
            self, "EmailIntelligenceRawData",
            bucket_name="email-intelligence-raw-data",
            versioned=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="archive-old-data",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30)
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90)
                        )
                    ]
                )
            ],
            removal_policy=RemovalPolicy.RETAIN
        )
        
        # Processed data bucket for ML results
        self.processed_data_bucket = s3.Bucket(
            self, "EmailIntelligenceProcessedData",
            bucket_name="email-intelligence-processed-data",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN
        )
        
        # Models bucket for storing trained models
        self.models_bucket = s3.Bucket(
            self, "EmailIntelligenceModels",
            bucket_name="email-intelligence-models",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN
        )
        
        # Notebooks bucket for Jupyter notebooks
        self.notebooks_bucket = s3.Bucket(
            self, "EmailIntelligenceNotebooks",
            bucket_name="email-intelligence-notebooks",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN
        )
        
        # Results bucket for final research outputs
        self.results_bucket = s3.Bucket(
            self, "EmailIntelligenceResults",
            bucket_name="email-intelligence-results",
            versioned=True,
            public_read_access=False,
            removal_policy=RemovalPolicy.RETAIN
        )
    
    def create_compute_infrastructure(self):
        """Create compute infrastructure for processing"""
        
        # VPC for secure compute
        self.vpc = ec2.Vpc(
            self, "EmailIntelligenceVPC",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ]
        )
        
        # Security group for compute instances
        self.compute_security_group = ec2.SecurityGroup(
            self, "ComputeSecurityGroup",
            vpc=self.vpc,
            description="Security group for email intelligence compute",
            allow_all_outbound=True
        )
        
        # Add inbound rules for Jupyter
        self.compute_security_group.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(8888),
            description="Jupyter notebook access"
        )
        
        # IAM role for compute instances
        self.compute_role = iam.Role(
            self, "EmailIntelligenceComputeRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchAgentServerPolicy")
            ]
        )
        
        # Add custom policies for Neo4j and other services
        self.compute_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:DescribeSecret",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                resources=["*"]
            )
        )
        
        # Launch template for research instances
        self.launch_template = ec2.LaunchTemplate(
            self, "ResearchInstanceTemplate",
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.M5,
                ec2.InstanceSize.XLARGE2
            ),
            machine_image=ec2.MachineImage.latest_amazon_linux2(),
            role=self.compute_role,
            security_group=self.compute_security_group,
            user_data=ec2.UserData.for_linux()
        )
        
        # Add user data for setting up research environment
        self.launch_template.user_data.add_commands(
            "yum update -y",
            "yum install -y python3 python3-pip git docker",
            "pip3 install --upgrade pip",
            "pip3 install jupyter pandas numpy scikit-learn transformers torch",
            "pip3 install bertopic sentence-transformers neo4j plotly",
            "pip3 install boto3 awscli",
            "systemctl start docker",
            "systemctl enable docker",
            "usermod -a -G docker ec2-user",
            
            # Setup Jupyter
            "mkdir -p /home/ec2-user/notebooks",
            "chown ec2-user:ec2-user /home/ec2-user/notebooks",
            
            # Download Enron dataset
            "cd /home/ec2-user",
            "wget -O enron_mail_20150507.tar.gz https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
            "tar -xzf enron_mail_20150507.tar.gz",
            "chown -R ec2-user:ec2-user maildir",
            
            # Start Jupyter as service
            "cat > /etc/systemd/system/jupyter.service << 'EOF'",
            "[Unit]",
            "Description=Jupyter Notebook Server",
            "After=network.target",
            "",
            "[Service]",
            "Type=simple",
            "User=ec2-user",
            "WorkingDirectory=/home/ec2-user/notebooks",
            "ExecStart=/usr/local/bin/jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root",
            "Restart=always",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "EOF",
            
            "systemctl daemon-reload",
            "systemctl enable jupyter",
            "systemctl start jupyter"
        )
    
    def create_ml_infrastructure(self):
        """Create ML processing infrastructure"""
        
        # SageMaker execution role
        self.sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
            ]
        )
        
        # SageMaker notebook instance for research
        self.sagemaker_notebook = sagemaker.CfnNotebookInstance(
            self, "EmailIntelligenceNotebook",
            instance_type="ml.t3.xlarge",
            role_arn=self.sagemaker_role.role_arn,
            notebook_instance_name="email-intelligence-research",
            default_code_repository="https://github.com/your-repo/email-intelligence-notebooks.git",
            volume_size_in_gb=100,
            root_access="Enabled"
        )
        
        # Lambda function for data preprocessing
        self.preprocessing_lambda = _lambda.Function(
            self, "EmailPreprocessingLambda",
            runtime=_lambda.Runtime.PYTHON_3_9,
            handler="lambda_function.lambda_handler",
            code=_lambda.Code.from_inline("""
import json
import boto3
import pandas as pd
from datetime import datetime

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    
    # Process email data
    bucket = event['bucket']
    key = event['key']
    
    # Download and process email file
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    
    # Basic email parsing
    email_data = {
        'processed_at': datetime.now().isoformat(),
        'file_key': key,
        'content_length': len(content),
        'status': 'processed'
    }
    
    # Upload processed data
    processed_key = f"processed/{key}.json"
    s3.put_object(
        Bucket='email-intelligence-processed-data',
        Key=processed_key,
        Body=json.dumps(email_data)
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Email processed successfully',
            'processed_key': processed_key
        })
    }
            """),
            timeout=Duration.minutes(15),
            memory_size=3008,
            environment={
                'RAW_BUCKET': self.raw_data_bucket.bucket_name,
                'PROCESSED_BUCKET': self.processed_data_bucket.bucket_name
            }
        )
        
        # Grant S3 permissions to Lambda
        self.raw_data_bucket.grant_read(self.preprocessing_lambda)
        self.processed_data_bucket.grant_write(self.preprocessing_lambda)
        
        # Batch compute environment for large-scale processing
        self.batch_compute_environment = batch.CfnComputeEnvironment(
            self, "EmailIntelligenceBatchCompute",
            type="MANAGED",
            state="ENABLED",
            compute_resources=batch.CfnComputeEnvironment.ComputeResourcesProperty(
                type="EC2",
                min_vcpus=0,
                max_vcpus=100,
                desired_vcpus=0,
                instance_types=["m5.large", "m5.xlarge", "m5.2xlarge"],
                subnets=[subnet.subnet_id for subnet in self.vpc.private_subnets],
                security_group_ids=[self.compute_security_group.security_group_id],
                instance_role=self.compute_role.role_arn
            )
        )
        
        # Batch job queue
        self.batch_job_queue = batch.CfnJobQueue(
            self, "EmailIntelligenceJobQueue",
            state="ENABLED",
            priority=1,
            compute_environment_order=[
                batch.CfnJobQueue.ComputeEnvironmentOrderProperty(
                    order=1,
                    compute_environment=self.batch_compute_environment.ref
                )
            ]
        )
    
    def create_monitoring(self):
        """Create monitoring and logging infrastructure"""
        
        # CloudWatch log group for application logs
        self.log_group = logs.LogGroup(
            self, "EmailIntelligenceLogGroup",
            log_group_name="/aws/email-intelligence/research",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY
        )
        
        # Custom metrics for tracking processing
        self.processing_metrics = logs.MetricFilter(
            self, "ProcessingMetrics",
            log_group=self.log_group,
            metric_namespace="EmailIntelligence",
            metric_name="EmailsProcessed",
            filter_pattern=logs.FilterPattern.literal("[timestamp, request_id, \"PROCESSED\", email_count]"),
            metric_value="$email_count"
        )
    
    def create_workflow(self):
        """Create Step Functions workflow for orchestration"""
        
        # Define preprocessing task
        preprocess_task = sfn_tasks.LambdaInvoke(
            self, "PreprocessEmails",
            lambda_function=self.preprocessing_lambda,
            output_path="$.Payload"
        )
        
        # Define batch processing task
        batch_task = sfn_tasks.BatchSubmitJob(
            self, "ProcessEmailsBatch",
            job_definition_arn="arn:aws:batch:*:*:job-definition/email-processing-job",
            job_name="email-intelligence-processing",
            job_queue_arn=self.batch_job_queue.ref
        )
        
        # Define workflow
        definition = preprocess_task.next(batch_task)
        
        # Create state machine
        self.state_machine = sfn.StateMachine(
            self, "EmailIntelligenceWorkflow",
            definition=definition,
            timeout=Duration.hours(2)
        )
        
        # Output important resources
        cdk.CfnOutput(
            self, "RawDataBucket",
            value=self.raw_data_bucket.bucket_name,
            description="S3 bucket for raw email data"
        )
        
        cdk.CfnOutput(
            self, "ProcessedDataBucket", 
            value=self.processed_data_bucket.bucket_name,
            description="S3 bucket for processed data"
        )
        
        cdk.CfnOutput(
            self, "NotebooksBucket",
            value=self.notebooks_bucket.bucket_name,
            description="S3 bucket for Jupyter notebooks"
        )
        
        cdk.CfnOutput(
            self, "ResultsBucket",
            value=self.results_bucket.bucket_name,
            description="S3 bucket for research results"
        )
        
        cdk.CfnOutput(
            self, "SageMakerNotebook",
            value=self.sagemaker_notebook.ref,
            description="SageMaker notebook instance for research"
        )


# CDK App
app = cdk.App()

EmailIntelligenceResearchStack(
    app, "EmailIntelligenceResearchStack",
    env=cdk.Environment(
        account="123456789012",  # Replace with your AWS account ID
        region="us-east-1"
    ),
    description="Complete AWS infrastructure for email intelligence research"
)

app.synth()