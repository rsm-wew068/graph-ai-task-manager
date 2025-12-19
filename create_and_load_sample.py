#!/usr/bin/env python3
"""
Create and Load Enron Sample Script
Creates a sample from enron.parquet and loads it into parsed_email table
"""

import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_enron_sample():
    """Load enron_sample.parquet into parsed_email table"""

    # Get database connection
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")

    logger.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(database_url)
    cursor = conn.cursor()

    try:
        # Read the sample parquet file
        logger.info("Reading enron_sample.parquet file...")
        df_sample = pd.read_parquet("enron_sample.parquet")
        logger.info(f"Loaded {len(df_sample)} emails from sample file")

        # Clean and prepare data
        logger.info("Preparing data for insertion...")

        # Convert date column to proper format
        df_sample["Date"] = pd.to_datetime(df_sample["Date"])

        # Replace NaN values with empty strings for text columns
        text_columns = [
            "Message-ID",
            "From",
            "To",
            "Subject",
            "content",
            "Name-From",
            "Name-To",
            "Name-cc",
            "Name-bcc",
        ]
        for col in text_columns:
            if col in df_sample.columns:
                df_sample[col] = df_sample[col].fillna("")

        # Prepare data for insertion - map to parsed_email table columns
        data_to_insert = []
        for _, row in df_sample.iterrows():
            data_to_insert.append(
                (
                    row.get("Message-ID", ""),  # message_id
                    row.get("Date"),  # date_received
                    row.get("From", ""),  # from_email
                    row.get("To", ""),  # to_email
                    "",  # cc_email (not in our data)
                    "",  # bcc_email (not in our data)
                    row.get("Name-From", ""),  # from_name
                    row.get("Name-To", ""),  # to_name
                    "",  # cc_name (not in our data)
                    "",  # bcc_name (not in our data)
                    row.get("Subject", ""),  # subject
                    row.get("content", ""),  # content
                    len(str(row.get("content", ""))),  # content_length
                    datetime.now(),  # processed_at
                    "enron_sample",  # file_source
                )
            )

        # Insert data in batches
        batch_size = 100
        total_rows = len(data_to_insert)

        logger.info(f"Inserting {total_rows} emails in batches of {batch_size}...")

        for i in range(0, total_rows, batch_size):
            batch = data_to_insert[i : i + batch_size]

            insert_sql = """
            INSERT INTO parsed_email 
            (message_id, date_received, from_email, to_email, cc_email, bcc_email, 
             from_name, to_name, cc_name, bcc_name, subject, content, content_length, 
             processed_at, file_source)
            VALUES %s
            """

            execute_values(cursor, insert_sql, batch)

            # Commit each batch
            conn.commit()

            logger.info(
                f"Inserted batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size} "
                f"({min(i + batch_size, total_rows)}/{total_rows} emails)"
            )

        # Verify the insertion
        cursor.execute("SELECT COUNT(*) FROM parsed_email")
        count = cursor.fetchone()[0]
        logger.info(f"Successfully inserted {count} emails into parsed_email table")

        # Show some sample data
        cursor.execute(
            "SELECT message_id, date_received, from_email, subject FROM parsed_email LIMIT 5"
        )
        samples = cursor.fetchall()
        logger.info("Sample data from database:")
        for sample in samples:
            logger.info(f"  {sample}")

        # Get database size
        cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        db_size = cursor.fetchone()[0]
        logger.info(f"📈 Database size after insertion: {db_size}")

    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
        logger.info("Database connection closed")


def check_current_data():
    """Check current data in the database"""

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")

    logger.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(database_url)
    cursor = conn.cursor()

    try:
        # Check row counts
        logger.info("📋 Current row counts:")
        # Whitelisted table names to prevent SQL injection
        allowed_tables = ["parsed_email", "tasks", "task_analysis"]
        for table_name in allowed_tables:
            # Use identifier for safe table name insertion
            from psycopg2 import sql

            query = sql.SQL("SELECT COUNT(*) FROM {}").format(
                sql.Identifier(table_name)
            )
            cursor.execute(query)
            count = cursor.fetchone()[0]
            logger.info(f"  {table_name}: {count} rows")

        # Get database size
        cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        db_size = cursor.fetchone()[0]
        logger.info(f"📈 Current database size: {db_size}")

    except Exception as e:
        logger.error(f"Error checking data: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    print("📊 Enron Sample Creation and Loading")
    print("=" * 40)

    action = input(
        "Choose action:\n1. Load enron_sample.parquet\n2. Check current data\nEnter choice (1-2): "
    ).strip()

    if action == "1":
        load_enron_sample()
    elif action == "2":
        check_current_data()
    else:
        print("Invalid choice. Loading enron_sample.parquet.")
        load_enron_sample()
