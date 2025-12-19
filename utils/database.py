import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import json

# Handle dotenv import gracefully for deployment environments
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class PostgreSQLDatabase:
    """
    PostgreSQL database manager for the automated task manager.
    Handles storing parsed emails and validated tasks.
    Optimized for Neon PostgreSQL cloud platform.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        username: str = None,
        password: str = None,
        connection_string: str = None,
    ):
        # Support both individual parameters and Neon connection string
        self.connection_string = connection_string or os.getenv("DATABASE_URL")

        if self.connection_string:
            # Use Neon connection string (preferred method)
            self.host = None
            self.port = None
            self.database = None
            self.username = None
            self.password = None
        else:
            # Fallback to individual parameters
            self.host = host or os.getenv("POSTGRES_HOST", "localhost")
            self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
            self.database = database or os.getenv("POSTGRES_DB", "task_manager")
            self.username = username or os.getenv("POSTGRES_USER", "postgres")
            self.password = password or os.getenv("POSTGRES_PASSWORD", "password")

        self.connection = None

    def connect(self) -> bool:
        """Establish connection to PostgreSQL database (Neon optimized)."""
        try:
            if self.connection_string:
                # Use Neon connection string (includes SSL and pooling)
                self.connection = psycopg2.connect(
                    self.connection_string,
                    cursor_factory=RealDictCursor,
                    sslmode="require",  # Neon requires SSL
                )
                logger.info("✅ Connected to Neon PostgreSQL database")
            else:
                # Fallback to individual parameters
                self.connection = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.username,
                    password=self.password,
                    cursor_factory=RealDictCursor,
                    sslmode="prefer",
                )
                logger.info("✅ Connected to PostgreSQL database")

            self.connection.autocommit = True
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            return False

    def close(self):
        """Close PostgreSQL connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def create_tables(self) -> bool:
        """Create the necessary tables if they don't exist."""
        if not self.connection:
            if not self.connect():
                return False

        try:
            with self.connection.cursor() as cursor:
                # Create parsed_email table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS parsed_email (
                        id SERIAL PRIMARY KEY,
                        message_id VARCHAR(255) UNIQUE NOT NULL,
                        date_received TIMESTAMP WITH TIME ZONE NOT NULL,
                        from_email VARCHAR(255),
                        to_email TEXT,
                        cc_email TEXT,
                        bcc_email TEXT,
                        from_name VARCHAR(255),
                        to_name TEXT,
                        cc_name TEXT,
                        bcc_name TEXT,
                        subject TEXT,
                        content TEXT NOT NULL,
                        content_length INTEGER,
                        processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        file_source VARCHAR(255)
                    )
                """
                )

                # Create indexes for parsed_email
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_parsed_email_message_id ON parsed_email(message_id)",
                    "CREATE INDEX IF NOT EXISTS idx_parsed_email_date_received ON parsed_email(date_received)",
                    "CREATE INDEX IF NOT EXISTS idx_parsed_email_from_email ON parsed_email(from_email)",
                    "CREATE INDEX IF NOT EXISTS idx_parsed_email_subject ON parsed_email(subject)",
                ]

                for index in indexes:
                    cursor.execute(index)

                # Create tasks table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        id SERIAL PRIMARY KEY,
                        email_id INTEGER,
                        message_id VARCHAR(255),
                        task_name TEXT NOT NULL,
                        task_description TEXT,
                        due_date DATE,
                        received_date DATE,
                        status VARCHAR(255) DEFAULT 'not started',
                        topic VARCHAR(255),
                        priority_level VARCHAR(255),
                        sender VARCHAR(255),
                        assigned_to VARCHAR(255),
                        email_source VARCHAR(255),
                        spam BOOLEAN DEFAULT false,
                        validation_status VARCHAR(255) DEFAULT 'llm',
                        confidence_score DOUBLE PRECISION,
                        raw_json JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (email_id) REFERENCES parsed_email(id) ON DELETE SET NULL
                    )
                """
                )

                # Create indexes for tasks
                task_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_tasks_email_id ON tasks(email_id)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_message_id ON tasks(message_id)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_topic ON tasks(topic)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_sender ON tasks(sender)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_assigned_to ON tasks(assigned_to)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_received_date ON tasks(received_date)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_validation_status ON tasks(validation_status)",
                ]

                for index in task_indexes:
                    cursor.execute(index)

                # Note: task_collaborators table removed - using flat structure instead

                logger.info("✅ Database tables created successfully")
                return True

        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            return False

    def store_parsed_emails(
        self, emails_df: pd.DataFrame, file_source: str = None
    ) -> List[int]:
        """
        Store parsed email data in the parsed_email table.

        Args:
            emails_df: DataFrame with parsed email data
            file_source: Name of the source .mbox file

        Returns:
            List of inserted email IDs
        """
        if not self.connection:
            if not self.connect():
                return []

        try:
            inserted_ids = []

            with self.connection.cursor() as cursor:
                for _, row in emails_df.iterrows():
                    # Convert date to proper format
                    date_received = pd.to_datetime(row["Date"])
                    # Ensure Message-ID is mapped to message_id in DB
                    message_id = row.get("Message-ID") or row.get("message_id") or None
                    cursor.execute(
                        """
                        INSERT INTO parsed_email (
                            message_id, date_received, from_email, to_email, cc_email, bcc_email,
                            from_name, to_name, cc_name, bcc_name, subject, content, 
                            content_length, file_source
                        ) VALUES (
                            %(message_id)s, %(date_received)s, %(from_email)s, %(to_email)s, 
                            %(cc_email)s, %(bcc_email)s, %(from_name)s, %(to_name)s, 
                            %(cc_name)s, %(bcc_name)s, %(subject)s, %(content)s, 
                            %(content_length)s, %(file_source)s
                        ) ON CONFLICT (message_id) DO UPDATE SET
                            processed_at = CURRENT_TIMESTAMP
                        RETURNING id
                        """,
                        {
                            "message_id": row[
                                "Message-ID"
                            ],  # <-- Always use Message-ID from DataFrame
                            "date_received": row["Date"],
                            "from_email": row["From"],
                            "to_email": row["To"],
                            "cc_email": row.get("Cc"),
                            "bcc_email": row.get("Bcc"),
                            "from_name": row.get("Name-From"),
                            "to_name": row.get("Name-To"),
                            "cc_name": row.get("Name-Cc"),
                            "bcc_name": row.get("Name-Bcc"),
                            "subject": row["Subject"],
                            "content": row["content"],
                            "content_length": (
                                len(row["content"]) if row["content"] else 0
                            ),
                            "file_source": file_source,
                        },
                    )

                    result = cursor.fetchone()
                    if result:
                        inserted_ids.append(result["id"])

            logger.info(f"✅ Stored {len(inserted_ids)} parsed emails")
            return inserted_ids

        except Exception as e:
            logger.error(f"❌ Error storing parsed emails: {e}")
            return []

    def store_validated_tasks(self, validated_tasks: List[Dict]) -> List[int]:
        """
        Store validated task data in the tasks table (flat structure).

        Args:
            validated_tasks: List of validated task dictionaries (flat structure)

        Returns:
            List of inserted task IDs
        """
        if not self.connection:
            if not self.connect():
                return []

        try:
            inserted_task_ids = []

            with self.connection.cursor() as cursor:
                for task_data in validated_tasks:
                    # Validate flat structure
                    required_fields = [
                        "task_name",
                        "task_description",
                        "topic",
                        "message_id",
                    ]

                    if not isinstance(task_data, dict):
                        logger.error("Invalid task JSON: Must be a dictionary")
                        continue

                    # Check required fields
                    missing_fields = []
                    for field in required_fields:
                        if field not in task_data or not task_data[field]:
                            missing_fields.append(field)

                    if missing_fields:
                        logger.error(
                            f"Invalid task JSON: Missing required fields: {missing_fields}"
                        )
                        continue

                    # Insert flat task data
                    task_id = self._insert_flat_task(cursor, task_data)
                    if task_id:
                        inserted_task_ids.append(task_id)

            logger.info(
                f"✅ Successfully stored {len(inserted_task_ids)} validated tasks"
            )
            return inserted_task_ids

        except Exception as e:
            logger.error(f"❌ Error storing validated tasks: {e}")
            return []

    # Old _insert_single_task method removed - using _insert_flat_task instead

    def _insert_flat_task(self, cursor, task_data: Dict) -> Optional[int]:
        """Insert a single flat task into the tasks table."""
        try:
            # Try to find corresponding email in parsed_email table
            email_id = None
            message_id = task_data.get("message_id")
            if message_id:
                cursor.execute(
                    "SELECT id FROM parsed_email WHERE message_id = %s LIMIT 1",
                    (message_id,),
                )
                result = cursor.fetchone()
                if result:
                    email_id = result["id"]
                else:
                    logger.warning(
                        f"Email with message_id {message_id} not found in parsed_email table"
                    )

            # Extract task details from flat structure
            task_name = safe_val(task_data.get("task_name"))
            task_description = safe_val(task_data.get("task_description"))
            topic = safe_val(task_data.get("topic"))
            due_date = safe_val(task_data.get("due_date"))
            received_date = safe_val(task_data.get("received_date"))
            status = safe_val(task_data.get("status", "not started"))
            priority_level = safe_val(task_data.get("priority_level"))
            sender = safe_val(task_data.get("sender"))
            assigned_to = safe_val(task_data.get("assigned_to"))
            spam = task_data.get("spam", False)
            validation_status = safe_val(task_data.get("validation_status", "llm"))
            confidence_score = task_data.get("confidence_score")

            # Convert date strings to proper dates
            if due_date:
                try:
                    due_date = pd.to_datetime(due_date).date()
                except:
                    due_date = None

            if received_date:
                try:
                    received_date = pd.to_datetime(received_date).date()
                except:
                    received_date = None

            # Insert task into database
            cursor.execute(
                """
                INSERT INTO tasks (
                    email_id, message_id, task_name, task_description, due_date, 
                    received_date, status, topic, priority_level, sender, assigned_to,
                    email_source, spam, validation_status, confidence_score, raw_json
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """,
                (
                    email_id,
                    message_id,
                    task_name,
                    task_description,
                    due_date,
                    received_date,
                    status,
                    topic,
                    priority_level,
                    sender,
                    assigned_to,
                    "enron_sample",
                    spam,
                    validation_status,
                    confidence_score,
                    json.dumps(task_data),
                ),
            )

            result = cursor.fetchone()
            if result:
                task_id = result["id"]
                logger.info(f"✅ Inserted task: {task_name}")
                return task_id
            else:
                logger.error("❌ Failed to insert task - no ID returned")
                return None

        except Exception as e:
            logger.error(f"❌ Error inserting flat task: {e}")
            return None

    def get_parsed_emails(self, limit: int = 100000, offset: int = 0) -> List[Dict]:
        """Retrieve parsed emails from database."""
        if not self.connection:
            if not self.connect():
                return []

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM parsed_email 
                    ORDER BY date_received DESC
                    LIMIT %s OFFSET %s
                """,
                    (limit, offset),
                )

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"❌ Error retrieving parsed emails: {e}")
            return []

    def get_tasks_with_collaborators(
        self, limit: int = 100000, offset: int = 0
    ) -> List[Dict]:
        """Retrieve tasks (simplified - no collaborators table)."""
        if not self.connection:
            if not self.connect():
                return []

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM tasks
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """,
                    (limit, offset),
                )

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"❌ Error retrieving tasks: {e}")
            return []

    def get_database_stats(self) -> Dict:
        """Get statistics about the database."""
        if not self.connection:
            if not self.connect():
                return {}

        try:
            with self.connection.cursor() as cursor:
                stats = {}

                # Count parsed emails
                cursor.execute("SELECT COUNT(*) as count FROM parsed_email")
                stats["parsed_emails"] = cursor.fetchone()["count"]

                # Count tasks
                cursor.execute("SELECT COUNT(*) as count FROM tasks")
                stats["tasks"] = cursor.fetchone()["count"]

                # Count collaborators (removed - using flat structure)
                stats["collaborators"] = 0

                # Recent activity
                cursor.execute(
                    """
                    SELECT DATE(processed_at) as date, COUNT(*) as count 
                    FROM parsed_email 
                    WHERE processed_at >= CURRENT_DATE - INTERVAL '7 days'
                    GROUP BY DATE(processed_at)
                    ORDER BY date DESC
                """
                )
                stats["recent_emails"] = [dict(row) for row in cursor.fetchall()]

                return stats

        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}

    def update_task_validation_status(self, task_id: int, status: str) -> bool:
        """Update the validation_status for a task."""
        if not self.connection:
            if not self.connect():
                return False
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE tasks SET validation_status = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                    (status, task_id),
                )
            return True
        except Exception as e:
            logger.error(f"❌ Error updating task validation_status: {e}")
            return False

    # Old update_collaborator_validation_status method removed - using flat structure


# Convenience functions for backward compatibility
def store_parsed_emails(
    emails_df: pd.DataFrame, file_source: str = None, connection_string: str = None
) -> List[int]:
    """Store parsed emails in PostgreSQL database (Neon compatible)."""
    db = PostgreSQLDatabase(connection_string=connection_string)
    if not db.create_tables():
        return []
    return db.store_parsed_emails(emails_df, file_source)


def store_validated_tasks(
    validated_tasks: List[Dict], connection_string: str = None
) -> List[int]:
    """Store validated tasks in PostgreSQL database (Neon compatible)."""
    db = PostgreSQLDatabase(connection_string=connection_string)
    if not db.create_tables():
        return []
    return db.store_validated_tasks(validated_tasks)


def safe_val(val):
    if pd.isnull(val) or val is None:
        return "Unknown"
    return str(val)


if __name__ == "__main__":
    # Test the database connection (Neon compatible)
    db = PostgreSQLDatabase()
    if db.connect():
        if db.connection_string:
            print("✅ Neon PostgreSQL connection successful")
        else:
            print("✅ PostgreSQL connection successful")

        if db.create_tables():
            print("✅ Database tables created")
            print("Database stats:", db.get_database_stats())
        db.close()
    else:
        print("❌ PostgreSQL connection failed")
        print(
            "💡 For Neon, set DATABASE_URL environment variable with your connection string"
        )
