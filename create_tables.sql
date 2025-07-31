-- Create Tables for Graph AI Task Manager
-- Run this in your Neon Console SQL Editor

-- 1. Create parsed_email table
CREATE TABLE IF NOT EXISTS parsed_email (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR NOT NULL,
    date_received TIMESTAMPTZ NOT NULL,
    from_email VARCHAR,
    to_email TEXT,
    cc_email TEXT,
    bcc_email TEXT,
    from_name VARCHAR,
    to_name TEXT,
    cc_name TEXT,
    bcc_name TEXT,
    subject TEXT,
    content TEXT NOT NULL,
    content_length INTEGER,
    processed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    file_source VARCHAR
);

-- 2. Create tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    email_id INTEGER,
    message_id VARCHAR(255),
    task_name TEXT NOT NULL,
    task_description TEXT,
    due_date DATE,
    received_date DATE,
    status VARCHAR(255) DEFAULT 'Pending',
    topic VARCHAR(255),
    priority_level VARCHAR(255),
    sender VARCHAR(255),
    assigned_to VARCHAR(255),
    email_source VARCHAR(255),
    spam BOOLEAN DEFAULT false,
    validation_status VARCHAR(255) DEFAULT 'llm',
    confidence_score DOUBLE PRECISION,
    raw_json JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (email_id) REFERENCES parsed_email(id) ON DELETE SET NULL
);

-- 3. Create task_analysis table (if needed)
CREATE TABLE IF NOT EXISTS task_analysis (
    id SERIAL PRIMARY KEY,
    task_name TEXT,
    task_description TEXT,
    due_date DATE,
    received_date DATE,
    status VARCHAR(255),
    topic VARCHAR(255),
    priority_level VARCHAR(255),
    sender VARCHAR(255),
    assigned_to VARCHAR(255),
    spam BOOLEAN,
    validation_status VARCHAR(255),
    confidence_score DOUBLE PRECISION,
    email_subject TEXT,
    sender_name VARCHAR(255),
    email_date TIMESTAMPTZ
);

-- 4. Create indexes and constraints for better performance
-- parsed_email indexes and constraints
CREATE UNIQUE INDEX IF NOT EXISTS idx_parsed_email_message_id_unique ON parsed_email(message_id);
CREATE INDEX IF NOT EXISTS idx_parsed_email_date_received ON parsed_email(date_received);
CREATE INDEX IF NOT EXISTS idx_parsed_email_from_email ON parsed_email(from_email);
CREATE INDEX IF NOT EXISTS idx_parsed_email_subject ON parsed_email(subject);

-- tasks indexes
CREATE INDEX IF NOT EXISTS idx_tasks_email_id ON tasks(email_id);
CREATE INDEX IF NOT EXISTS idx_tasks_message_id ON tasks(message_id);
CREATE INDEX IF NOT EXISTS idx_tasks_topic ON tasks(topic);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_tasks_assigned_to ON tasks(assigned_to);
CREATE INDEX IF NOT EXISTS idx_tasks_validation_status ON tasks(validation_status);

-- task_analysis indexes
CREATE INDEX IF NOT EXISTS idx_task_analysis_topic ON task_analysis(topic);
CREATE INDEX IF NOT EXISTS idx_task_analysis_status ON task_analysis(status);
CREATE INDEX IF NOT EXISTS idx_task_analysis_due_date ON task_analysis(due_date);

-- 5. Create sequences (if not auto-created)
-- These should be created automatically by SERIAL columns, but just in case:
CREATE SEQUENCE IF NOT EXISTS parsed_email_id_seq;
CREATE SEQUENCE IF NOT EXISTS tasks_id_seq;
CREATE SEQUENCE IF NOT EXISTS task_analysis_id_seq;

-- 6. Grant permissions (if needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;

-- 7. Verify tables were created
SELECT 
    table_name, 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name IN ('parsed_email', 'tasks', 'task_analysis')
ORDER BY table_name, ordinal_position; 