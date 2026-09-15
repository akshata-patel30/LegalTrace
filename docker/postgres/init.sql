-- LegalTrace PostgreSQL Initialization Script
-- Runs automatically on first container start via docker-entrypoint-initdb.d

-- Extensions for full-text search and JSON indexing
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Set timezone
ALTER DATABASE legaltrace SET timezone TO 'UTC';
