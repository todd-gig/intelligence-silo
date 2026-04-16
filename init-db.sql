-- Create databases for all services that declare DATABASE_URL
-- Runs automatically on first postgres container start via docker-entrypoint-initdb.d

CREATE DATABASE decision_logic;
CREATE DATABASE decision_execution;
CREATE DATABASE gigaton;
