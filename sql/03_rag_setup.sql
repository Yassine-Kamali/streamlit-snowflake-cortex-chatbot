-- Partie Bonus: mini-RAG (tables de documents/chunks)
USE DATABASE DB_LAB;
USE SCHEMA CHAT_APP;

CREATE TABLE IF NOT EXISTS RAG_DOCUMENTS (
  doc_id STRING,
  user_name STRING,
  doc_name STRING,
  source_type STRING,
  created_at TIMESTAMP_NTZ
);

CREATE TABLE IF NOT EXISTS RAG_CHUNKS (
  chunk_id STRING,
  doc_id STRING,
  user_name STRING,
  chunk_index NUMBER,
  chunk_text STRING,
  embedding VECTOR(FLOAT, 1024),
  created_at TIMESTAMP_NTZ
);

-- Verification rapide
DESC TABLE RAG_DOCUMENTS;
DESC TABLE RAG_CHUNKS;
