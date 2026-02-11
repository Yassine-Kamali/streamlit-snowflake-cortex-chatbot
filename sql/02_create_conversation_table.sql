-- Partie D: table de persistance des conversations
USE DATABASE DB_LAB;
USE SCHEMA CHAT_APP;

CREATE TABLE IF NOT EXISTS CHAT_MESSAGES (
  conversation_id STRING,
  user_name STRING,
  "timestamp" TIMESTAMP_NTZ,
  role STRING,
  content STRING
);

-- Verification rapide
DESC TABLE CHAT_MESSAGES;
