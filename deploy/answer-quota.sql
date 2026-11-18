-- Run once on the restored target as its migration owner.
-- A separate login named ayurip_answer_quota must already exist.
CREATE SCHEMA answer_quota;
REVOKE ALL ON SCHEMA answer_quota FROM PUBLIC;
CREATE TABLE answer_quota.usage (
    period integer NOT NULL CHECK (period IN (3600, 86400)),
    started bigint NOT NULL,
    subject text NOT NULL CHECK (subject = 'site' OR subject ~ '^[0-9a-f]{64}$'),
    used integer NOT NULL DEFAULT 1 CHECK (used BETWEEN 1 AND 30),
    PRIMARY KEY (period, started, subject)
);
REVOKE ALL ON TABLE answer_quota.usage FROM PUBLIC;
GRANT USAGE ON SCHEMA answer_quota TO ayurip_answer_quota;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE answer_quota.usage TO ayurip_answer_quota;
