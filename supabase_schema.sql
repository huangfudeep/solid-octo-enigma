-- Run this once in Supabase Dashboard > SQL Editor.
-- These statements are safe to rerun and do not delete existing data.
CREATE TABLE IF NOT EXISTS public.qa (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL UNIQUE,
    answer TEXT NOT NULL,
    source TEXT,
    created_at TEXT,
    embedding_api JSONB,
    embedding_local JSONB
);
ALTER TABLE public.qa ADD COLUMN IF NOT EXISTS embedding_api JSONB;
ALTER TABLE public.qa ADD COLUMN IF NOT EXISTS embedding_local JSONB;

CREATE TABLE IF NOT EXISTS public.unanswered (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS unanswered_question_status_idx ON public.unanswered (question, status);

CREATE TABLE IF NOT EXISTS public.app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT INTO public.app_settings (key, value) VALUES ('api_enabled', '1') ON CONFLICT (key) DO NOTHING;
