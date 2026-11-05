"""The OpenAI sessions should fall back to the key set via set_default_openai_key()."""

from __future__ import annotations

import pytest

from agents import SQLiteSession, set_default_openai_key
from agents.memory import OpenAIConversationsSession, OpenAIResponsesCompactionSession


@pytest.fixture(autouse=True)
def sdk_key_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    set_default_openai_key("sk-from-set-default-openai-key")


@pytest.mark.asyncio
async def test_conversations_session_uses_default_key():
    # constructing used to raise OpenAIError("Missing credentials") here
    session = OpenAIConversationsSession()

    assert session._openai_client.api_key == "sk-from-set-default-openai-key"


def test_compaction_session_uses_default_key():
    session = OpenAIResponsesCompactionSession("s", SQLiteSession("s"))

    assert session.client.api_key == "sk-from-set-default-openai-key"
