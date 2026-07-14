"""Regression tests for canonical chat messages and OpenAI adapters."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from analytics_agent.messages import (
    AssistantMessage,
    RefusalPart,
    SystemMessage,
    TextPart,
    from_openai_output_item,
    function_call_message,
    function_call_output_message,
    generate_initial_messages,
    get_message_text,
    to_openai_input_item,
)
from analytics_agent.providers.openai_provider import OpenAIProvider


class MessageModelTests(unittest.TestCase):
    """Tests for normalized message variants and adapters."""

    def test_generate_initial_messages_returns_typed_variants(self) -> None:
        """Initial prompt helpers should return typed canonical messages."""
        messages = generate_initial_messages("system prompt", "user prompt")

        self.assertIsInstance(messages[0], SystemMessage)
        self.assertEqual(get_message_text(messages[0]), "system prompt")
        self.assertEqual(get_message_text(messages[1]), "user prompt")

    def test_function_call_preserves_raw_and_parsed_arguments(self) -> None:
        """Function calls should keep both raw and parsed arguments."""
        message = function_call_message(
            call_id="call_123",
            name="search_rows",
            arguments_raw='{"query":"visiting hour","limit":5}',
        )

        self.assertEqual(message.arguments_raw, '{"query":"visiting hour","limit":5}')
        self.assertEqual(
            message.arguments_parsed,
            {"query": "visiting hour", "limit": 5},
        )

    def test_function_call_output_serializes_for_openai(self) -> None:
        """Function call outputs should serialize to OpenAI input items."""
        item = to_openai_input_item(
            function_call_output_message("call_123", "Found one row")
        )

        self.assertEqual(
            item,
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Found one row",
            },
        )

    def test_assistant_message_round_trip_preserves_metadata(self) -> None:
        """Assistant output items should round-trip with normalized metadata."""
        response_item = SimpleNamespace(
            type="message",
            role="assistant",
            id="msg_123",
            status="completed",
            phase="final_answer",
            content=[
                SimpleNamespace(
                    type="output_text",
                    text="Visiting hours are 10 AM to 8 PM.",
                    annotations=[],
                    model_dump=lambda: {
                        "type": "output_text",
                        "text": "Visiting hours are 10 AM to 8 PM.",
                        "annotations": [],
                    },
                ),
                SimpleNamespace(
                    type="refusal",
                    refusal="Cannot disclose private patient data.",
                    model_dump=lambda: {
                        "type": "refusal",
                        "refusal": "Cannot disclose private patient data.",
                    },
                ),
            ],
            model_dump=lambda: {
                "type": "message",
                "role": "assistant",
                "id": "msg_123",
                "status": "completed",
                "phase": "final_answer",
            },
        )

        message = from_openai_output_item(response_item)

        self.assertIsInstance(message, AssistantMessage)
        assert isinstance(message, AssistantMessage)
        self.assertEqual(message.id, "msg_123")
        self.assertEqual(message.status, "completed")
        self.assertEqual(message.phase, "final_answer")
        self.assertIsInstance(message.content[0], TextPart)
        self.assertIsInstance(message.content[1], RefusalPart)
        self.assertEqual(
            to_openai_input_item(message),
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Visiting hours are 10 AM to 8 PM.",
                        "annotations": [],
                    },
                    {
                        "type": "refusal",
                        "refusal": "Cannot disclose private patient data.",
                    },
                ],
                "id": "msg_123",
                "status": "completed",
                "phase": "final_answer",
            },
        )

    def test_function_call_from_openai_output_captures_identity(self) -> None:
        """Function call output items should preserve identity and arguments."""
        response_item = SimpleNamespace(
            type="function_call",
            id="fc_123",
            call_id="call_123",
            name="search_rows",
            arguments='{"query":"visiting hour","dataset_name":"Hospital Policy"}',
            status="completed",
            model_dump=lambda: {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_123",
                "name": "search_rows",
                "arguments": (
                    '{"query":"visiting hour","dataset_name":"Hospital Policy"}'
                ),
                "status": "completed",
            },
        )

        message = from_openai_output_item(response_item)

        self.assertEqual(message.call_id, "call_123")
        self.assertEqual(message.name, "search_rows")
        self.assertEqual(
            message.arguments_parsed,
            {"query": "visiting hour", "dataset_name": "Hospital Policy"},
        )


class ProviderHistoryTests(unittest.TestCase):
    """Tests for provider-side canonical history management."""

    def test_provider_history_stores_canonical_messages(self) -> None:
        """Provider history should keep canonical messages until serialization."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4o-mini",
            messages=generate_initial_messages("system prompt", "user prompt"),
        )
        provider.add_tool_output("call_123", "Found one row")

        history = provider.history

        self.assertEqual(len(history), 3)
        self.assertIsInstance(history[0], SystemMessage)
        self.assertEqual(
            provider.serialized_history("openai")[2],
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Found one row",
            },
        )


if __name__ == "__main__":
    unittest.main()
