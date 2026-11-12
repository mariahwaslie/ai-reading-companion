import unittest
from unittest.mock import MagicMock, patch

import httpx
import openai

from core import agent as agent_module


def out_of_credits_error():
    """The error the OpenAI client raises when the account has no credits left (HTTP 429)."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return openai.RateLimitError("You have no credits remaining.", response=httpx.Response(429, request=request),
                                 body=None)


class AnswerServiceErrorTests(unittest.TestCase):
    def test_failing_openai_call_returns_friendly_message(self):
        store = MagicMock()
        store.search.return_value = {"documents": ["Some chunk text."], "metadatas": [{"chunk_num": 0}], "ids": ["0"]}

        with patch.object(agent_module, "OpenAI") as fake_openai:
            fake_openai.return_value.responses.create.side_effect = out_of_credits_error()
            agent = agent_module.Agent(store)
            with self.assertLogs("core.agent", level="ERROR") as logs:
                answer = agent.run("What happens in chapter one?")

        self.assertEqual(answer, "The answer service is unavailable right now. Please try again later.")
        fake_openai.return_value.responses.create.assert_called_once()
        self.assertIn("OpenAI request failed", logs.output[0])
        self.assertIn("You have no credits remaining", logs.output[0])  # the real error reaches the log


if __name__ == "__main__":
    unittest.main()
