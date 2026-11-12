"""LLM-as-judge prompts and calls (OpenAI Responses API with structured output)."""

import json

FAITHFULNESS_PROMPT = """You are checking whether an AI assistant's answer is supported by the book passages it was given.

<question>
{question}
</question>

<passages>
{context}
</passages>

<answer>
{answer}
</answer>

1. List every distinct factual claim the answer makes about the book: about people, events, descriptions, quotations or numbers. Write each claim so it stands on its own, replacing pronouns with the person or thing they refer to (use the question to work out who "she", "he" or "it" is).
2. Check every claim, including claims made alongside "I don't know" or "the context doesn't say". An answer that declines to answer the question can still describe, quote or characterise people and events, and those statements are claims to check.
3. Leave out only: the statement that the passages lack some information, chunk citations such as "(Chunk 12)", and formatting.
4. A claim is supported only if the passages state it, or it follows directly from them, about the same person or thing. A description the passages give of one person, attributed in the answer to someone else, is not supported. Judge only against the passages, not your own knowledge of the book.
5. Return an empty list only if the answer makes no factual claims at all."""

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["claim", "reasoning", "supported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

ANSWER_PROMPT = """You are grading an AI reading assistant's answer to a question about a book.

<question>{question}</question>
<reference_answer>{reference}</reference_answer>
<assistant_answer>{answer}</assistant_answer>

Grade two things.

relevance (1-5): does the answer address the question that was asked? Judge focus, not accuracy.
5 = directly and fully addresses the question
4 = addresses it, with some padding or tangents
3 = addresses only part of the question
2 = mostly off-topic or evasive
1 = no real attempt (for example it says it doesn't know) or entirely off-topic

correctness: compare the assistant's answer with the reference answer, which was written from the book.
"correct" = has the reference's key facts and nothing that contradicts it
"partial" = some key facts right, others missing or wrong
"incorrect" = key facts missing or contradicted, or the assistant declined to answer
Extra details that are not in the reference are fine unless they contradict it."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "relevance": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "correctness": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
    },
    "required": ["reasoning", "relevance", "correctness"],
    "additionalProperties": False,
}

ABSTAIN_PROMPT = """The question below cannot be answered from the book: the book does not contain this information.

<question>{question}</question>
<why_unanswerable>{reference}</why_unanswerable>
<assistant_answer>{answer}</assistant_answer>

Did the assistant appropriately say it doesn't know?
abstained = true if the answer says, in substance, that the book or the provided context doesn't contain the requested information. It may still mention related things the book does say.
abstained = false if it supplies the requested information (a name, date, cause, event or opinion), even when hedged or attributed to general knowledge."""

ABSTAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "abstained": {"type": "boolean"},
    },
    "required": ["reasoning", "abstained"],
    "additionalProperties": False,
}


class Judge:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def _ask(self, prompt, name, schema):
        kwargs = {}
        # Reasoning models (o-series, gpt-5) reject a temperature setting.
        if not self.model.startswith(("o", "gpt-5")):
            kwargs["temperature"] = 0
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
            **kwargs,
        )
        return json.loads(response.output_text)

    def faithfulness(self, question, context, answer):
        prompt = FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer)
        return self._ask(prompt, "faithfulness", FAITHFULNESS_SCHEMA)["claims"]

    def answer_quality(self, question, reference, answer):
        prompt = ANSWER_PROMPT.format(question=question, reference=reference, answer=answer)
        return self._ask(prompt, "answer_quality", ANSWER_SCHEMA)

    def abstention(self, question, reference, answer):
        prompt = ABSTAIN_PROMPT.format(question=question, reference=reference, answer=answer)
        return self._ask(prompt, "abstention", ABSTAIN_SCHEMA)
