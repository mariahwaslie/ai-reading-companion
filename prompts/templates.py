QA_PROMPT = """
You are an AI reading assistant.

Use ONLY the provided context to answer the question.

Each context chunk includes a chunk ID like:
[Chunk 12], [Chunk 3], etc.

When you answer:
- You MUST cite chunk IDs in parentheses
- Example: (Chunk 12)
- If multiple chunks are used, include all relevant ones

Context:
{context}

Question:
{question}

Answer format:
Answer
[Citations]
"""

SUMMARY_PROMPT = """
Summarize the following passage from "{book_title}" in 3-5 sentences.
Focus on key events, characters, and themes.

PASSAGE:
{context}

SUMMARY:
"""

CHARACTER_PROMPT = """
Based on the passages below, describe the character "{character}" in:
- Personality traits (2-3)
- Relationships
- Notable actions
- Character growth (if any)

PASSAGES:
{context}
"""