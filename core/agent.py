from .vectorstore import VectorStore
from prompts.templates import QA_PROMPT
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from .parser import parse_epub
from .chunker import chunk_text
from .embeddings import embed_chunks

load_dotenv()

logger = logging.getLogger(__name__)

UNAVAILABLE_MESSAGE = "The answer service is unavailable right now. Please try again later."


class Agent:

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def retrieve(self, question, k=10):
        results = self.vectorstore.search(question, k=k)

        chunks = results["documents"]
        metadata = results["metadatas"]

        return  {

        "documents": results["documents"],

        "metadatas": results["metadatas"]

    }
    

    def generate_answer(self, question, chunks,metadata):
        
        context_parts = []

        for i in range(len(chunks)):
            context_parts.append(
                f"[Chunk {metadata[i]['chunk_num']}]: {chunks[i]}"
            )

        context = "\n\n".join(context_parts)
        prompt = QA_PROMPT.format(
            context=context,
            question=question

        )

        try:
            response = self.client.responses.create(
                model="gpt-4.1-mini",
                input=prompt

            )
        except OpenAIError:
            # Quota, outage, network or key problems: show a friendly message, log the details to the console.
            logger.exception("OpenAI request failed")
            return UNAVAILABLE_MESSAGE

        return response.output[0].content[0].text

    def run(self, question):
        results = self.retrieve(question,k=15)

        chunks = results["documents"]

        metadata = results["metadatas"]

        return self.generate_answer(question, chunks, metadata)

# -----------------------
# TEST PIPELINE
# -----------------------

# if __name__ == "__main__":

#     # 1. LOAD BOOK
#     text = parse_epub("sample_books/pg3296-images-3.epub")
#     print("Parsed characters:", len(text))

#     # 2. CHUNK
#     chunks = chunk_text(text)
#     print("Chunks:", len(chunks))

#     # 3. BUILD VECTOR DB (THIS IS THE FIX)
#     vs = VectorStore()
#     vs.build_index(chunks)

#     print("Done indexing book!")

#     # 4. TEST AGENT
#     agent = Agent(vs)

#     question = "What is this book about?"
#     print("\nQuestion:", question)

#     answer = agent.run(question)

#     print("\nAnswer:\n", answer)