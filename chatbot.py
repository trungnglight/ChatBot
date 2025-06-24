from openai import OpenAI
import os
import requests
import pypdf
import textwrap
import numpy as np
import faiss
import copy

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LL_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 512
PDF_PATH = "engineering-software-products-global.pdf"  # Change to PDF you want to use


class ChatBot_RAG:
    def __init__(self, pdf_path):
        response_llm = requests.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"model": EMBEDDING_MODEL},
        )
        if response_llm.ok:
            print("Model is being pulled or is ready.")
        else:
            print("Error pulling model:", response_llm.text)
        response_embed = requests.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"model": LL_MODEL},
        )
        if response_embed.ok:
            print("Model is being pulled or is ready.")
        else:
            print("Error pulling model:", response_embed.text)
        self.client = OpenAI(
            base_url=f"{OLLAMA_HOST}/v1/",
            api_key="ollama",
        )
        self.response = ""
        pdf_text = self.read_pdf(pdf_path)
        self.chunks = self.chunk_text(pdf_text)

        # Bước 2: Tạo embedding và index
        embeddings = self.embed_texts(self.chunks)
        self.index = self.create_faiss_index(embeddings)

    def read_pdf(self, file_path: str):
        pdf = pypdf.PdfReader(pdf)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text, chunk_size=CHUNK_SIZE):
        return textwrap.wrap(text, width=chunk_size, break_long_words=False)

    def embed_texts(self, texts):
        embeddings = []
        for i in range(0, len(texts), 10):
            batch = texts[i : i + 10]
            response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            batch_embeddings = [
                np.array(d.embedding, dtype="float32") for d in response.data
            ]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def create_faiss_index(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve_chunks(self, query, index, chunks, k=3):
        query_embedding = (
            self.client.embeddings.create(model=EMBEDDING_MODEL, input=query)
            .data[0]
            .embedding
        )
        query_vec = np.array(query_embedding, dtype="float32").reshape(1, -1)
        _, indices = index.search(query_vec, k)
        return [chunks[i] for i in indices[0]]

    def generate_answer(self, message: list[dict], context_chunks):
        context = "\n\n".join(context_chunks)
        prompt = copy.deepcopy(message)
        prompt[-1][
            "content"
        ] = f"""
            Dựa trên thông tin sau, hãy trả lời câu hỏi.

            Thông tin:
            {context}

            Câu hỏi:
            {[prompt[-1]["content"]]}

            Trả lời:
            """

        response = self.client.chat.completions.create(
            model=LL_MODEL,
            messages=prompt,
            max_tokens=800,
            reasoning_effort="low",
            temperature=0.2,
            top_p=0.9,
        )
        return response.choices[0].message.content

    def set_messages(self, messages):
        self.create_response(messages)

    def create_response(self, messages):

        # Bước 3: Truy vấn
        relevant_chunks = self.retrieve_chunks(
            messages[-1].get("content"), self.index, self.chunks
        )
        self.response = self.generate_answer(messages, relevant_chunks)

    def get_response(self):
        return self.response


class ChatBot:
    def __init__(self):
        response = requests.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"model": LL_MODEL},
        )
        if response.ok:
            print("Model is being pulled or is ready.")
        else:
            print("Error pulling model:", response.text)
        self.client = OpenAI(
            base_url=f"{OLLAMA_HOST}/v1/",
            api_key="ollama",
        )
        self.response = ""

    def set_messages(self, messages: list[dict]):
        self.create_response(messages)

    def create_response(self, prompt: list[dict]):
        chat_completion = self.client.chat.completions.create(
            model=LL_MODEL,
            messages=prompt,
            max_tokens=800,
            reasoning_effort="low",
            temperature=0.2,
            top_p=0.9,
        )
        result = chat_completion.choices[0].message.content
        self.response = result

    def get_response(self) -> str | None:
        return self.response


if __name__ == "__main__":
    active = True
    messages = []
    chatbot = ChatBot_RAG(PDF_PATH)
    system_message = "Bạn là một trợ lý ảo và câu trả lời bằng tiếng Việt của bạn được dịch từ thông tin được cung cấp bằng tiếng Anh. Nếu không thể lấy được câu trả lời trực tiếp từ thông tin được cung cấp, trả lời bằng: 'Tôi không có đủ thông tin để trả lời câu hỏi này'"
    while active:
        user_input = input("Viết câu hỏi\n")
        messages.append({"role": "user", "content": system_message})
        messages.append({"role": "user", "content": user_input})
        chatbot.set_messages(messages)
        print(chatbot.get_response())
    exit(0)
