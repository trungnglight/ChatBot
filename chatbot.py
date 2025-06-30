from openai import OpenAI
import os
import requests
import pypdf
import docx
from io import BytesIO
import magic
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from hashlib import sha256
from copy import deepcopy

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LL_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 512
PDF_PATH = "documents/"  # Change to PDF you want to use
OVERLAP_SIZE = 32


class ChromaDB:
    def __init__(self, database_name: str = "local_doc"):
        chroma_client = chromadb.PersistentClient(path="chroma")

        # Tạo embedding và thêm vào ChromaDB
        embedding_function = OpenAIEmbeddingFunction(
            api_key="ollama",
            api_base=f"{OLLAMA_HOST}/v1/",
            model_name=EMBEDDING_MODEL,
        )

        self.documents = chroma_client.get_or_create_collection(
            name=database_name, embedding_function=embedding_function
        )

    def get_documents(self):
        return self.documents

    def read_pdf(self, file_path) -> str:
        pdf = pypdf.PdfReader(file_path)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text

    def chunk_text(
        self, text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SIZE
    ) -> list[str]:
        chunks = []
        # Simple character-based chunking
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if chunk:  # Ensure we don't add empty chunks
                chunks.append(chunk)

        return chunks

    def detect_file_type(_self, file: BytesIO):
        # Reset stream to start
        file.seek(0)

        # Get MIME type from file header
        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(file.read(64))
        file.seek(0)

        # Map MIME type to friendly label
        mime_map = {
            "application/pdf": "pdf",
            "text/plain": "txt",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        }

        return mime_map.get(mime_type, "unknown")

    def add_data(self, document: BytesIO):
        data = ""
        file_type = self.detect_file_type(document)
        if file_type == "pdf":
            print("Pdf!")
            data = self.read_pdf(document)
        elif file_type == "txt":
            print("Txt!")
            data = document.read().decode(errors="ignore")
        elif file_type == "docx":
            print("Docx!")
            doc = docx.Document(document)
            data = "\n".join([p.text for p in doc.paragraphs])
        else:
            print("Unknown type")
            exit()
        chunks = self.chunk_text(data)
        [
            self.documents.upsert(
                documents=chunk, ids=sha256(chunk.encode("utf-8")).hexdigest()
            )
            for chunk in chunks.copy()
        ]
        print("Ready!")

    def get_chunks(self, query_text: str, top_k=3):
        chunks = self.documents.query(
            query_texts=query_text, n_results=top_k, include=["documents"]
        )
        return chunks


class ChatBot_RAG:
    def __init__(self, collection_name: str = "rag_documents"):
        # Pull the LL model and the embedding model.
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

        # Khởi tạo client cho các model
        self.model_client = OpenAI(
            base_url=f"{OLLAMA_HOST}/v1/",
            api_key="ollama",
        )

        # Xử lý pdf
        self.response = ""
        self.documents = ChromaDB(database_name=collection_name).get_documents()

    def generate_answer(self, message: list[dict], context_chunks: list[str]):
        context = "\n\n".join(context_chunks)
        prompt = deepcopy(message)
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

        response = self.model_client.chat.completions.create(
            model=LL_MODEL,
            messages=prompt,
            max_tokens=400,
            reasoning_effort="low",
            temperature=0.2,
            top_p=0.9,
        )
        return response.choices[0].message.content

    def get_chunks(self, query_text: str, top_k=3) -> list[list]:
        chunks = self.documents.query(
            query_texts=query_text, n_results=top_k, include=["documents"]
        )
        return chunks.get("documents")[0]

    def set_messages(self, messages: list[dict]):
        self.create_response(messages)

    def create_response(self, messages: list[dict[str, str]]):
        relevant_chunks = self.get_chunks(query_text=messages[-1]["content"], top_k=3)
        print(relevant_chunks)
        self.response = self.generate_answer(messages, relevant_chunks)

    def get_response(self):
        return self.response


if __name__ == "__main__":
    active = True
    messages = []
    # ChromaDB(document_path=PDF_PATH, update_data=True)
    chatbot = ChatBot_RAG()
    system_message = "Bạn là một trợ lý ảo và câu trả lời của bạn được dịch từ thông tin được cung cấp bằng tiếng Anh sang ngôn ngữ của câu hỏi. Nếu không thể lấy được câu trả lời trực tiếp từ thông tin được cung cấp, trả lời 'Tôi không có đủ thông tin để trả lời câu hỏi này' theo ngôn ngữ của câu hỏi"
    while active:
        user_input = input("Viết câu hỏi\n")
        messages.append({"role": "user", "content": system_message})
        messages.append({"role": "user", "content": user_input})
        chatbot.set_messages(messages)
        print(chatbot.get_response())
    exit(0)
