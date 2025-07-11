import os
import requests
import pypdf
import docx
from io import BytesIO
import magic

from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_ollama.embeddings import OllamaEmbeddings
import chromadb

from hashlib import sha256

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LL_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 512
PDF_PATH = "documents/"  # Change to PDF you want to use
OVERLAP_SIZE = 32

PROMPT = PromptTemplate.from_template(
    """
{system_message}

Lịch sử nói chuyện:
{chat_history}

Ngữ cảnh:
{context}

Câu hỏi người dùng:
{question}

Câu trả lời:
"""
)


class GraphState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    context_docs: List[Document]
    response: str


class LangChainChromaDB:
    def __init__(self, collection_name: str = "local_doc"):
        # Tạo embedding và thêm vào ChromaDB
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_HOST,
            model=EMBEDDING_MODEL,
        )

        chroma_client = chromadb.PersistentClient("chroma")
        self.doc_collection = chroma_client.get_or_create_collection(collection_name)

        self.vector_store = Chroma(
            persist_directory="chroma",
            collection_name=collection_name,
            embedding_function=embeddings,
        )

        self.retriever = self.vector_store.as_retriever()

    def get_retriever(self):
        return self.retriever

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
            self.vector_store.add_documents(
                documents=chunk, ids=sha256(chunk.encode("utf-8")).hexdigest()
            )
            for chunk in chunks.copy()
        ]
        print("Ready!")

    def delete_all_data(self):
        self.vector_store.delete()

    def get_chunks(self, query: str, k=3):
        chunks = self.vector_store.similarity_search(query=query, k=k)
        return chunks


class ChatBot_RAG:
    def __init__(self, collection_name: str = "local_doc", system_message: str = ""):
        # Pull the LL model and the embedding model.
        response_llm = requests.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"model": LL_MODEL},
        )
        if response_llm.ok:
            print("Model is being pulled or is ready.")
        else:
            print("Error pulling model:", response_llm.text)

        response_embed = requests.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"model": EMBEDDING_MODEL},
        )
        if response_embed.ok:
            print("Model is being pulled or is ready.")
        else:
            print("Error pulling model:", response_embed.text)

        self.system_message = system_message

        # Khởi tạo client cho các model
        graph_builder = StateGraph(GraphState)
        self.llm = ChatOpenAI(
            model=LL_MODEL,
            temperature=0.2,
            max_completion_tokens=400,
            timeout=None,
            max_retries=2,
            base_url=f"{OLLAMA_HOST}/v1/",
            api_key="ollama",
            reasoning_effort="low",
            top_p=0.9,
        )
        graph_builder.add_node("retrieve", self.retrieve_node)
        graph_builder.add_node("generate", self.generate_node)
        graph_builder.add_node("update_memory", self.update_memory_node)

        graph_builder.set_entry_point("retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", "update_memory")
        graph_builder.add_edge("update_memory", END)
        self.graph = graph_builder.compile()

        self.llm_chain = PROMPT | self.llm

        # Xử lý pdf
        self.response = ""
        self.retriever = LangChainChromaDB(collection_name).get_retriever()

        self.state: GraphState = {
            "input": "",
            "chat_history": [],
            "context_docs": [],
            "response": "",
        }

    def format_chat_history(self, history: List[BaseMessage]) -> str:
        return "\n".join(
            f"{'User' if m.type == 'human' else 'AI'}: {m.content}" for m in history
        )

    def retrieve_node(self, state: GraphState) -> GraphState:
        docs = self.retriever.invoke(state["input"])
        return {**state, "context_docs": docs}

    def generate_node(self, state: GraphState) -> GraphState:
        if not state["context_docs"]:
            return {
                **state,
                "response": "Tôi không có đủ thông tin để trả lời câu hỏi này.",
            }

        context = "\n\n".join(doc.page_content for doc in state["context_docs"])
        chat_str = self.format_chat_history(state["chat_history"])

        response = self.llm_chain.invoke(
            {
                "system_message": self.system_message,
                "question": state["input"],
                "context": context,
                "chat_history": chat_str,
            }
        )

        return {**state, "response": response.text()}

    def update_memory_node(self, state: GraphState) -> GraphState:
        updated = state["chat_history"] + [
            HumanMessage(content=state["input"]),
            AIMessage(content=state["response"]),
        ]
        return {**state, "chat_history": updated}

    def get_chat_history(self) -> List:
        return self.state["chat_history"]

    def generate_answer(self, message: str):
        self.state["input"] = message
        self.state["context_docs"] = []
        self.state = self.graph.invoke(self.state)

    def set_messages(self, message: str):
        self.generate_answer(message)

    def get_response(self):
        return self.state["response"]

    def get_current_state(self):
        return self.state

    def set_state(self, key: str, value):
        if key in ["input", "response", "context", "chat_history"]:
            self.state[key] = value
        else:
            print("Key not found!")


if __name__ == "__main__":
    active = True
    messages = []
    # ChromaDB(document_path=PDF_PATH, update_data=True)

    system_message = "Bạn là một trợ lý ảo và câu trả lời của bạn được dịch từ thông tin được cung cấp bằng tiếng Anh sang ngôn ngữ của câu hỏi. Nếu không thể lấy được câu trả lời trực tiếp từ thông tin được cung cấp, trả lời 'Tôi không có đủ thông tin để trả lời câu hỏi này' theo ngôn ngữ của câu hỏi"
    chatbot = ChatBot_RAG(system_message=system_message)
    while active:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("\nChat History:")
            print(chatbot.get_chat_history())
            break
        chatbot.set_messages(user_input)
        print(f"AI: {chatbot.get_response()}\n")
    exit(0)
