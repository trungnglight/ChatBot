import streamlit as st
from chatbot import ChatBot_RAG, LangChainChromaDB


class ChatBotPage:
    def __init__(_self):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        _self.system_message = "Bạn là một trợ lý ảo và câu trả lời của bạn được dịch từ thông tin được cung cấp bằng tiếng Anh sang ngôn ngữ của câu hỏi. Nếu không thể lấy được câu trả lời trực tiếp từ thông tin được cung cấp, trả lời 'Tôi không có đủ thông tin để trả lời câu hỏi này' theo ngôn ngữ của câu hỏi"

    @st.cache_resource(ttl=6000, max_entries=1, show_spinner="Initializing ChatBot...")
    def init_model(_self):
        return ChatBot_RAG("local_doc", _self.system_message)

    @st.cache_resource(
        ttl=6000, max_entries=1, show_spinner="Initializing local database..."
    )
    def init_database(_self):
        return LangChainChromaDB("local_doc")

    def load_chatbot(_self):

        st.header("Trò chuyện")

        chatbot = _self.init_model()

        st.sidebar.header("Thêm dữ liệu")

        with st.sidebar.form("add_documents", border=False):
            uploaded_files = st.file_uploader(
                label="Thêm dữ liệu",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed",
            )
            file_send = st.form_submit_button("Tải lên")
        database = _self.init_database()
        with st.spinner(text="Adding data...!", show_time=True):
            if file_send and uploaded_files is not []:
                for item in uploaded_files:
                    database.add_data(item)

        with st.spinner(text="Removing all uploaded data..."):
            if st.sidebar.button("Remove all data"):
                database.delete_all_data()

        with st.form("chat_message", border=False):
            st.text_input(
                label="Trò chuyện với trợ lý ảo",
                placeholder="...",
                key="message",
                label_visibility="collapsed",
            )
            chat_send = st.form_submit_button("Gửi")

        if chat_send and st.session_state.message != "":
            chatbot.set_messages(st.session_state.message)

        st.write(
            "\n\n".join(
                f"{'User' if m.type == 'human' else 'AI'}: {m.content}"
                for m in chatbot.get_chat_history()
            )
        )


ChatBotPage().load_chatbot()
