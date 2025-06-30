import streamlit as st
from chatbot import ChatBot_RAG, ChromaDB


class ChatBotPage:
    def __init__(_self):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        system_message = "Bạn là một trợ lý ảo và câu trả lời của bạn được dịch từ thông tin được cung cấp bằng tiếng Anh sang ngôn ngữ của câu hỏi. Những câu trả lời của bạn chỉ trả lời ý chính, ngắn gọn với thông tin đúng trọng tâm nhất. Nếu không thể lấy được câu trả lời trực tiếp từ thông tin được cung cấp, trả lời 'Tôi không có đủ thông tin để trả lời câu hỏi này' theo ngôn ngữ của câu hỏi"
        _self.system_messages = {"role": "user", "content": system_message}

    @st.cache_resource(ttl=6000, max_entries=1, show_spinner="Initializing ChatBot...")
    def init_model(_self):
        return ChatBot_RAG("local_doc")

    @st.cache_resource(
        ttl=6000, max_entries=1, show_spinner="Initializing local database..."
    )
    def init_database(_self):
        return ChromaDB("local_doc")

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

        with st.form("chat_message", border=False):
            st.text_input(
                label="Trò chuyện với trợ lý ảo",
                placeholder="...",
                key="message",
                label_visibility="collapsed",
            )
            messages = {"role": "user", "content": st.session_state.message}
            chat_send = st.form_submit_button("Gửi")

        if chat_send and st.session_state.message != "":
            st.session_state.chat_messages.append(_self.system_messages)
            st.session_state.chat_messages.append(messages)
            chatbot.set_messages(st.session_state.chat_messages)
            st.session_state.chat_messages.append(
                {"role": "model", "content": chatbot.get_response()}
            )

        for message in st.session_state.chat_messages:
            if message != _self.system_messages:
                with st.container(border=True):
                    st.write(
                        ("Bạn: " if message["role"] == "user" else "")
                        + "\n"
                        + message["content"]
                    )


ChatBotPage().load_chatbot()
