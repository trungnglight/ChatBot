import streamlit as st
from chatbot import ChatBot_RAG

PDF_PATH = "engineering-software-products-global.pdf"  # Change to PDF you want to use


class ChatBotPage:
    def __init__(_self):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        system_message = "Bạn là một trợ lý ảo và câu trả lời bằng tiếng Việt của bạn được dịch từ thông tin được cung cấp bằng tiếng Anh. Nếu không thể lấy được câu trả lời trực tiếp từ thông tin được cung cấp, trả lời bằng: 'Tôi không có đủ thông tin để trả lời câu hỏi này'"
        _self.system_messages = {"role": "user", "content": system_message}

    @st.cache_resource(ttl=6000, max_entries=1, show_spinner="Initializing ChatBot...")
    def load_model(_self):
        return ChatBot_RAG(PDF_PATH)

    def load(_self):
        st.header("# Trò chuyện")
        chatbot = _self.load_model()

        with st.form("chat_message"):
            st.text_input(
                label="Trò chuyện với trợ lý ảo",
                placeholder="...",
                key="message",
            )
            messages = {"role": "user", "content": st.session_state.message}
            send = st.form_submit_button("Gửi")
        if send and st.session_state.message != "":
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
                        ("Bạn: " if message["role"] == "user" else "Trợ lý ảo: ")
                        + "\n"
                        + message["content"]
                    )


ChatBotPage().load()
