import streamlit as st
from dotenv import load_dotenv
from chatbot import ChatBot

load_dotenv()


class ChatBotPage:
    def __init__(_self):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        system_message = "Chỉ được sử dụng tiếng Việt và trả lời với ít nhất 10 từ"
        _self.system_messages = {"role": "user", "content": system_message}

    @st.cache_resource(ttl=6000, max_entries=1, show_spinner="Initializing ChatBot...")
    def load_model(_self):
        return ChatBot()

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
        if send:
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
