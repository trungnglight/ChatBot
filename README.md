# ChatBot
Website trò chuyện cùng trợ lý ảo

# Cài đặt
Docker repository: [ChatBot](https://hub.docker.com/r/trungnglight/chatbot-app "Docker")

Cài đặt Python 3.11 và Ollama

Tải xuống:
```
git clone https://github.com/trungnglight/ChatBot.git
```

Cài đặt các thư viện cần thiết:
```
pip install -r requirements.txt
```

Cài đặt model để chạy ChatBot:
```
ollama serve
```

Khởi chạy máy chủ:
```
streamlit run app.py
```