from openai import OpenAI


class ChatBot:
    __END_TURN__ = "<end_of_turn>\n"
    __START_TURN_MODEL__ = "<start_of_turn>model\n"

    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )
        self.response = ""

    def set_messages(self, messages: list[dict]):
        self.create_response(messages)

    def create_response(self, prompt: list[dict]):
        chat_completion = self.client.chat.completions.create(
            model="gemma3:4b",
            messages=prompt,
            max_tokens=1200,
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
    chatbot = ChatBot()
    system_message = "Chỉ được sử dụng tiếng Việt. Bạn là một trợ lý ảo nhanh và thẳng thắn. Tuyệt đối không giả lập suy nghĩ hay nhập liệu. Luôn luôn trả lời ngay lập tức và vào trọng tâm vấn đề."
    while active:
        user_input = input()
        messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_input})
        chatbot.set_messages(messages)
        print(chatbot.get_response())
    exit(0)
