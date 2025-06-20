from dotenv import load_dotenv
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
import torch

load_dotenv()


# login(token=os.getenv("HF_READ_API_KEY"))


class ChatBot:
    __END_TURN__ = "<end_of_turn>\n"
    __START_TURN_MODEL__ = "<start_of_turn>model\n"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            device_map="auto",
            torch_dtype="auto",
        )
        self.response = ""

    def convert_message(self, role, message):
        return f"<start_of_turn>{role}\n" + message + self.__END_TURN__

    def get_full_prompt(self, messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return prompt

    def set_messages(self, messages: list[dict]):
        prompt = self.get_full_prompt(messages)
        self.create_response(prompt)

    def create_response(self, prompt):
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(
            self.model.device
        )
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=320)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        result = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")
        self.response = result.replace("\n<end_of_turn>", "")

    def get_response(self) -> str | None:
        return self.response


if __name__ == "__main__":
    active = True
    messages = []
    chatbot = ChatBot()
    system_message = "Chỉ được sử dụng tiếng Việt"
    while active:
        user_input = input()
        messages.append({"role": "user", "content": system_message})
        messages.append({"role": "user", "content": user_input})
        chatbot.set_messages(messages)
        print(chatbot.get_response())
    exit(0)
