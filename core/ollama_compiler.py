from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class NanbeigeCompiler:
    def __init__(self, model_id="Nanbeige/Nanbeige-Plus-Chat-v2-8B"):
        print(f"Model yükleniyor: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate_ir(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response