import torch

from simplellm.loader import Qwen3ModelLoader
from simplellm.generate import infer
from simplellm.tokenizer import Qwen3Tokenizer

def test_1(
        prompt = "Give me a short introduction to large language models.",
        max_tokens = 1024
    ):
    torch.manual_seed(123)
    model_path = rf"C:\Users\13766\.cache\modelscope\hub\models\Qwen\Qwen3-0___6B"
    qwen3_loader = Qwen3ModelLoader(model_path=model_path)
    device = qwen3_loader.device
    model = qwen3_loader.model
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=model_path+"/tokenizer.json",
        repo_id=f"Qwen/Qwen3-0.6B-Base",
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True
    )
    input_token_ids = tokenizer.encode(prompt)
    text = tokenizer.decode(input_token_ids)

    print(f"prompt: \"{prompt}\"")
    print(f"text: \"{text}\"")

    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)
    response, gen_tokens, elapsed_time = infer(model, tokenizer, input_token_ids_tensor, max_new_tokens=max_tokens)

    print(f"\n\nResponse: {response}")

    throughput = gen_tokens / elapsed_time
    print(f"\ngen tokens: {gen_tokens}, \n"
          f"elapsed: {elapsed_time:.3f} seconds, \n"
          f"throughput: {throughput:.3f} tokens/sec")


if __name__ == "__main__":
    test_1()
    # test_1(prompt="介绍中国历史")
    # test_1(prompt="什么是魑魅魍魉？")

