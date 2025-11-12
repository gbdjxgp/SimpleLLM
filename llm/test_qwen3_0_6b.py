import torch

from llm.loader import init_model, load_model
from llm.generate import infer

USE_REASONING_MODEL = True
USE_INSTRUCT_MODEL = False

def test_1(
        prompt = "Give me a short introduction to large language models.",
        max_tokens = 1024
    ):


    CHOOSE_MODEL = "0.6B"
    model, QWEN3_CONFIG, device = init_model(CHOOSE_MODEL)
    model, tokenizer = load_model(model, device, QWEN3_CONFIG, USE_REASONING_MODEL, USE_INSTRUCT_MODEL, CHOOSE_MODEL)

    input_token_ids = tokenizer.encode(prompt)
    text = tokenizer.decode(input_token_ids)

    print(f"\nprompt: \"{prompt}\"\n")
    print(f"\ntext: \"{text}\"\n")

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

