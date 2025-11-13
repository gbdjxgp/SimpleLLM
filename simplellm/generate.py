import torch
import time

from .kvcache import KVCache

def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                    and torch.all(next_token == eos_token_id)):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)

def generate_text_basic_stream_with_kvcache(model, token_ids, max_new_tokens, eos_token_id=None, context_size=None):
    model.eval()

    with torch.no_grad():
        cache = KVCache(n_layers=model.cfg["n_layers"])
        model.reset_kv_cache()

        # Prime the cache with the initial context
        logits = model(token_ids, cache=cache)

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)

            # Feed only the new token to the model; cache handles history
            logits = model(next_token, cache=cache)



def infer(model, tokenizer, input_ids, max_new_tokens = 500):
    start = time.time()
    gen_tokens = 0
    tokens = []
    response = ""
    for token in generate_text_basic_stream_with_kvcache(
            model=model,
            token_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).tolist()
        tokens.append(token_id)
        chunk = tokenizer.decode(token_id)
        print(
            chunk,
            end="",
            flush=True
        )
        gen_tokens += 1
        response += chunk
    elapsed = time.time() - start
    # response = tokenizer.decode(tokens)
    # TODO decode from id list.
    return response, gen_tokens, elapsed
