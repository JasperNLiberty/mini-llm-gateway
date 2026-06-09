from mlx_lm import load, generate

model, tokenizer = load('Qwen/Qwen2.5-7B')

# Single prompt
print("Single prompt:")
response = generate(model, tokenizer, prompt="count to 5")
print(response)

# Multiple prompts
print("\nMultiple prompts:")
try:
    responses = generate(model, tokenizer, prompt=["count to 5", "say hello"])
    print(responses)
except Exception as e:
    print(f"Error: {e}")