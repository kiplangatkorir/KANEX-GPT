import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from model import KANEX  

# Load GPT-2 configuration and set 'output_hidden_states' to True
config = GPT2Config.from_pretrained('gpt2', output_hidden_states=True)

# Load the pre-trained GPT-2 Small model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Initialize KANEX model with d_model=768 to match GPT-2's embedding size
kanex_model = KANEX(vocab_size=len(tokenizer), d_model=768, nhead=8, num_layers=6)
kanex_model.load_pretrained(gpt2_model)
kanex_model = kanex_model.to('cpu')  


# Function to generate text
def generate_text(prompt, model, tokenizer, max_length=50, temperature=1.0):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to('cpu')

    for _ in range(max_length):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)

# Generate text
prompt = "The history of AI began in"
generated_text = generate_text(prompt, kanex_model, tokenizer)
print(generated_text)
