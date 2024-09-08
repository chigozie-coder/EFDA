import torch
from torch.nn import functional as F
from transformers import GPT2Tokenizer
from model import StandardAutoregressiveModel, generate_square_subsequent_mask
from data_preparation import encode_text, decode_text

def load_model(model_path, num_tokens, d_model, nhead, num_decoder_layers, dim_feedforward):
    model = StandardAutoregressiveModel(num_tokens, d_model, nhead, num_decoder_layers, dim_feedforward)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.95):
    model.eval()
    encoded_prompt = encode_text(tokenizer, prompt)[0]
    generated = encoded_prompt.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            input_ids = torch.tensor(generated).unsqueeze(0)
            mask = generate_square_subsequent_mask(len(generated)).unsqueeze(0)
            
            output = model(input_ids, mask)
            next_token_logits = output[0, -1, :] / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Apply top-p (nucleus) filtering
            cumulative_probs = torch.cumsum(F.softmax(top_k_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(0, top_k_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
    
    return decode_text(tokenizer, generated)

def interactive_generation():
    # Model hyperparameters (should match the trained model)
    num_tokens = 50257  # GPT-2 tokenizer vocabulary size
    d_model = 512
    nhead = 8
    num_decoder_layers = 6
    dim_feedforward = 2048
    
    # Load the trained model
    model_path = "model.pt"
    model = load_model(model_path, num_tokens, d_model, nhead, num_decoder_layers, dim_feedforward)
    
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    print("Welcome to the text generation interface!")
    print("Enter your prompts below. Type 'quit' to exit.")
    
    while True:
        prompt = input("\nEnter your prompt: ")
        if prompt.lower() == 'quit':
            break
        
        max_length = int(input("Enter max generation length: "))
        temperature = float(input("Enter temperature (0.1-2.0, higher for more randomness): "))
        
        generated_text = generate_text(model, tokenizer, prompt, max_length, temperature)
        print("\nGenerated text:")
        print(generated_text)

if __name__ == "__main__":
    interactive_generation()