from transformers import LlamaTokenizer
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("json", data_files="dataset.jsonl")

# Split into training and validation sets
dataset = dataset["train"].train_test_split(test_size=0.2)

# Load the LLaMA tokenizer
tokenizer = LlamaTokenizer.from_pretrained("models--openlm-research--open_llama_3b")

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Define the preprocessing function
def preprocess_function(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

    # Tokenize outputs as labels
    # Use as_target_tokenizer for backward compatibility with some versions of transformers
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Ensures padding is properly set
    labels = tokenizer(outputs, truncation=True, padding="max_length", max_length=512)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the tokenization function to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

print("Tokenization completed!")
