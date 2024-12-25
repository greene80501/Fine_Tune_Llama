from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load the dataset
dataset = load_dataset("json", data_files="dataset.jsonl")  # Ensure "dataset.jsonl" exists and is in the correct format.

# Step 2: Split into training and validation sets
dataset = dataset["train"].train_test_split(test_size=0.2)

# Step 3: Load the LLaMA tokenizer
tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")  # Corrected path

# Step 4: Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  # Use EOS as the pad token if none exists

# Step 5: Define the preprocessing function
def preprocess_function(examples):
    inputs = examples["input"]
    outputs = examples["output"]

    # Tokenize inputs and outputs
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(outputs, truncation=True, padding="max_length", max_length=512)

    # Align input IDs and labels
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Step 6: Apply the tokenization function to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

print("Tokenization completed!")

# Step 7: Load the LLaMA model
model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b")  # Corrected path

# Resize embeddings if pad_token was added
model.resize_token_embeddings(len(tokenizer))

# Step 8: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",             # Where to save the model
    eval_strategy="epoch",        # Evaluate after every epoch
    per_device_train_batch_size=2,      # Adjust based on your GPU memory
    per_device_eval_batch_size=2,       # Match training batch size
    num_train_epochs=3,                 # Set number of epochs
    save_steps=500,                     # Save model every 500 steps
    save_total_limit=2,                 # Limit the number of checkpoints
    logging_dir="./logs",               # Directory for logs
    logging_steps=100,                  # Log every 100 steps
    learning_rate=2e-5,                 # Learning rate
    fp16=True,                          # Use mixed precision if available
)

# Step 9: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Step 10: Fine-tune the model
trainer.train()
print("Fine-tuning completed!")

# Step 11: Save the fine-tuned model
trainer.save_model("./fine_tuned_model")
print("Model saved!")
