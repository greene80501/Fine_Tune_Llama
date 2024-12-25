from transformers import AutoTokenizer

# Replace "openlm-research/open_llama_3b" with the correct model ID
tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")

# Save the downloaded files to your local directory
tokenizer.save_pretrained("D:/Lamma-3B_fine_tuned/models--openlm-research--open_llama_3b")
print("Tokenizer files downloaded and saved!")
