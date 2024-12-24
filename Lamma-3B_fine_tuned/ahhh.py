from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "openlm-research/open_llama_3b", cache_dir="d:/Lamma-3B_fine_tuned"
)
model = AutoModelForCausalLM.from_pretrained(
    "openlm-research/open_llama_3b", cache_dir="d:/Lamma-3B_fine_tuned"
)
