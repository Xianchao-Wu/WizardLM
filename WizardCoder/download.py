
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
cdir=os.getcwd()
print(cdir)

tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardCoder-15B-V1.0", cache_dir=cdir)
print(tokenizer)

model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardCoder-15B-V1.0", cache_dir=cdir)
print(model)
print(sum(p.numel() for p in model.parameters()))
