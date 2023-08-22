model="bigcode/starcoder"

cache_dir="/workspace/asr/WizardLM/WizardCoder"

import transformers

model = transformers.AutoModelForCausalLM.from_pretrained(
    model,
    cache_dir=cache_dir,
    use_auth_token=True
)

print(model)

print(sum(p.numel() for p in model.parameters()))
