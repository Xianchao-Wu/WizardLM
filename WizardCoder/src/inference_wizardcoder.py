import sys
import os
import fire
import torch
import transformers
import json
import jsonlines

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps" # metal performance shaders (MPS) 苹果Apple公司的MPS作为pytorch的后端
except:
    pass

def evaluate(
        batch_data, # 'Write a Python code to count 1 to 10.'
        tokenizer, # GPT2TokenizerFast(name_or_path='/workspace/asr/WizardLM/WizardCoder/models--WizardLM--WizardCoder-15B-V1.0/snapshots/926ca1b215c4631bc5f8c3e47173381452c23e5c', vocab_size=49152, model_max_length=2048, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'pad_token': '[PAD]', 'additional_special_tokens': ['<|endoftext|>', '<fim_prefix>', '<fim_middle>', '<fim_suffix>', '<fim_pad>', '<filename>', '<gh_stars>', '<issue_start>', '<issue_comment>', '<issue_closed>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<empty_output>', '<commit_before>', '<commit_msg>', '<commit_after>', '<reponame>']}, clean_up_tokenization_spaces=True)
        model, # type(model)=<class 'transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM'>
        input=None,
        temperature=1,
        top_p=0.9,
        top_k=40,
        num_beams=1,
        max_new_tokens=2048,
        **kwargs, # {}
):
    prompts = generate_prompt(batch_data, input)
    inputs = tokenizer(prompts, return_tensors="pt", max_length=256, truncation=True, padding=True) # ipdb> p inputs: {'input_ids': tensor([[27400,   438,   600, 12404,   688, 18872,   312,  2899,    32,  5950,           312,  1789,   688, 36808, 30772,   322,  1326,    32,   203,   203,          1482, 21081,    44,   203,  2538,   312,  4865,  1340,   372,  2385,           225,    35,   372,   225,    35,    34,    32,   203,   203,  1482,          5170,    44]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])} ipdb> inputs['input_ids'].shape = torch.Size([1, 42]), ipdb> inputs['attention_mask'].shape = torch.Size([1, 42])
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature, # 1 
        top_p=top_p, # 0.9
        top_k=top_k, # 40
        num_beams=num_beams, # 1
        eos_token_id=tokenizer.eos_token_id, # 0
        pad_token_id=tokenizer.pad_token_id, # 49152
        **kwargs, # {}
    )
    import ipdb; ipdb.set_trace()
    with torch.no_grad():
        generation_output = model.generate( # NOTE, > /opt/conda/lib/python3.8/site-packages/transformers/generation/utils.py(1160)generate()
            input_ids=input_ids, # [1, 42], batch-size=1
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens, # 2048
        )
        s = generation_output.sequences # size=[1, 220], alike=tensor([[27400,   438,   600, 12404,   688, 18872,   312,  2899,    32,  5950, ...]], device='cuda:0')
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output # ["Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a Python code to count 1 to 10.\n\n### Response:Here's the Python code to count 1 to 10:\r\n\r\n```python\r\nfor i in range(1, 11):\r\n    print(i)\r\n```\r\n\r\nOutput:\r\n\r\n```\r\n1\r\n2\r\n3\r\n4\r\n5\r\n6\r\n7\r\n8\r\n9\r\n10\r\n```\r\n\r\nExplanation:\r\n\r\n- The `range()` function generates a sequence of numbers from the starting value (inclusive) to the ending value (exclusive).\r\n- In this case, we start with 1 and go up to 11 (exclusive) because we want to count 10.\r\n- The `for` loop iterates over each number in the sequence and assigns it to the variable `i`.\r\n- The `print()` function outputs the value of `i` on a new line."] NOTE 可以看到这里的输出output，也包括了最初的输入的prompt和instruction.


def generate_prompt(instruction, input=None): # NOTE input这个参数没有被使用到
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:""" # out = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a Python code to count 1 to 10.\n\n### Response:'


def main(
    load_8bit: bool = False, # False
    base_model: str = "Model_Path", # '/workspace/asr/WizardLM/WizardCoder/models--WizardLM--WizardCoder-15B-V1.0/snapshots/926ca1b215c4631bc5f8c3e47173381452c23e5c'
    input_data_path = "Input.jsonl", # './data/in.jsonl'
    output_data_path = "Output.jsonl", # './data/out.jsonl'
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'" # NOTE 这里支持具体的model name或者本地的model path
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model) # GPT2TokenizerFast(name_or_path='/workspace/asr/WizardLM/WizardCoder/models--WizardLM--WizardCoder-15B-V1.0/snapshots/926ca1b215c4631bc5f8c3e47173381452c23e5c', vocab_size=49152, model_max_length=2048, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'pad_token': '[PAD]', 'additional_special_tokens': ['<|endoftext|>', '<fim_prefix>', '<fim_middle>', '<fim_suffix>', '<fim_pad>', '<filename>', '<gh_stars>', '<issue_start>', '<issue_comment>', '<issue_closed>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<empty_output>', '<commit_before>', '<commit_msg>', '<commit_after>', '<reponame>']}, clean_up_tokenization_spaces=True)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit, # False
            torch_dtype=torch.float16, # TODO 应该通过命令行管理
            device_map="auto",
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )

    model.config.pad_token_id = tokenizer.pad_token_id # 49152

    if not load_8bit:
        model.half() # NOTE in, 本来就是float16导入的，这个half()没有导致什么变化...

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32": # '1.13.0+cu116', 'linux'; so not in
        model = torch.compile(model)

    input_data = jsonlines.open(input_data_path, mode='r')
    output_data = jsonlines.open(output_data_path, mode='w')

    for num, line in enumerate(input_data):
        one_data = line # {'idx': 11, 'Instruction': 'Write a Python code to count 1 to 10.'}
        id = one_data["idx"]
        instruction = one_data["Instruction"]
        print(instruction)
        _output = evaluate(instruction, tokenizer, model) # NOTE
        final_output = _output[0].split("### Response:")[1].strip() # 因为这里是逐行循环的，所以_output中只可能有一个元素! NOTE final_output="Here's the Python code to count 1 to 10:\r\n\r\n```python\r\nfor i in range(1, 11):\r\n    print(i)\r\n```\r\n\r\nOutput:\r\n\r\n```\r\n1\r\n2\r\n3\r\n4\r\n5\r\n6\r\n7\r\n8\r\n9\r\n10\r\n```\r\n\r\nExplanation:\r\n\r\n- The `range()` function generates a sequence of numbers from the starting value (inclusive) to the ending value (exclusive).\r\n- In this case, we start with 1 and go up to 11 (exclusive) because we want to count 10.\r\n- The `for` loop iterates over each number in the sequence and assigns it to the variable `i`.\r\n- The `print()` function outputs the value of `i` on a new line."
        new_data = {
            "id": id, # 11
            "instruction": instruction, # 'Write a Python code to count 1 to 10.'
            "wizardcoder": final_output # "Here's the Python code to count 1 to 10:\r\n\r\n```python\r\nfor i in range(1, 11):\r\n    print(i)\r\n```\r\n\r\nOutput:\r\n\r\n```\r\n1\r\n2\r\n3\r\n4\r\n5\r\n6\r\n7\r\n8\r\n9\r\n10\r\n```\r\n\r\nExplanation:\r\n\r\n- The `range()` function generates a sequence of numbers from the starting value (inclusive) to the ending value (exclusive).\r\n- In this case, we start with 1 and go up to 11 (exclusive) because we want to count 10.\r\n- The `for` loop iterates over each number in the sequence and assigns it to the variable `i`.\r\n- The `print()` function outputs the value of `i` on a new line."
        }
        output_data.write(new_data) # {'id': 11, 'instruction': 'Write a Python code to count 1 to 10.', 'wizardcoder': "Here's the Python code to count 1 to 10:\r\n\r\n```python\r\nfor i in range(1, 11):\r\n    print(i)\r\n```\r\n\r\nOutput:\r\n\r\n```\r\n1\r\n2\r\n3\r\n4\r\n5\r\n6\r\n7\r\n8\r\n9\r\n10\r\n```\r\n\r\nExplanation:\r\n\r\n- The `range()` function generates a sequence of numbers from the starting value (inclusive) to the ending value (exclusive).\r\n- In this case, we start with 1 and go up to 11 (exclusive) because we want to count 10.\r\n- The `for` loop iterates over each number in the sequence and assigns it to the variable `i`.\r\n- The `print()` function outputs the value of `i` on a new line."}


if __name__ == "__main__":
    fire.Fire(main)
