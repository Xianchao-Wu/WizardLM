import jsonlines
import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from human_eval.data import write_jsonl, read_problems, stream_jsonl

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def read_mbpp(path):
    mbpp_problems = {}
    with jsonlines.open(path, "r") as fin:
        for obj in fin:
            mbpp_problems[obj["task_id"]] = obj
    return mbpp_problems # 500 个问题

def extract_text(prompt, remove_lines=True):
    token = '\"\"\"'
    start = token
    end = '>>>'

    start_idx = prompt.find(start) + len(start)
    end_idx = prompt.find(end)

    output = prompt[start_idx: end_idx]
    if remove_lines:
        output = output.replace('\n', ' ')
    output = re.sub(r"\s+", " ", output).strip()

    return output

def generate_prompt(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION # e.g., 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n\nWrite a python function to remove first and last occurrence of a given character from the string.\nTest examples:\nassert remove_Occ("hello","l") == "heo"\nassert remove_Occ("abcda","a") == "bcd"\nassert remove_Occ("PHP","P") == "H"\n\n### Response:'

def get_model(
    load_8bit: bool = False, # TODO for GPU memory saving
    base_model: str = "bigcode/starcoder",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model) # GPT2TokenizerFast(name_or_path='/workspace/asr/WizardLM/WizardCoder/models--WizardLM--WizardCoder-15B-V1.0/snapshots/926ca1b215c4631bc5f8c3e47173381452c23e5c', vocab_size=49152, model_max_length=2048, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'pad_token': '[PAD]', 'additional_special_tokens': ['<|endoftext|>', '<fim_prefix>', '<fim_middle>', '<fim_suffix>', '<fim_pad>', '<filename>', '<gh_stars>', '<issue_start>', '<issue_comment>', '<issue_closed>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<empty_output>', '<commit_before>', '<commit_msg>', '<commit_after>', '<reponame>']}, clean_up_tokenization_spaces=True)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit, # False
            torch_dtype=torch.float16,
            device_map="auto",
        ) # 15,517,462,528 -> 155亿参数的一个coding model NOTE
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    model.config.pad_token_id = tokenizer.pad_token_id # 49152

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    #import ipdb; ipdb.set_trace()
    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--overwrite', action='store_true', help='')
    parser.add_argument('--mbpp_path', type=str, help="")

    args = parser.parse_args() # Namespace(N=200, decoding_style='sampling', end_index=2, max_len=2048, mbpp_path='data/mbpp.test.jsonl', model='/workspace/asr/WizardLM/WizardCoder/models--WizardLM--WizardCoder-15B-V1.0/snapshots/926ca1b215c4631bc5f8c3e47173381452c23e5c', num_seqs_per_iter=2, output_path='preds/MBPP_T0.2_N200', overwrite=False, start_index=0, temperature=0.2)

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    STOP_SEQS = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']

    problems = read_mbpp(args.mbpp_path)

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = []
    for task_id in task_ids:
        prompt = f"\n{problems[task_id]['text']}\nTest examples:"
        if task_id == 493:
            # The test examples are too long. We choose to only include the function name.
            test_example = problems[task_id]['test_list'][0]
            prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
        else:
            for test_example in problems[task_id]['test_list']:
                prompt += f"\n{test_example}"
        prompts.append(prompt)
    # prompts = ['\nWrite a python function to remove first and last occurrence of a given character from the string.\nTest examples:\nassert remove_Occ("hello","l") == "heo"\nassert remove_Occ("abcda","a") == "bcd"\nassert remove_Occ("PHP","P") == "H"', '\nWrite a function to sort a given matrix in ascending order according to the sum of its rows.\nTest examples:\nassert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]\nassert sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])==[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]\nassert sort_matrix([[5,8,9],[6,4,3],[2,1,4]])==[[2, 1, 4], [6, 4, 3], [5, 8, 9]]'], for debugging only two examples, NOTE 
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    tokenizer, model = get_model(base_model=args.model)
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id, # 49152
        do_sample=True,
        temperature=args.temperature, # 0.2 
        max_length=args.max_len, # 2048
        num_return_sequences=args.num_seqs_per_iter, # 2
        eos_token_id=tokenizer.eos_token_id, # 0
        top_p=0.95
    )

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)
        # output_file = 'preds/MBPP_T0.2_N200/0.jsonl'
        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t') # '\nWrite a python function to remove first and last occurrence of a given character from the string.\nTest examples:\nassert remove_Occ("hello","l") == "heo"\nassert remove_Occ("abcda","a") == "bcd"\nassert remove_Occ("PHP","P") == "H"'
        prompt_batch = [generate_prompt(prompt)]

        ids_batch = [task_ids[i]]

        completion_seqs = []
        # encoding['input_ids'].shape=[1, 105]; encoding['attention_mask'].shape=[1, 105]
        encoding = tokenizer(prompt_batch, return_tensors="pt", 
                truncation=True, max_length=args.max_len).to(device)
        if args.decoding_style == 'sampling': # NOTE, yes, in
            loops = int(args.N / args.num_seqs_per_iter) # 100=loops
        else:
            loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):
            #import ipdb; ipdb.set_trace()
            with torch.no_grad():
                if args.decoding_style == 'sampling':
                    gen_tokens = model.generate(
                        **encoding, # 使用tokenizer准备好的，包括了'input_ids'和'attention_mask'
                        generation_config=generation_config # 生成算法的配置信息
                    ) # NOTE important, 在dgx1机器上，这个花了很长时间5minutes??? TODO
                    # gen_tokens.shape=[2, 404] tensor, 其中的2，是由num_return_sequences来控制的.
            if gen_tokens is not None:
                gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True) # '''['Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n\nWrite a python function to remove first and last occurrence of a given character from the string.\nTest examples:\nassert remove_Occ("hello","l") == "heo"\nassert remove_Occ("abcda","a") == "bcd"\nassert remove_Occ("PHP","P") == "H"\n\n### Response:Here\'s the Python function to remove first and last occurrence of a given character from the string:\r\n\r\n```python\r\ndef remove_Occ(string, char):\r\n    if char not in string:\r\n        return string\r\n    else:\r\n        if string.count(char) == 1:\r\n            return string.replace(char, "")\r\n        else:\r\n            return string[:string.index(char)] + string[string.index(char)+1:string.rindex(char)] + string[string.rindex(char)+1:]\r\n```\r\n\r\nHere\'s how the function works:\r\n\r\n1. First, we check if the given character is present in the string or not. If it\'s not present, we simply return the original string.\r\n2. If the character is present in the string, we check if it appears only once or multiple times.\r\n3. If it appears only once, we simply remove it using the `replace()` method.\r\n4. If it appears multiple times, we remove the first and last occurrence of the character using string slicing.\r\n\r\nLet\'s test the function with the given test examples:\r\n\r\n```python\r\nassert remove_Occ("hello","l") == "heo"\r\nassert remove_Occ("abcda","a") == "bcd"\r\nassert remove_Occ("PHP","P") == "H"\r\n```\r\n\r\nAll the test cases pass.', 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n\nWrite a python function to remove first and last occurrence of a given character from the string.\nTest examples:\nassert remove_Occ("hello","l") == "heo"\nassert remove_Occ("abcda","a") == "bcd"\nassert remove_Occ("PHP","P") == "H"\n\n### Response:Here\'s the Python function to remove first and last occurrence of a given character from the string:\r\n\r\n```python\r\ndef remove_Occ(string, char):\r\n    if char in string:\r\n        if string.count(char) == 1:\r\n            return string.replace(char, "")\r\n        else:\r\n            return string.replace(char, "", 1)[:-1]\r\n    else:\r\n        return string\r\n```\r\n\r\nThe function takes two arguments: `string` and `char`. It first checks if the given character is present in the string using the `in` operator. If it is present, it checks if the character appears only once in the string using the `count` method. If it appears only once, it removes the character using the `replace` method and returns the modified string. If the character appears more than once, it removes the first occurrence using the `replace` method with the `count` argument set to 1, and then removes the last character using slicing.\r\n\r\nIf the given character is not present in the string, the function simply returns the original string.\r\n\r\nHere are some test examples:\r\n\r\n```python\r\nassert remove_Occ("hello","l") == "heo"\r\nassert remove_Occ("abcda","a") == "bcd"\r\nassert remove_Occ("PHP","P") == "H"\r\n```\r\n\r\nAll the test cases pass.']'''
            else:
                gen_seqs = None

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]
                # NOTE 下面是把所有的2个候选输出，都放入'completion_seqs'中！
                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1] # 提取结果：'Here\'s the Python function to remove first and last occurrence of a given character from the string:\r\n\r\n```python\r\ndef remove_Occ(string, char):\r\n    if char not in string:\r\n        return string\r\n    else:\r\n        if string.count(char) == 1:\r\n            return string.replace(char, "")\r\n        else:\r\n            return string[:string.index(char)] + string[string.index(char)+1:string.rindex(char)] + string[string.rindex(char)+1:]\r\n```\r\n\r\nHere\'s how the function works:\r\n\r\n1. First, we check if the given character is present in the string or not. If it\'s not present, we simply return the original string.\r\n2. If the character is present in the string, we check if it appears only once or multiple times.\r\n3. If it appears only once, we simply remove it using the `replace()` method.\r\n4. If it appears multiple times, we remove the first and last occurrence of the character using string slicing.\r\n\r\nLet\'s test the function with the given test examples:\r\n\r\n```python\r\nassert remove_Occ("hello","l") == "heo"\r\nassert remove_Occ("abcda","a") == "bcd"\r\nassert remove_Occ("PHP","P") == "H"\r\n```\r\n\r\nAll the test cases pass.'
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id, # 11
                         'completion': completion_seq, # 'Here\'s the Python function to remove first and last occurrence of a given character from the string:\r\n\r\n```python\r\ndef remove_Occ(string, char):\r\n    if char not in string:\r\n        return string\r\n    else:\r\n        if string.count(char) == 1:\r\n            return string.replace(char, "")\r\n        else:\r\n            return string[:string.index(char)] + string[string.index(char)+1:string.rindex(char)] + string[string.rindex(char)+1:]\r\n```\r\n\r\nHere\'s how the function works:\r\n\r\n1. First, we check if the given character is present in the string or not. If it\'s not present, we simply return the original string.\r\n2. If the character is present in the string, we check if it appears only once or multiple times.\r\n3. If it appears only once, we simply remove it using the `replace()` method.\r\n4. If it appears multiple times, we remove the first and last occurrence of the character using string slicing.\r\n\r\nLet\'s test the function with the given test examples:\r\n\r\n```python\r\nassert remove_Occ("hello","l") == "heo"\r\nassert remove_Occ("abcda","a") == "bcd"\r\nassert remove_Occ("PHP","P") == "H"\r\n```\r\n\r\nAll the test cases pass.'
                         'all_code': all_code, # 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n\nWrite a python function to remove first and last occurrence of a given character from the string.\nTest examples:\nassert remove_Occ("hello","l") == "heo"\nassert remove_Occ("abcda","a") == "bcd"\nassert remove_Occ("PHP","P") == "H"\n\n### Response:Here\'s the Python function to remove first and last occurrence of a given character from the string:\r\n\r\n```python\r\ndef remove_Occ(string, char):\r\n    if char not in string:\r\n        return string\r\n    else:\r\n        if string.count(char) == 1:\r\n            return string.replace(char, "")\r\n        else:\r\n            return string[:string.index(char)] + string[string.index(char)+1:string.rindex(char)] + string[string.rindex(char)+1:]\r\n```\r\n\r\nHere\'s how the function works:\r\n\r\n1. First, we check if the given character is present in the string or not. If it\'s not present, we simply return the original string.\r\n2. If the character is present in the string, we check if it appears only once or multiple times.\r\n3. If it appears only once, we simply remove it using the `replace()` method.\r\n4. If it appears multiple times, we remove the first and last occurrence of the character using string slicing.\r\n\r\nLet\'s test the function with the given test examples:\r\n\r\n```python\r\nassert remove_Occ("hello","l") == "heo"\r\nassert remove_Occ("abcda","a") == "bcd"\r\nassert remove_Occ("PHP","P") == "H"\r\n```\r\n\r\nAll the test cases pass.'
                         }
                    )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    main()
