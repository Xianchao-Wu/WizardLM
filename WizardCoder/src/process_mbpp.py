import sys

sys.path.append('/workspace/asr/WizardLM/WizardCoder/human-eval')

from human_eval.data import stream_jsonl
import glob 
from tqdm import tqdm
import argparse
import jsonlines
import json

def read_mbpp(path):
    mbpp_problems = {}
    with jsonlines.open(path, "r") as fin:
        for obj in fin:
            mbpp_problems[obj["task_id"]] = obj
    return mbpp_problems

parser = argparse.ArgumentParser()

# Inputs
parser.add_argument(
    '--path',
    type=str,
    help="")
parser.add_argument(
    '--out_path',
    type=str,
    help="")
parser.add_argument(
    '--add_prompt',
    action='store_true',
    help='')
parser.add_argument('--mbpp_path', type=str, help="")

args = parser.parse_args()
# Namespace(add_prompt=True, mbpp_path='data/mbpp.test.jsonl', out_path='preds/MBPP_T0.2_N2.jsonl', path='preds/MBPP_T0.2_N2')

files = sorted(glob.glob(args.path + '/*.jsonl'))
print("{} files in {}".format(len(files), args.path))

problems = read_mbpp(args.mbpp_path)
output = [[] for _ in range(len(problems))]
a = 0
for code_file in tqdm(files, total=len(files)): # e.g., code_file='preds/MBPP_T0.2_N2/0.jsonl'
    codes = [c for c in stream_jsonl(code_file)]
    if args.add_prompt: # True, in, NOTE
        for code in codes:
            task_id = code['task_id']
            completion = code['completion']
            if '```python' in completion: 
                def_line = completion.index('```python')
                completion = completion[def_line:].strip() # e.g., '```python\r\ndef remove_Occ(string, char):\r\n    if char not in string:\r\n        return string\r\n    else:\r\n        return string.replace(char, "", 1)[1:-1]\r\n```\r\n\r\nThe function takes two arguments: `string` and `char`. It first checks if the given character is present in the string. If not, it returns the original string. If the character is present, it uses the `replace()` method to remove all occurrences of the character except the first and last one. Finally, it returns the modified string.\r\n\r\nHere are some test cases:\r\n\r\n```python\r\nassert remove_Occ("hello","l") == "heo"\r\nassert remove_Occ("abcda","a") == "bcd"\r\nassert remove_Occ("PHP","P") == "H"\r\n```\r\n\r\nAll test cases pass.'
                completion = completion.replace('```python', '') # '\r\ndef remove_Occ(string, char):\r\n    if char not in string:\r\n        return string\r\n    else:\r\n        return string.replace(char, "", 1)[1:-1]\r\n```\r\n\r\nThe function takes two arguments: `string` and `char`. It first checks if the given character is present in the string. If not, it returns the original string. If the character is present, it uses the `replace()` method to remove all occurrences of the character except the first and last one. Finally, it returns the modified string.\r\n\r\nHere are some test cases:\r\n\r\n\r\nassert remove_Occ("hello","l") == "heo"\r\nassert remove_Occ("abcda","a") == "bcd"\r\nassert remove_Occ("PHP","P") == "H"\r\n```\r\n\r\nAll test cases pass.'
                try:
                    next_line = completion.index('\n```')
                    completion = completion[:next_line].strip() # 'def remove_Occ(string, char):\r\n    if char not in string:\r\n        return string\r\n    else:\r\n        return string.replace(char, "", 1)[1:-1]'
                except:
                    a += 1
            if "__name__ == \"__main__\"" in completion:
                next_line = completion.index('if __name__ == "__main__":')
                completion = completion[:next_line].strip()
            
            if "# Example usage" in completion:
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            
            if "# Test examples" in completion:
                next_line = completion.index('# Test examples') # NOTE, e.g., 'def common_in_nested_lists(lst):\r\n    common = []\r\n    for i in range(len(lst)):\r\n        for j in range(len(lst[i])):\r\n            if lst[i][j] not in common:\r\n                common.append(lst[i][j])\r\n    return common\r\n\r\n# Test examples\r\nassert common_in_nested_lists([[12, 18, 23, 25, 45], [7, 12, 18, 24, 28], [1, 5, 8, 12, 15, 16, 18]])==[18, 12]\r\nassert common_in_nested_lists([[12, 5, 23, 25, 45], [7, 11, 5, 23, 28], [1, 5, 8, 18, 23, 16]])==[5,23]\r\nassert common_in_nested_lists([[2, 3,4, 1], [4, 5], [6,4, 8],[4, 5], [6, 8,4]])==[4]'
                completion = completion[:next_line].strip() # NOTE, e.g., -> 'def common_in_nested_lists(lst):\r\n    common = []\r\n    for i in range(len(lst)):\r\n        for j in range(len(lst[i])):\r\n            if lst[i][j] not in common:\r\n                common.append(lst[i][j])\r\n    return common'
            
            output[task_id-11].append(completion)
# NOTE 核心逻辑就是，只抽取最纯洁的代码部分，放入jsonl文件中。 
print("save to {}".format(args.out_path)) # save to preds/MBPP_T0.2_N2.jsonl
print(a)
with open(args.out_path, "w", encoding="utf-8") as fout:
    json.dump(output, fout)
