"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils

import fire

def encode_task_generation_prompt(prompt_file_name="./prompts/robot_task_prompt.txt"):
    """Encode prompt for task generation into a single string."""
    prompt = open(prompt_file_name).read() + "\n"
    # TODO: add seeded tasks
    # make it gpt-4 format
    message = [
        {"role": "user", "content": prompt}]
    return message

def post_process_task_response(response):
    if response is None:
        return []
    ### Parse the response into list of tasks ###
    raw_tasks = response['message']['content']
    raw_tasks = re.split("###", raw_tasks)
    print(raw_tasks)
    tasks = []
    for idx, task in enumerate(raw_tasks):
        # # if the decoding stops due to length, the last example is likely truncated so we discard it
        # if idx == len(raw_tasks) - 1 and response["finish_reason"] == "length":
        #     continue
        ##### Parse the response into task #####
        splitted_data = re.split(f"{idx}:\s+", task)
        if len(splitted_data) != 2:
            continue
        else:
            task = splitted_data[1].strip()

        ##### FILTER OUT Negative Examples #####
        # filter out too short or too long tasks
        if len(task.split()) <= 3 or len(task.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        # filter those starting with punctuation
        if task[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not task[0].isascii():
            continue
        tasks.append({"task": task})
    return tasks

def generate_task_data(
    output_dir="./gpt4_generation/",
    num_tasks_to_generate=1,
    model_name="gpt-4",
    request_batch_size=1,
    temperature=1.0,
    top_p=1.0,
):
    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    ### load the LM-generated previous tasks. ###
    machine_task_data = []
    if os.path.exists(os.path.join(output_dir, "task_regen.json")):
        machine_task_data = utils.jload(os.path.join(output_dir, "task_regen.json"))
        print(f"Loaded {len(machine_task_data)} machine-generated tasks")
    progress_bar = tqdm.tqdm(total=num_tasks_to_generate)
    while len(machine_task_data) < num_tasks_to_generate:
        ### Generate tasks ###
        request_idx += 1

        batch_input = []
        for _ in range(request_batch_size):
            prompt = encode_task_generation_prompt()
            batch_input.append(prompt)
        decoding_args = utils.OpenAIChatDecodingArguments(
            temperature=temperature,
            top_p=top_p,
            max_tokens=2048,
            stop=["\n10", "10."],
        )
        request_start_time = time.time()
        chatcompletions = utils.openai_chatcompletion(
            prompts=batch_input,
            model_name=model_name,
            decoding_args=decoding_args,
        )
        request_duration = time.time() - request_start_time

        tasks = []
        for completion in chatcompletions:
            new_tasks = post_process_task_response(completion)
            tasks += new_tasks

        # TODO: delete tasks based on similarity
        total = len(tasks)
        keep = 0

        for task_data in tasks:
            keep += 1
            machine_task_data.append(task_data)
            progress_bar.update(1)

        print(f"Request {request_idx} took {request_duration:.2f}s")
        print(f"Generated {total} tasks, kept {keep} instructions")
        utils.jdump(machine_task_data, os.path.join(output_dir, "task_regen.json"))


# def encode_instruct_prompt(task_description, prompt_file_name="./prompts/robot_instruct_prompt.txt"):
#     """Encode prompt for instruction following pairs into a single string."""
#     prompt = open(prompt_file_name).read() + "\n"
#     # TODO: add task name
#     # TODO: optional: add examples of subtasks
#     # TODO: add functions for subskills
#     # make it gpt-4 format
#     message = [
#         {"role": "user", "content": prompt}]
#     return message

# def post_process_chat_response(num_prompt_instructions, response):
#     if response is None:
#         return []
#     ### Parse the response into pairs of instruction, input, output ###
#     raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
#     raw_instructions = re.split("###", raw_instructions)
#     instructions = []
#     for idx, inst in enumerate(raw_instructions):
#         # if the decoding stops due to length, the last example is likely truncated so we discard it
#         if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
#             continue
#         ##### Parse the response into instruction, input, output #####
#         idx += num_prompt_instructions + 1
#         splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
#         if len(splitted_data) != 7:
#             continue
#         else:
#             inst = splitted_data[2].strip()
#             input = splitted_data[4].strip()
#             input = "" if input.lower() == "<noinput>" else input
#             output = splitted_data[6].strip()

#         ##### FILTER OUT Negative Examples #####
#         # filter out too short or too long instructions
#         if len(inst.split()) <= 3 or len(inst.split()) > 150:
#             continue
#         # filter based on keywords that are not suitable for language models.
#         # filter those starting with punctuation
#         if inst[0] in string.punctuation:
#             continue
#         # filter those starting with non-english character
#         if not inst[0].isascii():
#             continue
#         instructions.append({"instruction": inst, "input": input, "output": output})
#     return instructions

# def generate_instruction_following_chat_data(
#     output_dir="./gpt4_generation/",
#     seed_tasks_path="./seed_tasks.jsonl",
#     num_instructions_to_generate=1,
#     model_name="gpt-4",
#     num_prompt_instructions=0,
#     request_batch_size=1,
#     temperature=1.0,
#     top_p=1.0,
#     num_cpus=8,
# ):
#     ### Load the seed tasks. Optinal. TODO ###

#     os.makedirs(output_dir, exist_ok=True)
#     request_idx = 0
#     ### load the LM-generated previous instructions. ###
#     machine_instruction_data = []
#     # if os.path.exists(os.path.join(output_dir, "regen.json")):
#     #     machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
#     #     print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

#     # similarities = {}
#     scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

#     # now let's generate new instructions!
#     progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
#     if machine_instruction_data:
#         progress_bar.update(len(machine_instruction_data))

#     # first we tokenize all the seed instructions and generated machine instructions
#     # all_instructions = [d["instruction"] for d in seed_instruction_data] + [
#     #     d["instruction"] for d in machine_instruction_data
#     # ]
#     # all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

#     while len(machine_instruction_data) < num_instructions_to_generate:
#         request_idx += 1

#         batch_inputs = []
#         for _ in range(request_batch_size):
#             # only sampling from the seed tasks
#             # prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
#             prompt = encode_prompt()
#             batch_inputs.append(prompt)
#         decoding_args = utils.OpenAIDecodingArguments(
#             temperature=temperature,
#             n=1,
#             max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
#             top_p=top_p,
#             stop=["\n20", "20.", "20."],
#         )
#         request_start = time.time()
#         results = utils.openai_chatcompletion(
#             prompts=batch_inputs,
#             model_name=model_name,
#             batch_size=request_batch_size,
#             decoding_args=decoding_args,
#             logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
#         )
#         request_duration = time.time() - request_start
#         responss_text = results['choices'][0]['message']['content']
#         print(responss_text)
#         utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))

        # process_start = time.time()
        # instruction_data = []
        # for result in results:
        #     new_instructions = post_process_chat_response(num_prompt_instructions, result)
        #     instruction_data += new_instructions

        # total = len(instruction_data)
        # keep = 0
        # for instruction_data_entry in instruction_data:
        #     # computing similarity with the pre-tokenzied instructions
        #     new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
        #     with Pool(num_cpus) as p:
        #         rouge_scores = p.map(
        #             partial(rouge_scorer._score_lcs, new_instruction_tokens),
        #             all_instruction_tokens,
        #         )
        #     rouge_scores = [score.fmeasure for score in rouge_scores]
        #     most_similar_instructions = {
        #         all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
        #     }
        #     if max(rouge_scores) > 0.7:
        #         continue
        #     else:
        #         keep += 1
        #     instruction_data_entry["most_similar_instructions"] = most_similar_instructions
        #     instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
        #     machine_instruction_data.append(instruction_data_entry)
        #     all_instructions.append(instruction_data_entry["instruction"])
        #     all_instruction_tokens.append(new_instruction_tokens)
        #     progress_bar.update(1)
        # process_duration = time.time() - process_start
        # print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        # print(f"Generated {total} instructions, kept {keep} instructions")
        # utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))

def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
