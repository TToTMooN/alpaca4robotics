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


def encode_task_generation_prompt(prompt_file_name="./prompts_for_gpt/robot_task_prompt.txt"):
    """Encode prompt for task generation into a single string."""
    prompt = open(prompt_file_name).read() + "\n"
    # TODO: add seeded tasks
    # make it gpt-4 format
    message = [{"role": "user", "content": prompt}]
    return message


def post_process_task_response(response):
    if response is None:
        return []
    ### Parse the response into list of tasks ###
    raw_tasks = response["message"]["content"]
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
        # filter those starting with non-english character
        if not task[0].isascii():
            continue
        tasks.append({"task": task})
    return tasks


def generate_task_data(
    output_dir="./gpt4_generation/",
    prompt_file_name="./prompts_for_gpt/robot_task_prompt_v0.txt",
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
            prompt = encode_task_generation_prompt(prompt_file_name)
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


def encode_instruct_prompt(
    tasks,
    functions,
    examples,
    subskill_data=None,
    prompt_file_name="./prompts_for_gpt/robot_instruction_prompt.txt",
):
    """Encode prompt for instruction following pairs into a single string."""
    prompt = open(prompt_file_name).read() + "\n"
    task_placeholder = "{TASK_LIST_PLACEHOLDER}"
    function_placeholder = "{FUNCTION_LIST_PLACEHOLDER}"
    trajectory_placeholder = "{TRAJECTORY_PLACEHOLDER}"
    instruction_example_placeholder = "{INSTRUCTION_EXAMPLE_PLACEHOLDER}"
    # Add task names as a list
    task_string_to_replace = ""
    for idx, task in enumerate(tasks):
        task_string_to_replace += f"{idx + 1}. {task}\n"

    # Add add functions for subskills
    function_string_to_replace = ""
    for idx, function in enumerate(functions):
        function_info = function["function"]
        function_description = function["description"]
        function_string_to_replace += f"'''\n"
        function_string_to_replace += f"{function_info}\n"
        function_string_to_replace += f"'''\n"
        function_string_to_replace += f"{function_description}\n"
    # Replace the placeholder string with a new string
    prompt = prompt.replace(task_placeholder, task_string_to_replace)
    prompt = prompt.replace(function_placeholder, function_string_to_replace)

    # optional: add example trajectory of subskills
    if subskill_data is not None:
        subskill_data_string_to_replace = ""
        subskill_data_string_to_replace += \
            f"Ending observation of the robot finishing the provided skills:\n"
        for data in subskill_data:
            skill_name = data["skill_name"]
            skill_params = data["params"]
            skill_last_obs = data["last_obs"]

            subskill_data_string_to_replace += f"'''\n"
            subskill_data_string_to_replace += f"Skill name: {skill_name}\n"
            subskill_data_string_to_replace += f"Parameters: {skill_params}\n"
            subskill_data_string_to_replace += f"Final Obs: {skill_last_obs}\n"
            subskill_data_string_to_replace += f"'''\n"
        prompt = prompt.replace(trajectory_placeholder, subskill_data_string_to_replace)

    # Add examples of instruction pairs
    example_string = ""
    for example in examples:
        instruction, input, output, task_name = (
            example["instruction"],
            example["instances"][0]["input"],
            example["instances"][0]["output"],
            example["task_name"],
        )
        example_string += f"###\n"
        example_string += f" <Task> {task_name}\n"
        example_string += f" <Instruction> {instruction}\n"
        example_string += f" <Input> {input}\n"
        example_string += f" <Output> \n{output}\n"
    prompt = prompt.replace(instruction_example_placeholder, example_string)
    # make it gpt-4 format
    # System prompt: Environment information, prompt before [Tasks to solve]
    # User prompt: In-context examples.
    system_prompt = prompt.split("[Tasks to solve]")[0]
    user_prompt = prompt.split("[Tasks to solve]")[1]
    message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return message


def post_process_chat_response(response):
    if response is None:
        return []
    ### Parse the response into pairs of instruction, input, output ###
    raw_instructions = response["message"]["content"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        ##### Parse the response into instruction, input, output #####
        idx += 1
        splitted_data = re.split(f"(<Task>|<Instruction>|<Input>|<Output>)", inst)
        if len(splitted_data) != 9:
            continue
        else:
            task = splitted_data[2].strip()
            inst = splitted_data[4].strip()
            input = splitted_data[6].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[8].strip()
            # parse output into <verbal output> and <action output>
            output_splitted_data = re.split(f"(\[verbal\]|\[action\])", output)
            verbal_output = output_splitted_data[2].strip()
            action_output = output_splitted_data[4].strip()
            ##### FILTER OUT Negative Examples #####
            # filter out too short or too long instructions
            if len(inst.split()) <= 3 or len(inst.split()) > 200:
                print(splitted_data, len(inst.split()))

                continue
            # filter based on keywords that are not suitable for language models.
            # filter those starting with punctuation
            instructions.append(
                {
                    "task": task,
                    "instruction": inst,
                    "input": input,
                    "verbal_output": verbal_output,
                    "action_output": action_output,
                }
            )
    return instructions
def post_process_chat_response_gorilla_format(response):
    if response is None:
        return []
    ### Parse the response into pairs of instruction, input, output ###
    raw_instructions = response["message"]["content"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        ##### Parse the response into instruction, input, output #####
        idx += 1
        splitted_data = re.split(f"(<Task>|<Instruction>|<Input>|<Output>)", inst)
        if len(splitted_data) != 9:
            continue
        else:
            task = splitted_data[2].strip()
            inst = splitted_data[4].strip()
            input = splitted_data[6].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[8].strip()
            # parse output into <verbal output> and <action output>
            output_splitted_data = re.split(f"(\[verbal\]|\[action\])", output)
            verbal_output = output_splitted_data[2].strip()
            action_output = output_splitted_data[4].strip()
            ##### FILTER OUT Negative Examples #####
            # filter out too short or too long instructions
            if len(inst.split()) <= 3 or len(inst.split()) > 150:
                continue
            # filter based on keywords that are not suitable for language models.
            # filter those starting with punctuation
            if inst[0] in string.punctuation:
                continue
            # filter those starting with non-english character
            if not inst[0].isascii():
                continue
            # code in format of gorilla example
            code = f""
            code += f"###Instruction: {inst}\n"
            code += f"###Input: {input}\n"
            output = f""
            output += f"'explanation': {verbal_output}, 'code': {action_output}"
            code += f"###Output: {{{output}}}\n"
            instructions.append(
                {
                    "code": code,
                    "task": task,
                    # "environment": environment,
                    # "function_api": function_api,
                }
            )
    return instructions

def generate_instruction_following_chat_data(
    output_dir="./gpt4_generation/",
    instruction_prompt_file="./prompts_for_gpt/robot_instruction_prompt.txt",
    seed_tasks_path="./prompts_for_gpt/seeded_tasks.jsonl",
    seed_example_path="./prompts_for_gpt/seeded_example.jsonl",
    function_file_path="./prompts_for_gpt/skill_functions.jsonl",
    subskill_data_path="./subtask_data/sub_skill_example_data.json",
    use_sub_skill_data=False,
    num_instructions_to_generate=1,
    model_name="gpt-4",
    num_of_tasks_each_request=2,
    request_batch_size=1,
    temperature=1.0,
    top_p=1.0,
    output_file="instruct_regen.json",
    num_cpus=8,
):
    ### Load the seed tasks. ###

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instructions = [json.loads(l) for l in open(seed_example_path, "r")]
    functions = [json.loads(l) for l in open(function_file_path, "r")]

    ### load the LM-generated previous instructions. ###
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, output_file)):
        machine_instruction_data = utils.jload(
            os.path.join(output_dir, output_file)
        )
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")
    ### load subskill data ###
    if use_sub_skill_data:
        subskill_data = []
        examples = utils.jload(subskill_data_path)
        for example in examples:
            data = {"skill_name": "", "params": ""}
            data["skill_name"] = example["function_name"]
            data["params"] = example["params"] if example["params"] is not None else ""
            data["last_obs"] = example["last_obs"]
            # add data name and description
            subskill_data.append(data)
    else:
        subskill_data = None

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))
    start_task_index = 0
    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_input = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            # prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_instruct_prompt(
                prompt_file_name=instruction_prompt_file,
                tasks=seed_tasks[start_task_index:start_task_index+num_of_tasks_each_request],
                functions=functions,
                subskill_data=subskill_data,
                examples=seed_instructions,
            )
            batch_input.append(prompt)
        decoding_args = utils.OpenAIChatDecodingArguments(
            temperature=temperature,
            top_p=top_p,
            max_tokens=5000,
        )
        request_start_time = time.time()
        chatcompletions = utils.openai_chatcompletion(
            prompts=batch_input,
            model_name=model_name,
            decoding_args=decoding_args,
        )
        request_duration = time.time() - request_start_time
        instructions = []
        for completion in chatcompletions:
            new_instructions = post_process_chat_response(completion)
            instructions += new_instructions
        total = len(instructions)
        keep = 0

        for instruction_data in instructions:
            keep += 1
            machine_instruction_data.append(instruction_data)
            progress_bar.update(1)
        
        print(progress_bar)

        print(f"Request {request_idx} took {request_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(
            machine_instruction_data, os.path.join(output_dir, output_file)
        )
        start_task_index += num_of_tasks_each_request
    return chatcompletions

def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
