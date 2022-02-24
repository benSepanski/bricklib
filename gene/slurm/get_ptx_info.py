import re
import os.path

from csv import DictWriter
from typing import Callable, Any

with open("ptx_info.txt", 'r') as f:
    file_lines = f.readlines()

function_info_start_regex = re.compile("Compiling entry function")

function_info_start = [i for i, match in enumerate(map(function_info_start_regex.search, file_lines)) if
                       match is not None]
function_info = ['\n'.join(file_lines[start:stop])
                 for start, stop in zip(function_info_start, function_info_start[1:] + [len(file_lines)])
                 ]


def extract_group(regex_str, type_cast: Callable[[str], Any] = int):
    compiled_regex = re.compile(regex_str)
    match_objects = map(compiled_regex.search, function_info)
    actual_match = map(lambda x: None if x is None else type_cast(x.groups()[0]), match_objects)
    return list(actual_match)

brick_kernel_names = ["semiArakawaBrickKernelOptimized",
                      "semiArakawaBrickKernelSimple",
                      "ijDerivBrickKernel",
                      ]
possible_function_names = brick_kernel_names + ["cub\\d+EmptyKernel",
                                                "thrust.*kernel_agent",
                                                "kernel_assign_6",
                                                ]
function_name_regex = "(?<=Compiling entry function ).*(" + \
                      '|'.join(possible_function_names) + \
                      ")"
function_names = extract_group(function_name_regex, type_cast=str)

stat_name = ("cmem", "registers", "spill stores", "spill loads", "stack frame")
stat_regex = ("(\\d+)(?: bytes cmem)",
              "(?<=Used )(\\d+)(?= registers)",
              "(\\d+)(?= bytes spill stores)",
              "(\\d+)(?= bytes spill loads)",
              "(\\d+)(?= bytes stack frame)",
              )

function_stats = [{"name": name} for name in function_names]
for name, regex, in zip(stat_name, stat_regex):
    stats = extract_group(regex)
    for all_stats, stat in zip(function_stats, stats):
        all_stats[name] = stat

for function_stat in function_stats:
    print(function_stat)
    assert all(map(lambda x: x is not None, function_stat.values()))

# write to file
output_name = "ptx_info_brick_shape.csv"
file_exists = os.path.exists(output_name)
field_names = ("name",) + stat_name
with open(output_name, "a" if file_exists else "w") as csv_file:
    writer = DictWriter(csv_file, field_names)
    if not file_exists:
        writer.writeheader()
    for function_stat in filter(lambda x: x["name"] in brick_kernel_names, function_stats):
        writer.writerow(function_stat)
