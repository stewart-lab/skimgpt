from __future__ import annotations
import numpy as np
import pandas as pd
import json
from abstract_comprehension import read_tsv_to_dataframe
import os


class RaggedTensor:
    def __init__(self, data, break_point=[]):
        self.data = data
        self.break_point = break_point
        self.index = 0
        self.getShape()

    def getShape(self) -> None:
        if self.is2D():
            self.shape = [len(i) for i in self.data]
        else:
            self.shape = len(self.data)

    def is2D(self) -> bool:
        if not (len(self.data) == 0):
            return isinstance(self.data[0], list)
        else:
            return False

    # Duplicates each element in data according to the shape_list
    def expand(self, shape_list: list) -> None:
        assert (
            not self.is2D()
        ), "Data must be 1D before calling expand. Call flatten first?"
        assert self.shape == len(
            shape_list
        ), "The length of shape list must equal the length of data"

        expanded = []
        for idx, inp in enumerate(self.data):
            expanded.extend([inp] * shape_list[idx])

        return RaggedTensor(expanded)

    def flatten(self) -> RaggedTensor:
        if self.is2D():
            output = []
            for lst in self.data:
                output.extend(lst)
            return RaggedTensor(output)
        else:
            return self

    # Inverts the expand method
    def compress(self, shape_list: list):
        assert self.shape == sum(shape_list)
        self.data = list(set(self.data))
        self.getShape()

    # Splits the data depending on the index
    def split(self) -> list[RaggedTensor]:
        if len(self.break_point) == 0:
            print("Warning: No breakpoint was specified.")
            return self, RaggedTensor([])
        past_break_point = 0
        output = []
        for break_point in self.break_point:
            output.append(RaggedTensor(self.data[past_break_point:break_point]))
            past_break_point = break_point
        output.append(RaggedTensor(self.data[past_break_point:]))
        return output

    # Reshapes the data depending on the input
    def reshape(self, shape: list) -> list:
        assert not self.is2D(), "Reshape only works with 1D tensors."
        assert self.shape == sum(
            shape
        ), "The shape of the tensor should be equal to the sum of the wanted shape."
        output = []
        running_length = 0
        for length in shape:
            output.append(self.data[running_length : running_length + length])
            running_length += length

        self.data = output
        self.getShape()

    # Applies a mask to the tensor
    def applyFilter(self, mask: RaggedTensor) -> None:
        assert (
            self.shape == mask.shape
        ), "Filtering only works when the shapes are the same"
        if self.is2D():
            for i in range(len(self.data)):
                boolean_mask = np.array(mask[i]) == 1
                self.data[i] = list(np.array(self.data[i])[boolean_mask])
        else:
            boolean_mask = np.array(mask) == 1
            self.data = list(np.array(self.data)[boolean_mask])

    # Applies a function to the tensor
    def map(self, func: callable, *args) -> RaggedTensor:
        assert not self.is2D(), "Map only works with 1D tensors"
        return RaggedTensor([func(i, *args) for i in self.data], self.break_point)

    # Simply concatenates two ragged tensors and appends to the break_point list
    def __add__(self, other: RaggedTensor) -> RaggedTensor:
        assert not self.is2D(), "Adding only works with flattened tensors"
        break_point = self.shape
        return RaggedTensor(self.data + other.data, self.break_point + [break_point])

    def __str__(self):
        return str(self.data)

    def __iter__(self):
        return self.flatten().data.__iter__()

    def __getitem__(self, index: int) -> any:
        return self.data[index]


class Config:
    def __init__(self, args: dict):
        # Load configuration from a JSON file
        with open(args.config, "r") as config_file:
            self.job_config = json.load(config_file)

        # Load data from a TSV file
        self.data = read_tsv_to_dataframe(args.km_output)

        # Access global settings directly from the loaded JSON
        self.global_settings = self.job_config["GLOBAL_SETTINGS"]

        # Define output paths and ensure directories exist
        self.km_output_dir = os.path.dirname(args.km_output)
        self.km_output_base_name = os.path.splitext(os.path.basename(args.km_output))[0]
        if not os.path.exists(self.km_output_dir) and self.km_output_dir != "":
            os.makedirs(self.km_output_dir)

        self.filtered_tsv_name = os.path.join(
            self.km_output_dir, f"filtered_{self.km_output_base_name}.tsv"
        )
        self.debug_tsv_name = os.path.join(
            self.km_output_dir, f"debug_{self.km_output_base_name}.tsv"
        )

        # Hypotheses and job settings
        self.api_key = self.job_config["API_KEY"]
        self.pubmed_api_key = self.job_config["PUBMED_API_KEY"]
        self.km_hypothesis = self.job_config["KM_hypothesis"]
        self.skim_hypotheses = self.job_config["SKIM_hypotheses"]
        self.job_type = self.job_config.get("JOB_TYPE")
        self.filter_config = self.job_config["abstract_filter"]
        self.debug = self.filter_config["DEBUG"]
        self.test_leakage = self.filter_config["TEST_LEAKAGE"]
        self.is_skim_gpt = self.job_type == "skim_with_gpt"
        self.evaluate_single_abstract = self.job_config["Evaluate_single_abstract"]
        self.post_n = self.global_settings["POST_N"]

        # Additional checks for specific configurations
        self.has_ac = (
            "ac_pmid_intersection" in self.data.columns
            and len(self.data["ac_pmid_intersection"].value_counts()) > 0
        )
        self.max_cot_tokens = self.filter_config["MAX_COT_TOKENS"]

        print(f"Job type detected. Running {self.job_type}.")
        if self.is_skim_gpt:
            assert (
                "c_term" in self.data.columns
            ), "Input TSV must have c_term if running skim_with_gpt"
            assert (
                "bc_pmid_intersection" in self.data.columns
            ), "Input TSV must have a bc_pmid_intersection."

        assert (
            "ab_pmid_intersection" in self.data.columns
        ), "Input TSV must have an ab_pmid_intersection."
        assert "a_term" in self.data.columns, "Input TSV must have an a_term."
        assert "b_term" in self.data.columns, "Input TSV must have a b_term."
