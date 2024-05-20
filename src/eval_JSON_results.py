import os
import sys
import json
import re

COMMAND_LINE_DEF_FILE = "./eval_JSON_results_commandLine.txt"

def main():
    (start_time_secs, pretty_start_time, my_args, addl_logfile) = cmdlogtime.begin(COMMAND_LINE_DEF_FILE)
    out_file = os.path.join(my_args["out_dir"], "agg_results.txt")
    json_file = my_args["json_file"]
    a_file = my_args["a_file"]
    b_file = my_args["b_file"]
    agg_method = my_args["agg_method"]
    with open(json_file, "r") as f:
        json_data = json.load(f)
    
    a_terms = get_terms_from_infile(a_file)
    b_terms = get_terms_from_infile(b_file)  
    json_dict = process_json(json_data, a_terms, b_terms)

    write_json_dict(json_dict, out_file)
    
    cmdlogtime.end(addl_logfile, start_time_secs)

# ---------------- FUNCTIONS --------------------
def get_terms_from_infile(in_f):
    terms = []
    with open(in_f, "r") as f:
        for line in f.readlines():
            terms.append(line.strip())
    return terms

def process_json(json_data, a_terms, b_terms):
    nested_dict = {}
    try:
        term_scores = {}
        term_counts = {}
        ab_in_input = []
        ab_in_json = []
        
        for entry in json_data:
            a_term = entry
            for b in json_data[a_term]:
                b_term = b.get("Term", None)
                a_b = a_term + ":" + b_term
                ab_in_json.append(a_b)
                nested_dict[a_b] = {}  #hash by a and b term to guarantee uniqueness
                result_list = b.get("Result", None)
                total_score = 0
                n = 0
                for result in result_list:
                    matches = re.findall(r"Score: (-?\d+)", result)        
                    for match in matches:
                        #print(match)
                        total_score += int(match)
                        n = n + 1
                avg_score = total_score/n
                nested_dict[a_b] = {
                    "Total Score": total_score,
                    "Average Score": avg_score,
                    "Count": n,
                }
        for i, a in enumerate(a_terms):
            ab_in_input.append(a + ":" + b_terms[i])
        for ab in ab_in_input:
            if ab not in ab_in_json:
                nested_dict[ab] = {
                    "Total Score": 0,
                    "Average Score": 0 ,
                    "Count": 0,
                }        
        return nested_dict

    except Exception as e:
        print(f"An error occurred while processing the JSON data: {e}")
        return None

def write_json_dict(json_dict, out_file):
    try:
        term_scores = {}
        term_counts = {}
        with open(out_file, "w") as f:
            f.write("A_Term:B_Term\tAverage Score\tTotal Score\tCount\n")
            for a_b in json_dict:
                f.write(a_b + "\t" +
                 str(json_dict[a_b]["Average Score"]) + "\t" + 
                 str(json_dict[a_b]["Total Score"]) + "\t" + 
                 str(json_dict[a_b]["Count"]) + "\n")
                #import pdb
                #pdb.set_trace()
    except Exception as e:
        print(f"An error occurred while writing the JSON dictionary data: {e}")
        return None

def extract_scores(directory):
    results = []
    # Walk through all files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json') and file != 'config.json':
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Open and load the JSON file
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    # Iterate over each entry in the JSON array
                    for entry in data:
                        Relationship = entry.get("Relationship", "No Relationship Provided")
                        score_details = entry.get("Result", [])
                        for detail in score_details:
                            # Using regex to find the score pattern
                            match = re.search(r"Score: ([-+]?\d+)", detail)
                            if match:
                                score = match.group(1)
                                results.append({"Relationship": Relationship, "Score": score})
    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    score_results = extract_scores(directory_path)
    results_file_path = os.path.join(directory_path, "results.txt")

    # Saving the results to a text file in the specified directory
    with open(results_file_path, "w") as file:
        for result in score_results:
            file.write(f"Relationship: {result['Relationship']}, Score: {result['Score']}\n")

if __name__ == "__main__":
    main()

