import csv

# Define paths to your files
csv_path = '/w5home/jfreeman/kmGPT/input_lists/exercise5/skim_paper_table1_queries.csv'
a_term_path = '/w5home/jfreeman/kmGPT/input_lists/exercise5/skim_a.txt'
b_term_path = '/w5home/jfreeman/kmGPT/input_lists/exercise5/skim_b.txt'
c_term_path = '/w5home/jfreeman/kmGPT/input_lists/exercise5/skim_c.txt'

# Function to write terms to a file
def write_terms(terms, filename):
    with open(filename, 'w') as file:
        for term in terms:
            file.write(term + '\n')

# Read the CSV and extract columns
with open(csv_path, mode='r') as file:
    csv_reader = csv.DictReader(file)
    a_terms, b_terms, c_terms = [], [], []
    for row in csv_reader:
        a_terms.append(row['a_term'])
        b_terms.append(row['b_term'])
        c_terms.append(row['c_term'])

# Write terms to their respective files
write_terms(a_terms, a_term_path)
write_terms(b_terms, b_term_path)
write_terms(c_terms, c_term_path)
