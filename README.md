# GPT-based Paper Analysis

This repository provides tools to skim through scientific papers and analyze the effectiveness of treatment discussed within them using the OpenAI GPT model. The focus is on finding, consolidating, and categorizing abstracts into valid, ineffective, or inconclusive treatments.

## Directory Structure

```bash
├── pycache
├── requirements.txt
├── skim_crohn_bioprocess_drugs_but_not_km_crohn_colitis_drugs0.05.txt
├── skim_no_km.py
└── test_skim_no_km.py
```


## Requirements

- Python 3.x
- Libraries specified in `requirements.txt`

## Getting Started

1. **Setup**:
   Clone the repository to your local machine.
   
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**
   Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Before running the script, ensure you have set up your OpenAI API key in your environment. You can set it using:
  ```bash
    export OPENAI_API_KEY=your_api_key_here
```

## Important Notes

- The current script processes only the first 3 rows from the input file.
- Ensure you are adhering to the rate limits of the Semantic Scholar API and OpenAI API when making requests.
- Handle API keys with care. Avoid exposing them in public repositories or sharing them.

## Contributions

Feel free to contribute to this repository by submitting a pull request or opening an issue for suggestions and bugs.


## Acknowledgments

Thanks to OpenAI for providing the GPT model which powers the analysis in this tool.



