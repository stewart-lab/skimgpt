# GPT-based Paper Analysis

This repository provides tools to SKIM! through scientific papers and analyze the effectiveness of treatments discussed within them using the OpenAI GPT model. The primary goal is to extract, consolidate, and categorize abstracts from scientific papers into categories such as useful, harmful, ineffective, or inconclusive treatments for specific medical conditions.

## Directory Structure

```bash
├── km_results_of different_flavors.txt
├── requirements.txt
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

4. **Running the script**
   ```bash
   python skim_no_km.py
   ```

## Important Notes

- The script is designed to handle rate limits when making requests to external APIs.
- Always handle API keys with care. Do not expose them in public repositories or share them without necessary precautions.
- Handle API keys with care. Avoid exposing them in public repositories or sharing them.

## Contributions

Feel free to contribute to this repository by submitting a pull request or opening an issue for suggestions and bugs.


## Acknowledgments

Thanks to OpenAI for providing the GPT model which powers the analysis in this tool.



