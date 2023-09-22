import openai
import abstract_comprehension as ab 
import json
import os

TEST_JSON_FILE = 'test.json'
TEST_CONFIG_FILE = 'test_config.json'


def test_openai_connection():
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical research analyst."},
                {"role": "user", "content": "Test connection to OpenAI."},
            ],
        )
        print("Successfully connected to OpenAI!")
    except Exception as e:
        print(f"Failed to connect to OpenAI. Error: {e}")




def test_process_json():
    # Load test data and config from JSON files
    with open(TEST_JSON_FILE, 'r') as f:
        json_data = json.load(f)
    
    with open(TEST_CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    # Call the function with test data and config
    result = ab.process_json(json_data, config)
    
    # Perform assertions to check if the function works as expected
    assert result is not None, "Result should not be None"
    
    # Check if the keys in the result match the expected diseases and treatments
    assert set(result.keys()) == {"Crohn's disease"}, "Unexpected diseases in result"

    assert result["Crohn's disease"]['FERRIC MALTOL']['Total Score'] == 6, "Incorrect Total Score for FERRIC MALTOL"
    assert result["Crohn's disease"]['FERRIC MALTOL']['Average Score'] == 1.5, "Incorrect Average Score for FERRIC MALTOL"
    assert result["Crohn's disease"]['FERRIC MALTOL']['Count'] == 4, "Incorrect Count for FERRIC MALTOL"

