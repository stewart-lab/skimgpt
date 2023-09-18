import openai


def test_openai_connection(api_key):
    openai.api_key = api_key
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
