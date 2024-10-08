import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

client = OpenAI(
    project=os.getenv("OPENAI_PROJECT_ID"),
    api_key=os.getenv("OPENAI_API_KEY")
    )

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with a sentence in English, and your task is to translate it into French."
    },
    {
      "role": "user",
      "content": "My name is Jane. What is yours?"
    }
  ],
  temperature=0.7,
  max_tokens=64,
  top_p=1
)

print(response)
