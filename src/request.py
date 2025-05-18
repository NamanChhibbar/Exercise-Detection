from dotenv import load_dotenv
from openai import OpenAI

# Load the OpenAI API key from the .env file
load_dotenv()

def completion_request(
    prompt: str,
    model: str = 'gpt-4.1'
  ) -> str:
  '''
  Function to get a response from OpenAI's GPT-3.5-turbo model.
  
  Parameters:
    prompt (str): The prompt to send to the model.
      
  Returns:
    str: The response from the model.
  '''
  client = OpenAI()
  response = client.responses.create(
    model=model,
    input=prompt
  )
  return response.output_text
