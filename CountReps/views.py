import json

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from src.request import completion_request

@csrf_exempt
def openai_request(request: HttpRequest) -> JsonResponse:
  '''
  Returns a JSON response with the OpenAI response.
  
  Parameters:
    request: The HTTP request object.
  
  Returns:
    JsonRepsonse: Output from chat completion response.
  '''
  data = json.loads(request.body)
  prompt = data.get('prompt', '')
  model = data.get('model', 'gpt-4.1')
  
  try:
    response = completion_request(prompt, model)
    return JsonResponse({'response': response}, status=200)
  except Exception as e:
    return JsonResponse({'error': str(e)}, status=500)
