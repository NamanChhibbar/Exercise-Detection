import json

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def json_response(request: HttpRequest) -> JsonResponse:
  '''
  Returns a JSON response with the given data and status code.
  
  :param request: The HTTP request object.
  :return: A JsonResponse object with the given data and status code.
  '''
  print(json.loads(request.body))
  return JsonResponse({'message': 'Hello World!'}, status=200)
