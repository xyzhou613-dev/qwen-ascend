"""Validation shared by the HTTP service and CPU-only regression tests."""
import math


class RequestError(ValueError):
    def __init__(self, message, param=None, status=400):
        super().__init__(message)
        self.param = param
        self.status = status


def object_body(body):
    if not isinstance(body, dict):
        raise RequestError('request body must be a JSON object')
    return body


def positive_int(value, name, default):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RequestError(name + ' must be a positive integer', name)
    return value


def chat_parameters(body, model_id):
    object_body(body)
    if body.get('model', model_id) != model_id:
        raise RequestError('requested model is not loaded; use ' + model_id, 'model', 404)
    if type(body.get('n', 1)) is not int or body.get('n', 1) != 1:
        raise RequestError('this backend supports n=1 only', 'n')
    stream = body.get('stream', False)
    if not isinstance(stream, bool):
        raise RequestError('stream must be a boolean', 'stream')
    max_new = positive_int(body.get('max_tokens', body.get('max_completion_tokens')), 'max_tokens', 128)
    temperature = body.get('temperature', 0)
    if temperature is None:
        temperature = 0
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise RequestError('temperature must be a finite number between 0 and 2', 'temperature')
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages:
        raise RequestError('messages must be a nonempty list', 'messages')
    normalized = []
    for i, message in enumerate(messages):
        param = 'messages[%d]' % i
        if not isinstance(message, dict) or message.get('role') not in ('system', 'developer', 'user', 'assistant', 'tool'):
            raise RequestError('message must have a supported role', param)
        content = message.get('content')
        if isinstance(content, list):
            if not all(isinstance(p, dict) and p.get('type') == 'text' and isinstance(p.get('text'), str) for p in content):
                raise RequestError('only text content parts are supported', param + '.content')
            content = ''.join(p['text'] for p in content)
        if content is None and message.get('tool_calls'):
            content = ''          # OpenAI: 带 tool_calls 的 assistant 消息 content 可为 null
        if not isinstance(content, str):
            raise RequestError('content must be a string or list of text parts', param + '.content')
        normalized.append(dict(message, content=content))
    return normalized, max_new, float(temperature), stream
