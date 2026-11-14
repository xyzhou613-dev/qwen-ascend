"""Qwen XML to OpenAI tool calls. Pure Python; no model/device state."""
import ast
import json
import math
import re
import uuid

from .request_validation import RequestError

_CALL = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)


def select_tools(tools, choice=None):
    """Return selected declarations, forced continuation prefix, parse flag."""
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        raise RequestError("tools must be a list", "tools")
    names = {}
    for i, tool in enumerate(tools):
        param = "tools[%d]" % i
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise RequestError("only function tools are supported", param)
        fn = tool.get("function")
        if not isinstance(fn, dict):
            raise RequestError("tool.function must be an object", param + ".function")
        name = fn.get("name")
        if (not isinstance(name, str) or not name or name != name.strip()
                or re.search(r"[<>\r\n]", name)):
            raise RequestError("tool function name must be a nonempty XML-safe string", param + ".function.name")
        if name in names:
            raise RequestError("duplicate tool function name: " + name, param + ".function.name")
        if not isinstance(fn.get("parameters", {}), dict):
            raise RequestError("function.parameters must be an object", param + ".function.parameters")
        names[name] = tool
    if choice is None or choice == "auto":
        # Preserve legacy XML parsing for callers declaring tools in messages.
        return tools, "", True
    if choice == "none":
        return [], "", False
    if choice == "required":
        if not tools:
            raise RequestError("tool_choice=required needs at least one tool", "tool_choice")
        name = next(iter(names)) if len(names) == 1 else None
    elif isinstance(choice, dict) and choice.get("type") == "function":
        fn = choice.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or name not in names:
            raise RequestError("tool_choice must name a declared function", "tool_choice")
        tools = [names[name]]
    else:
        raise RequestError("invalid tool_choice", "tool_choice")
    prefix = "<tool_call>\n"
    if name is not None:
        prefix += "<function=" + name + ">\n"
    return tools, prefix, True


def completion_text(prefix, generated):
    """Reattach the prefilled envelope without duplicating model restarts."""
    if not prefix:
        return generated
    stripped = generated.lstrip()
    if stripped.startswith("<tool_call>"):
        return generated
    if stripped.startswith("<function="):
        return "<tool_call>\n" + generated
    return prefix + generated


def _resolve(schema, root):
    if not isinstance(schema, dict):
        return {}
    for _ in range(16):
        ref = schema.get("$ref")
        if ref is None:
            return schema
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise ValueError("only local schema references are supported")
        resolved = root
        for part in ref[2:].split("/"):
            resolved = resolved[part.replace("~1", "/").replace("~0", "~")]
        if not isinstance(resolved, dict):
            raise ValueError("schema reference must resolve to an object")
        schema = dict(resolved, **{k: v for k, v in schema.items() if k != "$ref"})
    raise ValueError("cyclic schema reference")


def _types(schema, root, depth=0):
    if depth > 24:
        raise ValueError("schema is too deep")
    schema = _resolve(schema, root)
    value = schema.get("type")
    found = {value} if isinstance(value, str) else set(value) if isinstance(value, list) else set()
    for keyword in ("anyOf", "oneOf", "allOf"):
        for branch in schema.get(keyword, []):
            found.update(_types(branch, root, depth + 1))
    if not found:
        if "properties" in schema:
            found.add("object")
        elif "items" in schema:
            found.add("array")
    return found


def _is_type(value, kind):
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return (isinstance(value, int) and not isinstance(value, bool)
                or isinstance(value, float) and math.isfinite(value))
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict) and all(isinstance(k, str) for k in value)
    if kind == "null":
        return value is None
    return False


def _json_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if a is None or b is None:
        return a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def _matches(value, schema, root, depth=0):
    if depth > 48:
        return False
    if isinstance(schema, bool):
        return schema
    schema = _resolve(schema, root)
    kind = schema.get("type")
    if isinstance(kind, str) and not _is_type(value, kind):
        return False
    if isinstance(kind, list) and not any(_is_type(value, k) for k in kind):
        return False
    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(keyword)
        if branches is not None:
            checks = [_matches(value, s, root, depth + 1) for s in branches]
            if keyword == "anyOf" and not any(checks):
                return False
            if keyword == "oneOf" and sum(checks) != 1:
                return False
            if keyword == "allOf" and not all(checks):
                return False
    if "enum" in schema and not any(_json_equal(value, v) for v in schema["enum"]):
        return False
    if "const" in schema and not _json_equal(value, schema["const"]):
        return False
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            return False
        props = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(k not in props for k in value):
            return False
        for k, v in value.items():
            subschema = props.get(k, schema.get("additionalProperties", {}))
            if isinstance(subschema, dict) and not _matches(v, subschema, root, depth + 1):
                return False
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        if not all(_matches(v, schema["items"], root, depth + 1) for v in value):
            return False
    return True


def _parameter(raw, schema, root):
    text = raw.strip()
    kinds = _types(schema, root)
    # String declarations must retain numeric/boolean/null-looking text.
    if (not kinds or kinds == {"string"}) and _matches(text, schema, root):
        return text
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
            if _matches(value, schema, root):
                json.dumps(value, allow_nan=False)
                return value
        except (ValueError, TypeError, SyntaxError, OverflowError, RecursionError):
            pass
    if _matches(text, schema, root):
        return text
    raise ValueError("tool parameter does not match its declared type")


def parse_tool_calls(text, tools=None):
    """Return text and complete, typed calls; retain malformed XML as text."""
    if not text or "<tool_call>" not in text:
        return text, None
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools or []}
    calls, spans = [], []
    for match in _CALL.finditer(text):
        name = match.group(1).strip()
        if schemas and name not in schemas:
            continue
        root = schemas.get(name, {})
        try:
            resolved = _resolve(root, root)
            props = resolved.get("properties", {})
            if not isinstance(props, dict):
                raise ValueError("schema properties must be an object")
            extra = resolved.get("additionalProperties", {})
            args = {}
            for key, raw in _PARAM.findall(match.group(2)):
                key = key.strip()
                if key in args:
                    raise ValueError("duplicate parameter")
                args[key] = _parameter(raw, props.get(key, extra if isinstance(extra, dict) else {}), root)
            if not _matches(args, root, root):
                continue
            arguments = json.dumps(args, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError, KeyError, SyntaxError, OverflowError, RecursionError):
            continue
        calls.append({"id": "call_" + uuid.uuid4().hex[:16], "type": "function",
                      "function": {"name": name, "arguments": arguments}})
        spans.append(match.span())
    if not calls:
        return text, None
    out, last = [], 0
    for start, end in spans:
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    return "".join(out).strip(), calls


def finish_reason(tokens, stop_ids, calls=None):
    if calls:
        return "tool_calls"
    return "stop" if tokens and tokens[-1] in stop_ids else "length"
