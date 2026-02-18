import json
import logging
import httpx

logger = logging.getLogger(__name__)


class OllamaRetryTransport(httpx.AsyncBaseTransport):
    """Sanitizes and retries requests to Ollama's OpenAI-compatible API.

    Handles two Ollama compatibility issues:

    1. HTTP 400 "invalid message content type: <nil>" — Ollama rejects assistant
       messages where content is null (valid per OpenAI spec but not supported by
       Ollama). Fixed by replacing null content with "" before sending.

    2. HTTP 500 "invalid character '\\n' in string literal" — the model sometimes
       generates tool call argument JSON with literal newline bytes, which Ollama's
       Go JSON parser rejects. Fixed by retrying (the model usually produces valid
       JSON on the next attempt).
    """

    MAX_RETRIES = 3

    def __init__(self):
        self._transport = httpx.AsyncHTTPTransport()

    def _sanitize_request(self, request: httpx.Request) -> httpx.Request:
        """Replace null message content with empty string for Ollama compatibility."""
        try:
            body = json.loads(request.content)
            if 'messages' not in body:
                return request

            changed = False
            for msg in body['messages']:
                if msg.get('content') is None:
                    msg['content'] = ''
                    changed = True

            if not changed:
                return request

            return self.fix_request(request, body)
        except Exception:
            return request

    def fix_request(self, request: httpx.Request, body: dict) -> httpx.Request:
        fixed = json.dumps(body, separators=(',', ':')).encode()
        headers = dict(request.headers)
        headers['content-length'] = str(len(fixed))
        return httpx.Request(
            method=request.method,
            url=request.url,
            headers=headers,
            content=fixed,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request = self._sanitize_request(request)
        for attempt in range(1, self.MAX_RETRIES + 1):
            response = await self._transport.handle_async_request(request)
            if response.status_code != 500 or attempt == self.MAX_RETRIES:
                return response
            await response.aread()  # drain body so connection can be reused
            logger.warning(
                f"Ollama 500 (attempt {attempt}/{self.MAX_RETRIES}), retrying..."
            )
        return response


ollama_http_client = httpx.AsyncClient(transport=OllamaRetryTransport())
