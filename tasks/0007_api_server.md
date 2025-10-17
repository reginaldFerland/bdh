# Task 0007: Implement OpenAI-Compatible API Server

## Priority
**High** - Recommended deployment solution for BDH models

## Purpose
Build a production-ready FastAPI server that wraps BDH inference with OpenAI-compatible API endpoints. This enables easy integration with existing tools and applications that use the OpenAI API format, provides a standard interface, and supports concurrent requests with proper queuing.

## Current State
- No API server implementation
- No standardized interface for model deployment
- No way to serve model over HTTP
- No support for concurrent requests
- No authentication or rate limiting

## Expected Outcome
After implementing this task, the project should have:
1. FastAPI-based server with OpenAI-compatible endpoints
2. Support for `/v1/completions` and `/v1/chat/completions` endpoints
3. Streaming responses using Server-Sent Events (SSE)
4. Request queuing and batching for efficiency
5. Authentication and rate limiting
6. Docker support for easy deployment
7. Configuration file for server settings
8. Health check and metrics endpoints

## Detailed Requirements

### 1. API Endpoints

#### Completions Endpoint
```
POST /v1/completions
Content-Type: application/json

{
  "model": "bdh-shakespeare",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 50,
  "n": 1,
  "stream": false,
  "stop": ["\n\n"]
}
```

#### Chat Completions Endpoint
```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "bdh-shakespeare",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "max_tokens": 100,
  "temperature": 0.8,
  "stream": false
}
```

#### Models Endpoint
```
GET /v1/models

{
  "object": "list",
  "data": [
    {
      "id": "bdh-shakespeare",
      "object": "model",
      "created": 1234567890,
      "owned_by": "user"
    }
  ]
}
```

### 2. Response Format

#### Non-streaming Response
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "bdh-shakespeare",
  "choices": [
    {
      "text": "Generated text...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 100,
    "total_tokens": 105
  }
}
```

#### Streaming Response (SSE)
```
data: {"id":"cmpl-abc123","object":"text_completion","created":1234567890,"choices":[{"text":"Once","index":0,"logprobs":null,"finish_reason":null}],"model":"bdh-shakespeare"}

data: {"id":"cmpl-abc123","object":"text_completion","created":1234567890,"choices":[{"text":" upon","index":0,"logprobs":null,"finish_reason":null}],"model":"bdh-shakespeare"}

data: [DONE]
```

### 3. Server Configuration

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1

model:
  path: "models/bdh-shakespeare"
  device: "cuda"  # or "cpu"
  dtype: "float16"  # or "float32", "bfloat16"

generation:
  max_tokens: 100
  temperature: 0.8
  top_p: 0.95
  top_k: 50

auth:
  enabled: false
  api_keys: []

rate_limiting:
  enabled: true
  requests_per_minute: 60
```

### 4. Docker Support

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "serve.py", "--config", "config.yaml"]
```

## Implementation Steps

### Step 1: Create serve.py with FastAPI Application

```python
#!/usr/bin/env python3
"""
BDH OpenAI-Compatible API Server
Provides /v1/completions and /v1/chat/completions endpoints.
"""
import argparse
import asyncio
import time
import uuid
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import yaml

from bdh import BDH


# Pydantic models for request/response
class CompletionRequest(BaseModel):
    model: str
    prompt: str | List[str]
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: Optional[float] = None
    top_k: Optional[int] = 50
    n: int = 1
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: Optional[float] = None
    top_k: Optional[int] = 50
    n: int = 1
    stream: bool = False
    stop: Optional[List[str]] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "user"


# Global model instance
model: Optional[BDH] = None
config: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, config
    
    # Load model
    print(f"Loading model from {config['model']['path']}...")
    model = BDH.from_pretrained(
        config['model']['path'],
        device=config['model']['device'],
        torch_dtype=getattr(torch, config['model']['dtype']),
    )
    model.eval()
    print("Model loaded successfully!")
    
    yield
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="BDH API Server", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key if authentication is enabled."""
    if not config.get('auth', {}).get('enabled', False):
        return True
    
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    # Extract key from "Bearer <key>"
    try:
        scheme, key = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid auth scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid auth header")
    
    if key not in config['auth']['api_keys']:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "BDH API Server", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/v1/models")
async def list_models(authorized: bool = Depends(verify_api_key)):
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": config['model']['path'].split('/')[-1],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
            }
        ]
    }


@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Generate text completion."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Handle streaming
    if request.stream:
        return StreamingResponse(
            generate_completion_stream(request),
            media_type="text/event-stream"
        )
    
    # Non-streaming generation
    try:
        # Generate
        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        generated_text = model.generate_text(
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        
        # Count tokens (approximate)
        prompt_tokens = len(prompt) // 4  # Rough estimate
        completion_tokens = len(generated_text) - len(prompt)
        completion_tokens = completion_tokens // 4
        
        # Format response
        response = {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "text": generated_text[len(prompt):],  # Only new text
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_completion_stream(request: CompletionRequest):
    """Generate streaming completion."""
    try:
        completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        
        for token in model.generate_stream(
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        ):
            if token != prompt:  # Skip echoing prompt
                chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "choices": [
                        {
                            "text": token,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None
                        }
                    ],
                    "model": request.model,
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run
        
        # Send done
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        error_chunk = {"error": str(e)}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Generate chat completion."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert messages to prompt
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    prompt += "\nassistant:"
    
    # Reuse completion endpoint logic
    completion_request = CompletionRequest(
        model=request.model,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        n=request.n,
        stream=request.stream,
        stop=request.stop,
    )
    
    if request.stream:
        return StreamingResponse(
            generate_completion_stream(completion_request),
            media_type="text/event-stream"
        )
    else:
        response = await create_completion(completion_request, authorized)
        # Convert to chat format
        data = response.body.decode() if hasattr(response, 'body') else response
        # Modify object type
        return response


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="BDH API Server")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    global config
    config = load_config(args.config)
    
    # Override with CLI args
    if args.host:
        config['server']['host'] = args.host
    if args.port:
        config['server']['port'] = args.port
    if args.model:
        config['model']['path'] = args.model
    
    # Run server
    import uvicorn
    uvicorn.run(
        app,
        host=config['server']['host'],
        port=config['server']['port'],
        workers=config['server'].get('workers', 1),
    )


if __name__ == "__main__":
    import json
    main()
```

### Step 2: Create Default Configuration File

Create `config.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1

model:
  path: "models/bdh-shakespeare"
  device: "auto"  # "auto", "cuda", "cpu"
  dtype: "float16"  # "float32", "float16", "bfloat16"

generation:
  max_tokens: 100
  temperature: 0.8
  top_p: 0.95
  top_k: 50

auth:
  enabled: false
  api_keys: []

rate_limiting:
  enabled: false
  requests_per_minute: 60
```

### Step 3: Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install server dependencies
RUN pip install --no-cache-dir fastapi uvicorn pyyaml

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "serve.py", "--config", "config.yaml"]
```

### Step 4: Create docker-compose.yml

```yaml
version: '3.8'

services:
  bdh-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./config.yaml:/app/config.yaml
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Step 5: Add Server Dependencies

Update `requirements.txt`:
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pyyaml>=6.0
pydantic>=2.0.0
```

### Step 6: Create Client Example

Create `examples/api_client.py`:
```python
"""Example client for BDH API server."""
import requests
import json

API_URL = "http://localhost:8000"

# Test completion
response = requests.post(
    f"{API_URL}/v1/completions",
    json={
        "model": "bdh-shakespeare",
        "prompt": "To be or not to be",
        "max_tokens": 100,
        "temperature": 0.8,
        "stream": False,
    }
)
print(response.json())

# Test streaming
response = requests.post(
    f"{API_URL}/v1/completions",
    json={
        "model": "bdh-shakespeare",
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "stream": True,
    },
    stream=True,
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data != '[DONE]':
                chunk = json.loads(data)
                print(chunk['choices'][0]['text'], end='', flush=True)

# Test chat completion
response = requests.post(
    f"{API_URL}/v1/chat/completions",
    json={
        "model": "bdh-shakespeare",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 100,
    }
)
print(response.json())
```

## Testing Plan

1. **Test Basic Endpoints**
   - GET / (root)
   - GET /health
   - GET /v1/models
   - Verify responses are correct

2. **Test Completions Endpoint**
   - POST with valid request
   - Verify response format
   - Check generated text quality
   - Test with different parameters

3. **Test Streaming**
   - POST with stream=true
   - Verify SSE format
   - Check tokens stream immediately
   - Verify [DONE] at end

4. **Test Chat Completions**
   - POST with messages
   - Verify prompt conversion
   - Check response format

5. **Test Authentication**
   - Enable auth in config
   - Test without API key (should fail)
   - Test with invalid key (should fail)
   - Test with valid key (should succeed)

6. **Test Error Handling**
   - Invalid model name
   - Malformed request
   - Server errors
   - Verify proper error responses

7. **Test Docker Deployment**
   - Build Docker image
   - Run container
   - Test API endpoints
   - Check logs

8. **Test Concurrent Requests**
   - Send multiple requests simultaneously
   - Verify all complete successfully
   - Check response times

## Code Example

Example usage after implementation:

```bash
# Start server
python serve.py --config config.yaml

# Or with Docker
docker-compose up

# Test with curl
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bdh-shakespeare",
    "prompt": "To be or not to be",
    "max_tokens": 100,
    "temperature": 0.8,
    "stream": false
  }'

# Test streaming
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bdh-shakespeare",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "stream": true
  }'
```

## Copilot Implementation Prompt

```
Create an OpenAI-compatible API server for BDH:

1. Create serve.py with FastAPI application:
   - Import fastapi, uvicorn, pydantic, torch, yaml
   - Define Pydantic models: CompletionRequest, ChatCompletionRequest, ChatMessage, ModelInfo
   - Global variables: model (BDH instance), config (dict)
   - Use @asynccontextmanager lifespan to load model on startup
   - Load model with BDH.from_pretrained() using config settings

2. Implement endpoints:
   - GET /: Return welcome message
   - GET /health: Return {"status": "healthy", "model_loaded": bool}
   - GET /v1/models: Return list of available models in OpenAI format
   - POST /v1/completions: Generate text completion
   - POST /v1/chat/completions: Generate chat completion (convert messages to prompt)

3. For /v1/completions endpoint:
   - Accept CompletionRequest with: model, prompt, max_tokens, temperature, top_p, top_k, stream, stop
   - If stream=true: return StreamingResponse with generate_completion_stream()
   - If stream=false: call model.generate_text(), format response in OpenAI format
   - Response should include: id (uuid), object, created (timestamp), model, choices (list), usage (token counts)
   - Each choice has: text, index, finish_reason ("length" or "stop")

4. Implement streaming:
   - async def generate_completion_stream() that yields SSE formatted chunks
   - Use model.generate_stream() (from Task 0006)
   - Format each chunk as: f"data: {json.dumps(chunk)}\\n\\n"
   - End with: "data: [DONE]\\n\\n"
   - Use asyncio.sleep(0) to yield control

5. Add authentication (optional):
   - verify_api_key() dependency function
   - Check Authorization header: "Bearer <key>"
   - Verify key against config['auth']['api_keys']
   - Return HTTPException(401) if invalid

6. Add CORS middleware:
   - Use CORSMiddleware from fastapi.middleware.cors
   - Allow all origins for development

7. Create config.yaml with sections:
   - server: host, port, workers
   - model: path, device, dtype
   - generation: max_tokens, temperature, top_p, top_k
   - auth: enabled, api_keys

8. Add main() function:
   - Parse args: --config, --host, --port, --model
   - Load config from YAML
   - Override config with CLI args
   - Run with uvicorn.run()

9. Create Dockerfile:
   - FROM python:3.10-slim
   - Install dependencies from requirements.txt
   - Install fastapi, uvicorn, pyyaml
   - COPY application files
   - EXPOSE 8000
   - CMD ["python", "serve.py", "--config", "config.yaml"]

10. Token counting:
    - Approximate: len(text) // 4 for prompt and completion
    - Include in usage field of response

Use the BDH.from_pretrained() method from Task 0005 and generate_text()/generate_stream() from Task 0006.

OpenAI response format example:
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "bdh-shakespeare",
  "choices": [{"text": "generated", "index": 0, "finish_reason": "length"}],
  "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
}
```
```

## Files to Modify/Create
- **Create**: `serve.py` - FastAPI server application
- **Create**: `config.yaml` - Server configuration
- **Create**: `Dockerfile` - Docker image definition
- **Create**: `docker-compose.yml` - Docker Compose configuration
- **Create**: `examples/api_client.py` - Example client code
- **Modify**: `requirements.txt` - Add server dependencies

## Dependencies
- `fastapi>=0.104.0`
- `uvicorn[standard]>=0.24.0`
- `pyyaml>=6.0`
- `pydantic>=2.0.0`
- Task 0005 (Model Export) for loading models
- Task 0006 (Inference Script) for generation methods

## Success Criteria
- [ ] Server starts and loads model successfully
- [ ] /v1/completions endpoint works
- [ ] /v1/chat/completions endpoint works
- [ ] Streaming responses work correctly
- [ ] OpenAI API format is compatible
- [ ] Docker deployment works
- [ ] Authentication works when enabled
- [ ] Concurrent requests are handled properly
- [ ] Error handling is robust
- [ ] Documentation includes examples

## Related Tasks
- **Task 0005**: Model Export (prerequisite)
- **Task 0006**: Inference Script (uses same generation methods)
- **Task 0010**: Quantization (can optimize server performance)

## Notes
- This is the recommended deployment solution for BDH
- OpenAI-compatible API makes integration easy
- Consider adding request queuing for high load
- Rate limiting can be added with slowapi library
- For production, use reverse proxy (nginx) in front
- Consider adding metrics endpoint for monitoring
- GPU sharing between requests needs careful management
