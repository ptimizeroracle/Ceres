# 🔬 Hermes - Complete Technical Reference

**Version**: 1.0.0
**Last Updated**: October 18, 2025
**Purpose**: Comprehensive technical documentation of every component, class, design decision, and relationship in the Hermes LLM Dataset Engine.

**Quick Navigation**:
- **Architecture Overview & Diagrams**: See [`ARCHITECTURE.md`](ARCHITECTURE.md) (auto-generated from `architecture/model.yaml`)
- **Design Decisions**: See [`architecture/decisions/`](architecture/decisions/) (ADRs)
- **Implementation Details**: This document (TECHNICAL_REFERENCE.md)

**Note**: This document provides detailed implementation information. For structural relationships and visual diagrams, see `ARCHITECTURE.md`. For design rationale and trade-offs, see the ADRs.

---

## 📚 Table of Contents

- [Part 1: Architecture Overview](#part-1-architecture-overview)
- [Part 2: External Dependencies](#part-2-external-dependencies)
- [Part 3: Layer 0 - Core Utilities](#part-3-layer-0---core-utilities)
- [Part 4: Core Models & Specifications](#part-4-core-models--specifications)
- [Part 5: Layer 1 - Infrastructure Adapters](#part-5-layer-1---infrastructure-adapters)
- [Part 6: Layer 2 - Processing Stages](#part-6-layer-2---processing-stages)
- [Part 7: Layer 3 - Orchestration Engine](#part-7-layer-3---orchestration-engine)
- [Part 8: Layer 4 - High-Level API](#part-8-layer-4---high-level-api)
- [Part 9: Configuration System](#part-9-configuration-system)
- [Part 10: CLI Interface](#part-10-cli-interface)
- [Part 11: Framework Integrations](#part-11-framework-integrations)
- [Part 12: Execution Flows](#part-12-execution-flows)
- [Part 13: Data Flows](#part-13-data-flows)
- [Part 14: Extension Points](#part-14-extension-points)

---

# Part 1: Architecture Overview

## 1.1 System Architecture

Hermes follows a **5-layer architecture**:

```mermaid
graph TB
    subgraph "Layer 4: High-Level API"
        API[Pipeline]
        Builder[PipelineBuilder]
        Composer[PipelineComposer]
        Processor[DatasetProcessor]
    end

    subgraph "Layer 3: Orchestration"
        Executor[PipelineExecutor]
        Strategy[ExecutionStrategy]
        SyncExec[SyncExecutor]
        AsyncExec[AsyncExecutor]
        StreamExec[StreamingExecutor]
        Context[ExecutionContext]
        StateManager[StateManager]
        Observers[Observers]
    end

    subgraph "Layer 2: Processing Stages"
        DataLoader[DataLoaderStage]
        PromptFormatter[PromptFormatterStage]
        LLMInvocation[LLMInvocationStage]
        ResponseParser[ResponseParserStage]
        ResultWriter[ResultWriterStage]
    end

    subgraph "Layer 1: Infrastructure Adapters"
        LLMClient[LLM Client]
        DataIO[Data I/O]
        Checkpoint[Checkpoint Storage]
    end

    subgraph "Layer 0: Utilities"
        Retry[RetryHandler]
        RateLimit[RateLimiter]
        Cost[CostTracker]
        Budget[BudgetController]
        Logging[Logging Utils]
    end

    API --> Executor
    Builder --> API
    Executor --> Strategy
    Strategy --> SyncExec
    Strategy --> AsyncExec
    Strategy --> StreamExec
    Executor --> Context
    Executor --> StateManager
    Executor --> Observers
    Executor --> DataLoader
    DataLoader --> PromptFormatter
    PromptFormatter --> LLMInvocation
    LLMInvocation --> ResponseParser
    ResponseParser --> ResultWriter
    LLMInvocation --> LLMClient
    DataLoader --> DataIO
    ResultWriter --> DataIO
    StateManager --> Checkpoint
    LLMInvocation --> Retry
    LLMInvocation --> RateLimit
    LLMInvocation --> Cost
    Executor --> Budget
```

### Layer Responsibilities

| Layer | Directory | Purpose | Dependencies |
|-------|-----------|---------|--------------|
| **Layer 0** | `utils/` | Cross-cutting concerns (retry, rate limiting, cost tracking) | External libraries only |
| **Layer 1** | `adapters/` | External system integrations (LLM providers, file I/O) | Layer 0 + external APIs |
| **Layer 2** | `stages/` | Data transformation logic (load, format, invoke, parse, write) | Layers 0-1 |
| **Layer 3** | `orchestration/` | Execution control and state management | Layers 0-2 |
| **Layer 4** | `api/` | User-facing interfaces (Pipeline, Builder) | All layers |

### Key Design Principles

1. **Dependency Rule**: Dependencies only point inward (higher layers depend on lower layers, never the reverse)
2. **Focused Components**: Each component has one clear purpose
3. **Extensible**: Open for extension, closed for modification
4. **Abstraction**: Depend on abstractions, not concretions
5. **Simple**: Keep it simple

## 1.2 Design Patterns Catalog

| Pattern | Where Used | Purpose | Implementation |
|---------|------------|---------|----------------|
| **Facade** | `Pipeline` | Simplify complex subsystem | Hides orchestration complexity |
| **Builder** | `PipelineBuilder` | Fluent construction API | Chainable method calls |
| **Strategy** | `ExecutionStrategy` | Pluggable execution modes | `SyncExecutor`, `AsyncExecutor`, `StreamingExecutor` |
| **Template Method** | `PipelineStage.execute()` | Standardized stage flow | Base class with hooks |
| **Observer** | `ExecutionObserver` | Monitoring hooks | `ProgressObserver`, `CostObserver`, `LoggingObserver` |
| **Adapter** | `LLMClient`, `DataReader` | Interface translation | Wrap external libraries |
| **Factory** | `ParserFactory` | Object creation | Create parsers by type |
| **Composite** | `PipelineComposer` | Tree structure | Compose pipelines |
| **Singleton** | Config instances | Single instance | Via module-level state |
| **Chain of Responsibility** | Stage pipeline | Sequential processing | Each stage processes then passes |
| **Protocol** | `TextCleaner` | Structural typing | Duck typing with validation |
| **Memento** | `ExecutionContext` | State capture | Serializable state |

## 1.3 Thread Safety Strategy

### Thread-Safe Components

| Component | Mechanism | Reason |
|-----------|-----------|--------|
| `CostTracker` | `threading.Lock` | Shared cost accumulation |
| `RateLimiter` | `threading.Lock` | Token bucket state |
| `CheckpointStorage` | File-based locks | Concurrent writes |
| `LLMInvocationStage` | `ThreadPoolExecutor` | Concurrent LLM calls |

### Thread-Unsafe Components (By Design)

| Component | Why Not Thread-Safe |
|-----------|-------------------|
| `PipelineBuilder` | Construction phase only, not shared |
| `Pipeline` (single exec) | Each execution is sequential |
| `DataLoaderStage` | Reads once, no shared state |

---

# Part 2: External Dependencies

## 2.1 Dependency Catalog

### Production Dependencies (20 libraries)

| Library | Version | Category | License | Why Chosen | Alternatives Considered |
|---------|---------|----------|---------|------------|------------------------|
| **llama-index** | >=0.12.0 | LLM | MIT | LLM provider clients (OpenAI, Anthropic, Groq). Hermes adds batch orchestration, cost tracking, checkpointing, YAML config. | LangChain (more complex), direct APIs (no abstraction) |
| **llama-index-llms-openai** | >=0.3.0 | LLM | MIT | Official OpenAI integration | `openai` package (less abstraction) |
| **llama-index-llms-azure-openai** | >=0.3.0 | LLM | MIT | Enterprise Azure support | Custom Azure client |
| **llama-index-llms-anthropic** | >=0.3.0 | LLM | MIT | Claude integration | `anthropic` package (less abstraction) |
| **llama-index-llms-groq** | >=0.3.0 | LLM | MIT | Fast, affordable inference | Direct Groq API |
| **pandas** | >=2.0.0 | Data | BSD-3 | Industry standard, rich API | Polars (less mature ecosystem) |
| **polars** | >=0.20.0 | Data | MIT | Fast Parquet reading | Pandas (slower for large files) |
| **pydantic** | >=2.0.0 | Validation | MIT | Validation + serialization + type hints | dataclasses (no validation), marshmallow (complex) |
| **python-dotenv** | >=1.0.0 | Config | BSD-3 | Simple .env loading | os.environ (manual) |
| **tqdm** | >=4.66.0 | UI | MPL/MIT | Simple, widely used progress bars | rich.progress (overkill), progressbar2 |
| **tenacity** | >=8.2.0 | Reliability | Apache-2.0 | Flexible retry logic | `backoff` (simpler but less flexible) |
| **openpyxl** | >=3.1.0 | Data | MIT | Excel file support | xlrd (deprecated), pyexcel |
| **pyarrow** | >=15.0.0 | Data | Apache-2.0 | Fast Parquet I/O | fastparquet (slower) |
| **tiktoken** | >=0.5.0 | LLM | MIT | Fast token counting, OpenAI-native | transformers (slower), estimate (inaccurate) |
| **structlog** | >=24.0.0 | Logging | MIT/Apache | Structured logging, JSON output | standard logging (unstructured) |
| **jinja2** | >=3.1.0 | Templating | BSD-3 | Powerful prompt templating | string.Template (too simple), mako |
| **prometheus-client** | >=0.20.0 | Monitoring | Apache-2.0 | Industry standard metrics | Custom metrics (reinvent wheel) |
| **click** | >=8.1.0 | CLI | BSD-3 | Decorator-based, simple | argparse (verbose), typer (overkill) |
| **rich** | >=13.0.0 | CLI | MIT | Beautiful tables and formatting | colorama (basic), termcolor |

### Dev Dependencies (8 libraries)

| Library | Purpose | Why Chosen |
|---------|---------|------------|
| **pytest** | Testing framework | Industry standard, rich plugins |
| **pytest-cov** | Coverage reporting | Integrates with pytest |
| **pytest-asyncio** | Async test support | Test async executors |
| **black** | Code formatting | Opinionated, consistent |
| **ruff** | Fast linting | Faster than flake8/pylint |
| **mypy** | Type checking | Catch type errors |
| **ipython** | Interactive shell | Better REPL |
| **jupyter** | Notebooks | Interactive exploration |

### Optional Dependencies (1 group)

| Group | Libraries | Purpose | Platform |
|-------|-----------|---------|----------|
| **mlx** | mlx>=0.29.0, mlx-lm>=0.28.0 | Apple Silicon local inference | macOS only (M1/M2/M3/M4) |

**Installation**: `pip install hermes[mlx]`

## 2.2 Dependency Graph

```mermaid
graph TD
    Hermes[Hermes Core]

    Hermes --> LI[llama-index]
    Hermes --> Pandas
    Hermes --> Pydantic
    Hermes --> Structlog
    Hermes --> Click
    Hermes --> Rich

    LI --> LIOAI[llama-index-llms-openai]
    LI --> LIAZ[llama-index-llms-azure]
    LI --> LIANT[llama-index-llms-anthropic]
    LI --> LIGR[llama-index-llms-groq]

    Pandas --> Openpyxl[openpyxl]
    Pandas --> Polars

    Hermes --> Tiktoken[tiktoken]
    Hermes --> Tenacity[tenacity]
    Hermes --> Tqdm[tqdm]
    Hermes --> Jinja2[jinja2]
    Hermes --> Prometheus[prometheus-client]
    Hermes --> Dotenv[python-dotenv]

    Hermes -.-> MLX[mlx + mlx-lm]
    MLX -.-> MLXMetal[mlx-metal]

    Polars --> PyArrow[pyarrow]

    style MLX stroke-dasharray: 5 5
    style MLXMetal stroke-dasharray: 5 5
```

**Note**: Dashed lines indicate optional dependencies (`pip install hermes[mlx]`)

### Critical Dependencies (Cannot Be Removed)

1. **llama-index** - LLM provider clients (OpenAI, Anthropic, Groq, Azure). Hermes wraps these with LLMSpec/LLMClient for batch processing, cost tracking, unified config.
2. **pandas** - Data manipulation backbone, used throughout
3. **pydantic** - Configuration validation, type safety
4. **structlog** - Structured logging, observability

### Optional Dependencies (Can Be Removed)

1. **polars** - Only for fast Parquet reading, pandas can handle it (slower)
2. **prometheus-client** - Only for metrics export, can be disabled
3. **rich** - Only for CLI pretty printing, can fall back to basic output
4. **jinja2** - Only for advanced templating, can use string.format()
5. **mlx + mlx-lm** - Only for Apple Silicon local inference, use cloud providers instead

---

# Part 5: Layer 1 - Infrastructure Adapters (LLM Providers)

## 5.1 LLM Provider Overview

Hermes supports multiple LLM providers through the **Adapter pattern**, allowing easy switching between providers without changing core logic.

### Supported Providers

| Provider | Category | Platform | Cost | Use Case |
|----------|----------|----------|------|----------|
| **OpenAI** | Cloud API | All | $$ | Production, high quality |
| **Azure OpenAI** | Cloud API | All | $$ | Enterprise, compliance |
| **Anthropic** | Cloud API | All | $$$ | Claude models, long context |
| **Groq** | Cloud API | All | Free tier | Fast inference, development |
| **OpenAI-Compatible** | Custom/Local/Cloud | All | Varies | Ollama, vLLM, Together.AI, custom APIs |

### Provider Selection Guide

**Choose OpenAI if**: Production quality, mature ecosystem, GPT-4
**Choose Azure if**: Enterprise compliance, private deployments
**Choose Anthropic if**: Claude models, 100K+ context
**Choose Groq if**: Fast inference, free tier, development
**Choose OpenAI-Compatible if**: Custom endpoints, Ollama, vLLM, Together.AI, self-hosted

---

## 5.2 OpenAI-Compatible Provider (Custom APIs)

### Purpose
Enable integration with any LLM API that implements the OpenAI chat completions format.

### Class: `OpenAICompatibleClient`

**Inheritance**: `LLMClient` (Adapter pattern)

**Platform**: All (platform-agnostic)

**Responsibility**: Connect to custom OpenAI-compatible API endpoints

**Supports**:
- ✅ **Ollama** (local LLM server)
- ✅ **vLLM** (self-hosted inference)
- ✅ **Together.AI** (cloud API)
- ✅ **Anyscale** (cloud API)
- ✅ **LocalAI** (self-hosted)
- ✅ **Any custom OpenAI-compatible API**

### Configuration

**Required Fields**:
- `provider: openai_compatible`
- `base_url`: Custom API endpoint URL
- `model`: Model identifier

**Optional Fields**:
- `provider_name`: Custom name for logging/metrics
- `api_key`: Authentication (or use env var, or "dummy" for local)
- `input_cost_per_1k_tokens`: Custom pricing
- `output_cost_per_1k_tokens`: Custom pricing

### Architecture

```python
class OpenAICompatibleClient(LLMClient):
    def __init__(self, spec: LLMSpec):
        # Validate base_url is provided
        # Initialize OpenAI client with custom base_url
        # Use provider_name for metrics

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        # Call custom API using OpenAI format
        # Return standardized LLMResponse
```

**Design Decision: Reuse OpenAI Client**

Instead of reimplementing HTTP calls, leverage llama-index's OpenAI client with custom `api_base`:

```python
self.client = OpenAI(
    model=spec.model,
    api_key=api_key or "dummy",
    api_base=spec.base_url,  # Custom URL
)
```

**Rationale**:
- ✅ DRY: Reuse existing, well-tested code
- ✅ Reliability: llama-index handles edge cases
- ✅ Compatibility: Ensures OpenAI format compliance
- ✅ Maintainability: Updates to OpenAI client benefit us

### Example Configurations

#### Local Ollama (Free)
```yaml
llm:
  provider: openai_compatible
  provider_name: "Ollama-Local"
  model: llama3.1:70b
  base_url: http://localhost:11434/v1
  # No API key needed
  input_cost_per_1k_tokens: 0.0
  output_cost_per_1k_tokens: 0.0
```

#### Together.AI (Cloud)
```yaml
llm:
  provider: openai_compatible
  provider_name: "Together.AI"
  model: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
  base_url: https://api.together.xyz/v1
  api_key: \${TOGETHER_API_KEY}
  input_cost_per_1k_tokens: 0.0006
  output_cost_per_1k_tokens: 0.0006
```

#### Self-Hosted vLLM
```yaml
llm:
  provider: openai_compatible
  provider_name: "vLLM-Custom"
  model: meta-llama/Llama-3.1-70B-Instruct
  base_url: http://your-vllm-server:8000/v1
  input_cost_per_1k_tokens: 0.0  # Self-hosted = free
  output_cost_per_1k_tokens: 0.0
```

### Validation

**Pydantic Model Validator**:
```python
@model_validator(mode="after")
def validate_provider_requirements(self) -> "LLMSpec":
    if self.provider == LLMProvider.OPENAI_COMPATIBLE and self.base_url is None:
        raise ValueError("base_url required for openai_compatible provider")
    return self
```

**Design**: Fail fast with clear error message

### Token Estimation

Uses tiktoken with cl100k_base encoding (approximation):
```python
self.tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = len(self.tokenizer.encode(text))
```

**Tradeoff**: Not model-specific, but consistent and fast

### Performance Characteristics

**Depends on backend**:
- **Ollama local**: ~10-20 tokens/sec (70B model, single GPU)
- **vLLM multi-GPU**: ~50-80 tokens/sec (70B model, tensor parallel)
- **Together.AI cloud**: ~30-50 tokens/sec (network latency)

### Authentication Handling

**Priority order**:
1. `spec.api_key` (explicit in config)
2. `OPENAI_COMPATIBLE_API_KEY` env var
3. `"dummy"` fallback (for local APIs without auth)

**Design**: Flexible authentication for different scenarios

### Provider Naming

The `provider_name` field appears in metrics/logging:
```python
model=f"{self.provider_name}/{self.model}"
# Example: "Together.AI/meta-llama/Llama-3.1-70B"
```

**Benefit**: Clear identification in logs and cost tracking

### Dependencies

```python
from llama_index.llms.openai import OpenAI  # Reuse OpenAI client
import tiktoken  # Token estimation
```

### Used By

- `create_llm_client()` factory
- Any pipeline with `provider: openai_compatible`
- Examples: Ollama, Together.AI, vLLM configs

### Testing Strategy

**Unit Tests (14 tests)**:
- Validation (base_url requirement)
- Provider naming
- Dummy API keys for local APIs
- Cost calculation with custom pricing
- Factory integration
- Backward compatibility

### Known Limitations

- Assumes OpenAI-compatible format (doesn't support exotic APIs)
- Token counting is approximate (cl100k_base encoding)
- Can't use provider-specific features (only OpenAI format)

### Future Improvements

- [ ] Support custom headers (for exotic auth)
- [ ] Add provider-specific tokenizers
- [ ] Support streaming responses
- [ ] Add connection pooling for performance

---

---

# Part 3: Layer 0 - Core Utilities

## 3.1 Overview

Layer 0 provides cross-cutting concerns that are used throughout the system:
- Retry logic with exponential backoff
- Rate limiting (token bucket algorithm)
- Cost tracking and accumulation
- Budget enforcement
- Structured logging
- Metrics export (Prometheus)
- Input text preprocessing

**Design Principle**: These utilities have **no dependencies on higher layers**, only on external libraries.

---

## 3.2 `utils/retry_handler.py`

### Purpose
Implements robust retry logic with exponential backoff for transient failures.

### Classes

#### `RetryableError` (Exception)

**Inheritance**: `Exception`

**Purpose**: Base class for errors that should be retried

**Design Decision**: Create custom exception hierarchy to distinguish retryable vs. non-retryable errors

```python
class RetryableError(Exception):
    """Base class for errors that should be retried."""
    pass
```

#### `RateLimitError` (Exception)

**Inheritance**: `RetryableError`

**Purpose**: Specific error for rate limit scenarios

**Usage**: Raised by LLM clients when rate limits are hit

#### `NetworkError` (Exception)

**Inheritance**: `RetryableError`

**Purpose**: Specific error for network-related failures

**Usage**: Raised when network calls fail transiently

#### `RetryHandler` (Class)

**Responsibility**: Handle retry logic with configurable exponential backoff

**Single Responsibility**: ONLY handles retry logic, nothing else

**Attributes**:
```python
max_attempts: int           # Maximum retry attempts (default: 3)
initial_delay: float        # Initial delay in seconds (default: 1.0)
max_delay: float            # Maximum delay cap (default: 60.0)
exponential_base: int       # Base for exponential calculation (default: 2)
retryable_exceptions: tuple # Which exceptions to retry
```

**Methods**:

##### `__init__(max_attempts, initial_delay, max_delay, exponential_base, retryable_exceptions)`

**Purpose**: Initialize retry configuration

**Parameters**:
- `max_attempts: int = 3` - How many times to retry
- `initial_delay: float = 1.0` - Starting delay
- `max_delay: float = 60.0` - Maximum delay cap
- `exponential_base: int = 2` - Exponent base (delay = initial * base^attempt)
- `retryable_exceptions: Optional[tuple] = None` - Which exceptions trigger retry

**Design Decision**: Make all parameters configurable for flexibility across different failure scenarios

**Default Behavior**: If `retryable_exceptions` is None, defaults to `(RetryableError, RateLimitError, NetworkError)`

##### `execute(func: Callable[[], T]) -> T`

**Purpose**: Execute a function with retry logic

**Algorithm**:
1. Create `Retrying` instance from tenacity library
2. Configure stop condition: `stop_after_attempt(max_attempts)`
3. Configure wait strategy: exponential backoff with `wait_exponential()`
4. Configure retry condition: `retry_if_exception_type(retryable_exceptions)`
5. Execute function through retryer
6. If all retries fail, reraise last exception

**Parameters**:
- `func: Callable[[], T]` - Zero-argument function to execute

**Returns**: `T` - Result from successful function execution

**Raises**: Last exception if all retries exhausted

**Example**:
```python
retry_handler = RetryHandler(max_attempts=3, initial_delay=1.0)

def risky_operation():
    # Might fail with RateLimitError
    return call_llm_api()

result = retry_handler.execute(risky_operation)
```

**Design Decision**: Use tenacity library instead of custom implementation
- **Why**: tenacity is battle-tested, handles edge cases, provides flexible configuration
- **Alternative**: Custom retry loop (simpler but less robust)
- **Trade-off**: Additional dependency vs. reliability

##### `calculate_delay(attempt: int) -> float`

**Purpose**: Calculate delay for given attempt number (for informational/testing purposes)

**Algorithm**:
```python
delay = initial_delay * (exponential_base ** (attempt - 1))
return min(delay, max_delay)
```

**Parameters**:
- `attempt: int` - Attempt number (1-based)

**Returns**: `float` - Delay in seconds

**Example**:
```python
handler = RetryHandler(initial_delay=1.0, exponential_base=2, max_delay=60.0)
handler.calculate_delay(1)  # 1.0 second
handler.calculate_delay(2)  # 2.0 seconds
handler.calculate_delay(3)  # 4.0 seconds
handler.calculate_delay(4)  # 8.0 seconds
handler.calculate_delay(10) # 60.0 seconds (capped at max_delay)
```

### Thread Safety

- ✅ **Thread-safe**: Yes (stateless operation, no shared mutable state)
- Each `execute()` call is independent
- tenacity handles thread safety internally

### Dependencies

```python
import time                  # For delays
from typing import Callable  # Type hints
from tenacity import (       # Retry library
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
```

### Used By

- `LLMInvocationStage` - Retries LLM API calls
- Any component needing retry logic

### Design Patterns

1. **Strategy Pattern**: Configurable retry strategy via parameters
2. **Dependency Inversion**: Uses abstract `Callable` type, not specific implementations

### Time Complexity

- **Exponential backoff**: O(2^n) time in worst case (where n = max_attempts)
- Actual delays: 1s, 2s, 4s, 8s, 16s, ...

### Testing Considerations

- Test with mocked functions that fail predictably
- Verify exponential backoff timing
- Test max_attempts enforcement
- Test exception filtering (retryable vs. non-retryable)

### Known Limitations

- No jitter (could add randomness to prevent thundering herd)
- Fixed exponential formula (could support other backoff strategies)

### Future Improvements

- [ ] Add jitter support for distributed systems
- [ ] Support custom backoff strategies (linear, polynomial)
- [ ] Add metrics for retry count tracking

---

## 3.3 `utils/rate_limiter.py`

### Purpose
Implements token bucket algorithm for rate limiting API calls.

### Classes

#### `RateLimiter` (Class)

**Responsibility**: Control request rate using token bucket algorithm

**Algorithm**: Token Bucket
- Bucket has maximum capacity (burst_size)
- Tokens refill at constant rate (requests_per_minute / 60)
- Each request consumes tokens
- If no tokens available, wait until refilled

**Thread Safety**: ✅ **Thread-safe** via `threading.Lock`

**Attributes**:
```python
rpm: int                    # Requests per minute limit
capacity: int               # Maximum burst size
tokens: float               # Current available tokens
last_update: float          # Last refill timestamp
lock: threading.Lock        # Thread safety
refill_rate: float          # Tokens per second
```

**Methods**:

##### `__init__(requests_per_minute: int, burst_size: Optional[int] = None)`

**Purpose**: Initialize rate limiter with capacity and refill rate

**Parameters**:
- `requests_per_minute: int` - Maximum requests allowed per minute
- `burst_size: Optional[int] = None` - Maximum burst size (defaults to `requests_per_minute`)

**Design Decision**: Separate burst_size from requests_per_minute
- **Why**: Allow bursts up to capacity, then throttle to sustained rate
- **Example**: `rpm=60, burst=120` allows 120 immediate requests, then throttles to 1/second

**Initialization**:
```python
self.rpm = requests_per_minute
self.capacity = burst_size or requests_per_minute
self.tokens = float(self.capacity)  # Start with full capacity
self.last_update = time.time()
self.lock = threading.Lock()
self.refill_rate = requests_per_minute / 60.0  # Tokens per second
```

##### `acquire(tokens: int = 1, timeout: Optional[float] = None) -> bool`

**Purpose**: Acquire tokens for making requests (blocks until available)

**Algorithm**:
1. Check if requested tokens <= capacity (raise ValueError if not)
2. Loop:
   a. Acquire lock
   b. Refill tokens based on elapsed time
   c. If sufficient tokens available:
      - Deduct tokens
      - Return True
   d. Release lock
   e. Check timeout
   f. Sleep briefly (0.1s) before retry

**Parameters**:
- `tokens: int = 1` - Number of tokens to acquire
- `timeout: Optional[float] = None` - Maximum wait time (None = wait forever)

**Returns**: `bool`
- `True` if tokens acquired
- `False` if timeout reached

**Raises**: `ValueError` if `tokens > capacity`

**Thread Safety**: Lock protects token bucket state during check-and-decrement

**Example**:
```python
limiter = RateLimiter(requests_per_minute=60, burst_size=120)

# Acquire 1 token (default)
if limiter.acquire():
    make_api_call()

# Acquire with timeout
if limiter.acquire(tokens=1, timeout=5.0):
    make_api_call()
else:
    print("Timeout waiting for rate limit")

# Acquire multiple tokens
if limiter.acquire(tokens=5):
    make_batch_api_call(batch_size=5)
```

##### `_refill() -> None`

**Purpose**: Refill tokens based on elapsed time (internal method)

**Algorithm**:
```python
now = time.time()
elapsed = now - self.last_update
tokens_to_add = elapsed * self.refill_rate
self.tokens = min(self.capacity, self.tokens + tokens_to_add)
self.last_update = now
```

**Design Decision**: Continuous refill vs. discrete intervals
- **Chosen**: Continuous refill (calculate tokens based on elapsed time)
- **Why**: More accurate, smoother rate limiting
- **Alternative**: Refill in discrete chunks (simpler but less accurate)

**Thread Safety**: Called only while holding lock

##### `available_tokens` (Property)

**Purpose**: Get current available tokens (thread-safe)

**Returns**: `float` - Current token count

**Thread Safety**: Acquires lock, refills, returns tokens

**Example**:
```python
limiter = RateLimiter(requests_per_minute=60)
print(f"Available: {limiter.available_tokens}")  # e.g., 45.3
```

##### `reset() -> None`

**Purpose**: Reset rate limiter to full capacity

**Thread Safety**: Acquires lock, resets tokens and timestamp

**Use Case**: Manual reset after known idle period

### Token Bucket Algorithm

**Visual**:
```
Bucket (capacity = 100)
┌─────────────┐
│ ▓▓▓▓▓▓▓▓    │  Current: 80 tokens
│             │  Refill rate: 60/minute = 1/second
└─────────────┘

After 5 seconds:
┌─────────────┐
│ ▓▓▓▓▓▓▓▓▓▓▓ │  Current: 85 tokens (80 + 5*1)
└─────────────┘

After acquire(10):
┌─────────────┐
│ ▓▓▓▓▓▓▓▓    │  Current: 75 tokens
└─────────────┘
```

**Properties**:
- Allows bursts up to capacity
- Refills continuously at constant rate
- Smooth rate limiting (no hard boundaries)

### Dependencies

```python
import threading          # Lock for thread safety
import time              # Timestamp tracking
from typing import Optional
```

### Used By

- `LLMInvocationStage` - Rate limit API calls
- Any component needing request throttling

### Thread Safety Details

```python
with self.lock:  # Critical section
    self._refill()
    if self.tokens >= tokens:
        self.tokens -= tokens
        return True
```

**Race Condition Prevention**:
- Without lock: Two threads could both see sufficient tokens, both decrement, exceed capacity
- With lock: Only one thread can check-and-decrement at a time

### Performance

- **Time Complexity**: O(1) per acquire
- **Space Complexity**: O(1) - constant space
- **Lock Contention**: Low (critical section is very short)

### Configuration Examples

```python
# Conservative (prevent rate limits)
limiter = RateLimiter(requests_per_minute=30, burst_size=30)

# Aggressive (maximize throughput)
limiter = RateLimiter(requests_per_minute=100, burst_size=200)

# Bursty workload (allow initial burst, then throttle)
limiter = RateLimiter(requests_per_minute=60, burst_size=180)
```

### Testing

```python
def test_rate_limiter():
    limiter = RateLimiter(requests_per_minute=60)

    # Should acquire immediately
    assert limiter.acquire(timeout=0.1)

    # Deplete tokens
    for _ in range(59):
        limiter.acquire()

    # Should timeout (no tokens left)
    assert not limiter.acquire(timeout=0.1)

    # Wait for refill
    time.sleep(2.0)
    assert limiter.acquire()  # Should have ~2 tokens refilled
```

### Known Limitations

- No distributed rate limiting (single-process only)
- Fixed refill rate (can't adjust dynamically)
- Polling-based waiting (could use condition variables)

### Future Improvements

- [ ] Add distributed rate limiting (Redis-based)
- [ ] Support dynamic rate adjustment
- [ ] Use condition variables instead of polling
- [ ] Add metrics for rate limit hits

---

## 3.4 `utils/cost_tracker.py`

### Purpose
Thread-safe cost tracking for LLM API usage with detailed breakdowns.

### Classes

#### `CostEntry` (Dataclass)

**Purpose**: Single cost tracking entry

**Attributes**:
```python
tokens_in: int      # Input tokens consumed
tokens_out: int     # Output tokens generated
cost: Decimal       # Cost for this entry
model: str          # Model identifier
timestamp: float    # When request occurred
```

**Design Decision**: Use dataclass for simplicity and immutability

#### `CostTracker` (Class)

**Responsibility**: Accumulate and track costs across LLM calls

**Single Responsibility**: ONLY handles cost accounting, not enforcement (that's BudgetController's job)

**Thread Safety**: ✅ **Thread-safe** via `threading.Lock`

**Attributes**:
```python
input_cost_per_1k: Decimal       # Input token price per 1K tokens
output_cost_per_1k: Decimal      # Output token price per 1K tokens
_total_input_tokens: int         # Cumulative input tokens
_total_output_tokens: int        # Cumulative output tokens
_total_cost: Decimal             # Cumulative cost (Decimal for precision)
_entries: list[CostEntry]        # Detailed entry log
_stage_costs: Dict[str, Decimal] # Cost breakdown by stage
_lock: threading.Lock            # Thread safety
```

**Design Decision**: Use `Decimal` for cost (not `float`)
- **Why**: Avoid floating-point precision errors (`0.1 + 0.2 != 0.3`)
- **Critical**: Financial calculations must be exact
- **Trade-off**: Slightly slower than float, but necessary for accuracy

**Methods**:

##### `__init__(input_cost_per_1k, output_cost_per_1k)`

**Purpose**: Initialize cost tracker with pricing

**Parameters**:
- `input_cost_per_1k: Optional[Decimal] = None` - Price per 1K input tokens
- `output_cost_per_1k: Optional[Decimal] = None` - Price per 1K output tokens

**Default**: Both default to `Decimal("0.0")` if not provided

**Initialization**:
```python
self.input_cost_per_1k = input_cost_per_1k or Decimal("0.0")
self.output_cost_per_1k = output_cost_per_1k or Decimal("0.0")
self._total_input_tokens = 0
self._total_output_tokens = 0
self._total_cost = Decimal("0.0")
self._entries = []
self._stage_costs = {}
self._lock = threading.Lock()
```

##### `add(tokens_in, tokens_out, model, timestamp, stage) -> Decimal`

**Purpose**: Add cost entry and return cost for this operation

**Algorithm**:
1. Calculate cost: `(tokens_in/1000) * input_price + (tokens_out/1000) * output_price`
2. Acquire lock
3. Create `CostEntry` and append to `_entries`
4. Accumulate tokens and cost
5. Update stage-specific costs
6. Release lock
7. Return calculated cost

**Parameters**:
- `tokens_in: int` - Input tokens used
- `tokens_out: int` - Output tokens generated
- `model: str` - Model identifier
- `timestamp: float` - Request timestamp
- `stage: Optional[str] = None` - Stage name for breakdown

**Returns**: `Decimal` - Cost for this entry

**Thread Safety**: Entire operation protected by lock to ensure atomic update

**Example**:
```python
tracker = CostTracker(
    input_cost_per_1k=Decimal("0.00005"),  # $0.05 per 1M input tokens
    output_cost_per_1k=Decimal("0.00008"),  # $0.08 per 1M output tokens
)

cost = tracker.add(
    tokens_in=1000,
    tokens_out=500,
    model="gpt-4o-mini",
    timestamp=time.time(),
    stage="llm_invocation"
)
# cost = (1000/1000 * 0.00005) + (500/1000 * 0.00008)
#      = 0.00005 + 0.00004
#      = 0.00009 ($0.00009)
```

##### `calculate_cost(tokens_in, tokens_out) -> Decimal`

**Purpose**: Calculate cost for given token counts (without recording)

**Algorithm**:
```python
input_cost = (Decimal(tokens_in) / 1000) * self.input_cost_per_1k
output_cost = (Decimal(tokens_out) / 1000) * self.output_cost_per_1k
return input_cost + output_cost
```

**Use Case**: Estimate cost before making request

**Example**:
```python
estimated = tracker.calculate_cost(tokens_in=500, tokens_out=200)
if estimated > Decimal("0.01"):  # More than 1 cent
    print("Expensive request!")
```

##### `total_cost` (Property)

**Purpose**: Get total accumulated cost (thread-safe)

**Returns**: `Decimal` - Total cost

**Thread Safety**: Acquires lock, reads `_total_cost`, releases lock

##### `total_tokens`, `input_tokens`, `output_tokens` (Properties)

**Purpose**: Get token counts (thread-safe)

**Returns**: `int` - Token count

##### `get_estimate(rows) -> CostEstimate`

**Purpose**: Create cost estimate object from current tracking

**Returns**: `CostEstimate` model with:
- `total_cost`
- `total_tokens`
- `input_tokens`
- `output_tokens`
- `rows` - Number of rows processed
- `breakdown_by_stage` - Dict of stage costs
- `confidence="actual"` - These are actual costs, not estimates

##### `reset() -> None`

**Purpose**: Clear all tracking data (thread-safe)

**Use Case**: Reset between test runs or pipeline executions

##### `get_stage_costs() -> Dict[str, Decimal]`

**Purpose**: Get cost breakdown by stage

**Returns**: Dictionary mapping stage names to costs

**Example**:
```python
stage_costs = tracker.get_stage_costs()
# {"llm_invocation": Decimal("0.15"), "retry": Decimal("0.02")}
```

### Dependencies

```python
import threading                      # Thread safety
from dataclasses import dataclass     # CostEntry
from decimal import Decimal           # Precise financial calculations
from typing import Dict, Optional
from src.core.models import CostEstimate  # Result model
```

### Used By

- `LLMInvocationStage` - Track costs during execution
- `PipelineExecutor` - Get total costs for result
- `BudgetController` - Check against budget limits

### Thread Safety Example

```python
# Two threads executing concurrently:
# Thread 1: tracker.add(tokens_in=1000, tokens_out=500, ...)
# Thread 2: tracker.add(tokens_in=800, tokens_out=300, ...)

# Without lock: Race condition
# 1. T1 reads _total_cost: 0.00
# 2. T2 reads _total_cost: 0.00
# 3. T1 writes _total_cost: 0.10
# 4. T2 writes _total_cost: 0.08  # LOST UPDATE! T1's cost is lost
# Final: 0.08 (WRONG, should be 0.18)

# With lock: Correct
# 1. T1 acquires lock
# 2. T1 reads _total_cost: 0.00, calculates 0.10, writes 0.10
# 3. T1 releases lock
# 4. T2 acquires lock
# 5. T2 reads _total_cost: 0.10, calculates 0.08, writes 0.18
# 6. T2 releases lock
# Final: 0.18 (CORRECT)
```

### Decimal Precision Example

```python
# Why Decimal is necessary:
from decimal import Decimal

# Float (WRONG):
float_cost = 0.1 + 0.2  # 0.30000000000000004 (WRONG!)

# Decimal (CORRECT):
decimal_cost = Decimal("0.1") + Decimal("0.2")  # 0.3 (CORRECT!)

# Real-world impact:
# If processing 1,000,000 rows with $0.0001 per row:
# Float: $100.00000000014 (accumulated error)
# Decimal: $100.00 (exact)
```

### Testing

```python
def test_cost_tracker():
    tracker = CostTracker(
        input_cost_per_1k=Decimal("0.00005"),
        output_cost_per_1k=Decimal("0.00008"),
    )

    # Add cost
    cost1 = tracker.add(tokens_in=1000, tokens_out=500, model="test", timestamp=0.0)
    assert cost1 == Decimal("0.00009")

    # Check accumulation
    assert tracker.total_cost == Decimal("0.00009")
    assert tracker.total_tokens == 1500

    # Add more
    cost2 = tracker.add(tokens_in=2000, tokens_out=1000, model="test", timestamp=1.0)
    assert tracker.total_cost == Decimal("0.00027")  # 0.00009 + 0.00018
```

### Known Limitations

- Stores all entries in memory (could grow large for long runs)
- No persistence (lost on crash)
- No cost cap enforcement (that's BudgetController's job)

### Future Improvements

- [ ] Add entry pruning for long-running pipelines
- [ ] Optional persistence to disk
- [ ] Add cost forecasting based on trends
- [ ] Support tiered pricing (cost changes with volume)

---

## 3.5 `utils/budget_controller.py`

### Purpose
Enforce budget limits and provide warnings during execution.

### Classes

#### `BudgetExceededError` (Exception)

**Inheritance**: `Exception`

**Purpose**: Raised when budget limit is exceeded

**Usage**: Allows caller to catch and handle budget overruns

####

 `BudgetController` (Class)

**Responsibility**: Monitor costs and enforce budget limits

**Single Responsibility**: ONLY handles budget management (cost tracking is CostTracker's job)

**Separation of Concerns**:
- `CostTracker`: Accumulates costs
- `BudgetController`: Enforces limits

**Attributes**:
```python
max_budget: Optional[Decimal]  # Maximum allowed budget (None = no limit)
warn_at_75: bool              # Warn at 75% of budget
warn_at_90: bool              # Warn at 90% of budget
fail_on_exceed: bool          # Raise error if budget exceeded
_warned_75: bool              # Track if 75% warning already shown
_warned_90: bool              # Track if 90% warning already shown
```

**Design Decision**: Separate warning flags from budget
- **Why**: Show each warning only once
- **Alternative**: Check every time (would spam logs)

**Methods**:

##### `__init__(max_budget, warn_at_75, warn_at_90, fail_on_exceed)`

**Purpose**: Initialize budget controller with limits

**Parameters**:
- `max_budget: Optional[Decimal] = None` - Maximum budget in USD
- `warn_at_75: bool = True` - Warn at 75% usage
- `warn_at_90: bool = True` - Warn at 90% usage
- `fail_on_exceed: bool = True` - Raise error when exceeded

**Default Behavior**: Warn twice, then fail

**Flexible Configuration**:
```python
# Strict (default): Fail on exceed
controller = BudgetController(max_budget=Decimal("10.0"))

# Warn only: Don't fail
controller = BudgetController(max_budget=Decimal("10.0"), fail_on_exceed=False)

# No warnings: Just hard limit
controller = BudgetController(
    max_budget=Decimal("10.0"),
    warn_at_75=False,
    warn_at_90=False,
)
```

##### `check_budget(current_cost: Decimal) -> None`

**Purpose**: Check if cost is within budget, warn or fail as configured

**Algorithm**:
1. If `max_budget` is None, return (no limit)
2. Calculate usage ratio: `current_cost / max_budget`
3. If >= 0.75 and not warned: Log warning, set `_warned_75 = True`
4. If >= 0.90 and not warned: Log warning, set `_warned_90 = True`
5. If > `max_budget`:
   - Log error
   - If `fail_on_exceed`: raise `BudgetExceededError`

**Parameters**:
- `current_cost: Decimal` - Current accumulated cost

**Raises**: `BudgetExceededError` if cost exceeds budget and `fail_on_exceed=True`

**Example**:
```python
controller = BudgetController(max_budget=Decimal("10.0"))

# Check after each operation
controller.check_budget(Decimal("5.0"))   # OK
controller.check_budget(Decimal("7.5"))   # Logs: "75% used"
controller.check_budget(Decimal("9.0"))   # Logs: "90% used"
controller.check_budget(Decimal("10.5"))  # Raises BudgetExceededError
```

**Design Decision**: Warnings at 75% and 90%
- **Why**: Give user time to react before hitting limit
- **75%**: Early warning
- **90%**: Final warning before failure
- **Alternative**: More granular warnings (every 10%) would spam logs

##### `get_remaining(current_cost: Decimal) -> Optional[Decimal]`

**Purpose**: Calculate remaining budget

**Returns**:
- `Decimal` - Remaining budget
- `None` if no budget limit set

**Example**:
```python
controller = BudgetController(max_budget=Decimal("10.0"))
remaining = controller.get_remaining(Decimal("7.5"))
# remaining = Decimal("2.5")
```

##### `get_usage_percentage(current_cost: Decimal) -> Optional[float]`

**Purpose**: Calculate budget usage as percentage

**Returns**:
- `float` - Usage percentage (0-100+)
- `None` if no budget limit set

**Example**:
```python
controller = BudgetController(max_budget=Decimal("10.0"))
pct = controller.get_usage_percentage(Decimal("7.5"))
# pct = 75.0
```

##### `can_afford(estimated_cost: Decimal, current_cost: Decimal) -> bool`

**Purpose**: Check if estimated additional cost fits within budget

**Algorithm**:
```python
if self.max_budget is None:
    return True
return (current_cost + estimated_cost) <= self.max_budget
```

**Use Case**: Pre-flight check before expensive operation

**Example**:
```python
controller = BudgetController(max_budget=Decimal("10.0"))
current = Decimal("9.0")
estimated_next = Decimal("0.5")

if controller.can_afford(estimated_next, current):
    proceed_with_operation()
else:
    print("Would exceed budget, skipping")
```

### Dependencies

```python
from decimal import Decimal       # Precise financial calculations
from typing import Optional
import structlog                  # Structured logging
```

### Used By

- `PipelineExecutor` - Check budget during execution
- Any component needing cost enforcement

### Integration with CostTracker

```python
# Typical usage pattern:
cost_tracker = CostTracker(...)
budget_controller = BudgetController(max_budget=Decimal("10.0"))

for row in data:
    # Make LLM call
    response = llm.invoke(prompt)

    # Track cost
    cost = cost_tracker.add(tokens_in=..., tokens_out=...)

    # Check budget
    budget_controller.check_budget(cost_tracker.total_cost)
```

### Testing

```python
def test_budget_warnings():
    controller = BudgetController(max_budget=Decimal("10.0"))

    # Should not warn
    controller.check_budget(Decimal("5.0"))

    # Should warn at 75%
    controller.check_budget(Decimal("7.5"))

    # Should warn at 90%
    controller.check_budget(Decimal("9.0"))

    # Should raise error
    with pytest.raises(BudgetExceededError):
        controller.check_budget(Decimal("10.1"))
```

### Error Messages

```python
# 75% warning:
"Budget warning: 75% used ($7.50 / $10.00)"

# 90% warning:
"Budget warning: 90% used ($9.00 / $10.00)"

# Budget exceeded:
"Budget exceeded: $10.50 > $10.00"
```

### Known Limitations

- No budget rollover (each execution is independent)
- No budget sharing across pipelines
- Fixed warning thresholds (75%, 90%)

### Future Improvements

- [ ] Configurable warning thresholds
- [ ] Support budget pools (shared across pipelines)
- [ ] Budget rollover support
- [ ] Budget forecasting (estimate when limit will be hit)

---

*This is Part 1 of the Technical Reference. Continue reading for Layer 0 (remaining utils), Core Models, and all other layers...*

---

# Part 5: Layer 1 - Infrastructure Adapters (LLM Providers)

## 5.1 LLM Provider Overview

Hermes supports multiple LLM providers through the **Adapter pattern**, allowing easy switching between providers without changing core logic.

### Supported Providers

| Provider | Category | Platform | Cost | Use Case |
|----------|----------|----------|------|----------|
| **OpenAI** | Cloud API | All | $$ | Production, high quality |
| **Azure OpenAI** | Cloud API | All | $$ | Enterprise, compliance |
| **Anthropic** | Cloud API | All | $$$ | Claude models, long context |
| **Groq** | Cloud API | All | Free tier | Fast inference, development |
| **MLX** | Local | macOS (Apple Silicon) | Free | Privacy, offline, no costs |

### Provider Selection Guide

**Choose OpenAI if**: Production quality, mature ecosystem, GPT-4
**Choose Azure if**: Enterprise compliance, private deployments
**Choose Anthropic if**: Claude models, 100K+ context
**Choose Groq if**: Fast inference, free tier, development
**Choose MLX if**: Apple Silicon Mac, privacy, free, offline capable

---

## 5.2 OpenAI-Compatible Provider (Custom APIs)

### Purpose
Enable integration with any LLM API that implements the OpenAI chat completions format.

### Class: `OpenAICompatibleClient`

**Inheritance**: `LLMClient` (Adapter pattern)

**Platform**: All (platform-agnostic)

**Responsibility**: Connect to custom OpenAI-compatible API endpoints

**Supports**:
- ✅ **Ollama** (local LLM server)
- ✅ **vLLM** (self-hosted inference)
- ✅ **Together.AI** (cloud API)
- ✅ **Anyscale** (cloud API)
- ✅ **LocalAI** (self-hosted)
- ✅ **Any custom OpenAI-compatible API**

### Configuration

**Required Fields**:
- `provider: openai_compatible`
- `base_url`: Custom API endpoint URL
- `model`: Model identifier

**Optional Fields**:
- `provider_name`: Custom name for logging/metrics
- `api_key`: Authentication (optional for local)
- `input_cost_per_1k_tokens`: Custom pricing
- `output_cost_per_1k_tokens`: Custom pricing

### Design Decision: Reuse OpenAI Client

Instead of reimplementing HTTP, leverage llama-index's OpenAI client:

```python
self.client = OpenAI(
    model=spec.model,
    api_key=api_key or "dummy",
    api_base=spec.base_url,  # Custom URL
)
```

**Rationale**:
- ✅ DRY: Reuse well-tested code
- ✅ Reliability: Edge cases handled
- ✅ Compatibility: Ensures OpenAI format
- ✅ Maintainability: Benefit from upstream updates

### Example: Together.AI
```yaml
llm:
  provider: openai_compatible
  provider_name: "Together.AI"
  model: meta-llama/Llama-3.1-70B
  base_url: https://api.together.xyz/v1
  api_key: \${TOGETHER_API_KEY}
  input_cost_per_1k_tokens: 0.0006
```

### Validation

```python
@model_validator(mode="after")
def validate_provider_requirements(self):
    if self.provider == OPENAI_COMPATIBLE and not self.base_url:
        raise ValueError("base_url required")
    return self
```

---

## 5.3 MLX Provider (Apple Silicon)

### Purpose
Enable fast, free, local LLM inference on Apple Silicon using Apple's MLX framework.

### Class: `MLXClient`

**Inheritance**: `LLMClient` (Adapter pattern)

**Platform**: macOS with M1/M2/M3/M4 chips only

**Responsibility**: Local in-process LLM inference using MLX framework

**Key Features**:
- ✅ In-process (no server management)
- ✅ Model caching (load once, use many times)
- ✅ Lazy imports (only when MLX provider used)
- ✅ Dependency injection (clean testing)
- ✅ Free inference ($0 cost)

### Architecture

```python
class MLXClient(LLMClient):
    def __init__(self, spec: LLMSpec, _mlx_lm_module=None):
        # Lazy import with helpful error
        # Load model once (cached in instance)

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        # Use cached model for inference
        # No server calls, all local
```

**Design Decision: Dependency Injection**

The `_mlx_lm_module` parameter enables clean testing:

```python
# Production usage (normal)
client = MLXClient(spec)  # Imports mlx_lm automatically

# Testing usage (mocked)
mock_mlx = MagicMock()
client = MLXClient(spec, _mlx_lm_module=mock_mlx)  # Inject mock
```

**Rationale**:
- ✅ Testable: No sys.modules hacks
- ✅ Clear: Explicit what's being injected
- ✅ Backward compatible: Parameter is optional

### Model Caching Strategy

**Problem**: Loading MLX models is expensive (~0.5-2 seconds)

**Solution**: Load once in `__init__()`, cache in instance

**Impact**:
- First invocation: ~0.7s (load) + ~0.7s (inference) = 1.4s
- Subsequent calls: ~0.7s (inference only)
- For 100 rows: Save ~70 seconds vs reload each time!

### Token Estimation

**Primary**: Use MLX tokenizer (model-specific, accurate)
```python
tokens = len(self.mlx_tokenizer.encode(text))
```

**Fallback**: Word count (if tokenizer fails)
```python
tokens = len(text.split())
```

**Rationale**: Graceful degradation, never fail on token estimation

### Error Handling

**Import Error**:
```
ImportError: MLX not installed. Install with:
  pip install hermes[mlx]
or:
  pip install mlx mlx-lm

Note: MLX only works on Apple Silicon (M1/M2/M3/M4 chips)
```

**Model Load Error**:
```
Exception: Failed to load MLX model 'model-name'.
Ensure the model exists on HuggingFace and you have access.
Error: [original error]
```

**Design**: Actionable error messages with context

### Performance Characteristics

**Tested on Apple Silicon (M2)**:
- Model: `mlx-community/Qwen3-1.7B-4bit`
- Load time: 0.74s (once per pipeline)
- Inference: 0.67s/prompt
- Throughput: 1.49 prompts/sec
- Memory: ~2GB (model in RAM)

**Comparison**:
- **vs Cloud APIs**: 10x faster (no network), free
- **vs vLLM**: Simpler (no server), Mac-compatible
- **vs Ollama**: Similar speed, more control

### Thread Safety

- ⚠️ **Not thread-safe**: MLX models are not thread-safe
- **Recommendation**: Use `concurrency=1` in ProcessingSpec
- **Why**: MLX optimized for Apple Neural Engine (single-threaded)

### Dependencies

```python
import mlx_lm  # Lazy imported
# mlx_lm depends on:
# - mlx>=0.29.0
# - mlx-metal (GPU acceleration)
# - transformers (HuggingFace)
```

### Used By

- `create_llm_client()` factory
- Any pipeline with `provider: mlx`
- Examples: `10_mlx_qwen3_local.py`, `10_mlx_qwen3_local.yaml`

### Testing Strategy

**Dependency Injection Pattern**:
```python
# Create mock module
mock_mlx = MagicMock()
mock_mlx.load.return_value = (mock_model, mock_tokenizer)
mock_mlx.generate.return_value = "response"

# Inject for testing
client = MLXClient(spec, _mlx_lm_module=mock_mlx)
```

**Coverage**:
- 14 unit tests
- Model loading/caching
- Token estimation with fallback
- Error handling
- Factory integration

### Known Limitations

- macOS only (MLX doesn't work on Linux/Windows)
- Single-threaded (not optimized for concurrency)
- Model download required (1-5GB per model)
- Requires HuggingFace token for some models

### Future Improvements

- [ ] Add MLX streaming support
- [ ] Support custom MLX generation parameters
- [ ] Add model warmup option
- [ ] Cache models globally (shared across pipelines)

---

## 5.4 LLM Provider Presets

### Purpose
Simplify LLM provider configuration by providing pre-configured specifications for common providers, eliminating boilerplate and configuration errors.

### Class: `LLMProviderPresets`

**Location**: `hermes/core/specifications.py` (lines 301-453)

**Responsibility**: Provide pre-validated LLMSpec instances for popular providers

**Design Pattern**: Static Registry Pattern

**Key Features**:
- ✅ Zero boilerplate for common providers (80% code reduction)
- ✅ Pre-validated configurations (correct URLs, pricing)
- ✅ No hardcoded API keys (security by design)
- ✅ IDE autocomplete support
- ✅ Pydantic validation throughout

### Available Presets

#### OpenAI Presets

```python
GPT4O_MINI = LLMSpec(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    input_cost_per_1k_tokens=Decimal("0.00015"),
    output_cost_per_1k_tokens=Decimal("0.0006"),
)

GPT4O = LLMSpec(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    input_cost_per_1k_tokens=Decimal("0.0025"),
    output_cost_per_1k_tokens=Decimal("0.01"),
)
```

#### Together.AI Presets

```python
TOGETHER_AI_LLAMA_70B = LLMSpec(
    provider=LLMProvider.OPENAI_COMPATIBLE,
    provider_name="Together.AI",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    base_url="https://api.together.xyz/v1",
    input_cost_per_1k_tokens=Decimal("0.0006"),
)

TOGETHER_AI_LLAMA_8B = LLMSpec(
    provider=LLMProvider.OPENAI_COMPATIBLE,
    provider_name="Together.AI",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    base_url="https://api.together.xyz/v1",
    input_cost_per_1k_tokens=Decimal("0.0001"),
)
```

#### Ollama Local Presets (Free)

```python
OLLAMA_LLAMA_70B = LLMSpec(
    provider=LLMProvider.OPENAI_COMPATIBLE,
    provider_name="Ollama-Local",
    model="llama3.1:70b",
    base_url="http://localhost:11434/v1",
    input_cost_per_1k_tokens=Decimal("0.0"),  # FREE!
    output_cost_per_1k_tokens=Decimal("0.0"),
)
```

#### Groq & Anthropic Presets

```python
GROQ_LLAMA_70B = LLMSpec(
    provider=LLMProvider.GROQ,
    model="llama-3.1-70b-versatile",
    input_cost_per_1k_tokens=Decimal("0.00059"),
)

CLAUDE_SONNET_4 = LLMSpec(
    provider=LLMProvider.ANTHROPIC,
    model="claude-sonnet-4-20250514",
    max_tokens=8192,
    input_cost_per_1k_tokens=Decimal("0.003"),
)
```

### Factory Method: `create_custom_openai_compatible()`

**Purpose**: Simplify custom provider configuration (vLLM, LocalAI, etc.)

```python
@classmethod
def create_custom_openai_compatible(
    cls,
    provider_name: str,
    model: str,
    base_url: str,
    input_cost_per_1k: float = 0.0,
    output_cost_per_1k: float = 0.0,
    **kwargs
) -> LLMSpec:
    """Factory for custom OpenAI-compatible providers."""
    return LLMSpec(
        provider=LLMProvider.OPENAI_COMPATIBLE,
        provider_name=provider_name,
        model=model,
        base_url=base_url,
        input_cost_per_1k_tokens=Decimal(str(input_cost_per_1k)),
        output_cost_per_1k_tokens=Decimal(str(output_cost_per_1k)),
        **kwargs
    )
```

### Usage with `PipelineBuilder.with_llm_spec()`

**New Method**: `with_llm_spec(spec: LLMSpec) -> PipelineBuilder`

**Location**: `hermes/api/pipeline_builder.py` (lines 260-311)

**Purpose**: Accept pre-built LLMSpec objects instead of individual parameters

**Example Usage**:

```python
from ceres.core.specifications import LLMProviderPresets

# Simple preset usage
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm_spec(LLMProviderPresets.TOGETHER_AI_LLAMA_70B)
    .build()
)

# Override preset settings
custom = LLMProviderPresets.GPT4O_MINI.model_copy(
    update={"temperature": 0.9, "max_tokens": 500}
)
pipeline.with_llm_spec(custom)

# Custom provider via factory
custom_vllm = LLMProviderPresets.create_custom_openai_compatible(
    provider_name="My vLLM Server",
    model="mistral-7b-instruct",
    base_url="http://my-server:8000/v1",
    temperature=0.7
)
pipeline.with_llm_spec(custom_vllm)
```

### Comparison: Before vs After

**Before (parameter-based)**:
```python
.with_llm(
    provider="openai_compatible",
    provider_name="Together.AI",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    base_url="https://api.together.xyz/v1",
    api_key="${TOGETHER_API_KEY}",
    input_cost_per_1k_tokens=0.0006,
    output_cost_per_1k_tokens=0.0006
)
```

**After (preset-based)**:
```python
.with_llm_spec(LLMProviderPresets.TOGETHER_AI_LLAMA_70B)
```

**Result**: 80% code reduction, zero configuration errors

### Design Decisions

**Why Static Class Instead of Enum?**
- Allows class methods (factory)
- Better IDE autocomplete
- Can include documentation in docstrings
- Easier to extend without breaking existing code

**Why No Hardcoded API Keys?**
- Security: API keys should never be in code
- All presets have `api_key=None` by default
- Users must provide via environment variables or `model_copy(update={"api_key": "..."})`

**Why Pydantic `model_copy()` for Overrides?**
- Immutability: Original presets unchanged
- Type safety: Validation on override
- Pythonic: Standard Pydantic pattern
- Flexible: Override any field

### Security Validation

**Test**: All presets must have `api_key=None`
```python
def test_presets_have_no_api_keys(self):
    """Security requirement: No hardcoded API keys."""
    presets = [
        LLMProviderPresets.GPT4O_MINI,
        LLMProviderPresets.TOGETHER_AI_LLAMA_70B,
        # ... all presets
    ]
    for preset in presets:
        assert preset.api_key is None
```

### Backward Compatibility

**100% backward compatible**: Existing `with_llm()` method unchanged
```python
# Old way still works
pipeline.with_llm(provider="openai", model="gpt-4o-mini")

# New way
pipeline.with_llm_spec(LLMProviderPresets.GPT4O_MINI)

# Both methods can be mixed (last call wins)
```

### Testing Coverage

**26 unit tests** (100% pass):
- Preset configuration validation
- Security checks (no API keys)
- Type safety verification
- `with_llm_spec()` method tests
- Override via `model_copy()` tests
- Factory method tests
- Backward compatibility tests

### Examples

**Complete example**: `examples/14_provider_presets.py`
- Demonstrates all preset usage patterns
- Shows customization via `model_copy()`
- Compares old vs new approach
- Lists all available presets

### Future Improvements

- [ ] Add more provider presets (Cohere, AI21, Mistral)
- [ ] Add model size variants (405B, 8B, etc.)
- [ ] Version-specific presets with deprecation warnings
- [ ] Auto-update pricing from provider APIs
- [ ] YAML preset files for user-defined presets

---

---

## 5.6 Observability Module

### Purpose
Provide distributed tracing with OpenTelemetry for production debugging and performance monitoring.

### Module: `hermes/observability/`

**Location**: `hermes/observability/` (4 files, 140 lines)

**Responsibility**: Optional distributed tracing via OpenTelemetry

**Key Features**:
- Opt-in tracing (disabled by default)
- PII-safe by default (prompts sanitized)
- Console & Jaeger exporters
- Per-stage latency tracking
- LLM token/cost tracking (ready for Phase 3)

### Classes

#### `TracingObserver` (Class)

**Inheritance**: `ExecutionObserver`

**Responsibility**: Create OpenTelemetry spans for pipeline execution

**Pattern**: Observer Pattern (non-invasive instrumentation)

**Attributes**:
```python
_include_prompts: bool           # If True, include prompts in spans (PII risk)
_spans: dict[str, trace.Span]    # Active spans by name
```

**Methods**:
- `on_pipeline_start()` - Create root span
- `on_stage_start()` - Create nested stage span
- `on_stage_complete()` - Close span with success metrics
- `on_stage_error()` - Close span with error details
- `on_pipeline_complete()` - Close root span
- `on_pipeline_error()` - Close root span with error

**Span Hierarchy**:
```
pipeline.execute (root)
├── stage.DataLoader
├── stage.PromptFormatter
├── stage.LLMInvocation
├── stage.ResponseParser
└── stage.ResultWriter
```

### Functions

#### `enable_tracing(exporter, endpoint, service_name)`

**Purpose**: Enable distributed tracing (opt-in)

**Parameters**:
- `exporter: str = "console"` - Exporter type (console or jaeger)
- `endpoint: str | None = None` - Jaeger endpoint URL
- `service_name: str = "hermes-pipeline"` - Service name for traces

**Example**:

```python
from ceres.observability import enable_tracing

# Console (development)
enable_tracing(exporter="console")

# Jaeger (production)
enable_tracing(exporter="jaeger", endpoint="http://localhost:14268/api/traces")
```

#### `disable_tracing()`

**Purpose**: Disable tracing and cleanup resources

#### `is_tracing_enabled() -> bool`

**Purpose**: Check if tracing is currently enabled

### PII Sanitization

#### `sanitize_prompt(prompt, include_prompts) -> str`

**Purpose**: Sanitize prompt text (hash by default)

**Algorithm**:
```python
if include_prompts:
    return prompt  # Opt-in: include actual prompt
else:
    return f"<sanitized-{hash(prompt) % 10000}>"  # Default: hash only
```

**Design Decision**: PII-safe by default
- Users must explicitly opt-in to include prompts
- Prevents accidental PII exposure in traces
- Hash allows duplicate detection without exposing content

### Dependencies

```python
opentelemetry-api>=1.20.0        # Tracing API
opentelemetry-sdk>=1.20.0        # SDK implementation
opentelemetry-exporter-jaeger>=1.20.0  # Jaeger export
```

**Installation**: `pip install hermes[observability]`

### Graceful Degradation

If OpenTelemetry not installed:

```python
from ceres.observability import enable_tracing

enable_tracing()  # Raises helpful ImportError with install instructions
is_tracing_enabled()  # Returns False (always)
```

### Thread Safety

- ✅ **Thread-safe**: Yes (OpenTelemetry handles concurrency)
- Span creation/completion uses OpenTelemetry's thread-local context

### Performance

- **Overhead**: <2% (measured with 10K row pipeline)
- **Span creation**: ~1-2ms per span
- **Export**: Async batch processing (non-blocking)

### Testing Coverage

**14 unit tests** (100% passing):
- Tracing enable/disable
- PII sanitization
- Observer integration
- Export failure handling
- Span lifecycle

**Integration tests**: Ready for Phase 3 (LLM instrumentation)

### Usage Example

```python
from ceres import PipelineBuilder
from ceres.observability import enable_tracing

# Enable tracing
enable_tracing(exporter="console")

# Build and execute pipeline (traces automatically created)
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)

result = pipeline.execute()  # Traces exported
```

### Future Enhancements (Phase 3)

- [ ] Instrument `LLMClient.invoke()` for LLM call tracing
- [ ] Add OTLP exporter (modern alternative to Jaeger)
- [ ] Add metrics integration
- [ ] Performance profiling

---

**Document Status**: 🚧 IN PROGRESS (Layer 0 complete, Layer 1 documented with MLX, Presets, and Observability)

**Next Sections**:
- 3.6 `utils/logging_utils.py`
- 3.7 `utils/metrics_exporter.py`
- 3.8 `utils/input_preprocessing.py`
- Part 4: Core Models & Specifications
- Part 5: Complete Layer 1 (remaining providers)
- Part 6+: Remaining layers...
