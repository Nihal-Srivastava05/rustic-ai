# Execution

Execution in Rustic AI Core is managed by execution engines, which control how agents are run, scheduled, and coordinated. This enables flexible deployment, from simple synchronous runs to advanced multithreaded or distributed setups.

## Purpose
- Manage the lifecycle and scheduling of agents
- Support different execution models (sync, multithreaded, distributed)
- Integrate with messaging and state management
- Provide agent tracking and monitoring capabilities
- Handle graceful shutdown and resource cleanup

## Execution Engines

Rustic AI provides three built-in execution engines and supports custom implementations:

- **SyncExecutionEngine**: Runs agents synchronously in the main thread/process.
- **MultiThreadedEngine**: Runs agents in separate threads for concurrency.
- **RayExecutionEngine**: Runs agents as distributed Ray actors for scalable, distributed execution.
- **Custom Engines**: Extendable for specialized execution models.

| Engine | Concurrency Model | Suitable For | Key Features |
|--------|------------------|--------------|-------------|
| `SyncExecutionEngine` | Single-thread | Tutorials, deterministic tests, debugging | Simple, predictable execution order |
| `MultiThreadedEngine` | Thread-per-agent | IO-bound tasks, WebSocket bots, concurrent processing | Thread-safe agent tracking |
| `RayExecutionEngine` | Distributed actors | CPU-heavy workloads, distributed systems, scalable deployments | Cross-machine execution, fault tolerance |
| *Custom* | User-defined | Specialized workloads | Implement `ExecutionEngine` interface |

### SyncExecutionEngine
```python
from rustic_ai.core.guild.execution import SyncExecutionEngine

# Default execution - runs agents sequentially in main thread
engine = SyncExecutionEngine(guild_id="my-guild")
```

**Key characteristics:**
- Uses `SyncAgentWrapper` for direct execution
- Employs `InMemorySyncAgentTracker` for agent management
- Ideal for development, testing, and simple workflows

### MultiThreadedEngine
```python
from rustic_ai.core.guild.execution.multithreaded import MultiThreadedEngine

# Concurrent execution - each agent runs in its own thread
engine = MultiThreadedEngine(guild_id="my-guild")
```

**Key characteristics:**
- Uses `MultiThreadedAgentWrapper` with separate threads
- Employs `InMemoryMTAgentTracker` (thread-safe) for agent management
- Suitable for IO-bound operations and concurrent processing

### RayExecutionEngine
```python
from rustic_ai.ray import RayExecutionEngine
import ray

# Initialize Ray cluster first
ray.init()

# Distributed execution - agents run as Ray actors
engine = RayExecutionEngine(guild_id="my-guild")
```

**Key characteristics:**
- Uses `RayAgentWrapper` decorated with `@ray.remote`
- Agents run as named Ray actors with namespace isolation
- Supports distributed execution across multiple machines
- Includes built-in observability and tracing setup

## Agent Wrappers

Agent wrappers encapsulate the logic for initializing, running, and shutting down agents within an execution engine. All wrappers inherit from the base `AgentWrapper` class.

### Common Wrapper Functionality
- **Dependency injection**: Resolves and injects agent dependencies
- **Messaging client setup**: Configures messaging clients and subscriptions
- **State and guild context**: Provides access to guild specifications and state
- **Resource management**: Handles initialization and cleanup

### Wrapper Types
- **SyncAgentWrapper**: Executes `initialize_agent()` directly in the current thread
- **MultiThreadedAgentWrapper**: Starts a new thread running `initialize_agent()`
- **RayAgentWrapper**: Runs as a Ray actor with distributed execution capabilities

## Configuration and Usage

### Default Engine Selection
The default execution engine can be configured via environment variable or guild properties:

```python
# Via environment variable
export RUSTIC_AI_EXECUTION_ENGINE="rustic_ai.core.guild.execution.multithreaded.MultiThreadedEngine"

# Via guild properties
guild_spec.properties["execution_engine"] = "rustic_ai.core.guild.execution.sync.SyncExecutionEngine"
```

### Example: Running Agents with Different Engines
```python
from rustic_ai.core.guild import AgentBuilder, Guild
from rustic_ai.core.guild.execution import SyncExecutionEngine
from rustic_ai.core.guild.execution.multithreaded import MultiThreadedEngine

# Create a guild and agent spec
guild = Guild(...)
agent_spec = AgentBuilder(...).set_name("Agent1").set_description("...").build_spec()

# Option 1: Use default execution engine
guild.launch_agent(agent_spec)

# Option 2: Use specific execution engine
sync_engine = SyncExecutionEngine(guild_id=guild.id)
guild.launch_agent(agent_spec, execution_engine=sync_engine)

# Option 3: Use multithreaded engine
mt_engine = MultiThreadedEngine(guild_id=guild.id)
guild.launch_agent(agent_spec, execution_engine=mt_engine)
```

## Agent Lifecycle Management

### Agent Tracking
All execution engines provide methods for tracking and managing running agents:

```python
# Check if agent is running
is_running = engine.is_agent_running(guild_id, agent_id)

# Get all agents in guild
agents = engine.get_agents_in_guild(guild_id)

# Find agents by name
matching_agents = engine.find_agents_by_name(guild_id, "MyAgent")

# Stop specific agent
engine.stop_agent(guild_id, agent_id)
```

### Graceful Shutdown
All engines respect graceful stop semantics:

1. **Stop Request**: Call `guild.shutdown()` or `engine.shutdown()`
2. **Agent Cleanup**: Each agent's wrapper handles resource cleanup
3. **Messaging Cleanup**: Unsubscribe from topics and unregister clients
4. **Engine Cleanup**: Engine-specific cleanup (thread joining, Ray actor termination)

## Advanced Topics

### Extending Execution Engines
To create a custom execution engine, implement the `ExecutionEngine` abstract base class:

```python
from rustic_ai.core.guild.execution.execution_engine import ExecutionEngine

class CustomExecutionEngine(ExecutionEngine):
    def __init__(self, guild_id: str):
        super().__init__(guild_id=guild_id)
        # Custom initialization
    
    def run_agent(self, guild_spec, agent_spec, messaging_config, machine_id, **kwargs):
        # Custom agent execution logic
        pass
    
    def get_agents_in_guild(self, guild_id: str):
        # Return running agents
        pass
    
    # Implement other required methods...
```

### Error Handling and Observability
- **Ray Integration**: RayExecutionEngine includes OpenTelemetry tracing setup
- **Logging**: All engines provide structured logging for agent lifecycle events
- **Exception Handling**: Proper error propagation and cleanup on failures

### Performance Considerations

| Scenario | Recommended Engine | Reasoning |
|----------|-------------------|-----------|
| Development/Testing | `SyncExecutionEngine` | Predictable, debuggable execution |
| IO-bound applications | `MultiThreadedEngine` | Concurrent processing without GIL issues |
| CPU-intensive workloads | `RayExecutionEngine` | True parallelism across cores/machines |
| Distributed systems | `RayExecutionEngine` | Built-in fault tolerance and scaling |

> See the [Guilds](guilds.md) and [Agents](agents.md) sections for how execution integrates with agent and guild lifecycles. 