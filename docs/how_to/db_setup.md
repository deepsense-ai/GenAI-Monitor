# Database and Persistence Configuration

## Overview

GenAI Monitor stores both structured data (in a SQLite database) and unstructured data (model inputs/outputs and other artifacts) on disk. This documentation explains how to configure these storage mechanisms using environment variables.

## Configuration Options

GenAI Monitor uses two main environment variables to configure data storage:

1. `GENAI_MONITOR_DB_URL`: Defines the SQLite database location
2. `GENAI_MONITOR_PERSISTENCY_PATH`: Specifies the directory where unstructured data is stored

## Setting Up the Database

### SQLite Database URL

The `GENAI_MONITOR_DB_URL` environment variable controls where your SQLite database is stored. This database contains metadata about model calls, relationships between artifacts, and other structured information.

```bash
# Example: Store the database in a specific location
export GENAI_MONITOR_DB_URL="sqlite:///path/to/your/genai_monitor.db"

# Default: If not specified, a default location (current working directory) will be used
```

### Persistence Path for Unstructured Data

The `GENAI_MONITOR_PERSISTENCY_PATH` environment variable specifies the absolute path to a directory where all large objects (like model inputs, outputs, embeddings, and other artifacts) will be stored. This separation allows for efficient database operations while still preserving all valuable data.

```bash
# Example: Store persistent objects in a specific directory
export GENAI_MONITOR_PERSISTENCY_PATH="/path/to/your/persistency/directory"
```

## Initialization Process

When you first import GenAI Monitor, the system performs several setup operations:

1. The database is initialized according to `GENAI_MONITOR_DB_URL` (or using the default location if not specified)
2. The persistence path from `GENAI_MONITOR_PERSISTENCY_PATH` is recorded in the database
3. In subsequent runs, the persistence path is read from the database, ensuring consistency

```python
# First import triggers initialization
import genai_monitor.auto

# Now GenAI Monitor is initialized with your configured storage locations
```