"""HTTP server example.

This spins up the FastAPI app and starts serving on 127.0.0.1:8765.
The server is optional — install with:

    pip install herding-cats[server]

Then run:

    python examples/server_demo.py

In another terminal, try:

    # one-shot
    curl -X POST http://127.0.0.1:8765/crew/run \\
         -H 'Content-Type: application/json' \\
         -d '{"question":"What is 2 + 2?"}'

    # streaming
    curl -N -X POST http://127.0.0.1:8765/crew/stream \\
         -H 'Content-Type: application/json' \\
         -d '{"question":"What is the meaning of life?"}'

    # session
    curl -X POST http://127.0.0.1:8765/sessions/myid/ask \\
         -H 'Content-Type: application/json' \\
         -d '{"question":"Hello"}'
    curl -X POST http://127.0.0.1:8765/sessions/myid/ask \\
         -H 'Content-Type: application/json' \\
         -d '{"question":"And tomorrow?"}'
    curl http://127.0.0.1:8765/sessions/myid
"""

from __future__ import annotations

import sys

from herding_cats import Crew, OllamaClient
from herding_cats.tools_builtin import builtin_tools

try:
    import uvicorn
except ImportError:
    print(
        "This example requires the 'server' extra. Install with:\n"
        "  pip install herding-cats[server]",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    ollama = OllamaClient()
    if not ollama.health():
        print(
            "Ollama not reachable. Start it with `docker compose up -d`.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Use the built-in tool registry (with pydantic schemas).
    specs = builtin_tools()
    crew = Crew(
        ollama=ollama,
        tool_specs=specs,
    )

    from herding_cats.server import create_app

    app = create_app(crew=crew)

    print("Starting herding-cats HTTP server on 127.0.0.1:8765")
    print("OpenAPI docs: http://127.0.0.1:8765/docs")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
