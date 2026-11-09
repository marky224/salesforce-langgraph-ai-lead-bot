"""
LangGraph Studio entrypoint: the TARS graph for ``langgraph dev`` (dev only).

``langgraph.json`` at the repo root points its ``tars`` graph at ``graph`` below,
so ``langgraph dev`` can load it and LangGraph Studio can step through it (see
the README's "Debug in LangGraph Studio").  Nothing in the app imports this
module and the production image never runs it; ``server.py`` still builds its
own graph in the lifespan.

On import it does what the server's lifespan does for the graph, minus the parts
``langgraph dev`` owns or the server adds around a run:

- The LLM is built from settings (the env ``langgraph.json`` loads) and injected
  with ``set_llm``, so a missing provider key fails here, at graph load.
- The graph is compiled with no checkpointer: ``langgraph dev`` supplies its own
  persistence and refuses a graph compiled with one (``build_graph`` would
  otherwise attach a ``MemorySaver``).
- No masked tracer is attached, so if the env turns LangSmith tracing on, runs
  are traced unmasked (unlike ``server._run_config``).
- The site index is not loaded, so a site question gets the fixed
  ``SITE_UNAVAILABLE_REPLY``.
- ``salesforce_node`` is the real one: a run that reaches an accepted
  confirmation writes a Lead and its Task to the org the env's credentials name.
"""

from __future__ import annotations

from app.config import get_llm
from app.graph.graph import build_graph
from app.graph.nodes import set_llm

set_llm(get_llm())

graph = build_graph(default_checkpointer=False)
