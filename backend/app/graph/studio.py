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
- No masked tracer is attached (unlike ``server._run_config``), so the module
  refuses to load while LangSmith tracing is on: a run would upload visitor
  emails and phones unmasked.  ``langgraph dev`` turns tracing on by itself
  when a LangSmith API key is set and no tracing flag is, so a ``.env`` with a
  key needs ``LANGSMITH_TRACING=false``.  The check is langsmith's own
  ``langsmith.utils.tracing_is_enabled``.
- The site index is not loaded, so a site question gets the fixed
  ``SITE_UNAVAILABLE_REPLY``.
- ``salesforce_node`` is the real one: a run that reaches an accepted
  confirmation writes a Lead and its Task to the org the env's credentials name.
"""

from __future__ import annotations

from langsmith.utils import get_env_var, tracing_is_enabled

from app.config import get_llm
from app.graph.graph import build_graph
from app.graph.nodes import set_llm

# langsmith caches each env lookup; read the env as it is now (langgraph dev may
# have set LANGSMITH_TRACING after an earlier lookup), so the check fails closed.
get_env_var.cache_clear()
if tracing_is_enabled():
    raise RuntimeError(
        "LangSmith tracing is on, and Studio runs have no masked tracer, so they "
        "would upload visitor emails and phones unmasked. langgraph dev turns "
        "tracing on by itself when a LangSmith API key is set and no tracing flag "
        "is. Set LANGSMITH_TRACING=false in .env to load the TARS graph in Studio."
    )

set_llm(get_llm())

graph = build_graph(default_checkpointer=False)
