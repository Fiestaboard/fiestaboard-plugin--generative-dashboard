"""Keep pytest from importing plugin modules as top-level modules.

They are reached through the ``plugins/`` symlink as
``plugins.generative_dashboard.*``, where relative imports resolve.
"""

collect_ignore = [
    "__init__.py",
    "catalog.py",
    "charset.py",
    "fallback.py",
    "gate.py",
    "layout.py",
    "llm.py",
    "validation.py",
]
