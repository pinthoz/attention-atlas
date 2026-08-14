"""HTTP service layer.

Wraps the existing analysis pipeline behind a small JSON interface and mounts
the Shiny dashboard unchanged underneath it. Complementary tooling: nothing in
``attention_app`` imports from here, so the dashboard runs with or without this
package present.
"""
