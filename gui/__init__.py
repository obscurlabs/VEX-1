"""PySide6 presentation layer over the existing pipeline.

The GUI owns no pipeline logic. It calls ``main.run()`` on a worker thread and
renders the events the Phase 7.A reporter seam delivers.
"""
