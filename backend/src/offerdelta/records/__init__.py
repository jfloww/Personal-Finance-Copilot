"""Shapes that cross a layer boundary.

Neither ingest nor persistence owns these. A CSV row, a web form, and a
repository write all have to agree on what a transaction looks like before it
is stored, and the agreement itself is not the property of either side.

They lived under ``infrastructure.postgres`` first, which forced the CSV
adapter to import persistence to describe its own output - the mirror image of
the coupling this package exists to prevent, and invisible to a contract that
only watched one direction.
"""
