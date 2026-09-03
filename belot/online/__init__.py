"""Playing on belot.md: the composite player adapted to a live table.

The `belotmd` SDK owns everything platform-shaped -- joining, auth, state
reconstruction from the server's partial feed, declarations, the seven-swap,
retrying refused moves, seat-takeover detection, frame recording. This package
owns only the decision, and reuses the offline player's own encoder, network
and search rather than carrying copies of them.
"""
