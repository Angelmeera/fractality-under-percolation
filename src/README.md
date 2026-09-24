# Source

A flat directory, not a package: every script does
`sys.path.insert(0, dirname(abspath(__file__)))` and imports its siblings, so the files
have to stay together but can be run from anywhere. Run them from the repository root.

The top-level `README.md` has the file-by-file table. Two entry points:

- `run_snap.py` — the main per-network analysis. `--fsfn N` needs no data.
- `conn_audit.py`, `m5_control_t5.py`, `subgrid_daic.py` — the audits that produced the
  findings in `docs/METHODS_NOTES.md`.

`snap_networks.py` is the core module. Worth reading before modifying anything:
`cover_stream` (the greedy cover and its ordering contract), `_box_connected` and
`_box_diameter` (why a disconnected box has no well-defined diameter), `joint_mass_law`
(the direct route — the docstring is explicit that it touches neither γ nor δ), and
`_ccdf_exponent` next to `_tail_exponent_mle` and `_discrete_mle` (the three exponent
estimators the paper compares, and they disagree by a factor approaching two).
