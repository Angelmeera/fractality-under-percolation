# Data

Nothing in this directory is committed. `.gitignore` keeps everything here out of the
repository except this file and `get_data.py`.

```bash
python data/get_data.py            # download the two SNAP graphs, verify checksums
python data/get_data.py --check    # verify what is already here, download nothing
```

| file | size | source | fetched automatically |
|---|---|---|---|
| `web-NotreDame.txt.gz` | 3.6 MB | [SNAP](https://snap.stanford.edu/data/web-NotreDame.html) — Albert, Jeong & Barabási (1999) | yes |
| `as20000102.txt.gz` | 40 KB | [SNAP](https://snap.stanford.edu/data/as-733.html) — Internet AS graph, 2 Jan 2000 | yes |
| `dblp_weighted.tsv.gz` | 51 MB | built locally from a DBLP or AMiner dump | no — see below |

The FSFN model sections need no data at all.

## Building the DBLP file

It is a derived product — authors joined by an edge weighted by their number of joint
publications — rather than a dataset anyone redistributes, and it depends on which DBLP
snapshot you start from.

```bash
# Option A: the official DBLP XML dump (~1 GB compressed, no account needed)
wget https://dblp.org/xml/dblp.xml.gz
python src/run_dblp.py --xml dblp.xml.gz --save-weighted data/dblp_weighted.tsv.gz

# Option B: the AMiner DBLP-Citation-network V12 used by Fronczak et al.
#   https://www.aminer.org/citation
python src/run_dblp.py --v12 dblp.v12.json --save-weighted data/dblp_weighted.tsv.gz
```

Expect the parse to take a while and to want several GB of RAM. The output is a
tab-separated `u <tab> v <tab> weight` file with a `# u\tv\tweight` header.

**Your file will probably not match ours, and that is expected.** DBLP has roughly
doubled since the 2020 snapshot Fronczak *et al.* used, so a fixed joint-paper threshold
picks out a different backbone on every dump. This is exactly why the paper *sweeps* the
threshold from `w ≥ 25` to `w ≥ 60` and reports `d_B` across an 82-fold range of
backbone size, rather than quoting one cut.

The file behind the published numbers:

```
sha256  0ba307e2b03c78e38df6237ef3b6efad7f64622412b4d53098d55bb9eb1f44ab
bytes   52,577,017
edges   6,407,862 weighted
```

`get_data.py --check` compares against this and tells you plainly if you have a
different snapshot. If you do, record your own checksum alongside your results.
