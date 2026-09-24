#!/usr/bin/env python3
"""Fetch the empirical networks used in the paper.

None of the three datasets is redistributed here. Two are public downloads and this
script gets them; the third has to be built from a bibliographic dump, and the script
tells you how.

    python data/get_data.py            # download the two SNAP graphs and verify them
    python data/get_data.py --check    # verify what is already on disk, download nothing

What you end up with
--------------------
    data/web-NotreDame.txt.gz     3.6 MB   nd.edu web crawl (Albert, Jeong & Barabasi 1999)
    data/as20000102.txt.gz         40 KB   Internet autonomous-system graph, 2 Jan 2000
    data/dblp_weighted.tsv.gz      51 MB   DBLP co-authorship, built locally (see below)

The DBLP file
-------------
It is a derived product -- authors weighted by joint publications -- not a dataset we
can hand you, and it is rebuilt from whichever DBLP snapshot you use. Because DBLP has
roughly doubled since the 2020 snapshot used by Fronczak et al., a fixed weight
threshold does NOT reproduce their backbone size on a current dump; the paper sweeps
the threshold for exactly this reason. Expect your numbers to differ from ours if your
snapshot differs from ours -- that is the honest situation, not a bug.

Build it with either source:

    # official DBLP XML dump (~1 GB compressed, no account needed)
    wget https://dblp.org/xml/dblp.xml.gz
    python src/run_dblp.py --xml dblp.xml.gz --save-weighted data/dblp_weighted.tsv.gz

    # or the AMiner DBLP-Citation-network V12 used by Fronczak et al.
    #   https://www.aminer.org/citation
    python src/run_dblp.py --v12 dblp.v12.json --save-weighted data/dblp_weighted.tsv.gz

The file this paper used has sha256
0ba307e2b03c78e38df6237ef3b6efad7f64622412b4d53098d55bb9eb1f44ab (52,577,017 bytes,
6,407,862 weighted edges, header line `# u\tv\tweight`). If yours differs, that is a
different DBLP snapshot; record its checksum alongside your results.
"""
import argparse
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> (url, sha256, approximate size in bytes)
DOWNLOADS = {
    "web-NotreDame.txt.gz": (
        "https://snap.stanford.edu/data/web-NotreDame.txt.gz",
        "d4f4916db2b45060d4dde21fca2d6d757f245f4397528f72230eff3bb57e499e",
        3708538),
    "as20000102.txt.gz": (
        "https://snap.stanford.edu/data/as20000102.txt.gz",
        "b5bbd9318d2e3d7fa64be29068f53d5553f0af4729cda4e53203a125f1e414f0",
        40325),
}

DERIVED = {
    "dblp_weighted.tsv.gz":
        "0ba307e2b03c78e38df6237ef3b6efad7f64622412b4d53098d55bb9eb1f44ab",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(name, expected):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return "missing", None
    got = sha256(path)
    return ("ok" if got == expected else "MISMATCH"), got


def download(name, url, expected):
    path = os.path.join(HERE, name)
    print("  fetching %s" % url, flush=True)
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:                                   # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        print("  FAILED: %s" % exc)
        print("  Download it by hand into %s/ and re-run with --check." % HERE)
        return False
    got = sha256(tmp)
    if got != expected:
        os.remove(tmp)
        print("  checksum MISMATCH (got %s)" % got)
        print("  SNAP may have re-published the file; do not use it silently.")
        return False
    os.replace(tmp, path)
    print("  ok  %s" % name)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify files already present; download nothing")
    args = ap.parse_args()

    bad = 0
    print("SNAP graphs")
    for name, (url, digest, _size) in DOWNLOADS.items():
        state, got = verify(name, digest)
        if state == "ok":
            print("  ok       %s" % name)
        elif state == "missing":
            if args.check:
                print("  missing  %s" % name)
                bad += 1
            elif not download(name, url, digest):
                bad += 1
        else:
            print("  MISMATCH %s  (got %s)" % (name, got))
            bad += 1

    print("DBLP (built locally -- see the module docstring)")
    for name, digest in DERIVED.items():
        state, got = verify(name, digest)
        if state == "ok":
            print("  ok       %s  (same snapshot as the paper)" % name)
        elif state == "missing":
            print("  missing  %s  -- build it, or skip the DBLP sections" % name)
        else:
            print("  differs  %s" % name)
            print("           got      %s" % got)
            print("           paper's  %s" % digest)
            print("           A different DBLP snapshot. Record this checksum with "
                  "your results; the DBLP numbers will not match ours.")

    if bad:
        print("\n%d problem(s). The FSFN model sections need no data at all and will "
              "still run." % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
