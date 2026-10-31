"""Pull SEC EDGAR companyfacts for a few companies and show what was mapped.

Run from the repository root:

    export SEC_USER_AGENT="Your Name your@email"
    .venv/bin/python notebooks/fetch_edgar.py --as-of 2026-10-31 320193 789019 21344

Raw JSON goes to data/raw/edgar_companyfacts/<as_of>/ (fetched once, then reused);
the parsed table to data/interim/edgar_companyfacts/<as_of>/facts.parquet.
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from Toolbox.skatkis.io.config import load_config
from Toolbox.skatkis.io.snapshots import SnapshotStore
from Toolbox.skatkis.io.tables import write_table
from Toolbox.value_investing.sources.edgar import (
    EdgarCompanyFacts,
    edgar_client,
    load_companyfacts,
)
from Toolbox.value_investing.sources.tag_map import TAG_MAP

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--as-of", required=True, type=date.fromisoformat, help="snapshot date, YYYY-MM-DD")
parser.add_argument("ciks", nargs="+", help="company CIKs, e.g. 320193 (Apple)")
args = parser.parse_args()

user_agent = os.environ.get("SEC_USER_AGENT", "")
if not user_agent:
    sys.exit('Set SEC_USER_AGENT, e.g. export SEC_USER_AGENT="Your Name your@email"')

config = load_config(ROOT / "config" / "value_investing.toml")["sources"]["edgar"]
client = edgar_client(user_agent, **config)
facts = load_companyfacts(
    args.ciks,
    provider=EdgarCompanyFacts(client),
    store=SnapshotStore(ROOT / "data" / "raw"),
    as_of=args.as_of,
)

out = ROOT / "data" / "interim" / "edgar_companyfacts" / args.as_of.isoformat() / "facts.parquet"
try:
    write_table(facts, out)
    print(f"Wrote {len(facts):,} rows to {out.relative_to(ROOT)}")
except ImportError as error:
    print(f"Parquet not written ({error}); install pyarrow.")

annual = facts[facts["form"] == "10-K"]
coverage = annual.groupby(["entity_name", "field"])["fiscal_year"].agg(["min", "max", "nunique"])
coverage.columns = ["first_fy", "last_fy", "years"]
with pd.option_context("display.max_rows", 500, "display.width", 120):
    print("\nAnnual (10-K) coverage per company and field:")
    print(coverage)
missing = sorted({f.field for f in TAG_MAP} - set(facts["field"]))
print("\nFields with no data for any company:", missing or "none")
