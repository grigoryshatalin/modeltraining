#!/usr/bin/env python
"""Provision one funded Alpaca Broker API (sandbox) account per model.

Usage:
    python scripts/provision_accounts.py provision   # create + fund missing accounts
    python scripts/provision_accounts.py status      # show each account's status/balances
    python scripts/provision_accounts.py close-all    # close every provisioned account (reset)

Accounts activate and ACH deposits settle asynchronously in the sandbox, so run
`status` a couple of minutes after `provision` to confirm each is ACTIVE with
$1,000 of buying power. The model -> account map is saved to state/accounts.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modeltraining.broker.broker_api import BrokerAPI, BrokerAPIError  # noqa: E402
from modeltraining.config import get_settings  # noqa: E402
from modeltraining.tournament.roster import default_roster  # noqa: E402

MAP_PATH = Path("state/accounts.json")
CAPITAL = 1000.0


def _load_map() -> dict:
    if MAP_PATH.exists():
        return json.loads(MAP_PATH.read_text())
    return {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "capital": CAPITAL, "accounts": {}}


def _save_map(m: dict) -> None:
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(m, indent=2) + "\n")


def _account_payload(label: str) -> dict:
    email = f"{label.replace('-', '.')}.{uuid.uuid4().hex[:8]}@example.com"
    ssn = f"{random.randint(100, 665):03d}-{random.randint(10, 99):02d}-{random.randint(1000, 9999):04d}"
    return {
        "contact": {
            "email_address": email, "phone_number": "+15551234567",
            "street_address": ["20 N San Mateo Dr"], "city": "San Mateo",
            "state": "CA", "postal_code": "94401", "country": "USA",
        },
        "identity": {
            "given_name": "Model", "family_name": label[:30], "date_of_birth": "1990-01-01",
            "tax_id": ssn, "tax_id_type": "USA_SSN", "country_of_citizenship": "USA",
            "country_of_birth": "USA", "country_of_tax_residence": "USA",
            "funding_source": ["employment_income"],
        },
        "disclosures": {
            "is_control_person": False, "is_affiliated_exchange_or_finra": False,
            "is_politically_exposed": False, "immediate_family_exposed": False,
        },
        "agreements": [{
            "agreement": "customer_agreement",
            "signed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "ip_address": "127.0.0.1",
        }],
        "_email": email,  # local convenience; stripped before send below
    }


def provision(api: BrokerAPI) -> None:
    m = _load_map()
    for c in default_roster():
        if c.id in m["accounts"]:
            print(f"  {c.id:<20} exists -> {m['accounts'][c.id]['account_id']}")
            continue
        payload = _account_payload(c.id)
        email = payload.pop("_email")
        try:
            acct = api.create_account(payload)
            api.fund_via_ach(acct["id"], CAPITAL)
        except BrokerAPIError as e:
            print(f"  {c.id:<20} FAILED: {e.status} {e.body[:160]}")
            continue
        m["accounts"][c.id] = {
            "account_id": acct["id"], "account_number": acct.get("account_number"), "email": email,
        }
        _save_map(m)
        print(f"  {c.id:<20} created {acct['id']}  #{acct.get('account_number')}  (funding ${CAPITAL:,.0f})")
    print(f"\nSaved {len(m['accounts'])} accounts to {MAP_PATH}")


def status(api: BrokerAPI) -> None:
    m = _load_map()
    if not m["accounts"]:
        print("No accounts provisioned yet. Run:  python scripts/provision_accounts.py provision")
        return
    print(f"{'MODEL':<20}{'STATUS':<12}{'CASH':>12}{'BUYING_POWER':>14}{'EQUITY':>12}")
    for cid, info in m["accounts"].items():
        try:
            acct = api.get_account(info["account_id"])
            ta = api.get_trade_account(info["account_id"])
            print(f"{cid:<20}{acct.get('status',''):<12}"
                  f"{float(ta.get('cash',0)):>12,.2f}{float(ta.get('buying_power',0)):>14,.2f}"
                  f"{float(ta.get('equity',0)):>12,.2f}")
        except BrokerAPIError as e:
            print(f"{cid:<20} error {e.status}")


def close_all(api: BrokerAPI) -> None:
    m = _load_map()
    for cid, info in list(m["accounts"].items()):
        try:
            api.close_account(info["account_id"])
            print(f"  closed {cid} ({info['account_id']})")
        except BrokerAPIError as e:
            print(f"  {cid}: {e.status} {e.body[:120]}")
    m["accounts"] = {}
    _save_map(m)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["provision", "status", "close-all"])
    args = parser.parse_args()
    api = BrokerAPI(get_settings())
    {"provision": provision, "status": status, "close-all": close_all}[args.command](api)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
