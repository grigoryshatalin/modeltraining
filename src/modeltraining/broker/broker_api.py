"""Alpaca Broker API client authenticated via OAuth2 client-credentials.

The credentials (CLIENT_ID / CLIENT_SECRET) are exchanged at the authx endpoint
for a short-lived (~15 min) Bearer token, which authorizes the Broker API. This
is a thin REST wrapper because alpaca-py's BrokerClient only supports Basic auth.
"""

from __future__ import annotations

import time

import httpx

from ..config import Settings

_TOKEN_REFRESH_MARGIN = 60  # refresh this many seconds before expiry


class BrokerAuthError(RuntimeError):
    pass


class BrokerAPIError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Broker API {status}: {body}")
        self.status = status
        self.body = body


class _TokenManager:
    """Fetches and caches the OAuth2 client-credentials Bearer token."""

    def __init__(self, client_id: str, client_secret: str, sandbox: bool) -> None:
        self._id = client_id
        self._secret = client_secret
        self._authx = (
            "https://authx.sandbox.alpaca.markets/v1"
            if sandbox
            else "https://authx.alpaca.markets/v1"
        )
        self._token: str | None = None
        self._expires_at = 0.0

    def token(self) -> str:
        if self._token and time.time() < self._expires_at - _TOKEN_REFRESH_MARGIN:
            return self._token
        try:
            resp = httpx.post(
                f"{self._authx}/oauth2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._id,
                    "client_secret": self._secret,
                },
                timeout=20,
            )
        except httpx.HTTPError as exc:  # pragma: no cover - network
            raise BrokerAuthError(f"token request failed: {exc}") from exc
        if resp.status_code != 200:
            raise BrokerAuthError(f"token exchange failed ({resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        self._token = data["access_token"]
        self._expires_at = time.time() + float(data.get("expires_in", 900))
        return self._token


class BrokerAPI:
    """Minimal Alpaca Broker API client (accounts, journals, trading)."""

    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        settings.require_broker_keys()
        self._sandbox = settings.alpaca_broker_sandbox
        self._auth = _TokenManager(settings.client_id, settings.client_secret, self._sandbox)
        self._base = (
            "https://broker-api.sandbox.alpaca.markets"
            if self._sandbox
            else "https://broker-api.alpaca.markets"
        )
        self._http = http or httpx.Client(timeout=30)

    # --- transport ---
    def _request(self, method: str, path: str, **kw):
        headers = {"Authorization": f"Bearer {self._auth.token()}", "accept": "application/json"}
        resp = self._http.request(method, f"{self._base}{path}", headers=headers, **kw)
        if resp.status_code >= 400:
            raise BrokerAPIError(resp.status_code, resp.text[:400])
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    def _get(self, path: str, params: dict | None = None):
        return self._request("GET", path, params=params)

    def _post(self, path: str, json: dict | None = None):
        return self._request("POST", path, json=json)

    # --- accounts ---
    def list_accounts(self) -> list[dict]:
        return self._get("/v1/accounts") or []

    def get_account(self, account_id: str) -> dict:
        return self._get(f"/v1/accounts/{account_id}")

    def create_account(self, payload: dict) -> dict:
        return self._post("/v1/accounts", json=payload)

    def get_trade_account(self, account_id: str) -> dict:
        """Balances/equity for a trading account."""
        return self._get(f"/v1/trading/accounts/{account_id}/account")

    # --- funding (sandbox: ACH deposit; settles asynchronously) ---
    def get_ach_relationships(self, account_id: str) -> list[dict]:
        return self._get(f"/v1/accounts/{account_id}/ach_relationships") or []

    def create_ach_relationship(self, account_id: str) -> dict:
        return self._post(
            f"/v1/accounts/{account_id}/ach_relationships",
            json={
                "account_owner_name": "Model Trader",
                "bank_account_type": "CHECKING",
                "bank_account_number": "32131231864",
                "bank_routing_number": "121000358",
                "nickname": "funding",
            },
        )

    def create_transfer(self, account_id: str, relationship_id: str, amount: float) -> dict:
        return self._post(
            f"/v1/accounts/{account_id}/transfers",
            json={
                "transfer_type": "ach",
                "relationship_id": relationship_id,
                "amount": str(amount),
                "direction": "INCOMING",
                "timing": "immediate",
            },
        )

    def fund_via_ach(self, account_id: str, amount: float) -> dict:
        """Deposit `amount` into the account via ACH (reusing/creating a relationship)."""
        rels = self.get_ach_relationships(account_id)
        rel_id = rels[0]["id"] if rels else self.create_ach_relationship(account_id)["id"]
        return self.create_transfer(account_id, rel_id, amount)

    def create_journal(self, payload: dict) -> dict:
        return self._post("/v1/journals", json=payload)

    def close_account(self, account_id: str) -> None:
        self._request("DELETE", f"/v1/accounts/{account_id}")

    # --- trading ---
    def submit_order(self, account_id: str, order: dict) -> dict:
        return self._post(f"/v1/trading/accounts/{account_id}/orders", json=order)

    def get_positions(self, account_id: str) -> list[dict]:
        return self._get(f"/v1/trading/accounts/{account_id}/positions") or []

    def get_portfolio_history(self, account_id: str, params: dict | None = None) -> dict:
        return self._get(f"/v1/trading/accounts/{account_id}/account/portfolio/history", params=params)
