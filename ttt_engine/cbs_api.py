"""
cbs_api.py — CBS Data API + Backend Services
==============================================
Reuses customer_master_data from astor_collection_agent.
Also includes simulated backend services (PTP, Verify, SMS).
"""

from __future__ import annotations

import csv
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from logger import get_logger
from error_logger import log_error

logger = get_logger(__name__)

from config import cfg

DATA_DIR = cfg.CBS_DATA_DIR if (hasattr(cfg, "CBS_DATA_DIR") and os.path.exists(cfg.CBS_DATA_DIR)) else os.path.join(os.path.dirname(__file__), "customer_master_data")
if not os.path.exists(DATA_DIR):
    alt = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "customer_master_data")
    if os.path.exists(alt):
        DATA_DIR = alt

cbs_router = APIRouter(prefix="/api/cbs", tags=["CBS Data"])
services_router = APIRouter(prefix="/api/services", tags=["Backend Services"])


# ── CSV Loader ───────────────────────────────────────────────────────────────

def _load_csv(filename: str) -> List[Dict]:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


class CBSData:
    def __init__(self):
        self._mtime = 0.0
        self.reload()

    def reload(self):
        csv_path = os.path.join(DATA_DIR, "customer_master_data.csv")
        try:
            self._mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0.0
        except Exception:
            self._mtime = 0.0

        self.customers = _load_csv("customer_master_data.csv")
        self.accounts = _load_csv("account_data.csv")
        self.loans = _load_csv("loan_data (1).csv")
        self.transactions = _load_csv("transaction_history.csv")

        self._by_mobile = {c.get("mobile", "").strip(): c for c in self.customers}
        self._by_id = {c.get("customer_id", "").strip(): c for c in self.customers}

        logger.info(f"[CBS] Loaded: {len(self.customers)} customers, {len(self.loans)} loans")

    def _ensure_fresh(self):
        csv_path = os.path.join(DATA_DIR, "customer_master_data.csv")
        try:
            if os.path.exists(csv_path) and os.path.getmtime(csv_path) != self._mtime:
                self.reload()
        except Exception:
            pass

    def get_customer_by_mobile(self, mobile: str) -> Optional[Dict]:
        self._ensure_fresh()
        return self._by_mobile.get(mobile.strip())

    def get_summary(self, mobile: str) -> Optional[Dict]:
        self._ensure_fresh()
        customer = self._by_mobile.get(mobile.strip())
        if not customer:
            return None

        cid = customer.get("customer_id", "")
        loans = [l for l in self.loans if l.get("customer_id") == cid]
        accounts = [a for a in self.accounts if a.get("customer_id") == cid]

        all_txns = []
        for acc in accounts:
            txns = [t for t in self.transactions if t.get("account_no") == acc.get("account_no", "")]
            all_txns.extend(txns[:5])

        total_overdue = sum(float(l.get("emi", 0)) for l in loans if l.get("status", "").lower() == "overdue")

        return {
            "customer": customer,
            "loans": loans,
            "accounts": accounts,
            "transactions": all_txns,
            "total_overdue": total_overdue,
        }


cbs_data = CBSData()


# ── CBS Endpoints ────────────────────────────────────────────────────────────

@cbs_router.get("/customers")
def list_customers():
    cbs_data._ensure_fresh()
    return {
        "total": len(cbs_data.customers),
        "customers": [
            {"customer_id": c.get("customer_id"), "name": c.get("name"), "mobile": c.get("mobile"), "city": c.get("city"), "risk_profile": c.get("risk_profile")}
            for c in cbs_data.customers
        ],
    }


@cbs_router.get("/summary/{mobile}")
def get_summary(mobile: str):
    summary = cbs_data.get_summary(mobile)
    if not summary:
        log_error(
            "cbs_customer_not_found",
            f"CBS customer lookup failed — no record for mobile '{mobile[-4:].rjust(len(mobile), '*')}'",
            severity="WARNING",
            error_detail={"mobile_last4": mobile[-4:] if len(mobile) >= 4 else mobile},
            source_module="cbs_api",
            source_function="get_summary",
        )
        raise HTTPException(status_code=404, detail=f"Customer not found: {mobile}")
    return summary


@cbs_router.get("/transactions/{account_no}")
def get_transactions(account_no: str, limit: int = 10):
    txns = [t for t in cbs_data.transactions if t.get("account_no") == account_no]
    txns.sort(key=lambda x: x.get("date", ""), reverse=True)
    return {"account_no": account_no, "transactions": txns[:limit]}


# ── Backend Services (simulated) ─────────────────────────────────────────────

ptp_store: List[Dict] = []


class VerifyReq(BaseModel):
    customer_id: str
    dob: str


@services_router.post("/verify")
def verify_customer(req: VerifyReq):
    verified = bool(req.dob and len(req.dob) >= 4)
    return {"verified": verified, "customer_id": req.customer_id}


class PTPReq(BaseModel):
    customer_id: str
    ptp_date: str
    ptp_amount: float
    mobile: Optional[str] = None


@services_router.post("/ptp/create")
def create_ptp(req: PTPReq):
    ptp_id = f"ptp_{uuid.uuid4().hex[:6]}"
    record = {
        "ptp_id": ptp_id,
        "customer_id": req.customer_id,
        "ptp_date": req.ptp_date,
        "ptp_amount": req.ptp_amount,
        "status": "Created",
        "created_at": datetime.now().isoformat(),
    }
    ptp_store.append(record)
    logger.info(f"[PTP] Created: {ptp_id} | Amount: {req.ptp_amount}")
    return record


class SMSReq(BaseModel):
    mobile: str
    message: str


@services_router.post("/sms/send")
def send_sms(req: SMSReq):
    sms_id = f"sms_{uuid.uuid4().hex[:6]}"
    logger.info(f"[SMS] Sent to {req.mobile}: {req.message[:50]}...")
    return {"sms_id": sms_id, "mobile": req.mobile, "status": "Sent", "sent_at": datetime.now().isoformat()}
