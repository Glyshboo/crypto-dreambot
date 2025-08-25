# -*- coding: utf-8 -*-
"""
execution_setup.py — Patch 02
- Småfix: sikrer leverage/margin init uten exception-stopp i paper mode.
- Utløser ikke live-handling (det kommer i Patch 03).
"""
from typing import Dict, Any

class ExecutionSetup:
    def __init__(self, paper: bool = True, default_leverage: int = 12):
        self.paper = paper
        self.default_leverage = default_leverage

    def prepare_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Returnerer en trygg metadata for symbolet.
        I live-mode vil dette sette leverage/margin via børse-API (kommende patch).
        """
        meta = {
            "symbol": symbol,
            "leverage": self.default_leverage,
            "ok": True
        }
        # TODO: implementer live leverage/margin i Patch 03
        return meta
