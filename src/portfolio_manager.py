# src/portfolio_manager.py
"""
Simplified portfolio manager – no heavy external dependencies.
Implements:
- cash + monthly deposits
- position handling (BUY / SELL)
- daily price updates
- rebalance on a specific day of month
- KPI calculations (ROI, Sharpe, max‑drawdown, VaR 95, ES 95)
- transaction log (date, symbol, type, price, quantity, cash_change)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
#   Data structures
# ----------------------------------------------------------------------
@dataclass
class Position:
    """Représente une position détenue sur une crypto."""
    symbol: str
    quantity: float
    avg_price: float
    purchase_date: datetime
    last_price: float = field(default=0.0)

    @property
    def market_value(self) -> float:
        """Valeur actuelle = qty × dernier prix connu."""
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        """Profit / loss non réalisé."""
        return (self.last_price - self.avg_price) * self.quantity

    @property
    def unrealized_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return 100.0 * (self.last_price / self.avg_price - 1.0)


# ----------------------------------------------------------------------
#   Portfolio core
# ----------------------------------------------------------------------
class Portfolio:
    """
    Gestionnaire de portefeuille minimaliste.
    - cash  : USDT disponible
    - positions : dict(symbol → Position)
    - history : list[dict] → “date”, “total_value”, “cash”, “positions_value”,
                 “unrealized_pnl”, “unrealized_pct”, “positions_count”.
    - transactions : list[dict] → “date”, “symbol”, “type”, “price”, “quantity”,
                 “cash_change”.
    - monthly_amount : montant ajouté le jour d’investissement (default 10 → 50 USDT)
    - investment_day : jour du mois où le dépôt / le rebalance a lieu
    - start_date    : date de la première observation (exemple : 22/01/2026)
    """

    def __init__(
        self,
        initial_capital: float = 200.0,
        monthly_amount: float = 50.0,
        investment_day: int = 10,
        start_date: datetime | None = None,
    ) -> None:
        self.cash = float(initial_capital)
        self.monthly_amount = float(monthly_amount)
        self.investment_day = int(investment_day)
        self.start_date = (
            start_date if start_date is not None else datetime.now()
        ).replace(hour=0, minute=0, second=0, microsecond=0)

        self.positions: Dict[str, Position] = {}
        self.history: List[dict] = []
        self.transactions: List[dict] = []          # ← journal des trades

        # tracking du cash injecté (utile pour le ROI)
        self.total_capital_added = self.cash

    # ------------------------------------------------------------------
    # 1️⃣  Gestion des apports mensuels
    # ------------------------------------------------------------------
    def add_monthly_deposit(self, cur_date: datetime) -> None:
        """
        Ajoute le dépôt mensuel si ``cur_date`` correspond au jour d’investissement.
        Le dépôt du mois précédent (si la date de début est après le jour 10)
        n’est **pas** rétro‑actif : le premier dépôt survient le premier
        10 > start_date.
        """
        if cur_date.day == self.investment_day and cur_date >= self.start_date:
            self.cash += self.monthly_amount
            self.total_capital_added += self.monthly_amount
            log.debug(
                f"[{cur_date.date()}] dépôt mensuel +{self.monthly_amount:.2f} USDT"
            )

    # ------------------------------------------------------------------
    # 2️⃣  Mise à jour quotidienne des cours
    # ------------------------------------------------------------------
    def update_prices(self, cur_date: datetime, price_dict: Dict[str, float]) -> None:
        """
        Met à jour le dernier cours de chaque position.
        Si le prix d’une crypto n’est pas disponible, on garde l’ancien prix.
        """
        for sym, pos in self.positions.items():
            if sym in price_dict and price_dict[sym] > 0:
                pos.last_price = price_dict[sym]

        # on crée un snapshot du jour, même si aucune position n’est ouverte
        self._snapshot(cur_date)

    # ------------------------------------------------------------------
    # 3️⃣  Journal des transactions
    # ------------------------------------------------------------------
    def _record_transaction(
        self,
        cur_date: datetime,
        symbol: str,
        type_: str,
        price: float,
        quantity: float,
        cash_change: float,
    ) -> None:
        """Enregistre une ligne de transaction dans ``self.transactions``."""
        self.transactions.append(
            {
                "date": cur_date.date(),
                "symbol": symbol,
                "type": type_,               # "BUY" ou "SELL"
                "price": price,
                "quantity": quantity,
                "cash_change": cash_change,   # valeur nette ajoutée (+) ou retirée (‑)
            }
        )

    # ------------------------------------------------------------------
    # 4️⃣  Logique de trade (BUY / SELL) – date explicite
    # ------------------------------------------------------------------
    def _execute_buy(
        self,
        cur_date: datetime,
        symbol: str,
        price: float,
        cash_to_spend: float,
        fee_pct: float = 0.001,
    ) -> None:
        """Achète autant que possible avec ``cash_to_spend`` (après frais)."""
        if price <= 0:
            log.warning(f"{cur_date.date()} – prix nul pour {symbol} → achat ignoré")
            return

        fee = cash_to_spend * fee_pct
        net = cash_to_spend - fee
        qty = net / price
        if qty <= 0:
            return

        if symbol in self.positions:
            pos = self.positions[symbol]
            new_qty = pos.quantity + qty
            new_avg = (pos.avg_price * pos.quantity + price * qty) / new_qty
            pos.quantity = new_qty
            pos.avg_price = new_avg
            pos.last_price = price
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=qty,
                avg_price=price,
                purchase_date=cur_date,
                last_price=price,
            )
        self.cash -= cash_to_spend
        self._record_transaction(
            cur_date,
            symbol,
            "BUY",
            price,
            qty,
            -cash_to_spend,
        )
        log.debug(
            f"{cur_date.date()} – BUY {qty:.6f} {symbol} @ {price:.4f} "
            f"(cash‑used {cash_to_spend:.2f})"
        )

    def _execute_sell(
        self,
        cur_date: datetime,
        symbol: str,
        price: float,
        fee_pct: float = 0.001,
    ) -> None:
        """Vente intégrale d’une position (si elle existe)."""
        pos = self.positions.get(symbol)
        if pos is None or pos.quantity <= 0:
            return

        proceeds = pos.quantity * price
        fee = proceeds * fee_pct
        net = proceeds - fee
        self.cash += net
        self._record_transaction(
            cur_date,
            symbol,
            "SELL",
            price,
            pos.quantity,
            net,
        )
        log.debug(
            f"{cur_date.date()} – SELL {pos.quantity:.6f} {symbol} @ {price:.4f} "
            f"(cash +{net:.2f})"
        )
        del self.positions[symbol]

    # ------------------------------------------------------------------
    # 5️⃣  Rebalancing – exécuté uniquement le jour d’investissement
    # ------------------------------------------------------------------
    def rebalance(
        self,
        cur_date: datetime,
        predictions: pd.DataFrame,
        fee_pct: float = 0.001,
        top_n: int = 2,
    ) -> None:
        """
        - Vend les positions dont le signal = SELL (0).
        - Sélectionne les ``top_n`` cryptos avec le meilleur *score* où le signal = BUY (1).
        - Alloue le cash disponible à parts égales.
        - N’est invoqué que le ``investment_day``; sinon il sort immédiatement.
        """
        if cur_date.day != self.investment_day:
            return

        # ------------ 1️⃣ VENTES ------------
        for sym, pos in list(self.positions.items()):
            row = predictions.loc[predictions["symbol"] == sym]
            if not row.empty and int(row["signal"].iloc[0]) == 0:
                price = row["price"].iloc[0]
                self._execute_sell(cur_date, sym, price, fee_pct)

        # ------------ 2️⃣ ACHATS ------------
        buy_candidates = predictions[predictions["signal"] == 1]
        if buy_candidates.empty:
            log.info(f"{cur_date.date()} – Aucun signal BUY, rien à acheter")
            self._snapshot(cur_date)
            return

        top = buy_candidates.nlargest(top_n, "score")
        cash_available = self.cash
        if cash_available <= 0:
            log.info(f"{cur_date.date()} – Pas de cash disponible pour rebalance")
            self._snapshot(cur_date)
            return

        allocation = cash_available / len(top)
        for _, row in top.iterrows():
            sym = row["symbol"]
            price = row["price"]
            self._execute_buy(cur_date, sym, price, allocation, fee_pct)

        # ------------ 3️⃣ SNAPSHOT ------------
        self._snapshot(cur_date)

    # ------------------------------------------------------------------
    # 6️⃣  Snapshot – on enregistre la date du jour simulé
    # ------------------------------------------------------------------
    def _snapshot(self, cur_date: datetime) -> None:
        total_positions = sum(p.market_value for p in self.positions.values())
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total = self.cash + total_positions

        self.history.append(
            {
                "date": cur_date.date(),
                "total_value": total,
                "cash": self.cash,
                "positions_value": total_positions,
                "unrealized_pnl": unrealized,
                "unrealized_pct": (
                    100.0 * unrealized / total_positions
                    if total_positions > 0
                    else 0.0
                ),
                "positions_count": len(self.positions),
            }
        )

    # ------------------------------------------------------------------
    # 7️⃣  Métriques (ROI, Sharpe, Max‑drawdown, VaR 95, ES 95)
    # ------------------------------------------------------------------
    def metrics(self, risk_free_rate: float = 0.0) -> dict:
        """Retourne un dictionnaire de métriques basées sur l’historique."""
        if not self.history:
            return {}

        df = pd.DataFrame(self.history).set_index("date")
        values = df["total_value"]

        # ---- Retour sur investissement ------------------------------------------------
        roi = 100.0 * (values.iloc[-1] - self.total_capital_added) / self.total_capital_added

        # ---- Rendements quotidiens ----------------------------------------------------
        daily_ret = values.pct_change().dropna()
        if daily_ret.empty:
            sharpe = max_dd = vol = var95 = es95 = np.nan
        else:
            # Volatilité annualisée (252 jours de trading)
            vol = daily_ret.std() * np.sqrt(252) * 100

            # Sharpe (annualisé)
            excess_ret = daily_ret - risk_free_rate / 252
            sharpe = (
                np.mean(excess_ret) / np.std(excess_ret) * np.sqrt(252)
                if np.std(excess_ret) > 0
                else np.nan
            )

            # Max‑drawdown
            cum = (1 + daily_ret).cumprod()
            running_max = cum.cummax()
            drawdown = (cum - running_max) / running_max
            max_dd = drawdown.min() * 100

            # ---- VaR 95 % historique (percentile 5) ----
            var95 = -np.percentile(daily_ret.values, 5) * 100

            # ---- Expected Shortfall (ES) à 95 % ----
            tail_losses = daily_ret[daily_ret <= np.percentile(daily_ret, 5)]
            es95 = -tail_losses.mean() * 100 if not tail_losses.empty else np.nan

        return {
            "roi_pct": roi,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd,
            "volatility_pct": vol,
            "var_95_pct": var95,
            "es_95_pct": es95,
            "final_value": values.iloc[-1],
        }

    # ------------------------------------------------------------------
    # 8️⃣  Export / import JSON (facultatif)
    # ------------------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        """Sauvegarde de l’état du portefeuille au format JSON."""
        import json

        data = {
            "cash": self.cash,
            "positions": {
                sym: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "last_price": pos.last_price,
                    "purchase_date": pos.purchase_date.isoformat(),
                }
                for sym, pos in self.positions.items()
            },
            "history": self.history,
            "transactions": self.transactions,
            "total_capital_added": self.total_capital_added,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Portfolio":
        """Charge l’état du portefeuille depuis un fichier JSON."""
        import json

        d = json.loads(Path(path).read_text(encoding="utf-8"))
        start_date = datetime.strptime(d["history"][0]["date"], "%Y-%m-%d")
        port = cls(initial_capital=0.0, start_date=start_date)
        port.cash = d["cash"]
        port.total_capital_added = d.get("total_capital_added", 0.0)

        for sym, pos in d["positions"].items():
            port.positions[sym] = Position(
                symbol=sym,
                quantity=pos["quantity"],
                avg_price=pos["avg_price"],
                purchase_date=datetime.fromisoformat(pos["purchase_date"]),
                last_price=pos["last_price"],
            )
        port.history = d["history"]
        port.transactions = d.get("transactions", [])
        return port
