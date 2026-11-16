"""Backtest metrics, independent of the GPU/model runtime."""
import numpy as np


def metrics_from_infos(infos, start_equity=100_000):
    if not infos:
        return {}
    nets = np.array([x["net_return"] for x in infos], dtype=np.float64)
    final = float(infos[-1]["equity"])
    total = final / start_equity - 1
    volatility = float(np.std(nets))
    result = {
        "final_equity": final,
        "peak_equity": float(max(start_equity, max(x["equity"] for x in infos))),
        "return_total": total,
        "return_annualized": float((final / start_equity) ** (252 / len(infos)) - 1),
        "annualization_basis": "geometric, 252 trading days per year",
        "mean_net_return": float(np.mean(nets)),
        "mean_gross_return": float(np.mean([x["gross_return"] for x in infos])),
        "mean_turnover": float(np.mean([x["turnover"] for x in infos])),
        "mean_trading_cost": float(np.mean([x["trading_cost"] for x in infos])),
        "max_drawdown": float(max(x["drawdown"] for x in infos)),
        "volatility_daily": volatility,
        "sharpe_proxy": float(np.mean(nets) / volatility * np.sqrt(252)) if volatility > 0 else None,
        "steps": len(infos),
    }
    if "return_date" in infos[0]:
        result.update(first_decision_date=infos[0]["decision_date"],
                      first_return_date=infos[0]["return_date"],
                      last_return_date=infos[-1]["return_date"])
    return result
