"""The energy page: parsing EIA's files, summarizing them, scoring the result.

No network and no workbooks: the parsers take rows as the readers return them,
so the tests build those rows by hand.
"""

import numpy as np
import pandas as pd
import pytest

from src import energy as en
from src.sources import eia
from src.sources.public import (as_observations, bra_month, forward_capacity_price,
                                parse_rpm_prices, us_power_path)


def steo_sheet(codes: dict[str, list[float]], start_year=2024, years=3):
    """A table sheet as STEO lays it out: years over months, codes in column 0."""
    n = 12 * years
    year_row = ("Forecast date:", None) + tuple(
        start_year + i // 12 if i % 12 == 0 else None for i in range(n))
    month_row = ("Thursday, September 4, 2025", None) + tuple(
        m.title() for _ in range(years) for m in eia.MONTHS)
    rows = [("Table of Contents", "Table 7a"), (None, "STEO"), year_row, month_row]
    rows += [(code, "label") + tuple(vals) for code, vals in codes.items()]
    return {"7atab": rows}


def test_steo_rows_become_monthly_series_and_codes_ignore_case():
    sheets = steo_sheet({"eltctwh": list(range(36)), "OTHER": [1.0] * 36})
    out = eia.parse_steo(sheets)
    assert set(out) == {"ELTCTWH"}
    s = out["ELTCTWH"]
    assert s.index[0] == pd.Timestamp("2024-01-01") and s.index[-1] == pd.Timestamp("2026-12-01")
    assert s[pd.Timestamp("2025-03-01")] == 14


def test_the_release_date_is_read_from_the_sheet():
    assert str(eia.release_date(steo_sheet({}), pd.Timestamp("2025-09-01"))) == "2025-09-04"


def test_twelve_months_ahead_against_the_twelve_before():
    # Sales of 100 a month through August 2025, 110 from September.
    vals = [100.0] * 20 + [110.0] * 16
    out = eia.steo_summary(eia.parse_steo(steo_sheet({"ELTCTWH": vals})), pd.Timestamp("2025-09-01"))
    assert out["elec_growth_12m"] == pytest.approx(10.0)
    # Codes the release does not carry are missing, not zero.
    assert np.isnan(out["opec_spare_12m"])


def test_older_releases_use_the_daily_sales_code():
    vals = [10.0] * 20 + [11.0] * 16
    out = eia.steo_summary(eia.parse_steo(steo_sheet({"EXTCPUS": vals})), pd.Timestamp("2025-09-01"))
    assert out["elec_growth_12m"] == pytest.approx(10.0)


def gens(rows):
    df = pd.DataFrame(rows)
    df["mw"] = df["Net Summer Capacity (MW)"].astype(float)
    return df


def test_capacity_due_inside_three_years_less_retirements():
    month = pd.Timestamp("2026-08-01")
    planned = gens([
        {"Net Summer Capacity (MW)": 100, "Energy Source Code": "NG",
         "Planned Operation Year": 2027, "Planned Operation Month": 6},
        {"Net Summer Capacity (MW)": 300, "Energy Source Code": "SUN",
         "Planned Operation Year": 2028, "Planned Operation Month": 1},
        {"Net Summer Capacity (MW)": 999, "Energy Source Code": "NG",   # outside the window
         "Planned Operation Year": 2031, "Planned Operation Month": 1},
    ])
    operating = gens([
        {"Net Summer Capacity (MW)": 1000, "Planned Retirement Year": 2027, "Planned Retirement Month": 3},
        {"Net Summer Capacity (MW)": 9000, "Planned Retirement Year": None, "Planned Retirement Month": None},
    ])
    canceled = gens([{"Net Summer Capacity (MW)": 601}])
    out = eia.gen_summary(planned, operating, canceled, month)
    assert out["add_36m_gw"] == pytest.approx(0.4)
    assert out["gas_add_36m_gw"] == pytest.approx(0.1)
    assert out["clean_add_36m_gw"] == pytest.approx(0.3)
    assert out["net_add_36m_pct"] == pytest.approx((400 - 1000) / 10000 * 100)
    assert out["cancel_share_pct"] == pytest.approx(601 / (601 + 1399) * 100)
    # Firm: gas at 0.75 and solar at 0.10 come in; the retiring unit has no
    # fuel recorded, so it and the fleet count at the default 0.60.
    firm_in = 100 * 0.75 + 300 * 0.10
    assert out["firm_net_add_36m_pct"] == pytest.approx((firm_in - 600) / 6000 * 100)


def test_a_solar_megawatt_counts_for_less_than_a_gas_one_at_the_peak():
    units = pd.DataFrame({"Energy Source Code": ["SUN", "NG", "NG", "NUC", "XYZ"],
                          "Prime Mover Code": ["PV", "CT", "GT", "ST", "ST"]})
    assert eia.firm_credit(units).tolist() == [0.10, 0.75, 0.60, 0.95, 0.60]


def test_each_site_holds_its_power_until_its_next_milestone():
    sites = pd.DataFrame({"Name": ["A", "B", "C"], "Country": ["United States", "United States", "China"]})
    tl = pd.DataFrame({
        "Data center": ["A", "A", "B", "C"],
        "Date": ["2025-01-15", "2027-01-01", "2026-06-01", "2025-01-01"],
        "Power (MW)": [100, 500, 200, 9999],
    })
    gw = us_power_path(sites, tl, start="2025-01", end="2027-12")
    assert gw[pd.Timestamp("2025-01-01")] == 0          # A starts mid-month
    assert gw[pd.Timestamp("2025-02-01")] == pytest.approx(0.1)
    assert gw[pd.Timestamp("2026-06-01")] == pytest.approx(0.3)
    assert gw[pd.Timestamp("2027-12-01")] == pytest.approx(0.7)  # China excluded


def test_observations_carry_a_lag_or_a_fixed_vintage():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-02-01"]))
    lagged = as_observations("X", s, lag_days=45)
    assert str(lagged["vintage_date"].iloc[0]) == "2026-02-15"
    fixed = as_observations("X", s, vintage="2026-10-02")
    assert set(map(str, fixed["vintage_date"])) == {"2026-10-02"}


def test_a_value_in_hand_is_never_dated_after_today():
    today = pd.Timestamp.today().normalize()
    s = pd.Series([1.0], index=[today - pd.Timedelta(days=10)])
    assert as_observations("X", s, lag_days=45)["vintage_date"].iloc[0] == today.date()


@pytest.mark.parametrize("demand, supply, want", [
    (0.6, 0.55, "ai_boom"),           # surge, supply keeping pace
    (0.6, 0.2, "energy_first"),       # surge, supply falling behind
    (0.0, -0.4, "energy_first"),      # no surge, supply shrinking
    (0.05, 0.3, "current_policies"),
    (np.nan, 0.3, None),
])
def test_the_two_markers_read_as_asr_scenarios(demand, supply, want):
    assert en.scenario_for(demand, supply) == want


def synthetic_wide():
    idx = pd.date_range("2009-01-31", "2026-08-31", freq="ME")
    rng = np.random.default_rng(0)
    growth = np.where(idx > "2021-12-31", 0.025, 0.0) / 12 + rng.normal(0, 0.0005, len(idx))
    return pd.DataFrame({
        "EN_STEO_ELEC_FWD": 3700 * np.exp(np.cumsum(growth)),
        "EN_860M_FIRM_NET_ADD": 3 + rng.normal(0, 0.5, len(idx)),
        "EN_860M_CANCEL_SHARE": 30 + rng.normal(0, 2, len(idx)),
        "PCU335311335311": np.exp(np.linspace(0, 1.2, len(idx))) * 100,
        "PCU335313335313": np.exp(np.linspace(0, 1.0, len(idx))) * 100,
        "PPIACO": np.exp(np.linspace(0, 0.4, len(idx))) * 100,
        "PCOPPUSDM": 8000 + rng.normal(0, 300, len(idx)),
        "EN_STEO_OPEC_SPARE": 3 + rng.normal(0, 0.5, len(idx)),
        "EN_STEO_OIL_DRAW": rng.normal(0, 0.5, len(idx)),
        "EN_STEO_HENRY_HUB": 4 + rng.normal(0, 0.5, len(idx)),
        "EN_CENSUS_DC": np.where(idx > "2022-12-31", 60000.0, 5000.0) + rng.normal(0, 300, len(idx)),
        "PRPWRCONS": 100000 + rng.normal(0, 3000, len(idx)),
        "GDP": 20000 * np.exp(np.linspace(0, 0.5, len(idx))),
        "WPU0542": np.exp(np.linspace(0, 0.5, len(idx))) * 100,
        "WPU0543": np.exp(np.linspace(0, 0.5, len(idx))) * 100,
        "EN_GPR": 100 + rng.normal(0, 15, len(idx)),
        "B935RC1Q027SBEA": 100 + rng.normal(0, 5, len(idx)),
        "PRMFGCONS": 80000 + rng.normal(0, 3000, len(idx)),
        "EN_EIA_CI_SALES": 250000 * np.exp(np.cumsum(growth)),
    }, index=idx)


def test_a_demand_surge_scores_tight_and_everything_stays_in_range():
    cfg = en.load_config()
    res = en.compute(cfg, synthetic_wide())
    comp, blocks = res["composite"], res["blocks"]
    assert set(blocks.columns) == {"power_demand", "power_supply", "power_prices", "fuels"}
    assert blocks["power_demand"].iloc[-1] > 0.3
    assert comp["power"].iloc[-1] > 0
    for frame in (blocks, comp):
        assert frame.abs().max().max() <= 1
    # Grid equipment outran all producer prices, so it scores tight.
    assert res["values"]["power_prices::grid_equipment"].iloc[-1] > 0


def test_supply_enters_power_tightness_with_its_sign_flipped():
    cfg = en.load_config()
    wide = synthetic_wide()
    loose = en.compute(cfg, wide)["composite"]["power"].iloc[-1]
    wide["EN_860M_FIRM_NET_ADD"] = wide["EN_860M_FIRM_NET_ADD"].where(wide.index < "2025-01-01", 9.0)
    more_supply = en.compute(cfg, wide)["composite"]["power"].iloc[-1]
    assert more_supply < loose


def test_every_series_the_config_names_is_required():
    cfg = en.load_config()
    req = set(en.required_series(cfg))
    assert {"PPIACO", "EN_EPOCH_US_POWER", "EN_STEO_ELEC_FWD"} <= req
    for _, ind in en.indicators(cfg):
        assert {"id", "label", "series", "transform", "sign", "weight", "why", "link"} <= set(ind)


def test_a_flow_against_gdp_and_a_twelve_month_average():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    wide = pd.DataFrame({"X": 5000.0, "GDP": 25000.0, "R": np.arange(24.0)}, index=idx)
    share = en.indicator_value({"series": ["X"], "transform": "share", "denominator": "GDP",
                                "factor": 0.1}, wide, idx[-1])
    assert share.iloc[-1] == pytest.approx(5000 / 25000 * 0.1 * 1)  # 0.02% of GDP
    avg = en.indicator_value({"series": ["R"], "transform": "mean_12m"}, wide, idx[-1])
    assert avg.index[0] == idx[11] and avg.iloc[0] == pytest.approx(5.5)


def test_every_indicator_explains_itself_briefly():
    """Two short paragraphs at most, so the page stays readable."""
    cfg = en.load_config()
    for _, ind in en.indicators(cfg):
        why = ind["why"] if isinstance(ind["why"], list) else [ind["why"]]
        assert len(why) <= 2 and all(len(p) <= 140 for p in why), ind["id"]


def test_a_block_approaches_but_never_sticks_at_the_bound():
    """More indicators at an extreme must still read higher, and never reach 1."""
    idx = pd.date_range("2010-01-31", periods=200, freq="ME")
    calm = np.r_[np.zeros(120), np.zeros(80)]
    cfg = {"meta": {"center_to": "2019-12-31", "min_history_months": 36},
           "blocks": {"b": {"indicators": [
               {"id": f"i{k}", "series": [f"S{k}"], "transform": "level", "sign": 1, "weight": 1}
               for k in range(3)]}},
           "combine": {"power": {"b": 1}, "energy": {"power": 1}}}
    wide = pd.DataFrame({f"S{k}": calm + np.r_[np.random.default_rng(k).normal(0, 1, 120),
                                                np.full(80, 50.0)] for k in range(3)}, index=idx)
    three = en.compute(cfg, wide)["blocks"]["b"].iloc[-1]
    wide["S2"] = wide["S2"].where(wide.index < idx[120], 0.0)
    two = en.compute(cfg, wide)["blocks"]["b"].iloc[-1]
    assert two < three < 1


def test_a_quiet_score_is_told_apart_from_two_big_forces_canceling():
    assert en.split({"spare": 1.4, "balance": -2.5, "gas": -0.2}, 1.0) == ("spare", "balance")
    assert en.split({"spare": 0.4, "balance": -2.5}, 1.0) is None
    assert en.split({"demand": 0.78, "supply": -0.67, "prices": 0.31}, 0.5) == ("demand", "supply")
    assert en.split({"only": 2.0}, 1.0) is None


def test_inventories_are_read_against_the_same_month_in_past_years():
    idx = pd.date_range("2015-01-31", periods=84, freq="ME")
    seasonal = np.tile([100.0, 110.0] * 6, 7)          # an even-odd monthly swing
    wide = pd.DataFrame({"S": seasonal}, index=idx)
    wide.iloc[-1, 0] = seasonal[-1] * 0.9                # last month 10% below its usual
    v = en.indicator_value({"series": ["S"], "transform": "dev_5y_same_month_pct"}, wide, idx[-1])
    assert v.iloc[-2] == pytest.approx(0.0)              # seasonal swing cancels
    assert v.iloc[-1] == pytest.approx(-10.0)


IMM_TABLE = """Table 5-24 RPM revenue by delivery year: 2007/2008 through 2027/2028
Delivery Year
Weighted Average RPM
Price ($ per MW-day)
2007/2008
$89.78
129,409.2
366
$4,252,287,381
2024/2025
$45.57
154,362.5
365
$2,567,425,124
2025/2026
$296.98
137,733.6
365
$14,930,075,226
"""


def test_the_imm_revenue_table_reads_as_prices_by_delivery_year():
    prices = parse_rpm_prices("intro mentions RPM revenue by delivery year in text\n" + IMM_TABLE)
    assert prices.to_dict() == {2007: 89.78, 2024: 45.57, 2025: 296.98}


def test_a_capacity_price_is_known_from_its_base_auction():
    assert bra_month(2015) == pd.Timestamp("2012-05-01")   # standard: May, three years ahead
    assert bra_month(2025) == pd.Timestamp("2024-07-01")   # the delayed schedule
    prices = pd.Series({2024: 45.57, 2025: 296.98})
    f = forward_capacity_price(prices, end=pd.Timestamp("2024-08-01"))
    assert f[pd.Timestamp("2024-06-01")] == 45.57         # 2025/26 not yet auctioned
    assert f[pd.Timestamp("2024-07-01")] == 296.98
