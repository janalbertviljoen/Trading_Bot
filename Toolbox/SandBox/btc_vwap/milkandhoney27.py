#!/usr/bin/env python3
"""
milkandhoney15_pulse_fixed_v9.py
FIX: ValueError in Plotly add_hline (opacity moved to top level).
FIX: Visual Volume data loading on launch (Robust 72h loop).
Logic: 66.9/18 Synergy + 96% Deep Fake + 8.8 Bridge.
"""

import time
import uuid
import datetime as dt
from typing import Tuple, List, Dict

import ccxt
import numpy as np
import pandas as pd
import pytz
import requests

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# -------------------
# CONFIG
# -------------------
SYMBOL_SPOT = "BTC/USDT"
TIMEZONE_LON = pytz.timezone("Europe/London")
BASE_TF = "3m"
REFRESH_INTERVAL_MS = 15_000
MAX_ROWS = 5000

# HARMONIC CONSTANTS
NATURAL_OCTAVE_STEP = 0.25
TERNARY_SEGMENT_MINS = 53.33333
SNIPER_WEDGES = [0, 3, 7, 10, 13, 17, 18, 20, 24]

# RAMANUJAN SINGULARITY MARKERS
S_LOW = 3.6
S_BRIDGE = 8.8
S_BEAT_MICRO = 18.0
S_MID = 36.3
S_BEAT_MACRO = 66.9
S_FAKE = 96.0


# -------------------
# ENGINES
# -------------------
def bootstrap_72h(ex, symbol, tz):
    """Forcefully fetches 72 hours of history by looping API calls."""
    all_ohlcv = []
    # 72 hours back in ms
    since = int((time.time() - (3 * 24 * 60 * 60)) * 1000)
    print(f"[*] Bootstrapping 72h Volume Intensity for {symbol}...")

    try:
        while len(all_ohlcv) < 1440:
            batch = ex.fetch_ohlcv(symbol, timeframe=BASE_TF, since=since, limit=500)
            if not batch:
                break
            all_ohlcv.extend(batch)
            since = batch[-1][0] + 1000
            time.sleep(0.1)
    except Exception as e:
        print(f"[!] Bootstrap partial failure: {e}")

    df = pd.DataFrame(all_ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(tz)
    df.set_index("ts", inplace=True)
    return df.astype(float).sort_index().tail(MAX_ROWS)


def get_time_resonance(now_lon: dt.datetime) -> Tuple[float, int, str]:
    total_mins = now_lon.hour * 60 + now_lon.minute
    wedge_idx = int(total_mins // TERNARY_SEGMENT_MINS)
    progress = ((total_mins / TERNARY_SEGMENT_MINS) % 1.0) * 100
    status = "NEUTRAL"
    if wedge_idx in SNIPER_WEDGES:
        if progress < 12:
            status = "🔥 INCEPTION"
        elif progress > 88:
            status = "⌛ CULMINATION"
        else:
            status = "ACTIVE PULSE"
    return progress, wedge_idx, status


def get_natural_octaves(current_price: float) -> List[float]:
    if current_price <= 0:
        return []
    root = np.sqrt(current_price)
    base_root = round(root / NATURAL_OCTAVE_STEP) * NATURAL_OCTAVE_STEP
    return sorted(
        [(base_root + (i * NATURAL_OCTAVE_STEP)) ** 2 for i in range(-12, 13)]
    )


# -------------------
# DASH UI
# -------------------
RUN_ID = uuid.uuid4().hex[:6]
URL_BASE = f"/mh15/{RUN_ID}/"
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    requests_pathname_prefix=URL_BASE,
    routes_pathname_prefix=URL_BASE,
)

app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id="store-cache"),
        dcc.Store(id="store-alert-state"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(
                            id="hud-price",
                            style={
                                "fontWeight": "900",
                                "fontSize": "64px",
                                "color": "#00ffcc",
                                "lineHeight": "0.8",
                            },
                        ),
                        html.Div(
                            id="hud-synergy",
                            style={
                                "fontSize": "18px",
                                "color": "#ffcc00",
                                "fontWeight": "bold",
                                "marginTop": "10px",
                            },
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H2(
                                    id="hud-objective",
                                    className="text-center",
                                    style={"fontWeight": "900"},
                                ),
                                html.Div(
                                    id="hud-substatus",
                                    className="text-center h5",
                                    style={"color": "#00ffcc"},
                                ),
                            ]
                        ),
                        id="objective-card",
                        style={"border": "4px solid #444"},
                    ),
                    md=5,
                ),
                dbc.Col(
                    [
                        html.Div(
                            id="hud-clock",
                            className="text-end h3",
                            style={"marginBottom": "0px"},
                        ),
                        html.Div(
                            id="hud-wedge",
                            className="text-end text-info h5",
                            style={"fontWeight": "bold"},
                        ),
                    ],
                    md=3,
                ),
            ],
            className="mt-3 g-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        "SNIPER FORECAST (LON)",
                                        className="small text-muted mb-2",
                                    ),
                                    html.Div(id="hud-forecast"),
                                ]
                            ),
                            style={"height": "100%"},
                        )
                    ],
                    md=3,
                ),
                dbc.Col(html.Div(id="m-lethality"), md=3),
                dbc.Col(html.Div(id="m-4h-tide"), md=3),
                dbc.Col(html.Div(id="m-3m-vel"), md=3),
            ],
            className="mt-2 g-2",
        ),
        dcc.Graph(
            id="alpha-chart", style={"height": "75vh"}, config={"displayModeBar": False}
        ),
        dcc.Interval(id="timer", interval=REFRESH_INTERVAL_MS, n_intervals=0),
    ],
)


def mk_meter(l, v, color="info", note=""):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(l, className="small text-muted"),
                html.Div(
                    f"{v:.1f}%", className="h3 mb-0", style={"fontWeight": "bold"}
                ),
                dbc.Progress(value=v, color=color, style={"height": "8px"}),
                html.Div(note, className="small text-muted mt-1") if note else None,
            ]
        ),
        className="p-0 shadow-sm",
    )


@app.callback(
    [
        Output("store-cache", "data"),
        Output("store-alert-state", "data"),
        Output("hud-price", "children"),
        Output("hud-synergy", "children"),
        Output("hud-objective", "children"),
        Output("hud-substatus", "children"),
        Output("objective-card", "style"),
        Output("hud-clock", "children"),
        Output("hud-wedge", "children"),
        Output("hud-forecast", "children"),
        Output("m-lethality", "children"),
        Output("m-4h-tide", "children"),
        Output("m-3m-vel", "children"),
        Output("alpha-chart", "figure"),
    ],
    [Input("timer", "n_intervals")],
    [State("store-cache", "data"), State("store-alert-state", "data")],
)
def update(n, cache_json, alert_state):
    now_lon = dt.datetime.now(TIMEZONE_LON)
    alert_state = alert_state or {"last_res_ts": 0, "fake_active": False}
    ex = ccxt.okx()

    # 1. BOOTSTRAP / FETCH
    try:
        if not cache_json:
            cache = bootstrap_72h(ex, SYMBOL_SPOT, TIMEZONE_LON)
        else:
            cache = pd.read_json(cache_json, orient="split")
            cache.index = pd.to_datetime(cache.index).tz_convert(TIMEZONE_LON)
            ohlcv = ex.fetch_ohlcv(SYMBOL_SPOT, timeframe=BASE_TF, limit=100)
            new_df = pd.DataFrame(
                ohlcv, columns=["ts", "o", "h", "l", "c", "v"]
            ).astype(float)
            new_df["ts"] = pd.to_datetime(
                new_df["ts"], unit="ms", utc=True
            ).dt.tz_convert(TIMEZONE_LON)
            new_df.set_index("ts", inplace=True)
            cache = pd.concat([cache, new_df]).sort_index()
            cache = cache[~cache.index.duplicated(keep="last")].tail(MAX_ROWS)
        price = float(cache["c"].iloc[-1])
    except:
        return [None] * 14

    # 2. CALCULATIONS
    time_prog, wedge_idx, time_status = get_time_resonance(now_lon)
    gann_levels = get_natural_octaves(price)
    closest_gann = min(gann_levels, key=lambda x: abs(x - price))
    in_hg = abs(price - closest_gann) < (price * 0.0012)

    def get_rank(series):
        return (
            series.rolling(200)
            .apply(lambda x: 100 * (np.sum(x <= x[-1]) / x.size), raw=True)
            .ewm(span=2)
            .mean()
        )

    t4h = get_rank((cache["v"] * cache["c"]).rolling(80).sum())
    t3m = get_rank((cache["v"] * cache["c"]).rolling(1).sum())
    v4h, v3 = t4h.iloc[-1], t3m.iloc[-1]

    # Logic Markers
    is_fake_spike = v3 > S_FAKE
    if is_fake_spike:
        alert_state["fake_active"] = True

    is_synergy = (abs(v4h - S_BEAT_MACRO) < 2.5) and (abs(v3 - S_BEAT_MICRO) < 2.5)
    is_cooled = v3 < S_BRIDGE  # Energy has left the trap

    # 3. LETHALITY SCORING
    lethality = 0
    if wedge_idx in SNIPER_WEDGES:
        lethality += 30
    if in_hg:
        lethality += 30
    if is_synergy:
        lethality += 20
    if alert_state["fake_active"] and is_cooled:
        lethality += 20  # The post-fake reversal logic

    resonance = lethality >= 70

    # 4. OBJECTIVE & ALERTS
    objective, status, bg = "OBSERVE", f"{time_status}", "#1a1a1a"
    synergy_text = f"BEAT: {'SYNCED' if is_synergy else 'WAITING'}"

    if is_fake_spike:
        objective, status, bg = (
            "⚠️ DEEP FAKE",
            "SINGULARITY CLIMAX | NO ENTRY",
            "#4d3d00",
        )
    elif resonance:
        direction = "LONG" if v4h > 50 else "SHORT"
        objective = f"🎯 SNIPER {direction}"
        status = "HARMONIC CONVERGENCE | PULSE LOADED"
        bg = "#004d40" if direction == "LONG" else "#4d000a"

        if time.time() - alert_state["last_res_ts"] > 1200:
            msg = f"🎯 *Sniper {direction} Alert*\nWedge: {wedge_idx} ({time_status})\nLethality: {lethality}%\nOctave: {closest_gann:,.0f}"
            requests.post(
                f"https://api.telegram.org/bot8372044720:AAHx9w6YqaygxgL_vgs_L_JYDGQZGHlxBac/sendMessage",
                json={"chat_id": "-4924532972", "text": msg, "parse_mode": "Markdown"},
            )
            alert_state["last_res_ts"] = time.time()
            alert_state["fake_active"] = False
    ###########################################
    # 5. UI ELEMENTS & CHART FIX
    midnight = now_lon.replace(hour=0, minute=0, second=0, microsecond=0)
    forecast = [
        html.Div(
            f"W{w} @ {(midnight + dt.timedelta(minutes=w * TERNARY_SEGMENT_MINS)).strftime('%H:%M')}",
            className="small",
        )
        for w in SNIPER_WEDGES
        if (midnight + dt.timedelta(minutes=w * TERNARY_SEGMENT_MINS)) > now_lon
    ][:5]

    afig = go.Figure().update_layout(
        template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), uirevision="const"
    )
    afig.add_trace(
        go.Scatter(x=t4h.index, y=t4h, name="4H TIDE", line=dict(color="cyan", width=3))
    )
    afig.add_trace(
        go.Scatter(
            x=t3m.index, y=t3m, name="3m VELOCITY", line=dict(color="yellow", width=1)
        )
    )

    # FIXED: Opacity moved to top level
    levels = [S_LOW, S_BRIDGE, S_BEAT_MICRO, S_MID, S_BEAT_MACRO, S_FAKE]
    labels = ["3.6", "8.8 COOL", "18.0 BEAT", "36.3", "66.9 BEAT", "96.0 FAKE"]
    for s, lab in zip(levels, labels):
        afig.add_hline(
            y=s,
            line=dict(
                color="orange" if "BEAT" in lab else "white", dash="dash", width=1
            ),
            opacity=0.3,
            annotation_text=lab,
        )

    afig.update_xaxes(range=[now_lon - dt.timedelta(hours=72), now_lon], type="date")
    afig.update_yaxes(range=[0, 100])

    return [
        cache.to_json(date_format="iso", orient="split"),
        alert_state,
        f"${price:,.2f}",
        synergy_text,
        objective,
        status,
        {"backgroundColor": bg, "border": "2px solid #00ffcc"},
        f"{now_lon:%H:%M:%S}",
        f"WEDGE {wedge_idx} | {time_prog:.1f}%",
        forecast,
        mk_meter("LETHALITY", lethality, color="danger" if resonance else "info"),
        mk_meter("4H TIDE", v4h, color="info"),
        mk_meter("3m VELOCITY", v3, note=f"Status: {time_status}"),
        afig,
    ]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
