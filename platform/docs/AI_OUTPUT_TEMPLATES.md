# AI Output Templates

Every AI response in Sentinel follows one of the templates below. Templates exist to
prevent hallucination drift, enable consistent UI rendering, and keep compliance auditable.

---

## Template A — Stock Analysis (full)

```
{
  "summary": "{SYMBOL} composite score {X}/100. {LONG|WAIT|AVOID} bias.
              Entry {a}-{b}, stop {s}, targets {t1}/{t2}, R:R {r}.",
  "blocks": [
    { "agent": "Fundamentals", "tone": "bull|bear|neutral",
      "headline": "Revenue {x}%, EPS {y}%.",
      "bullets": ["Gross margin …", "FCF …", "Forward PE …", "Catalysts: …"] },
    { "agent": "Technical", "tone": "...", "headline": "...", "bullets": [...] },
    { "agent": "News",      "tone": "...", "headline": "...", "bullets": [...] },
    { "agent": "Risk",      "tone": "...", "headline": "...", "bullets": [...] }
  ],
  "disclaimer": "For educational and research purposes only..."
}
```

## Template B — Macro / Regime

```
{
  "summary": "Macro regime: {RISK-ON|RISK-OFF|MIXED}. Trend stage: {1|2|3|sideways|bear}.",
  "blocks": [
    { "agent": "Macro", "tone": "...", "headline": "...", "bullets": [
        "QQQ above 21/50/200 EMA…",
        "VIX < 16 — risk-on…",
        "10Y rolling over — supportive…",
        "USD weak — global liquidity tailwind…"
    ]}
  ],
  "disclaimer": "..."
}
```

## Template C — Sector Rotation

```
{
  "summary": "Leadership: {sector1}, {sector2}, {sector3}.",
  "blocks": [
    { "agent": "Sector Rotation", "tone": "bull",
      "headline": "Top 3 sectors by RS …",
      "bullets": [
        "{Sector} — RS {x}, flow {y}, {narrative}",
        "Laggards: {…}; avoid or hedge."
      ]
    }
  ],
  "disclaimer": "..."
}
```

## Template D — Risk Alert

```
{
  "summary": "ALERT — {severity}: {message}",
  "blocks": [
    { "agent": "Risk", "tone": "bear",
      "headline": "Concentration / Vol / Earnings tail …",
      "bullets": ["Specific exposure …", "Recommended action …", "Hedge instrument …"]
    }
  ]
}
```

## Template E — Journal Behavioral Summary

```
{
  "summary": "Your largest leak this month: {emotion} on {N} trades, avg -{x}%.",
  "blocks": [
    { "agent": "Behavioral",
      "headline": "Pattern detected: {pattern}",
      "bullets": ["Recommendation 1", "Recommendation 2", "Rule to add"]
    }
  ]
}
```

---

## Tone Rules

| Tone | UI color | When to use |
|---|---|---|
| `bull` | green border + tint | Net positive evidence |
| `bear` | red border + tint | Net negative evidence |
| `neutral` | plain border | Mixed or insufficient data |

## Mandatory Postscript

Every response ends with:

> "For educational and research purposes only. Not investment advice. AI outputs may contain errors."

This appears in the UI as small italic dim text under the response.
