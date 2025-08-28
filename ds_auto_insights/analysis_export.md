# Analysis Export

## Narrative
- The strongest correlations with goals scored (npg) are observed with an absolute correlation of 0.937, indicating a significant relationship between non-penalty goals and total goals.
- Mean goals scored by position show that Attacking Midfielders (AMR) lead with an average of 0.4 goals, while several positions, including Defensive Midfielders (DMR) and Fullbacks (FWL), average 0.0 goals.
- Among home teams, West Bromwich Albion, Watford, and Norwich each average 1.0 goals, while teams like Manchester City and Manchester United average 0.0 goals.
- For away teams, Brentford tops the list with an average of 1.0 goals, while several teams, including Liverpool and Aston Villa, average 0.0 goals.
- Next steps should focus on analyzing player performance metrics to identify specific factors that drive goal-scoring efficiency for Bukayo Saka.

## Plan
```json
{
  "steps": [
    {
      "op": "correlation"
    },
    {
      "op": "groupby_aggregate",
      "group_col": "position",
      "agg_col": "goals",
      "agg_fn": "mean"
    },
    {
      "op": "groupby_aggregate",
      "group_col": "h_team",
      "agg_col": "goals",
      "agg_fn": "mean"
    },
    {
      "op": "groupby_aggregate",
      "group_col": "a_team",
      "agg_col": "goals",
      "agg_fn": "mean"
    }
  ]
}
```

## Results
### Step 1: correlation
**Top correlation pairs (by |corr|)**
| col_a     | col_b     |   abs_corr |
|:----------|:----------|-----------:|
| season    | id        |   0.993551 |
| id        | season    |   0.993551 |
| roster_id | id        |   0.958292 |
| id        | roster_id |   0.958292 |
| season    | roster_id |   0.946927 |
| roster_id | season    |   0.946927 |
| npg       | goals     |   0.937092 |
| goals     | npg       |   0.937092 |
| npxG      | xG        |   0.894507 |
| xG        | npxG      |   0.894507 |

### Step 2: groupby_aggregate
**Group:** `position` — **Metric:** `goals` — **Agg:** `mean`
| position   |   mean(goals) |
|:-----------|--------------:|
| AMR        |      0.4      |
| FWR        |      0.333333 |
| MC         |      0.333333 |
| MR         |      0.2      |
| AMC        |      0        |
| AML        |      0        |
| DL         |      0        |
| DML        |      0        |
| DMR        |      0        |
| FWL        |      0        |
| ML         |      0        |
| Sub        |      0        |

### Step 3: groupby_aggregate
**Group:** `h_team` — **Metric:** `goals` — **Agg:** `mean`
| h_team                  |   mean(goals) |
|:------------------------|--------------:|
| West Bromwich Albion    |      1        |
| Watford                 |      1        |
| Norwich                 |      1        |
| Leeds                   |      0.5      |
| Wolverhampton Wanderers |      0.333333 |
| Southampton             |      0.333333 |
| Chelsea                 |      0.333333 |
| Aston Villa             |      0.333333 |
| Arsenal                 |      0.16     |
| Everton                 |      0        |
| Leicester               |      0        |
| Crystal Palace          |      0        |
| Manchester City         |      0        |
| Manchester United       |      0        |
| Newcastle United        |      0        |

### Step 4: groupby_aggregate
**Group:** `a_team` — **Metric:** `goals` — **Agg:** `mean`
| a_team               |   mean(goals) |
|:---------------------|--------------:|
| Brentford            |      1        |
| Newcastle United     |      0.666667 |
| Tottenham            |      0.5      |
| Sheffield United     |      0.5      |
| Manchester United    |      0.5      |
| Manchester City      |      0.333333 |
| Chelsea              |      0.333333 |
| Arsenal              |      0.191489 |
| West Ham             |      0        |
| West Bromwich Albion |      0        |
| Watford              |      0        |
| Southampton          |      0        |
| Norwich              |      0        |
| Liverpool            |      0        |
| Aston Villa          |      0        |
