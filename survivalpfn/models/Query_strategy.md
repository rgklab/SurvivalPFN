# Query Strategies

`InContextModel` appends a delta feature to every context/query row. For query rows, this feature tells the model which distribution to predict:

- `1`: event-time distribution
- `0`: censor-time distribution

Let:

- `E_query`: latent event time
- `C_query`: latent censor time
- `T_obs = min(E_query, C_query)`: observed survival time
- `delta_event = 1[E_query < C_query]`: event is observed before censoring
- `delta_censor = 1 - delta_event`: censoring is observed before event

## Summary

| Strategy | Query input delta | Target time | Loss | Query length |
| --- | --- | --- | --- | --- |
| `random` | Randomly choose event (`1`) or censor (`0`) per query | `E_query` if event query, otherwise `C_query` | KL / cross-entropy on fully observed synthetic target | `query_len` |
| `event` | Always event (`1`) | `T_obs` | Survival NLL with `delta_event` | `query_len` |
| `both` | Duplicate every query: one event (`1`) and one censor (`0`) | `T_obs` for both copies | Survival NLL with `[delta_event, delta_censor]` | `2 * query_len` |
| `both_fix_len` | Randomly choose event (`1`) or censor (`0`) per query | `T_obs` | Survival NLL for the chosen process | `query_len` |

## `random`

`random` asks the model for exactly one fully observed latent process time per query row.

```text
query_input_delta ~ Bernoulli(event_rate)
T_query = E_query if query_input_delta == 1 else C_query
loss = KL(logits, T_query)
```

This is valid in synthetic training because both `E_query` and `C_query` are known. It does not train with right-censored likelihood; it directly supervises the selected latent time.

## `event`

`event` always asks for the event-time distribution and trains from the observed survival outcome.

```text
query_input_delta = 1
T_query = min(E_query, C_query)
delta_query = 1[E_query < C_query]
loss = survival NLL(logits, T_query, delta_query)
```

If the event happens first, the event time is observed. If censoring happens first, the event time is right-censored at `C_query`.

## `both`

`both` asks for both latent process distributions for every query row by duplicating the query set.

```text
query_input_delta = [1, 0]
T_query = [T_obs, T_obs]
delta_query = [delta_event, delta_censor]
loss = survival NLL(logits, T_query, delta_query)
```

For the event query, censoring first means the event time is right-censored. For the censor query, event first means the censor time is right-censored. This is the most complete NLL supervision, but it doubles the query sequence length.

## `both_fix_len`

`both_fix_len` is a fixed-length approximation of `both`. It chooses one process query per original query row, then applies the same NLL semantics as `both` for that chosen process.

```text
query_input_delta ~ Bernoulli(event_rate)
T_query = min(E_query, C_query)
delta_query = delta_event if query_input_delta == 1 else delta_censor
loss = survival NLL(logits, T_query, delta_query)
```

It keeps the same `context_len + query_len` sequence length as `random` and `event`, so it is suitable when training uses multiple train/query split buckets. In that setting, `query_strategy="both"` is automatically changed to `both_fix_len`.

## Example

For one query row with `E_query = 5` and `C_query = 8`:

```text
T_obs = 5
delta_event = 1
delta_censor = 0
```

- `random` with event query trains directly on `T_query = 5`.
- `random` with censor query trains directly on `T_query = 8`.
- `event` trains event NLL with observed event at `5`.
- `both` trains event NLL with observed event at `5`, and censor NLL with censor time right-censored at `5`.
- `both_fix_len` trains either the event-query NLL or the censor-query NLL for this row, depending on the sampled `query_input_delta`.
