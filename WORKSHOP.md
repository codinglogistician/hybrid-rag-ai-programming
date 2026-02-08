
# Routing w RAG — warsztat

## Cel

Zbudować workflow, który:

* klasyfikuje pytanie użytkownika
* zapisuje kategorię (`route`)
* wykonuje retrieval
* generuje odpowiedź

Pipeline:

```
query → router → retrieve → generate
```

---

## Krok 1 — Uruchom workflow

Uruchom:

```python
result = graph_routing.invoke({
    "query": "Jak uruchomić kontener Docker?"
})

print(result["answer"])
```

Sprawdź:

* czy router zwraca `route`
* czy generowana jest odpowiedź

---

## Krok 2 — Podejrzyj decyzję routera

Dodaj debug:

```python
def router_node(state: RoutingState):
    decision = router_llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=state["query"]),
        ]
    )
    print("ROUTE:", decision.step)
    return {"route": decision.step}
```

Uruchom:

```python
graph_routing.invoke({"query": "Jak działa Docker networking?"})
```

---

## Krok 3 — Test różnych zapytań  z langsmith

Przetestuj:

```
Jak zbudować obraz Docker?
Co to jest kontener?
Jak zainstalować Docker Desktop?
Docker port mapping
```

Sprawdź:

* czy router zmienia kategorię
* czy retrieval działa poprawnie

---

## Krok 4 — Debug retrieval

Dodaj:

```python
print("CONTEXT:", context[:200])
```

w:

```python
routing_retrieve_then_generate
```

Uruchom workflow ponownie.

---

## Krok 5 — Eksperyment z routerem

Zmodyfikuj prompt routera (`ROUTER_SYSTEM`), np.:

```
Zwracaj jedną z kategorii:
installation
containers
networking
images
```

Uruchom workflow i sprawdź:

* czy klasyfikacja się zmienia
* czy wpływa to na odpowiedź

---

## Krok 6 — Rozszerzenie (opcjonalne)

Dodaj kategorię:

```
direct_answer
```

i obsługę w workflow:

* jeśli `route == "direct_answer"` → pomiń retrieval
