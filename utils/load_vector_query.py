import yaml

def load_vector_query(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "query" not in config:
        raise ValueError(f"'query' key not found in {path}")

    return {
        "query": config["query"],
        "k": config.get("k", 10),  # default fallback
        "name": config.get("name", None),
        "description": config.get("description", None)
    }