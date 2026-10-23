from __future__ import annotations
import json
from pathlib import Path
import server


def export_tree(estimator, labels):
    tree = estimator.tree_
    def visit(node):
        left, right = int(tree.children_left[node]), int(tree.children_right[node])
        if left == right:
            raw = tree.value[node][0].tolist()
            total = sum(raw) or 1.0
            probs = [0.0] * len(labels)
            for source_index, source_label in enumerate(estimator.classes_.tolist()):
                label = labels[int(source_label)] if isinstance(source_label, (int, float)) else str(source_label)
                probs[labels.index(label)] = float(raw[source_index] / total)
            return {"p": probs}
        return {"f": int(tree.feature[node]), "t": float(tree.threshold[node]), "l": visit(left), "r": visit(right)}
    return visit(0)

labels = [str(x) for x in server.models["forest"].classes_.tolist()]
metrics = {"rows": server.metrics["rows"], "testRows": server.metrics["testRows"], "split": server.metrics["split"]}
for key in ("tree", "forest"):
    metrics[key] = {"accuracy": server.metrics[key]["accuracy"], "f1": server.metrics[key]["f1"]}
payload = {
    "features": server.FEATURES,
    "classes": labels,
    "tree": export_tree(server.models["tree"], labels),
    "forest": [export_tree(t, labels) for t in server.models["forest"].estimators_],
    "metrics": metrics,
}
Path("models.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
print("Exported trained model bundle", len(payload["forest"]), "forest trees")
