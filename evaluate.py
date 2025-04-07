import json
from recommender import get_top_k_recommendations

def recall_at_k(relevant, predicted, k):
    predicted_k = predicted[:k]
    relevant_set = set(map(lambda x: x.lower().strip(), relevant))
    predicted_set = set(map(lambda x: x.lower().strip(), predicted_k))
    retrieved_relevant = relevant_set.intersection(predicted_set)
    return len(retrieved_relevant) / len(relevant) if relevant else 0.0

def average_precision_at_k(relevant, predicted, k):
    relevant_set = set(map(lambda x: x.lower().strip(), relevant))
    score = 0.0
    num_hits = 0

    for i, p in enumerate(predicted[:k]):
        if p.lower().strip() in relevant_set:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)
            score += precision_at_i

    return score / min(k, len(relevant)) if relevant else 0.0

def evaluate(test_queries_path="data/test_queries.json", k=3):
    with open(test_queries_path, "r") as f:
        test_queries = json.load(f)

    total_recall = 0.0
    total_map = 0.0
    num_queries = len(test_queries)

    for test in test_queries:
        query = test["query"]
        relevant = test["relevant"]

        recommendations = get_top_k_recommendations(query, k=k)
        predicted = [rec["assessment_name"] for rec in recommendations]

        recall = recall_at_k(relevant, predicted, k)
        ap = average_precision_at_k(relevant, predicted, k)

        print(f"Query: {query}")
        print(f"  Predicted: {predicted}")
        print(f"  Expected: {relevant}")
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  AP@{k}: {ap:.4f}\n")

        total_recall += recall
        total_map += ap

    mean_recall = total_recall / num_queries
    mean_ap = total_map / num_queries

    print("====== Overall Evaluation ======")
    print(f"Mean Recall@{k}: {mean_recall:.4f}")
    print(f"MAP@{k}: {mean_ap:.4f}")

if __name__ == "__main__":
    evaluate()
