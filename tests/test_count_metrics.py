from src.ai.metrics.count_metrics import compute_tracklet_evaluation_metrics


def test_compute_tracklet_evaluation_metrics_counts_valid_false_and_duplicates():
    tracklets = [
        {"track_id": 1, "animal_id": 10, "enrolled": True, "num_embeddings": 12},
        {"track_id": 2, "animal_id": 10, "enrolled": True, "num_embeddings": 11},
        {"track_id": 3, "animal_id": 11, "enrolled": True, "num_embeddings": 15},
        {"track_id": 4, "animal_id": None, "enrolled": False, "num_embeddings": 20},
        {"track_id": 5, "animal_id": 12, "enrolled": True, "num_embeddings": 9},
        {"track_id": 6, "animal_id": 13, "enrolled": False, "num_embeddings": 14},
    ]

    metrics = compute_tracklet_evaluation_metrics(tracklets)

    assert metrics.valid_animals == 3
    assert metrics.false_tracks == 2
    assert metrics.duplicate_animals == 1
    assert metrics.predicted_count == 2
