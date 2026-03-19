import numpy as np

from src.ai.reid.identity_decision import IdentityDecision


class _FakeStore:
    def __init__(self, candidates):
        self._candidates = candidates

    def __len__(self):
        return len(self._candidates)

    def search_animal_ids(self, _embedding, k=5):
        return self._candidates[:k]


def test_decide_rejects_when_top1_top2_margin_is_small():
    decision = IdentityDecision(similarity_threshold=0.45, top1_top2_margin=0.10)
    store = _FakeStore([(10, 0.78), (11, 0.74)])

    animal_id, score = decision.decide(np.array([1.0, 0.0], dtype=np.float32), store, top_k=5)

    assert animal_id is None
    assert score == 0.78


def test_decide_accepts_when_threshold_and_margin_are_satisfied():
    decision = IdentityDecision(similarity_threshold=0.45, top1_top2_margin=0.10)
    store = _FakeStore([(10, 0.82), (11, 0.66)])

    animal_id, score = decision.decide(np.array([1.0, 0.0], dtype=np.float32), store, top_k=5)

    assert animal_id == 10
    assert score == 0.82
