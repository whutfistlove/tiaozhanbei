"""
llm_cost_model 的单元测试（创新点 C）。

测试策略：
- 物理过滤用确定性逻辑，直接验证
- LLM 预测用 mock predict_fn 模拟
- 双层过滤的拒绝/通过逻辑
- 限流（max_survivors）
- 校准机制

运行：pytest tests/test_llm_cost_model.py -v
"""
import pytest
from agent_system.llm_cost_model import (
    Candidate, FilterStats, PredictionRecord,
    physics_filter, two_stage_filter, calibrate,
)


def make_candidate(cid, speedup=None, confidence=None, split_k=1, tile_n=16):
    return Candidate(
        candidate_id=cid,
        description=f"split_k={split_k}, tile_n={tile_n}",
        params={"split_k": split_k, "tile_n": tile_n},
        predicted_speedup=speedup,
        confidence=confidence,
    )


class TestPhysicsFilter:
    def test_feasible_speedup_passes(self):
        c = make_candidate("c1", speedup=1.5)
        ok, _ = physics_filter(c, baseline_util=0.45)
        assert ok

    def test_infeasible_speedup_rejected(self):
        c = make_candidate("c1", speedup=20.0)
        ok, reason = physics_filter(c, baseline_util=0.45)
        assert not ok
        assert "roofline" in reason

    def test_no_speedup_checks_params(self):
        c = make_candidate("c1", speedup=None, split_k=1, tile_n=16)
        ok, _ = physics_filter(c, baseline_util=0.45)
        assert ok

    def test_invalid_split_k(self):
        c = make_candidate("c1", split_k=0)
        ok, reason = physics_filter(c, baseline_util=0.45)
        assert not ok
        assert "split_k" in reason

    def test_invalid_tile_n(self):
        c = make_candidate("c1", tile_n=999)
        ok, reason = physics_filter(c, baseline_util=0.45)
        assert not ok
        assert "tile_n" in reason


class TestTwoStageFilter:
    def test_all_pass_without_predict(self):
        """无 LLM 预测时，只走物理过滤"""
        candidates = [make_candidate(f"c{i}", split_k=i+1) for i in range(5)]
        survivors, stats = two_stage_filter(candidates, baseline_util=0.45)
        assert stats.total == 5
        assert stats.rejected_by_physics == 0
        assert len(survivors) == 5

    def test_physics_rejection(self):
        candidates = [
            make_candidate("good", speedup=1.5, confidence=0.8),
            make_candidate("bad", speedup=50.0, confidence=0.9),  # 违反 roofline
        ]
        survivors, stats = two_stage_filter(candidates, baseline_util=0.45)
        assert stats.rejected_by_physics == 1
        assert len(survivors) == 1
        assert survivors[0].candidate_id == "good"

    def test_confidence_filter(self):
        candidates = [
            make_candidate("hi", confidence=0.9),
            make_candidate("lo", confidence=0.2),
        ]
        survivors, stats = two_stage_filter(
            candidates, baseline_util=0.45, min_confidence=0.5
        )
        assert stats.rejected_by_confidence == 1
        assert len(survivors) == 1
        assert survivors[0].candidate_id == "hi"

    def test_max_survivors_limit(self):
        candidates = [make_candidate(f"c{i}", confidence=0.9-i*0.05) for i in range(10)]
        survivors, stats = two_stage_filter(
            candidates, baseline_util=0.45, max_survivors=3
        )
        assert len(survivors) == 3
        # 应保留置信度最高的 3 个
        confs = [c.confidence for c in survivors]
        assert confs == sorted(confs, reverse=True)

    def test_mock_predict_fn(self):
        """用 mock LLM 预测函数"""
        def mock_predict(cand, util):
            # split_k 越大预测越快，但有上限
            sk = cand.params.get("split_k", 1)
            return (min(sk * 0.5, 3.0), 0.7)
        candidates = [make_candidate(f"c{i}", split_k=i+1) for i in range(6)]
        survivors, stats = two_stage_filter(
            candidates, baseline_util=0.45, predict_fn=mock_predict
        )
        # 所有候选都被 mock 预测
        for c in candidates:
            assert c.predicted_speedup is not None
            assert c.confidence is not None

    def test_empty_candidates(self):
        survivors, stats = two_stage_filter([], baseline_util=0.45)
        assert len(survivors) == 0
        assert stats.survival_rate == 0.0

    def test_survival_rate(self):
        candidates = [
            make_candidate("p1", confidence=0.9),
            make_candidate("p2", confidence=0.8),
            make_candidate("r1", confidence=0.1),  # 被置信度拒
        ]
        _, stats = two_stage_filter(candidates, baseline_util=0.45, min_confidence=0.5)
        assert stats.survival_rate == pytest.approx(2/3, abs=0.01)


class TestCalibration:
    def test_empty_records(self):
        r = calibrate([])
        assert r["degrade_to_full"] is False

    def test_accurate_predictions(self):
        records = [
            PredictionRecord("c1", 2.0, 1.9, 0.1, True),
            PredictionRecord("c2", 1.5, 1.6, 0.1, True),
        ]
        r = calibrate(records)
        assert r["direction_accuracy"] == 1.0
        assert not r["degrade_to_full"]

    def test_bad_direction_suggests_degrade(self):
        """方向准确率<50% 应建议退化到全量评测"""
        records = [
            PredictionRecord("c1", 3.0, 0.5, 2.5, False),   # 预测快实际慢
            PredictionRecord("c2", 0.5, 2.0, 1.5, False),   # 预测慢实际快
            PredictionRecord("c3", 1.0, 1.0, 0.0, True),
        ]
        r = calibrate(records)
        assert r["direction_accuracy"] < 0.5
        assert r["degrade_to_full"]

    def test_mae_calculation(self):
        records = [
            PredictionRecord("c1", 2.0, 1.5, 0.5, True),
            PredictionRecord("c2", 1.0, 1.5, 0.5, True),
        ]
        r = calibrate(records)
        assert r["mae"] == pytest.approx(0.5)
