from agent_system.strategy_schema import ChangeProposal, PatchApplyError, apply_change_proposal


def test_apply_change_proposal_replaces_unique_snippet():
    src = "static constexpr int NUM_SPLITS = 12;\nint x = 1;\n"
    proposal = ChangeProposal(
        proposal_id="p1",
        target="NUM_SPLITS",
        before="static constexpr int NUM_SPLITS = 12;",
        after="static constexpr int NUM_SPLITS = 8;",
    )

    out = apply_change_proposal(src, proposal)

    assert "NUM_SPLITS = 8" in out
    assert "NUM_SPLITS = 12" not in out


def test_apply_change_proposal_rejects_missing_snippet():
    proposal = ChangeProposal(
        proposal_id="p1",
        target="NUM_SPLITS",
        before="static constexpr int NUM_SPLITS = 16;",
        after="static constexpr int NUM_SPLITS = 8;",
    )

    try:
        apply_change_proposal("static constexpr int NUM_SPLITS = 12;", proposal)
    except PatchApplyError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("expected PatchApplyError")


def test_apply_change_proposal_rejects_ambiguous_snippet():
    proposal = ChangeProposal(
        proposal_id="p1",
        target="x",
        before="int x = 1;",
        after="int x = 2;",
    )

    try:
        apply_change_proposal("int x = 1;\nint x = 1;\n", proposal)
    except PatchApplyError as exc:
        assert "matched 2 places" in str(exc)
    else:
        raise AssertionError("expected PatchApplyError")
