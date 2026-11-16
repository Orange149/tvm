import pytest
from summarize_common_tail_comparison import average_ranks, spearman, method_statistics


def test_tied_prediction_scores_do_not_create_spurious_correlation():
    assert average_ranks([3, 1, 1, 4]) == [3, 1.5, 1.5, 4]
    assert spearman([1, 1, 1], [2, 3, 4]) is None
    assert spearman([1, 2, 3], [6, 4, 2]) == pytest.approx(-1)


def test_static_first_and_best_have_different_meanings():
    rows = [dict(configuration=str(i), predicted_II_ns=100+i, predicted_II_ms=(100+i)/1e6,
                 actual_II_ms=40-i, FPS=1000/(40-i), static_rank=i+1,
                 signed_II_error_percent=-50) for i in range(20)]
    result = method_statistics(rows)
    assert result['first_rank'] == '0' and result['actual_best_static_rank'] == 20
    assert result['best_FPS'] > result['first_FPS']
    with pytest.raises(ValueError):
        method_statistics(rows[:-1])
