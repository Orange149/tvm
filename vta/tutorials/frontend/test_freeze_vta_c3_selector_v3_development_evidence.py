from freeze_vta_c3_selector_v3_development_evidence import resnet_decision


def development(median, wins):
    return {
        "boot_id": "boot",
        "all_outputs_equal": True,
        "paired_improvement": {
            "composable_vs_stock": {
                "median_pct": median, "wins": wins, "pairs": 9,
            }
        },
    }


def test_resnet_negative_result_falls_back_without_scheduling_confirmation():
    result = resnet_decision(development(0.05, 6))
    assert result["development_classification"] == "FALL_BACK_TO_INCUMBENT"
    assert result["formal_confirmation_scheduled"] is False


def test_resnet_result_must_pass_both_gain_and_win_gates():
    assert resnet_decision(development(3.0, 4))["formal_confirmation_scheduled"] is False
    assert resnet_decision(development(3.0, 5))["formal_confirmation_scheduled"] is True
