import subprocess
import sys
from pathlib import Path


def test_reproduce_local_mvp_script_has_valid_bash_syntax():
    script = Path("scripts/reproduce_local_mvp.sh")
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduce_final_diagnostic_script_has_valid_bash_syntax():
    script = Path("scripts/reproduce_final_diagnostic.sh")
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduce_missing_baselines_script_has_valid_bash_syntax():
    script = Path("scripts/reproduce_missing_baselines.sh")
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduce_syntactic_immediate_matched_script_has_valid_bash_syntax():
    script = Path("scripts/reproduce_syntactic_immediate_matched.sh")
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduce_heldout_stress_script_has_valid_bash_syntax():
    script = Path("scripts/reproduce_heldout_stress.sh")
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduce_update_budget_stress_script_has_valid_bash_syntax():
    script = Path("scripts/reproduce_update_budget_stress.sh")
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduce_gpt2_lora_smoke_script_has_valid_bash_syntax():
    script = Path("scripts/reproduce_gpt2_lora_smoke.sh")
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduce_gpt2_hlc_k2_smoke_script_has_valid_bash_syntax():
    script = Path("scripts/reproduce_gpt2_hlc_k2_smoke.sh")
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduce_gpt2_long_horizon_script_has_valid_bash_syntax():
    script = Path("scripts/reproduce_gpt2_long_horizon.sh")
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduce_pythia410m_lora_v100_script_has_valid_bash_syntax():
    script = Path("scripts/reproduce_pythia410m_lora_v100_smoke.sh")
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduce_qwen3_lora_v100_script_has_valid_bash_syntax():
    script = Path("scripts/reproduce_qwen3_lora_v100_smoke.sh")
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_syntactic_immediate_sweep_script_has_valid_python_syntax():
    script = Path("scripts/run_syntactic_immediate_sweep.py")
    assert script.exists()
    result = subprocess.run([sys.executable, "-m", "py_compile", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_grad_ascent_sweep_script_has_valid_python_syntax():
    script = Path("scripts/run_grad_ascent_sweep.py")
    assert script.exists()
    result = subprocess.run([sys.executable, "-m", "py_compile", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_make_heldout_stress_relearn_pools_script_has_valid_python_syntax():
    script = Path("scripts/make_heldout_stress_relearn_pools.py")
    assert script.exists()
    result = subprocess.run([sys.executable, "-m", "py_compile", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_selected_baseline_stress_eval_script_has_valid_python_syntax():
    script = Path("scripts/run_selected_baseline_stress_eval.py")
    assert script.exists()
    result = subprocess.run([sys.executable, "-m", "py_compile", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_strong_npo_sweep_supports_non_default_eval_model():
    script = Path("scripts/run_strong_npo_sweep.py")
    text = script.read_text(encoding="utf-8")
    assert "--base_model" in text
    assert "--dtype" in text
    assert "--eval_max_length" in text


def test_grad_ascent_sweep_supports_non_default_eval_model():
    script = Path("scripts/run_grad_ascent_sweep.py")
    text = script.read_text(encoding="utf-8")
    assert "--base_model" in text
    assert "--dtype" in text
    assert "--eval_max_length" in text


def test_selected_baseline_stress_eval_supports_non_default_eval_model():
    script = Path("scripts/run_selected_baseline_stress_eval.py")
    text = script.read_text(encoding="utf-8")
    assert "--base_model" in text
    assert "--dtype" in text
    assert "--max_length" in text


def test_update_budget_stress_report_script_has_valid_python_syntax():
    script = Path("scripts/report_update_budget_stress.py")
    assert script.exists()
    result = subprocess.run([sys.executable, "-m", "py_compile", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduce_final_diagnostic_verify_does_not_retrain():
    script = Path("scripts/reproduce_final_diagnostic.sh")
    text = script.read_text(encoding="utf-8")
    forbidden = [
        "scripts/train_full.py",
        "scripts/train_retrain_minus_f.py",
        "scripts/unlearn.py",
        "scripts/run_survival_curve.py",
        "scripts/run_hlc_stress_sweep.py",
    ]
    for command in forbidden:
        assert command not in text
    assert "scripts/make_diagnostic_report.py" in text
    assert "docs/EVALUATION_PROTOCOL.md" in text
    assert "docs/MVP_CONCLUSION.md" in text


def test_reproduce_missing_baselines_defaults_to_paraphrase_stress():
    script = Path("scripts/reproduce_missing_baselines.sh")
    text = script.read_text(encoding="utf-8")
    assert 'BASELINES="${BASELINES:-npo_paraphrase}"' in text
    assert "forget_declarative" in text
    assert "forget_paraphrase_qa" in text
    assert "forget_mixed_stress" in text
    assert "configs/method/npo_paraphrase_fullft_smoke.yaml" in text
    assert "configs/method/syntactic_diversification_fullft_smoke.yaml" in text
    assert "baseline_family_label" in text
    assert "scripts/report_conditional_durability.py" in text


def test_reproduce_syntactic_immediate_matched_runs_selection_and_stress():
    script = Path("scripts/reproduce_syntactic_immediate_matched.sh")
    text = script.read_text(encoding="utf-8")
    assert "scripts/run_syntactic_immediate_sweep.py" in text
    assert "scripts/select_immediate_matched.py" in text
    assert "scripts/run_selected_baseline_stress_eval.py" in text
    assert "runs/survival/syntactic_sweep_match_selected.csv" in text
    assert "runs/reports/syntactic_immediate_matched_stress" in text


def test_reproduce_heldout_stress_runs_all_selected_families():
    script = Path("scripts/reproduce_heldout_stress.sh")
    text = script.read_text(encoding="utf-8")
    assert "scripts/make_heldout_stress_relearn_pools.py" in text
    assert "heldout_forget_declarative" in text
    assert "heldout_forget_paraphrase_qa" in text
    assert "heldout_forget_mixed_stress" in text
    assert "runs/survival/strong_npo_sweep_match_selected.csv" in text
    assert "runs/survival/syntactic_sweep_match_selected.csv" in text
    assert "npo_paraphrase_fullft_tofu50_seed" in text


def test_reproduce_update_budget_stress_runs_stronger_heldout_budget():
    script = Path("scripts/reproduce_update_budget_stress.sh")
    text = script.read_text(encoding="utf-8")
    assert 'BUDGET_LRS="${BUDGET_LRS:-0.01}"' in text
    assert "heldout_forget_declarative" in text
    assert "heldout_forget_paraphrase_qa" in text
    assert "heldout_forget_mixed_stress" in text
    assert "runs/survival/strong_npo_sweep_match_selected.csv" in text
    assert "runs/survival/syntactic_sweep_match_selected.csv" in text
    assert "npo_paraphrase_fullft_tofu50_seed" in text
    assert "scripts/report_update_budget_stress.py" in text


def test_reproduce_gpt2_lora_smoke_runs_larger_model_lane():
    script = Path("scripts/reproduce_gpt2_lora_smoke.sh")
    text = script.read_text(encoding="utf-8")
    assert "configs/model/gpt2_lora_smoke.yaml" in text
    assert "configs/method/strong_npo_gpt2_lora_smoke.yaml" in text
    assert "configs/method/hlc_sg_k1_gpt2_lora_smoke.yaml" in text
    assert "--base_model gpt2" in text
    assert "checkpoints/gpt2_lora_full_seed" in text
    assert "thresholds/tofu50_gpt2_lora_seed" in text
    assert "heldout_forget_mixed_stress" in text
    assert "scripts/report_conditional_durability.py" in text


def test_reproduce_gpt2_hlc_k2_smoke_reuses_base_lane_and_compares_k1():
    script = Path("scripts/reproduce_gpt2_hlc_k2_smoke.sh")
    text = script.read_text(encoding="utf-8")
    assert "configs/method/hlc_sg_k2_gpt2_lora_smoke.yaml" in text
    assert "checkpoints/gpt2_lora_full_seed" in text
    assert "checkpoints/unlearned/npo_strong_gpt2_lora_tofu50_seed" in text
    assert "checkpoints/unlearned/hlc_sg_k1_gpt2_lora_tofu50_seed" in text
    assert "Run scripts/reproduce_gpt2_lora_smoke.sh --skip-tests first." in text
    assert "runs/survival/gpt2_lora_smoke/hlc_sg_k1_gpt2_lora_seed" in text
    assert "HLC-SG K2 GPT-2" in text


def test_reproduce_gpt2_long_horizon_reuses_k1_k2_and_strong_npo():
    script = Path("scripts/reproduce_gpt2_long_horizon.sh")
    text = script.read_text(encoding="utf-8")
    assert 'STEPS="${STEPS:-0,1,2,4,8,16,32,64,128}"' in text
    assert 'REPORT_HORIZONS="${REPORT_HORIZONS:-32,128}"' in text
    assert "checkpoints/unlearned/npo_strong_gpt2_lora_tofu50_seed" in text
    assert "checkpoints/unlearned/hlc_sg_k1_gpt2_lora_tofu50_seed" in text
    assert "checkpoints/unlearned/hlc_sg_k2_gpt2_lora_tofu50_seed" in text
    assert "heldout_forget_mixed_stress" in text
    assert "runs/reports/gpt2_long_horizon" in text
    assert "scripts/report_conditional_durability.py" in text


def test_reproduce_pythia410m_lora_v100_uses_fp16_and_pythia_base():
    script = Path("scripts/reproduce_pythia410m_lora_v100_smoke.sh")
    text = script.read_text(encoding="utf-8")
    assert 'BASE_MODEL="${BASE_MODEL:-EleutherAI/pythia-410m}"' in text
    assert 'DTYPE="${DTYPE:-float16}"' in text
    assert 'SKIP_PREP="${SKIP_PREP:-0}"' in text
    assert "configs/model/pythia410m_lora_v100.yaml" in text
    assert "configs/method/strong_npo_pythia410m_lora_v100.yaml" in text
    assert "configs/method/hlc_sg_k2_pythia410m_lora_v100.yaml" in text
    assert "thresholds/tofu50_pythia410m_lora_seed" in text
    assert "heldout_forget_mixed_stress" in text
    assert "scripts/report_conditional_durability.py" in text


def test_reproduce_qwen3_lora_v100_uses_modern_qwen_defaults():
    script = Path("scripts/reproduce_qwen3_lora_v100_smoke.sh")
    text = script.read_text(encoding="utf-8")
    assert 'MODEL_TAG="${MODEL_TAG:-qwen3_0p6b}"' in text
    assert 'BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"' in text
    assert 'DTYPE="${DTYPE:-float16}"' in text
    assert "configs/model/qwen3_0p6b_lora_v100.yaml" in text
    assert "configs/model/qwen3_1p7b_lora_v100.yaml" in text
    assert "configs/method/strong_npo_qwen3_lora_v100.yaml" in text
    assert "configs/method/hlc_sg_k2_qwen3_lora_v100.yaml" in text
    assert "thresholds/tofu50_${MODEL_TAG}_lora_seed" in text
    assert "heldout_forget_mixed_stress" in text
    assert "scripts/report_conditional_durability.py" in text
