#!/usr/bin/env python3
"""Integration test script to verify agent-engine-API integration.

Run this before starting with systemd to catch issues early.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 50)
    print("Testing imports...")

    errors = []

    # Core components
    try:
        from src.features.registry import FeatureRegistry
        print("  ✓ FeatureRegistry")
    except Exception as e:
        errors.append(f"FeatureRegistry: {e}")
        print(f"  ✗ FeatureRegistry: {e}")

    try:
        from src.features.calculator import FeatureCalculator
        print("  ✓ FeatureCalculator")
    except Exception as e:
        errors.append(f"FeatureCalculator: {e}")
        print(f"  ✗ FeatureCalculator: {e}")

    try:
        from src.risk.manager import RiskManager
        print("  ✓ RiskManager")
    except Exception as e:
        errors.append(f"RiskManager: {e}")
        print(f"  ✗ RiskManager: {e}")

    try:
        from src.settings.runtime import RuntimeSettings
        print("  ✓ RuntimeSettings")
    except Exception as e:
        errors.append(f"RuntimeSettings: {e}")
        print(f"  ✗ RuntimeSettings: {e}")

    # Agent components
    try:
        from src.agent.decision import AgentAction, ActionType, AutonomyLevel
        print("  ✓ AgentDecision")
    except Exception as e:
        errors.append(f"AgentDecision: {e}")
        print(f"  ✗ AgentDecision: {e}")

    try:
        from src.agent.action import ActionExecutor
        print("  ✓ ActionExecutor")
    except Exception as e:
        errors.append(f"ActionExecutor: {e}")
        print(f"  ✗ ActionExecutor: {e}")

    try:
        from src.agent.claude_client import ClaudeClient
        print("  ✓ ClaudeClient")
    except Exception as e:
        errors.append(f"ClaudeClient: {e}")
        print(f"  ✗ ClaudeClient: {e}")

    return len(errors) == 0, errors


def test_feature_registry():
    """Test FeatureRegistry functionality."""
    print("\n" + "=" * 50)
    print("Testing FeatureRegistry...")

    errors = []

    try:
        from src.features.registry import FeatureRegistry

        # Create registry (will use temp file)
        registry = FeatureRegistry(config_path="data/test_registry.json")

        # Test get_enabled_features
        enabled = registry.get_enabled_features()
        print(f"  ✓ Enabled features: {len(enabled)}")

        # Test enable/disable
        test_feature = "adx_14"  # Extended feature

        # Enable
        registry.enable_feature(test_feature)
        assert test_feature in registry.get_enabled_features(), "Feature should be enabled"
        print(f"  ✓ Enable feature: {test_feature}")

        # Disable
        registry.disable_feature(test_feature)
        assert test_feature not in registry.get_enabled_features(), "Feature should be disabled"
        print(f"  ✓ Disable feature: {test_feature}")

        # Test reload
        registry.force_reload()
        print("  ✓ Force reload")

        # Cleanup
        import os
        if os.path.exists("data/test_registry.json"):
            os.remove("data/test_registry.json")

    except Exception as e:
        errors.append(f"FeatureRegistry: {e}")
        print(f"  ✗ Error: {e}")

    return len(errors) == 0, errors


def test_risk_manager():
    """Test RiskManager functionality."""
    print("\n" + "=" * 50)
    print("Testing RiskManager...")

    errors = []

    try:
        from src.risk.manager import RiskManager

        rm = RiskManager()

        # Test initial values
        print(f"  ✓ Initial long_confidence_threshold: {rm.long_config.confidence_threshold:.2%}")
        print(f"  ✓ Initial short_confidence_threshold: {rm.short_config.confidence_threshold:.2%}")

        # Test update_runtime_settings
        old_threshold = rm.long_config.confidence_threshold
        updates = rm.update_runtime_settings(long_confidence_threshold=0.85)

        assert rm.long_config.confidence_threshold == 0.85, "Threshold should be updated"
        print(f"  ✓ Updated long_confidence_threshold: {rm.long_config.confidence_threshold:.2%}")
        print(f"  ✓ Update result: {updates}")

        # Restore
        rm.update_runtime_settings(long_confidence_threshold=old_threshold)

    except Exception as e:
        errors.append(f"RiskManager: {e}")
        print(f"  ✗ Error: {e}")

    return len(errors) == 0, errors


def test_feature_calculator_integration():
    """Test FeatureCalculator with FeatureRegistry."""
    print("\n" + "=" * 50)
    print("Testing FeatureCalculator + FeatureRegistry integration...")

    errors = []

    try:
        from src.features.registry import FeatureRegistry
        from src.features.calculator import FeatureCalculator

        # Create registry and calculator
        registry = FeatureRegistry(config_path="data/test_registry2.json")
        calculator = FeatureCalculator(registry=registry)

        # Test active_feature_names property
        features = calculator.active_feature_names
        print(f"  ✓ Active features with registry: {len(features)}")

        # Enable an extended feature
        registry.enable_feature("adx_14")
        features_after = calculator.active_feature_names

        assert "adx_14" in features_after, "adx_14 should be in active features"
        print(f"  ✓ After enabling adx_14: {len(features_after)} features")

        # Cleanup
        import os
        if os.path.exists("data/test_registry2.json"):
            os.remove("data/test_registry2.json")

    except Exception as e:
        errors.append(f"FeatureCalculator integration: {e}")
        print(f"  ✗ Error: {e}")

    return len(errors) == 0, errors


def test_action_executor():
    """Test ActionExecutor can be instantiated."""
    print("\n" + "=" * 50)
    print("Testing ActionExecutor...")

    errors = []

    try:
        from src.agent.action import ActionExecutor
        from src.features.registry import FeatureRegistry

        registry = FeatureRegistry(config_path="data/test_registry3.json")

        executor = ActionExecutor(
            api_base_url="http://localhost:8088",
            feature_registry=registry,
        )

        print("  ✓ ActionExecutor created with FeatureRegistry")

        # Cleanup
        import os
        if os.path.exists("data/test_registry3.json"):
            os.remove("data/test_registry3.json")

    except Exception as e:
        errors.append(f"ActionExecutor: {e}")
        print(f"  ✗ Error: {e}")

    return len(errors) == 0, errors


def test_runtime_settings():
    """Test RuntimeSettings functionality."""
    print("\n" + "=" * 50)
    print("Testing RuntimeSettings...")

    errors = []

    try:
        from src.settings.runtime import RuntimeSettings

        rs = RuntimeSettings(persist_path="data/test_runtime.json")

        # Test set/get
        rs.set("long_confidence_threshold", 0.80)
        value = rs.get("long_confidence_threshold")
        assert value == 0.80, "Value should be set"
        print(f"  ✓ Set/Get: long_confidence_threshold = {value}")

        # Test modifiable settings
        modifiable = rs.MODIFIABLE_SETTINGS
        print(f"  ✓ Modifiable settings: {len(modifiable)}")

        # Cleanup
        rs.delete("long_confidence_threshold")

        import os
        if os.path.exists("data/test_runtime.json"):
            os.remove("data/test_runtime.json")

    except Exception as e:
        errors.append(f"RuntimeSettings: {e}")
        print(f"  ✗ Error: {e}")

    return len(errors) == 0, errors


def test_syntax_check():
    """Check syntax of key files."""
    print("\n" + "=" * 50)
    print("Checking file syntax...")

    import py_compile

    files = [
        "src/core/engine.py",
        "src/risk/manager.py",
        "src/api/main.py",
        "src/features/calculator.py",
        "src/features/registry.py",
        "src/agent/core.py",
        "src/agent/action.py",
        "src/agent/claude_client.py",
    ]

    errors = []

    for file_path in files:
        full_path = project_root / file_path
        if not full_path.exists():
            errors.append(f"{file_path}: File not found")
            print(f"  ✗ {file_path}: Not found")
            continue

        try:
            py_compile.compile(str(full_path), doraise=True)
            print(f"  ✓ {file_path}")
        except py_compile.PyCompileError as e:
            errors.append(f"{file_path}: {e}")
            print(f"  ✗ {file_path}: Syntax error")

    return len(errors) == 0, errors


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("  BITDAYTRADER INTEGRATION TEST")
    print("=" * 60)

    all_passed = True
    all_errors = []

    # Run tests
    tests = [
        ("Syntax Check", test_syntax_check),
        ("Imports", test_imports),
        ("FeatureRegistry", test_feature_registry),
        ("RiskManager", test_risk_manager),
        ("FeatureCalculator Integration", test_feature_calculator_integration),
        ("ActionExecutor", test_action_executor),
        ("RuntimeSettings", test_runtime_settings),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed, errors = test_func()
            results.append((name, passed, errors))
            if not passed:
                all_passed = False
                all_errors.extend(errors)
        except Exception as e:
            results.append((name, False, [str(e)]))
            all_passed = False
            all_errors.append(f"{name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for name, passed, errors in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if errors:
            for err in errors:
                print(f"         - {err}")

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL TESTS PASSED ✓")
        print("  Safe to start with systemd")
    else:
        print("  SOME TESTS FAILED ✗")
        print("  Fix errors before starting systemd")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
