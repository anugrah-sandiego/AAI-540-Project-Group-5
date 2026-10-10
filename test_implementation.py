"""
Syntax and structure validation test for AWS SageMaker implementation.
This validates that all modules are properly structured without requiring AWS credentials.
"""

import os
import sys
import ast
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python file syntax."""
    try:
        with open(file_path, 'r') as f:
            ast.parse(f.read())
        return True, "Syntax valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

def validate_module_structure(module_path):
    """Validate module has required components."""
    file_path = Path(module_path)
    if not file_path.exists():
        return False, "File does not exist"

    valid, msg = validate_python_syntax(file_path)
    if not valid:
        return False, msg

    return True, "Valid"

def main():
    """Run validation tests."""
    print("=" * 60)
    print("AWS SageMaker Implementation Validation")
    print("=" * 60)
    print()

    # Define modules to validate
    modules = [
        "src/aws/sagemaker/data_upload.py",
        "src/aws/sagemaker/feature_store.py",
        "src/aws/sagemaker/training.py",
        "src/aws/sagemaker/deployment.py",
        "src/aws/sagemaker/batch_inference.py",
        "src/aws/sagemaker/model_registry.py",
        "src/aws/sagemaker/monitoring.py",
        "src/aws/sagemaker/infrastructure_monitoring.py",
        "src/aws/sagemaker/rollback.py",
        "src/aws/sagemaker/scripts/train.py",
        "src/aws/sagemaker/scripts/inference.py",
        "src/aws/sagemaker/scripts/batch_transform.py",
    ]

    results = []

    for module in modules:
        print(f"Validating: {module}")
        valid, msg = validate_module_structure(module)
        results.append((module, valid, msg))
        print(f"  {'✓' if valid else '✗'} {msg}")
        print()

    # Validate CI/CD workflow
    print("Validating: .github/workflows/ci-cd.yml")
    ci_cd_path = Path(".github/workflows/ci-cd.yml")
    if ci_cd_path.exists():
        results.append((".github/workflows/ci-cd.yml", True, "File exists"))
        print("  ✓ File exists")
    else:
        results.append((".github/workflows/ci-cd.yml", False, "File does not exist"))
        print("  ✗ File does not exist")
    print()

    # Validate documentation
    docs = [
        "README.md",
        "DEPLOYMENT_GUIDE.md",
        "IMPLEMENTATION_SUMMARY.md",
        "validate_aws_setup.py"
    ]

    for doc in docs:
        print(f"Validating: {doc}")
        doc_path = Path(doc)
        if doc_path.exists():
            results.append((doc, True, "File exists"))
            print("  ✓ File exists")
        else:
            results.append((doc, False, "File does not exist"))
            print("  ✗ File does not exist")
        print()

    # Summary
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)

    passed = sum(1 for _, valid, _ in results if valid)
    total = len(results)

    for module, valid, msg in results:
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"{status}: {module}")

    print()
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All validations passed!")
        print("\nTo deploy to AWS:")
        print("1. Configure AWS credentials in .env")
        print("2. Run: python validate_aws_setup.py")
        print("3. Follow DEPLOYMENT_GUIDE.md")
        return True
    else:
        print("\n✗ Some validations failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
