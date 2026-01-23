#!/usr/bin/env python3
"""
Submission Validator for NLP Challenge

Checks if your submission.zip is valid before uploading.
Catches common errors that would cause evaluation to fail.

Usage:
    python check_submission.py                    # Check submission.zip
    python check_submission.py my_submission.zip  # Check specific file
"""

import argparse
import importlib.util
import sys
import tempfile
import zipfile
from pathlib import Path

# Constraints (must match evaluator)
MAX_ZIP_SIZE_MB = 1.0
MAX_UNCOMPRESSED_MB = 50.0
BATCH_SIZE = 1024
NUM_BYTES = 256


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def print_check(name: str, passed: bool, detail: str = ""):
    """Print a check result."""
    icon = "\u2713" if passed else "\u2717"
    status = "PASS" if passed else "FAIL"
    msg = f"[{icon}] {status}: {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    return passed


def check_file_exists(zip_path: Path) -> bool:
    """Check if the zip file exists."""
    exists = zip_path.exists()
    print_check("File exists", exists, str(zip_path))
    return exists


def check_is_zipfile(zip_path: Path) -> bool:
    """Check if it's a valid zip file."""
    is_zip = zipfile.is_zipfile(zip_path)
    print_check("Valid ZIP format", is_zip)
    return is_zip


def check_zip_size(zip_path: Path) -> bool:
    """Check if zip is under size limit."""
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    passed = size_mb <= MAX_ZIP_SIZE_MB
    print_check(f"ZIP size <= {MAX_ZIP_SIZE_MB} MB", passed, f"{size_mb:.3f} MB")
    return passed


def check_uncompressed_size(zip_path: Path) -> bool:
    """Check if uncompressed size is under limit."""
    with zipfile.ZipFile(zip_path) as zf:
        total = sum(info.file_size for info in zf.infolist())
    size_mb = total / (1024 * 1024)
    passed = size_mb <= MAX_UNCOMPRESSED_MB
    print_check(f"Uncompressed size <= {MAX_UNCOMPRESSED_MB} MB", passed, f"{size_mb:.3f} MB")
    return passed


def check_no_path_traversal(zip_path: Path) -> bool:
    """Check for path traversal attacks."""
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.filename.startswith('/') or '..' in info.filename:
                print_check("No path traversal", False, f"Bad path: {info.filename}")
                return False
    print_check("No path traversal", True)
    return True


def check_model_py_exists(zip_path: Path) -> bool:
    """Check if model.py exists in the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    has_model = 'model.py' in names
    print_check("Contains model.py", has_model, f"Files: {names[:5]}{'...' if len(names) > 5 else ''}")
    return has_model


def check_model_class(extract_dir: Path) -> bool:
    """Check if model.py has a Model class with correct interface."""
    model_file = extract_dir / "model.py"

    # Load module
    try:
        spec = importlib.util.spec_from_file_location("submission", model_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["submission"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        print_check("model.py loads without error", False, str(e))
        return False
    print_check("model.py loads without error", True)

    # Check Model class exists
    if not hasattr(module, 'Model'):
        print_check("Model class exists", False, "No 'Model' class found")
        return False
    print_check("Model class exists", True)

    Model = module.Model

    # Check __init__ signature
    import inspect
    sig = inspect.signature(Model.__init__)
    params = list(sig.parameters.keys())
    if 'submission_dir' not in params and len(params) < 2:
        print_check("__init__(self, submission_dir)", False, f"Got params: {params}")
        return False
    print_check("__init__(self, submission_dir)", True)

    # Check predict method exists
    if not hasattr(Model, 'predict'):
        print_check("predict() method exists", False)
        return False
    print_check("predict() method exists", True)

    return True


def check_model_instantiation(extract_dir: Path) -> bool:
    """Check if Model can be instantiated."""
    model_file = extract_dir / "model.py"

    try:
        spec = importlib.util.spec_from_file_location("submission", model_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["submission"] = module
        spec.loader.exec_module(module)

        model = module.Model(submission_dir=extract_dir)
    except Exception as e:
        print_check("Model instantiates", False, str(e)[:100])
        return False

    print_check("Model instantiates", True)
    return True


def check_predict_output(extract_dir: Path) -> bool:
    """Check if predict() returns correct format."""
    model_file = extract_dir / "model.py"

    try:
        spec = importlib.util.spec_from_file_location("submission", model_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["submission"] = module
        spec.loader.exec_module(module)

        model = module.Model(submission_dir=extract_dir)

        # Test with sample contexts
        test_contexts = [
            [],                           # Empty context
            [72, 101, 108, 108, 111],     # "Hello"
            [32, 119, 111, 114, 108, 100], # " world"
            list(range(100)),              # 100 bytes
        ]

        result = model.predict(test_contexts)

    except Exception as e:
        print_check("predict() runs without error", False, str(e)[:100])
        return False

    print_check("predict() runs without error", True)

    # Check return type
    if not isinstance(result, list):
        print_check("predict() returns list", False, f"Got {type(result)}")
        return False
    print_check("predict() returns list", True)

    # Check batch size
    if len(result) != len(test_contexts):
        print_check("Output batch size matches input", False, f"Expected {len(test_contexts)}, got {len(result)}")
        return False
    print_check("Output batch size matches input", True)

    # Check each row has 256 values
    for i, row in enumerate(result):
        if not isinstance(row, (list, tuple)):
            print_check(f"Row {i} is list/tuple", False, f"Got {type(row)}")
            return False
        if len(row) != NUM_BYTES:
            print_check(f"Row {i} has {NUM_BYTES} values", False, f"Got {len(row)}")
            return False
    print_check(f"Each row has {NUM_BYTES} logit values", True)

    # Check values are numeric
    try:
        for row in result:
            for val in row:
                float(val)
    except (TypeError, ValueError) as e:
        print_check("All values are numeric", False, str(e))
        return False
    print_check("All values are numeric", True)

    return True


def check_batch_performance(extract_dir: Path) -> bool:
    """Check if model can handle full batch size."""
    model_file = extract_dir / "model.py"

    try:
        spec = importlib.util.spec_from_file_location("submission", model_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["submission"] = module
        spec.loader.exec_module(module)

        model = module.Model(submission_dir=extract_dir)

        # Test with full batch of 1024 contexts
        import random
        test_contexts = [
            [random.randint(0, 255) for _ in range(random.randint(0, 512))]
            for _ in range(BATCH_SIZE)
        ]

        import time
        start = time.time()
        result = model.predict(test_contexts)
        elapsed = time.time() - start

    except Exception as e:
        print_check(f"Handles batch of {BATCH_SIZE}", False, str(e)[:100])
        return False

    if len(result) != BATCH_SIZE:
        print_check(f"Handles batch of {BATCH_SIZE}", False, f"Got {len(result)} results")
        return False

    print_check(f"Handles batch of {BATCH_SIZE}", True, f"{elapsed:.2f}s")

    # Warn if slow
    if elapsed > 5.0:
        print(f"  WARNING: Batch took {elapsed:.1f}s - may timeout on full evaluation")

    return True


def validate_submission(zip_path: Path) -> bool:
    """Run all validation checks."""
    print("=" * 60)
    print(f"VALIDATING: {zip_path}")
    print("=" * 60)
    print()

    all_passed = True

    # Basic file checks
    print("--- File Checks ---")
    if not check_file_exists(zip_path):
        return False
    all_passed &= check_is_zipfile(zip_path)
    if not all_passed:
        return False
    all_passed &= check_zip_size(zip_path)
    all_passed &= check_uncompressed_size(zip_path)
    all_passed &= check_no_path_traversal(zip_path)
    all_passed &= check_model_py_exists(zip_path)

    if not all_passed:
        print("\n" + "=" * 60)
        print("VALIDATION FAILED - Fix errors above before submitting")
        print("=" * 60)
        return False

    # Extract and check model
    print()
    print("--- Model Checks ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        extract_dir = Path(tmpdir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        all_passed &= check_model_class(extract_dir)
        if not all_passed:
            print("\n" + "=" * 60)
            print("VALIDATION FAILED - Fix errors above before submitting")
            print("=" * 60)
            return False

        all_passed &= check_model_instantiation(extract_dir)
        if not all_passed:
            print("\n" + "=" * 60)
            print("VALIDATION FAILED - Fix errors above before submitting")
            print("=" * 60)
            return False

        print()
        print("--- Output Checks ---")
        all_passed &= check_predict_output(extract_dir)

        print()
        print("--- Performance Checks ---")
        all_passed &= check_batch_performance(extract_dir)

    print()
    print("=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED - Ready to submit!")
    else:
        print("VALIDATION FAILED - Fix errors above before submitting")
    print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate submission zip file")
    parser.add_argument("zip_file", nargs="?", default="submission.zip",
                        help="Path to submission zip (default: submission.zip)")
    args = parser.parse_args()

    zip_path = Path(args.zip_file)
    success = validate_submission(zip_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
