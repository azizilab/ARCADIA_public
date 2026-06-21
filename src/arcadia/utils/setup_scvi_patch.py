"""
Utility script to patch scVI library for ARCADIA custom training plans.

This script patches scvi-tools==1.2.2.post2 to add support for custom training plans
by inserting 'self._training_plan = training_plan' at line 131 in
scvi/model/base/_training_mixin.py.

Usage:
    python -m arcadia.utils.setup_scvi_patch
    or
    from arcadia.utils.setup_scvi_patch import patch_scvi_library
    patch_scvi_library()
"""

import os
import sys


def patch_scvi_library():
    """
    Patch scVI library to support custom training plans.

    Adds 'self._training_plan = training_plan' at line 131 (after the
    training_plan = self._training_plan_cls(...) assignment).

    Returns:
        bool: True if patching was successful or already patched, False otherwise

    Raises:
        FileNotFoundError: If scVI library files cannot be found
        RuntimeError: If patching fails
    """
    try:
        import scvi

        # Verify scVI version
        if scvi.__version__ != "1.2.2.post2":
            print(f"Warning: Expected scVI version 1.2.2.post2, got {scvi.__version__}")
            print("Patching may not work correctly with other versions.")

        # Get scVI package path
        scvi_path = scvi.__file__
        base_dir = os.path.dirname(os.path.dirname(scvi_path))
        training_mixin_path = os.path.join(base_dir, "scvi", "model", "base", "_training_mixin.py")

        if not os.path.exists(training_mixin_path):
            raise FileNotFoundError(
                f"Could not find _training_mixin.py at {training_mixin_path}\n"
                f"scVI may not be installed correctly."
            )

        # Read the file
        with open(training_mixin_path, "r") as f:
            lines = f.readlines()

        # Check if already patched
        if any("self._training_plan = training_plan" in line for line in lines):
            print("✓ scVI library already patched")
            return True

        # Find the line with training_plan assignment and add the required line after it
        patched = False
        new_lines = []

        for i, line in enumerate(lines):
            new_lines.append(line)
            # Look for the pattern: training_plan = self._training_plan_cls(...)
            if "training_plan = self._training_plan_cls" in line and not patched:
                # Check if line 131 (index 130) is empty or whitespace
                # We're currently at the line with training_plan assignment
                # The next line (i+1) should be line 131
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # If next line is empty/whitespace, insert there
                    if not next_line.strip():
                        # Add the required line with proper indentation
                        indent = len(line) - len(line.lstrip())
                        new_lines.append(" " * indent + "self._training_plan = training_plan\n")
                        patched = True
                    else:
                        # Insert after current line even if next line is not empty
                        indent = len(line) - len(line.lstrip())
                        new_lines.append(" " * indent + "self._training_plan = training_plan\n")
                        patched = True

        if patched:
            # Write back the patched file
            with open(training_mixin_path, "w") as f:
                f.writelines(new_lines)
            print(f"✓ Successfully patched scVI library at {training_mixin_path}")
            print(f"  Added 'self._training_plan = training_plan' after training_plan assignment")
            return True
        else:
            raise RuntimeError(
                f"Could not find training_plan assignment line in {training_mixin_path}\n"
                f"The scVI library structure may have changed."
            )

    except ImportError:
        print("Error: scvi-tools is not installed. Please install it first.")
        return False
    except Exception as e:
        print(f"Error patching scVI library: {e}")
        raise


def main():
    """Main entry point for command-line usage."""
    try:
        success = patch_scvi_library()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Failed to patch scVI library: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
