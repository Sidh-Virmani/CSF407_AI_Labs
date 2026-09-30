import subprocess
import sys


def run(script, output_file):
    print(f"Running {script}...")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        check=True,
    )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    print(result.stdout)
    print(f"Saved output to {output_file}\n")


def main():
    run("first_order_model.py", "first_order_output.txt")
    run("second_order_model.py", "second_order_output.txt")
    run("compare_models.py", "comparison_output.txt")


if __name__ == "__main__":
    main()
