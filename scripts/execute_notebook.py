"""Execute this project's notebook without requiring Jupyter to be installed.

This small fallback runner supports the notebook's deliberately simple cells:
stdout and Matplotlib figures are captured and written as standard notebook
outputs.  Users with Jupyter installed should normally use ``jupyter nbconvert``
or run the notebook interactively instead.
"""

import argparse
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import traceback
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
)


def execute_notebook(path):
    notebook_path = Path(path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    execution_count = 0

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        execution_count += 1
        cell["execution_count"] = execution_count
        cell["outputs"] = []
        source = "".join(cell.get("source", []))
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        try:
            with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
                exec(compile(source, f"{notebook_path.name}:cell-{execution_count}", "exec"), namespace)
        except Exception as error:
            standard_output = captured_stdout.getvalue()
            standard_error = captured_stderr.getvalue()
            if standard_output:
                cell["outputs"].append(
                    {"name": "stdout", "output_type": "stream", "text": standard_output.splitlines(True)}
                )
            if standard_error:
                cell["outputs"].append(
                    {"name": "stderr", "output_type": "stream", "text": standard_error.splitlines(True)}
                )
            formatted_traceback = traceback.format_exc().splitlines()
            cell["outputs"].append(
                {
                    "ename": type(error).__name__,
                    "evalue": str(error),
                    "output_type": "error",
                    "traceback": formatted_traceback,
                }
            )
            notebook_path.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            raise

        standard_output = captured_stdout.getvalue()
        standard_error = captured_stderr.getvalue()
        if standard_output:
            cell["outputs"].append(
                {"name": "stdout", "output_type": "stream", "text": standard_output.splitlines(True)}
            )
        if standard_error:
            cell["outputs"].append(
                {"name": "stderr", "output_type": "stream", "text": standard_error.splitlines(True)}
            )

        for figure_number in plt.get_fignums():
            figure = plt.figure(figure_number)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            cell["outputs"].append(
                {
                    "data": {"image/png": encoded, "text/plain": ["<Matplotlib Figure>"]},
                    "metadata": {},
                    "output_type": "display_data",
                }
            )
        plt.close("all")

    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Executed {execution_count} code cells: {notebook_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", help="Path to the .ipynb file to execute in place")
    arguments = parser.parse_args()
    execute_notebook(arguments.notebook)


if __name__ == "__main__":
    main()
