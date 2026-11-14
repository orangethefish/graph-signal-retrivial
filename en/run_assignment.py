"""Execute the self-contained English notebook, retaining outputs and exporting CSVs."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    folder = Path(__file__).resolve().parent
    path = folder / "Student_Graph_Signal_Assignment_MSSV_MODULO.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(folder)}},
    )
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text, end="")
    print(f"Executed and saved {path}")


if __name__ == "__main__":
    main()
