"""Build the prompt JARVIS gives the AI to complete the assignment.

The assignment itself is UPLOADED as a file (Claude/claude.ai read it natively),
so the prompt references the attachment rather than embedding the whole text. It
steers the AI to follow every instruction and to emit the answer in a shape our
assembler can turn into the requested file cleanly.
"""
from __future__ import annotations

# How the answer should be shaped for each target format, so assemble.py can build
# a clean file from it.
_SHAPE = {
    "ipynb": (
        "Produce a Jupyter notebook: explanations as prose and ALL code inside "
        "```python fenced blocks, in the order the cells should run."
    ),
    "py": "Produce a single, complete, runnable Python program inside one ```python fenced block.",
    "docx": "Produce a well-structured written document with clear ## section headings.",
    "pptx": "Produce slides: use a `## Slide title` heading per slide followed by concise bullet points.",
    "md": "Produce clean Markdown with headings and, where relevant, fenced code blocks.",
    "txt": "Produce a clear plain-text answer.",
}


def build_prompt(user_instructions: str, out_format: str, assignment_name: str = "the assignment") -> str:
    shape = _SHAPE.get(out_format, _SHAPE["md"])
    extra = (user_instructions or "").strip()
    extra_line = f"\nAdditional instructions from me: {extra}" if extra else ""
    return (
        f"The attached file, '{assignment_name}', is an academic assignment. Read ALL of "
        "it carefully and complete it fully, following every instruction, question, and "
        "constraint precisely and in order. Show working where appropriate and make it "
        f"correct and submission-ready.{extra_line}\n\n{shape}\n\n"
        "Return ONLY the finished answer content in a single message, with nothing before "
        "or after it, so it can be saved directly as the answer file."
    )
