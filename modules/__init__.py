"""Financial Document AI modules.

Keep this package import lightweight.

Many submodules load heavyweight GPU libraries (e.g., PyTorch / Paddle) and can
conflict with each other on Windows if imported eagerly. Import what you need
explicitly, e.g.:

	from modules.pipeline import FinancialTablePipeline
"""

__all__: list[str] = []
