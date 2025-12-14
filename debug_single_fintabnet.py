import os
import traceback

from modules.pipeline import FinancialTablePipeline


def main() -> None:
    images_dir = r"D:\datasets\FinTabNet_c\FinTabNet.c-Structure\FinTabNet.c-Structure\images"
    filename = "SNPS_2014_page_76_table_1.jpg"
    path = os.path.join(images_dir, filename)

    print("image:", path)
    print("exists:", os.path.exists(path))

    pipe = FinancialTablePipeline(use_v1_1=True)

    try:
        out = pipe.process_image(path)
        print("OK")
        print("file:", out.get("file"))
        grid = out.get("grid") or []
        print("grid rows:", len(grid))
        print("grid cols:", len(grid[0]) if grid else 0)
    except Exception as exc:
        print("FAILED:", repr(exc))
        traceback.print_exc()


if __name__ == "__main__":
    main()
