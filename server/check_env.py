from __future__ import annotations

import multiprocessing
import os
import platform
import sys
import time

OK, BAD, WARN = "  [ok]  ", "  [FAIL]", "  [warn]"
problems: list[str] = []


def fail(msg: str, fix: str):
    print(f"{BAD} {msg}")
    problems.append(f"{msg}\n         fix: {fix}")


def main() -> int:
    print("=" * 68)
    print(" Soccer Form AI — server preflight")
    print("=" * 68)
    print(f"  {platform.python_implementation()} {platform.python_version()} "
          f"on {platform.system()} {platform.machine()}")

    if sys.version_info < (3, 9):
        fail(f"Python {platform.python_version()} is too old.",
             "Install Python 3.10-3.12 and recreate the virtualenv.")
    else:
        print(f"{OK} Python version")

    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        print(f"{WARN} Intel Mac: mediapipe 1.x ships no x86_64 wheel.")
        print("         If install failed, use: pip install 'mediapipe==0.10.14' "
              "(needs Python <= 3.12)")

    deps = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("multipart", "python-multipart"),
        ("mediapipe", "mediapipe"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
    ]
    for mod, pkg in deps:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{OK} {pkg:18s} {ver}")
        except ImportError as e:
            hint = ("On a headless Linux box use opencv-python-headless instead."
                    if mod == "cv2" else "pip install -r requirements.txt")
            fail(f"{pkg} missing or broken ({e}).", hint)
        except Exception as e:
            fail(f"{pkg} imported but raised {type(e).__name__}: {e}",
                 "On headless Linux: pip uninstall opencv-python && "
                 "pip install opencv-python-headless")

    if problems:
        return report()

    import models_cache
    print(f"  model cache: {models_cache.CACHE_DIR}")
    try:
        t = time.perf_counter()
        paths = models_cache.prefetch_all()
        dt = time.perf_counter() - t
        for k, p in paths.items():
            mb = os.path.getsize(p) / 1e6
            print(f"{OK} {k:9s} {os.path.basename(p)} ({mb:.1f}MB)")
        if dt > 3:
            print(f"         downloaded in {dt:.0f}s")
    except Exception as e:
        fail(f"Could not fetch models ({type(e).__name__}: {e}).",
             "This machine needs internet access on first run, or copy an "
             "existing server/models/ folder across and re-run.")
        return report()

    import numpy as np
    frame = np.zeros((360, 240, 3), dtype=np.uint8)

    try:
        import pose
        t = time.perf_counter()
        with pose.PoseSession(heavy=True) as s:
            build = time.perf_counter() - t
            t = time.perf_counter()
            s.detect_sequence([{"image": frame, "time": 0.0}], rescue=False)
            run = time.perf_counter() - t
        print(f"{OK} pose graph      built {build:.1f}s, 1 frame {run * 1000:.0f}ms")
    except Exception as e:
        fail(f"PoseLandmarker failed to build/run ({type(e).__name__}: {e}).",
             "Usually a mediapipe/protobuf version clash — recreate the venv.")

    try:
        import ball_detection
        t = time.perf_counter()
        with ball_detection.BallSession() as s:
            build = time.perf_counter() - t
            t = time.perf_counter()
            s.detect_sequence([{"image": frame, "time": 0.0}], [None])
            run = time.perf_counter() - t
        print(f"{OK} ball graph      built {build:.1f}s, 1 frame {run * 1000:.0f}ms")
    except Exception as e:
        fail(f"ObjectDetector failed to build/run ({type(e).__name__}: {e}).",
             "Usually a mediapipe/protobuf version clash — recreate the venv.")

    cores = multiprocessing.cpu_count()
    workers = int(os.environ.get("SFAI_WORKERS", "3"))
    print(f"  {cores} logical cores, SFAI_WORKERS={workers}")
    if cores < 4:
        print(f"{WARN} Few cores — expect slow analysis. Try SFAI_WORKERS=1 "
              "and SFAI_MAX_EDGE=1280.")

    return report()


def report() -> int:
    print("-" * 68)
    if problems:
        print(f" {len(problems)} problem(s):\n")
        for p in problems:
            print(f"  - {p}")
        print("\n RESULT: not ready.")
        return 1
    print(" RESULT: ready. Start with:")
    print("   python -m uvicorn main:app --host 0.0.0.0 --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
