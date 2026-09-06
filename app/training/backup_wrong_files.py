from pathlib import Path
import shutil

LIST = Path("data/include50/wrong_files.txt")
BACKUP = Path("data/include50/wrong_backup")

BACKUP.mkdir(parents=True, exist_ok=True)

files = [
    Path(line.strip())
    for line in LIST.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

moved = 0

for src in files:

    if not src.exists():
        print("NOT FOUND:", src)
        continue

    # Preserve category/class structure
    relative = src.relative_to(Path("data/include50/sequences"))
    dest = BACKUP / relative

    dest.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(src), str(dest))

    moved += 1
    print("MOVED:", src)

print()
print("=" * 70)
print(f"Moved: {moved}/{len(files)}")
print("Backup:", BACKUP)
print("=" * 70)
