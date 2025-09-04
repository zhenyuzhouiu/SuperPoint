

#!/usr/bin/env python3
"""
Scan an images root, validate each image. If corrupted/unreadable => delete the image
and also delete the corresponding ground-truth .npz under --gt-root with the same
relative path and base name. If valid => convert to JPEG (RGB) in-place (save <name>.jpg),
then delete the original non-jpeg file.

Usage:
  python check_image_.py \
      --images-root /path/to/images \
      --gt-root /path/to/gt_npz_root \
      [--dry-run]

Notes:
- Corresponding GT path is constructed as:  relpath(image, images_root) -> change extension to .npz, and join with gt_root.
- For images with alpha channel, this script composites on a white background before saving JPEG.
"""
import argparse
import os
import sys
import shutil
from typing import Tuple

try:
    from PIL import Image
except Exception as e:
    print("[ERR] Pillow not installed. Run: pip install Pillow", file=sys.stderr)
    raise

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="Validate images, delete bad ones with GT, convert good ones to JPG")
    p.add_argument("--images-root", required=True, help="Root directory of images to scan")
    p.add_argument("--gt-root", required=True, help="Root directory containing GT .npz files in mirrored structure")
    p.add_argument("--dry-run", action="store_true", help="Do not write/delete, only print actions")
    return p.parse_args()


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VALID_EXTS


def verify_image(path: str) -> Tuple[bool, str]:
    """Return (ok, msg). ok=False when corrupted or unreadable."""
    try:
        with Image.open(path) as im:
            im.verify()  # light check (header)
        # reopen to fully load pixel data
        with Image.open(path) as im:
            im.load()
        return True, "ok"
    except Exception as e:
        return False, f"bad: {e.__class__.__name__}: {e}"


def ensure_rgb_noalpha(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "L"):
        return img.convert("RGB")
    # composite alpha onto white background if present
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha = None
        if img.mode in ("RGBA", "LA"):
            base = Image.new("RGB", img.size, (255, 255, 255))
            alpha = img.split()[-1]
            base.paste(img.convert("RGB"), mask=alpha)
            return base
        else:  # paletted with transparency
            return img.convert("RGBA").convert("RGB")
    # fallback
    return img.convert("RGB")


def convert_to_jpg_inplace(path: str, dry: bool = False) -> str:
    """Save <basename>.jpg next to original; remove original if different ext. Return new path."""
    root, ext = os.path.splitext(path)
    new_path = root + ".jpg"
    if os.path.abspath(new_path) == os.path.abspath(path) and ext.lower() in {".jpg", ".jpeg"}:
        # already jpg; still normalize encoding to RGB
        with Image.open(path) as im:
            im = ensure_rgb_noalpha(im)
            if not dry:
                im.save(path, format="JPEG", quality=95, subsampling=1)
        return path
    with Image.open(path) as im:
        im = ensure_rgb_noalpha(im)
        if not dry:
            im.save(new_path, format="JPEG", quality=95, subsampling=1)
    if os.path.exists(new_path) and (os.path.abspath(new_path) != os.path.abspath(path)):
        if not dry:
            os.remove(path)
    return new_path


def delete_with_gt(img_path: str, images_root: str, gt_root: str, dry: bool = False) -> None:
    # delete image
    if dry:
        print(f"[DEL] {img_path}")
    else:
        try:
            os.remove(img_path)
        except FileNotFoundError:
            pass
    # delete corresponding GT .npz
    rel = os.path.relpath(img_path, images_root)
    base = os.path.splitext(rel)[0]
    gt_path = os.path.join(gt_root, base + ".npz")
    if os.path.exists(gt_path):
        if dry:
            print(f"[DEL] {gt_path}")
        else:
            os.makedirs(os.path.dirname(gt_path), exist_ok=True)
            try:
                os.remove(gt_path)
            except FileNotFoundError:
                pass
    else:
        print(f"[MISS GT] {gt_path}")


def main():
    args = parse_args()
    images_root = os.path.abspath(args.images_root)
    gt_root = os.path.abspath(args.gt_root)

    if not os.path.isdir(images_root):
        print(f"[ERR] images-root not found: {images_root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(gt_root):
        print(f"[WARN] gt-root not found: {gt_root} (will still attempt relpath mapping)")

    n_total = n_ok = n_bad = n_converted = 0

    for r, _, files in os.walk(images_root):
        for fn in files:
            p = os.path.join(r, fn)
            if not is_image_file(p):
                continue
            n_total += 1
            ok, msg = verify_image(p)
            if not ok:
                n_bad += 1
                print(f"[BAD] {p} :: {msg}")
                delete_with_gt(p, images_root, gt_root, dry=args.dry_run)
                continue
            # valid image -> convert to jpg
            try:
                newp = convert_to_jpg_inplace(p, dry=args.dry_run)
                if os.path.abspath(newp) != os.path.abspath(p):
                    print(f"[CONVERT] {p} -> {newp}")
                else:
                    print(f"[REWRITE] {p}")
                n_ok += 1
                n_converted += 1
            except Exception as e:
                # If conversion fails, treat as bad
                n_bad += 1
                print(f"[BAD-CONVERT] {p} :: {e}")
                delete_with_gt(p, images_root, gt_root, dry=args.dry_run)

    print(f"[DONE] total={n_total} ok+converted={n_ok} bad_deleted={n_bad}")

if __name__ == "__main__":
    main()