import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps


@dataclass
class StyleGAN3Config:
    run_dir: str
    data_root: str
    image_size: int = 256
    stylegan3_repo: str = "./external/stylegan3"
    outdir_name: str = "stylegan3_runs"
    gpus: int = 1
    batch: int = 8
    gamma: float = 6.6
    kimg: int = 500
    mirror: bool = False
    cfg: str = "stylegan3-t"
    snap: int = 10
    metrics: str = "fid50k_full"
    cond: bool = False
    workers: int = 3
    seed: int = 0
    target_class: Optional[str] = None


class StyleGAN3Model:
    """
    Wrapper around the official NVLabs StyleGAN3 repository.

    This prepares a 256x256 dataset zip and then calls the official
    StyleGAN3 scripts for training, generation, and evaluation.
    """

    def __init__(self, args):
        self.config = StyleGAN3Config(
            run_dir=args.run_dir,
            data_root=args.dataroot,
            image_size=getattr(args, "image_size", 256),
            stylegan3_repo=getattr(args, "stylegan3_repo", "./external/stylegan3"),
            outdir_name=getattr(args, "stylegan3_outdir", "stylegan3_runs"),
            gpus=getattr(args, "gpus", 1),
            batch=getattr(args, "batch_size", 8),
            gamma=getattr(args, "gamma", 6.6),
            kimg=getattr(args, "kimg", 500),
            mirror=getattr(args, "mirror", False),
            cfg=getattr(args, "cfg", "stylegan3-t"),
            snap=getattr(args, "snap", 10),
            metrics=getattr(args, "metrics", "fid50k_full"),
            cond=getattr(args, "cond", False),
            workers=getattr(args, "workers", 3),
            seed=getattr(args, "seed", 0),
            target_class=getattr(args, "target_class", None),
        )

        self.run_dir = Path(self.config.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.sample_dir = self.run_dir / "samples"
        self.eval_dir = self.run_dir / "evaluation"
        self.train_plot_dir = self.run_dir / "training_plots"
        self.metadata_dir = self.run_dir / "metadata"

        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.train_plot_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.repo_dir = Path(self.config.stylegan3_repo).resolve()
        self.training_outdir = (self.run_dir / self.config.outdir_name).resolve()
        self.training_outdir.mkdir(parents=True, exist_ok=True)

        self.dataset_zip = (self.run_dir / f"dataset_{self.config.image_size}x{self.config.image_size}.zip").resolve()

        print("Initializing StyleGAN3 wrapper...")
        print(f"StyleGAN3 repo: {self.repo_dir}")
        print(f"Dataset root: {self.config.data_root}")
        print(f"Target image size: {self.config.image_size}x{self.config.image_size}")
        if self.config.target_class is not None:
            print(f"Target class: {self.config.target_class}")

    def _run_subprocess(self, cmd, cwd: Optional[Path] = None):
        print("Running command:")
        print(" ".join(str(x) for x in cmd))
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

    def _check_repo(self):
        required = [
            self.repo_dir / "train.py",
            self.repo_dir / "dataset_tool.py",
            self.repo_dir / "gen_images.py",
        ]

        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"StyleGAN3 repo not found at: {self.repo_dir}. Clone it first with: "
                "git clone https://github.com/NVlabs/stylegan3.git external/stylegan3"
            )

        for file_path in required:
            if not file_path.exists():
                raise FileNotFoundError(f"Missing required StyleGAN3 file: {file_path}")

    def _find_input_root(self) -> Path:
        data_root = Path(self.config.data_root)
        train_dir = data_root / "train"
        base_root = train_dir if train_dir.exists() else data_root

        if self.config.target_class is not None:
            class_dir = base_root / self.config.target_class
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Target class folder not found: {class_dir}. "
                    "Check --dataroot and --target_class."
                )
            return class_dir

        return base_root

    def prepare_dataset(self):
        """
        Convert the current dataset into a StyleGAN3-compatible zip.

        Supported source layouts:
        - root/train/class_name/*.png
        - root/class_name/*.png
        - root/*.png
        """
        self._check_repo()

        input_root = self._find_input_root()
        temp_dir = self.run_dir / "_stylegan3_dataset_temp"

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        image_paths = []

        entries = list(input_root.iterdir()) if input_root.exists() else []
        has_subdirs = any(p.is_dir() for p in entries)

        if has_subdirs and self.config.target_class is None:
            for class_dir in sorted([p for p in entries if p.is_dir()]):
                for img_path in sorted(class_dir.rglob("*")):
                    if img_path.suffix.lower() in valid_exts:
                        image_paths.append((class_dir.name, img_path))
        else:
            class_name = self.config.target_class if self.config.target_class is not None else None
            for img_path in sorted(input_root.rglob("*")):
                if img_path.suffix.lower() in valid_exts:
                    image_paths.append((class_name, img_path))

        if len(image_paths) == 0:
            raise FileNotFoundError(f"No images found under: {input_root}")

        class_to_idx = {}
        if self.config.cond:
            class_names = sorted({class_name for class_name, _ in image_paths if class_name is not None})
            if len(class_names) == 0:
                class_names = ["unlabeled"]
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        labels = []
        for idx, (class_name, img_path) in enumerate(image_paths):
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = ImageOps.fit(
                    img,
                    (self.config.image_size, self.config.image_size),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )

                file_name = f"img_{idx:06d}.png"
                img.save(temp_dir / file_name)

            if self.config.cond:
                label_name = class_name if class_name is not None else "unlabeled"
                if label_name not in class_to_idx:
                    class_to_idx[label_name] = len(class_to_idx)
                labels.append([file_name, class_to_idx[label_name]])

        if self.config.cond:
            with open(temp_dir / "dataset.json", "w") as f:
                json.dump({"labels": labels}, f, indent=2)

        if self.dataset_zip.exists():
            self.dataset_zip.unlink()

        with zipfile.ZipFile(self.dataset_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            for file_path in sorted(temp_dir.iterdir()):
                zf.write(file_path, arcname=file_path.name)

        shutil.rmtree(temp_dir)

        info = {
            "image_count": len(image_paths),
            "image_size": self.config.image_size,
            "dataset_zip": str(self.dataset_zip),
            "conditional": self.config.cond,
            "target_class": self.config.target_class,
            "class_to_idx": class_to_idx,
        }
        with open(self.metadata_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"Prepared StyleGAN3 dataset zip: {self.dataset_zip}")
        print(f"Number of images: {len(image_paths)}")
        return str(self.dataset_zip)

    def train(self, train_loader=None, test_loader=None):
        del train_loader, test_loader
        self._check_repo()

        dataset_zip = str(Path(self.prepare_dataset()).resolve())
        outdir = str(self.training_outdir.resolve())

        cmd = [
            sys.executable,
            str(self.repo_dir / "train.py"),
            f"--outdir={outdir}",
            f"--cfg={self.config.cfg}",
            f"--data={dataset_zip}",
            f"--gpus={self.config.gpus}",
            f"--batch={self.config.batch}",
            f"--gamma={self.config.gamma}",
            f"--kimg={self.config.kimg}",
            f"--snap={self.config.snap}",
            f"--metrics={self.config.metrics}",
            f"--workers={self.config.workers}",
            f"--seed={self.config.seed}",
            "--mirror=1" if self.config.mirror else "--mirror=0",
            "--cond=1" if self.config.cond else "--cond=0",
        ]

        self._run_subprocess(cmd, cwd=self.repo_dir)
        print("StyleGAN3 training finished.")
        print(f"Outputs saved in: {self.training_outdir}")

        # Automatically find and copy the latest trained model
        try:
            latest_model = self._find_latest_network_pkl()

            # main saved model in run directory
            saved_model_path = self.run_dir / "stylegan3_trained_model.pkl"

            # also store inside a dedicated models folder
            models_dir = self.run_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            models_copy_path = models_dir / "stylegan3_trained_model.pkl"

            shutil.copy(latest_model, saved_model_path)
            shutil.copy(latest_model, models_copy_path)

            print("Latest StyleGAN3 model saved to:")
            print(saved_model_path)
            print("Backup model saved to:")
            print(models_copy_path)

        except Exception as e:
            print("Warning: could not copy trained model automatically.")
            print(e)

    def _find_latest_network_pkl(self) -> Path:
        run_dirs = sorted([p for p in self.training_outdir.iterdir() if p.is_dir()])
        if not run_dirs:
            raise FileNotFoundError(f"No StyleGAN3 run directories found in: {self.training_outdir}")

        latest_run = run_dirs[-1]
        snapshots = sorted(latest_run.glob("network-snapshot-*.pkl"))
        if snapshots:
            return snapshots[-1]

        final_pkl = latest_run / "network-final.pkl"
        if final_pkl.exists():
            return final_pkl

        raise FileNotFoundError(f"No StyleGAN3 network pickle found in: {latest_run}")

    def load_model(self, g_path=None, d_path=None):
        del d_path
        self._check_repo()

        if g_path is None:
            g_path = str(self._find_latest_network_pkl())

        self.loaded_network_pkl = Path(g_path)
        print(f"Using StyleGAN3 network pickle: {self.loaded_network_pkl}")

    def generate_images(self, num_images=16, trunc=1.0, translate="0,0", rotate=0):
        self._check_repo()

        network_pkl = getattr(self, "loaded_network_pkl", None)
        if network_pkl is None:
            network_pkl = self._find_latest_network_pkl()

        seeds = ",".join(str(i) for i in range(num_images))
        cmd = [
            sys.executable,
            str(self.repo_dir / "gen_images.py"),
            f"--network={network_pkl}",
            f"--seeds={seeds}",
            f"--outdir={self.sample_dir}",
            f"--trunc={trunc}",
            f"--translate={translate}",
            f"--rotate={rotate}",
        ]

        self._run_subprocess(cmd, cwd=self.repo_dir)
        print(f"Generated images saved to: {self.sample_dir}")

    def evaluate(self, test_loader=None):
        del test_loader
        self._check_repo()

        network_pkl = getattr(self, "loaded_network_pkl", None)
        if network_pkl is None:
            network_pkl = self._find_latest_network_pkl()

        if not self.dataset_zip.exists():
            self.prepare_dataset()

        calc_metrics_py = self.repo_dir / "calc_metrics.py"
        metrics_helper = self.eval_dir / "stylegan3_eval_command.txt"
        cmd = [
            sys.executable,
            str(calc_metrics_py),
            f"--network={network_pkl}",
            f"--metrics={self.config.metrics}",
            f"--data={self.dataset_zip.resolve()}",
        ]

        with open(metrics_helper, "w") as f:
            f.write(" ".join(str(x) for x in cmd))
            f.write("\n")

        if calc_metrics_py.exists():
            print("Running StyleGAN3 evaluation...")
            self._run_subprocess(cmd, cwd=self.repo_dir)
        else:
            print("StyleGAN3 calc_metrics.py not found. Wrote helper command instead.")

        print(f"See: {metrics_helper}")