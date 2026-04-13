# crypto_ml/read_fichiers.py
# script qui parcourt le répertoire du projet et affiche l'arborescence
# en ne conservant que les fichiers d'extension utiles (.py, .csv, .parquet, .yaml/.yml, .html, .css, .js)
# et la taille de chaque fichier (format humain : B, KB, MB, GB).

import os
from pathlib import Path

# ---------------------------------------------------------
# helper : conversion en taille lisible
# ---------------------------------------------------------
def _human_readable(num_bytes: int) -> str:
    """Convertit un nombre d’octets en chaîne lisible (B, KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes/1024:.1f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes/1024**2:.1f} MB"
    else:
        return f"{num_bytes/1024**3:.1f} GB"

# ---------------------------------------------------------
# extensions que l’on veut afficher
# ---------------------------------------------------------
ALLOWED_EXT = {
    ".py",       # code source
    ".csv",      # jeux de données légers
    ".parquet",  # OHLCV
    ".yaml",     # configuration
    ".yml",      # alias yaml
    ".html",     # templates
    ".css",      # styles
    ".js",       # scripts front-end
}

# ---------------------------------------------------------
# dossiers à ignorer complètement
# ---------------------------------------------------------
IGNORE_DIRS = {
    "__pycache__",
    ".venv",
    ".git",
}

# ---------------------------------------------------------
# fonction récursive d’affichage
# ---------------------------------------------------------
def _print_tree(root: Path, prefix: str = "") -> None:
    """
    Parcourt `root` et imprime chaque répertoire / fichier autorisé.
    `prefix` contient les caractères de branche (│, ├─, └─) pour le rendu.
    """
    # tri alphabétique, dossiers avant fichiers
    all_entries = sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

    # on filtre :
    entries = []
    for e in all_entries:
        if e.is_dir():
            if e.name in IGNORE_DIRS or e.name.startswith("."):
                continue
            entries.append(e)
        else:
            if e.suffix.lower() in ALLOWED_EXT:
                entries.append(e)

    for idx, entry in enumerate(entries):
        is_last = idx == len(entries) - 1
        branch = "└── " if is_last else "├── "

        if entry.is_dir():
            # affichage du répertoire
            print(f"{prefix}{branch}{entry.name}/")
            extension = "    " if is_last else "│   "
            _print_tree(entry, prefix + extension)
        else:
            size = _human_readable(entry.stat().st_size)
            print(f"{prefix}{branch}{entry.name}  ({size})")

# ---------------------------------------------------------
# point d’entrée du script
# ---------------------------------------------------------
def main() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"{project_root.name}/")
    _print_tree(project_root, prefix="")

if __name__ == "__main__":
    main()