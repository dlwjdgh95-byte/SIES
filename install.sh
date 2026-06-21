#!/usr/bin/env sh
# SIES — 망각된-가치 검색기 install script
# Usage: curl install.caveman.sh | sh
set -eu

REPO="https://github.com/dlwjdgh95-byte/sies"
INSTALL_DIR="${SIES_DIR:-$HOME/.sies}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

# ── helpers ──────────────────────────────────────────────────────────────────
info()  { printf '\033[0;34m==> %s\033[0m\n' "$*"; }
ok()    { printf '\033[0;32m    ✓ %s\033[0m\n' "$*"; }
die()   { printf '\033[0;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

need() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1. Please install it and retry."
}

# ── OS / arch detection ───────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Linux)  ;;
    Darwin) ;;
    *)      die "Unsupported OS: $OS" ;;
esac

# ── Python 3.12 ───────────────────────────────────────────────────────────────
info "Checking Python 3.12..."
PYTHON=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
        case "$ver" in
            3.12) PYTHON="$candidate"; break ;;
        esac
    fi
done
# uv will manage Python itself, so this is just an advisory check

# ── uv ───────────────────────────────────────────────────────────────────────
info "Checking uv..."
if ! command -v uv >/dev/null 2>&1; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this script session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv installation failed — please install manually: https://docs.astral.sh/uv/getting-started/installation/"
fi
ok "uv $(uv --version 2>&1 | head -1)"

# ── clone / update ────────────────────────────────────────────────────────────
need git
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating existing installation at $INSTALL_DIR ..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    info "Cloning SIES into $INSTALL_DIR ..."
    git clone --depth 1 "$REPO" "$INSTALL_DIR"
fi
ok "Source at $INSTALL_DIR"

# ── sync dependencies ─────────────────────────────────────────────────────────
info "Installing dependencies (CPU-only torch, sentence-transformers, sqlite-vec …)"
info "This may take a few minutes on first run."
cd "$INSTALL_DIR"
uv sync --no-dev
ok "Dependencies installed"

# ── wrapper scripts ───────────────────────────────────────────────────────────
install_wrapper() {
    name="$1"
    module="$2"
    mkdir -p "$BIN_DIR"
    cat > "$BIN_DIR/$name" <<EOF
#!/usr/bin/env sh
exec uv run --project "$INSTALL_DIR" python -m $module "\$@"
EOF
    chmod +x "$BIN_DIR/$name"
}

info "Installing CLI wrappers to $BIN_DIR ..."
install_wrapper sies-index  sies.index
install_wrapper sies-search sies.search
install_wrapper sies-ab     sies.ab
install_wrapper sies-stats  sies.stats
install_wrapper sies-bench  sies.bench
ok "Wrappers: sies-index  sies-search  sies-ab  sies-stats  sies-bench"

# ── PATH reminder ─────────────────────────────────────────────────────────────
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        info "Add $BIN_DIR to your PATH:"
        printf '\n    # Add to ~/.bashrc or ~/.zshrc:\n'
        printf '    export PATH="%s:$PATH"\n\n' "$BIN_DIR"
        ;;
esac

# ── corpus dir ────────────────────────────────────────────────────────────────
CORPUS_DIR="$INSTALL_DIR/corpus"
if [ ! -d "$CORPUS_DIR" ]; then
    mkdir -p "$CORPUS_DIR"
fi

# ── done ──────────────────────────────────────────────────────────────────────
printf '\n\033[0;32m━━━ SIES installed! ━━━\033[0m\n\n'
printf '  1) Drop your notes/essays into:\n'
printf '         %s/\n' "$CORPUS_DIR"
printf '     (sub-folders per source, .md / .txt preferred)\n\n'
printf '  2) Index:\n'
printf '         sies-index\n\n'
printf '  3) Search:\n'
printf '         sies-search "할머니에 대한 기억"\n'
printf '         sies-search "관성에 대하여" --invert\n\n'
printf '  4) A/B test (kill-test):\n'
printf '         sies-ab "관성에 대하여" --judge\n'
printf '         sies-stats\n\n'
printf '  Docs: %s/README.md\n\n' "$INSTALL_DIR"
