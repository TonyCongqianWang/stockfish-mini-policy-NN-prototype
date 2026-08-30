"""
Python Installation & Verification Script for Monty Chess Engine.
Resolves Monty target directory via `paths.py` and automates toolchain verification, cloning, and building.
"""

import os
import shutil
import subprocess
import sys

from paths import MONTY_BIN, REPO_ROOT


def run_command(cmd, cwd=None, env=None, check=True):
    print(f"--> Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    res = subprocess.run(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str))
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with returncode {res.returncode}: {cmd}")
    return res


def main():
    print("=" * 80)
    print("           MONTY CHESS ENGINE INSTALLER & BUILD VERIFIER")
    print("=" * 80)

    monty_exe = MONTY_BIN
    monty_dir = os.path.dirname(monty_exe)

    print(f"[1/5] Target Monty Binary:    {monty_exe}")
    print(f"      Target Monty Directory: {monty_dir}\n")

    # 1. Check system tools
    print("[2/5] Checking system prerequisites (make, git, curl)...")
    for tool in ["make", "git", "curl"]:
        if not shutil.which(tool):
            print(f"Error: Required tool '{tool}' is not installed or not in PATH.", file=sys.stderr)
            sys.exit(1)
    print("      System prerequisites verified.\n")

    # 2. Check Rust / Cargo
    print("[3/5] Checking Rust / Cargo toolchain...")
    cargo_path = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    rustc_path = shutil.which("rustc") or os.path.expanduser("~/.cargo/bin/rustc")

    env = os.environ.copy()
    cargo_bin_dir = os.path.expanduser("~/.cargo/bin")
    if os.path.isdir(cargo_bin_dir):
        env["PATH"] = cargo_bin_dir + os.pathsep + env.get("PATH", "")

    if not shutil.which("cargo", path=env["PATH"]):
        print("      Rust/Cargo not found. Installing rustup...")
        run_command("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y", env=env)

    rustc_ver = subprocess.check_output(["rustc", "--version"], env=env, text=True).strip()
    print(f"      Rust toolchain verified: {rustc_ver}\n")

    # 3. Clone repository if needed
    print("[4/5] Preparing Monty source code...")
    if not os.path.isdir(monty_dir) or not os.path.isfile(os.path.join(monty_dir, "Cargo.toml")):
        os.makedirs(os.path.dirname(monty_dir), exist_ok=True)
        print(f"      Cloning official Monty repository into {monty_dir}...")
        run_command(["git", "clone", "https://github.com/official-monty/Monty.git", monty_dir], env=env)
    else:
        print("      Monty repository source present.")
    print()

    # 4. Build Monty
    print("[5/5] Building Monty native release binary (make)...")
    num_cores = os.cpu_count() or 4
    env["CARGO_BUILD_JOBS"] = str(num_cores)
    run_command(["make", f"-j{num_cores}"], cwd=monty_dir, env=env)

    if not os.path.isfile(monty_exe):
        print(f"Error: Build completed but binary not found at {monty_exe}", file=sys.stderr)
        sys.exit(1)

    # 5. Sanity check UCI
    print("\nRunning UCI validation check...")
    proc = subprocess.Popen(
        [monty_exe],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    proc.stdin.write("uci\nisready\nquit\n")
    proc.stdin.flush()
    stdout_data, _ = proc.communicate(timeout=10)

    if "readyok" in stdout_data:
        print("UCI validation SUCCESS: Monty is responding to UCI protocol commands.")
    else:
        print("Warning: Monty binary compiled, but 'readyok' was not detected in output.")

    print("=" * 80)
    print("Monty is ready for telemetry and policy extraction!")
    print(f"Binary path: {monty_exe}")
    print("=" * 80)


if __name__ == "__main__":
    main()
