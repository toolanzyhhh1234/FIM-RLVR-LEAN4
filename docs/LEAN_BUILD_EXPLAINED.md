# Understanding the Lean 4 Build Process

This document explains why our build script skips `lake build` and how the Lean 4 compilation pipeline works. Written for beginners who want to understand what's happening under the hood.

---

## Table of Contents
1. [The Problem We Solved](#the-problem-we-solved)
2. [Lean 4 Build Concepts for Beginners](#lean-4-build-concepts-for-beginners)
3. [Why Our Build Was Slow](#why-our-build-was-slow)
4. [The Solution](#the-solution)
5. [Technical Deep Dive](#technical-deep-dive)

---

## The Problem We Solved

**Original issue:** Our build script took a very long time (potentially 30+ minutes) even though we were downloading prebuilt mathlib caches.

**Root cause:** We were running `lake build` which compiled our `VerificationEnv` library, but our verification workflow doesn't actually need that.

**Solution:** Skip `lake build` entirely—our verifier only needs the mathlib cache.

---

## Lean 4 Build Concepts for Beginners

### What is Lean 4?

Lean 4 is a **proof assistant** and **programming language**. When you write Lean code, the compiler doesn't just check syntax—it mathematically verifies that your proofs are correct. This is what makes Lean powerful for formal verification.

### Key Files in a Lean Project

| File | Purpose |
|------|---------|
| `lakefile.lean` | Project configuration (like `package.json` in Node.js or `Cargo.toml` in Rust) |
| `lake-manifest.json` | Lock file with exact dependency versions |
| `*.lean` | Lean source files |
| `.lake/` | Build artifacts and downloaded dependencies |

### What is Lake?

**Lake** is Lean 4's build system and package manager (similar to Cargo for Rust or npm for JavaScript). Key commands:

```bash
lake update      # Update dependencies listed in lakefile.lean
lake build       # Compile the project
lake exe <tool>  # Run an executable from a dependency
lake env <cmd>   # Run a command with the project's environment
```

### What are `.olean` Files?

When Lean compiles a `.lean` file, it produces an **`.olean` file** (Object Lean). Think of these like:
- `.pyc` files in Python
- `.o` files in C/C++
- `.class` files in Java

These contain the compiled/type-checked version of the code. Loading an `.olean` is **much faster** than re-compiling the `.lean` source.

### What is Mathlib?

**Mathlib** (specifically `mathlib4` for Lean 4) is a massive community-maintained library of formalized mathematics. It contains:
- ~4 million lines of Lean code
- Thousands of theorems covering algebra, analysis, topology, number theory, etc.
- Definitions for mathematical objects (groups, rings, real numbers, etc.)

**The catch:** Mathlib is huge. Compiling it from source takes **hours** on a fast machine.

---

## Why Our Build Was Slow

### Our Project Structure

```
verification_env/
├── lakefile.lean          # Declares dependency on mathlib
├── lake-manifest.json     # Locks mathlib to v4.15.0
└── VerificationEnv/
    └── Basic.lean         # Our code: "import Mathlib"
```

### The `Basic.lean` File

```lean
import Mathlib

theorem sanity_check : 1 + 1 = 2 := by simp
```

**`import Mathlib`** is the problem. This single line imports the **entire** mathlib library—all ~4 million lines of it.

### What Happens During `lake build`

When you run `lake build`, Lake does the following:

1. **Check dependencies:** Ensure mathlib is downloaded
2. **Load mathlib's `.olean` files:** Even with prebuilt caches, Lean must load and link all the compiled modules
3. **Compile your code:** Build `VerificationEnv/Basic.lean`
4. **Produce `.olean`:** Create `.lake/build/lib/VerificationEnv/Basic.olean`

**Step 2 and 3** are slow because:
- Mathlib has ~5,000+ modules
- Each module must be loaded into memory
- Dependencies must be resolved and linked
- Type checking happens across module boundaries

### The Cache Helps, But Doesn't Eliminate Work

`lake exe cache get` downloads prebuilt `.olean` files from mathlib's CI servers. This saves you from **compiling** mathlib (hours of work), but Lean still must:
- Download ~2GB of cached files
- Load and link them when building code that imports them
- Verify compatibility with your Lean toolchain version

---

## The Solution

### Why We Can Skip `lake build`

Our verification workflow uses this command (from `lean_verifier.py`):

```python
subprocess.run(["lake", "env", "lean", file_path], ...)
```

**`lake env lean <file>`** does something different from `lake build`:

| Command | What It Does |
|---------|--------------|
| `lake build` | Compiles your entire project, producing `.olean` files |
| `lake env lean <file>` | Type-checks a single file using the project's environment |

The key insight: **`lake env lean` only needs the dependencies' `.olean` files to exist**. It does NOT need your project's `.olean` files.

### What Our Build Script Does Now

```bash
# 0. Enter the directory containing lakefile.lean (CRITICAL)
cd verification_env

# 1. Update dependencies (downloads mathlib source if needed)
lake update

# 2. Download prebuilt mathlib cache (~2GB of .olean files)
lake exe cache get

# 3. We SKIP 'lake build' because:
#    - Our verifier uses 'lake env lean <file>'
#    - That command only needs mathlib's cache (step 2)
#    - It does NOT need VerificationEnv to be compiled
```

### Time Savings

| Step | With `lake build` | Without `lake build` |
|------|-------------------|----------------------|
| `lake update` | ~30 seconds | ~30 seconds |
| `lake exe cache get` | ~2-5 minutes | ~2-5 minutes |
| `lake build` | ~10-30 minutes | **SKIPPED** |
| **Total** | ~15-35 minutes | **~3-6 minutes** |

---

## Technical Deep Dive

### How `lake env lean` Works

When you run `lake env lean MyFile.lean`:

1. Lake reads `lakefile.lean` to understand the project structure
2. Lake sets up environment variables pointing to dependency locations:
   - `LEAN_PATH` includes `.lake/packages/mathlib/.lake/build/lib/`
   - This tells Lean where to find `.olean` files
3. Lake invokes the `lean` compiler on your file
4. Lean parses, elaborates, and type-checks the file
5. If successful, exit code 0; if errors, exit code 1

**Crucially:** This works even if your project has never been built, as long as the dependencies are cached.

### Why `import Mathlib` is Expensive

```lean
import Mathlib
```

This is equivalent to importing ALL of mathlib's public modules. When Lean processes this:

1. It must resolve what "Mathlib" means (the root module)
2. Load the `.olean` for `Mathlib.lean` (which re-exports everything)
3. Transitively load ALL modules that `Mathlib` exports
4. This means loading thousands of `.olean` files

**Better practice for production code:**
```lean
-- Import only what you need
import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic
```

But for our use case (verifying arbitrary proofs that may use any part of mathlib), `import Mathlib` is necessary.

### Why We Still Need `lake exe cache get`

Even though we skip `lake build`, we MUST run `lake exe cache get` because:

1. Our verification files do `import Mathlib`
2. When `lake env lean` runs on those files, Lean needs mathlib's `.olean` files
3. Without the cache, Lean would try to compile mathlib from source (hours)
4. The cache provides the prebuilt `.olean` files that make verification fast

### The Verification Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    BUILD PHASE (one-time)                       │
├─────────────────────────────────────────────────────────────────┤
│  lake update         → Download mathlib source code             │
│  lake exe cache get  → Download prebuilt .olean files (~2GB)    │
│  [lake build]        → SKIPPED (not needed)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               VERIFICATION PHASE (per proof)                    │
├─────────────────────────────────────────────────────────────────┤
│  1. Python creates temp file: Verify_<uuid>.lean                │
│  2. Writes: import Mathlib + theorem + proof                    │
│  3. Runs: lake env lean Verify_<uuid>.lean                      │
│  4. Lean loads mathlib .olean files (cached)                    │
│  5. Lean type-checks the proof                                  │
│  6. Returns: success (exit 0) or failure (exit 1)               │
│  7. Python deletes temp file                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Characteristics

Each verification call (`lake env lean <file>`) takes approximately:
- **~1-3 seconds** on first run (loading mathlib into memory)
- **~0.5-2 seconds** for subsequent runs (OS file caching helps)

This is why our training loop uses parallel verification—we can verify multiple proofs simultaneously across CPU cores.

---

## Summary

| Concept | Explanation |
|---------|-------------|
| **Lake** | Lean's build system and package manager |
| **`.olean` files** | Compiled Lean modules (like `.pyc` or `.o` files) |
| **Mathlib** | Huge math library (~4M lines); takes hours to compile |
| **`lake exe cache get`** | Downloads prebuilt mathlib `.olean` files |
| **`lake build`** | Compiles YOUR project (we don't need this) |
| **`lake env lean`** | Type-checks a single file using project environment |
| **Our optimization** | Skip `lake build`; only need mathlib cache for verification |

---

## Further Reading

- [Lean 4 Documentation](https://lean-lang.org/lean4/doc/)
- [Lake Documentation](https://github.com/leanprover/lean4/tree/master/src/lake)
- [Mathlib4 Repository](https://github.com/leanprover-community/mathlib4)
- [Mathlib Cache System](https://github.com/leanprover-community/mathlib4/wiki/Using-mathlib4-as-a-dependency)
