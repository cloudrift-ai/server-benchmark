{
  description = "emmy dev shell: toolchain + runtime libs for venv binary wheels";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      # A second instance, because ``cudaSupport`` is an evaluation-time config rather than a
      # per-package option: it has to be set where nixpkgs is imported. The CUDA packages are
      # unfree, so that flag rides along. Nothing else in the shell comes from here, which keeps
      # the ordinary toolchain on the cached ``legacyPackages`` instance.
      #
      # These packages are NOT in the official binary cache, so a first ``nix develop`` builds
      # them from source — hours, not minutes. Adding the community CUDA cache to the HOST
      # configuration avoids that; this flake deliberately does not, because substituters are a
      # trust decision that belongs to whoever runs the machine, not to a checked-in dev shell:
      # https://wiki.nixos.org/wiki/CUDA#Setting_up_CUDA_Binary_Cache
      cudaPkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.pkg-config
          pkgs.gcc
          pkgs.cmake
          # The interpreter the venv's symlinks point at. Without it in the shell's closure
          # nothing keeps that store path alive, and a garbage collection leaves
          # ``venv/bin/python`` dangling — the venv looks present and every call fails.
          pkgs.python313
          # ``pip install ruff`` puts a generic-linux binary in the venv, which NixOS cannot
          # exec. The nixpkgs build is what ``make lint`` finds on PATH here.
          pkgs.ruff
          # ``nvcc``, which the CUDA backend shells out to for every kernel it compiles.
          cudaPkgs.cudatoolkit
        ];

        # The compiler reads this to find the toolkit root; without it a CUDA build falls back to
        # whatever ``/usr/local/cuda`` happens to be, which on NixOS is nothing.
        CUDA_HOME = "${cudaPkgs.cudatoolkit}";

        # NOT covered here: the loop backend JIT-compiles kernels in-process through cppyy, whose
        # bundled ``rootcling`` is a generic-linux executable. NixOS stubs the loader such
        # binaries need, so Cling fails to build its precompiled header and then FAULTS — the
        # symptom is a segfault during pytest collection ("node down: Not properly terminated"),
        # which reads like a broken checkout rather than a missing setting. ``programs.nix-ld.enable``
        # in the HOST configuration supplies a real loader and fixes it (verified: cppyy imports and
        # ``tests/compiler/ir/loop/`` passes). A devShell cannot set it for its user, which is why
        # it is named here rather than solved here.
        #
        # venv-installed wheels (numpy, torch, ...) dlopen these at run time;
        # makeLibraryPath picks each package's lib output, so plain `zlib` is
        # correct here (no `.out` footgun).
        LD_LIBRARY_PATH = nixpkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc
          pkgs.zlib
        ];
      };
    };
}
