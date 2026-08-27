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
