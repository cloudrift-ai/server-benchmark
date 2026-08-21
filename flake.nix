{
  description = "emmy dev shell: toolchain + runtime libs for venv binary wheels";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          pkg-config
          gcc
          cmake
        ];

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
