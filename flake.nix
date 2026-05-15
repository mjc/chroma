{
  description = "Development shell for Chroma";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };
        lib = pkgs.lib;
        python = pkgs.python313.withPackages (ps: with ps; [
          bcrypt
          build
          fastapi
          grpcio
          httpx
          hypothesis
          chroma-hnswlib
          importlib-resources
          jsonschema
          kubernetes
          mmh3
          numpy
          onnxruntime
          opentelemetry-api
          opentelemetry-exporter-otlp-proto-grpc
          opentelemetry-instrumentation-fastapi
          opentelemetry-sdk
          orjson
          overrides
          pandas
          psutil
          pybase64
          pydantic
          pydantic-settings
          pypika
          pytest
          pytest-asyncio
          pytest-rerunfailures
          pytest-xdist
          pyyaml
          rich
          setuptools
          setuptools-scm
          tenacity
          tokenizers
          tqdm
          typer
          typing-extensions
          uvicorn
          virtualenv
          wheel
        ]);
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python
            rustToolchain
            protobuf
            pkg-config
            openssl
            jemalloc
            cmake
            gnumake
            gcc
            git
            perl
          ] ++ lib.optionals pkgs.stdenv.isLinux [
            pkgs.stdenv.cc.cc.lib
          ];

          LD_LIBRARY_PATH = lib.makeLibraryPath [
            pkgs.jemalloc
            pkgs.openssl
            pkgs.stdenv.cc.cc.lib
          ];
          JEMALLOC_OVERRIDE = "${pkgs.jemalloc}/lib/libjemalloc.a";
          PROTOC = "${pkgs.protobuf}/bin/protoc";
          CC = "${pkgs.gcc}/bin/gcc";
          CFLAGS = "-D_GNU_SOURCE";
          CPPFLAGS = "-D_GNU_SOURCE";
          PIP_DISABLE_PIP_VERSION_CHECK = "1";

          shellHook = ''
            export CHROMA_DEFAULT_VENV="''${CHROMA_DEFAULT_VENV:-.venv}"
            echo "Chroma dev shell"
            echo "Python: $(python --version)"
            echo "Rust: $(rustc --version)"
            echo "Python deps supplied by nix shell"
          '';
        };
      });
}
