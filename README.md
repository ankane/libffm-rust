# LIBFFM Rust

[LIBFFM](https://github.com/ycjuan/libffm) - field-aware factorization machines - in Rust

[![Build Status](https://github.com/ankane/libffm-rust/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/libffm-rust/actions)

## Getting Started

LIBFFM Rust is available as a Rust library and a command line tool.

- [Rust library](#rust-library)
- [Command line tool](#command-line-tool)

## Rust Library

### Installation

Add this line to your application’s `Cargo.toml` under `[dependencies]`:

```toml
libffm = "0.2"
```

### How to Use

Prep your data in LIBFFM format

```txt
0 0:0:1 1:1:1
1 0:2:1 1:3:1
```

Train a model

```rust
let model = libffm::Model::train("train.ffm").unwrap();
```

Use a validation set and early stopping to prevent overfitting

```rust
let model = libffm::Model::params()
    .auto_stop(true)
    .train_eval("train.ffm", "valid.ffm")
    .unwrap();
```

Make predictions

```rust
let (predictions, loss) = model.predict("test.ffm").unwrap();
```

Save the model to a file

```rust
model.save("model.bin").unwrap();
```

Load a model from a file

```rust
let model = libffm::Model::load("model.bin").unwrap();
```

### Training Options

```rust
let model = libffm::Model::params()
    .learning_rate(0.2)      // learning rate
    .lambda(0.00002)         // regularization parameter
    .iterations(15)          // number of iterations
    .factors(4)              // number of latent factors
    .quiet(false)            // quiet mode (no output)
    .normalization(true)     // use instance-wise normalization
    .auto_stop(false)        // stop at the iteration that achieves the best validation loss
    .on_disk(false)          // on-disk training
    .train("train.ffm");     // train or train_eval
```

## Command Line Tool

### Installation

Run:

```sh
cargo install libffm --features cli
```

### How to Use

Prep your data in LIBFFM format

```txt
0 0:0:1 1:1:1
1 0:2:1 1:3:1
```

Train a model

```sh
ffm-train train.ffm model.bin
```

Use a validation set and early stopping to prevent overfitting

```sh
ffm-train -p valid.ffm --auto-stop train.ffm model.bin
```

Make predictions

```sh
ffm-predict test.ffm model.bin output.txt
```

### Training Options

```txt
FLAGS:
        --auto-stop    Stop at the iteration that achieves the best validation loss (must be used with -p)
        --in-memory    Enable in-memory training
        --no-norm      Disable instance-wise normalization
        --quiet        Quiet mode (no output)

OPTIONS:
    -r <eta>               Set learning rate [default: 0.2]
    -k <factor>            Set number of latent factors [default: 4]
    -t <iteration>         Set number of iterations [default: 15]
    -l <lambda>            Set regularization parameter [default: 0.00002]
    -s <nr-threads>        Set number of threads [default: 1]
    -p <va-path>           Set path to the validation set
```

## Credits

This library was ported from the [LIBFFM C++ library](https://github.com/ycjuan/libffm) and is available under the same license.

## History

View the [changelog](https://github.com/ankane/libffm-rust/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/libffm-rust/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/libffm-rust/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/libffm-rust.git
cd libffm-rust
cargo test
cargo run --bin ffm-train --features cli
cargo run --bin ffm-predict --features cli
```
