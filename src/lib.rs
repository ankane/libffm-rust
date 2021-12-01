//! Field-aware factorization machines in Rust
//!
//! [View the docs](https://github.com/ankane/libffm-rust)

mod disk;
mod error;
mod model;
mod node;
mod params;
mod timer;

pub use error::Error;
pub use model::Model;
pub use params::Params;

#[cfg(test)]
mod tests {
    use crate::*;
    use std::path::PathBuf;

    #[test]
    fn test_str() {
        let model = Model::train("data/train.ffm").unwrap();
        assert!(model.predict("data/test.ffm").is_ok());
    }

    #[test]
    fn test_string() {
        let model = Model::train("data/train.ffm".to_string()).unwrap();
        assert!(model.predict("data/test.ffm".to_string()).is_ok());
    }

    #[test]
    fn test_pathbuf() {
        let train_path = PathBuf::from("data/train.ffm");
        let test_path = PathBuf::from("data/test.ffm");
        let model = Model::train(&train_path).unwrap();
        assert!(model.predict(&test_path).is_ok());
    }

    #[test]
    fn test_train_eval() {
        Model::params()
            .auto_stop(true)
            .train_eval("data/train.ffm", "data/valid.ffm")
            .unwrap();
    }

    #[test]
    fn test_empty() {
        assert!(Model::train("data/empty.ffm").is_ok());
    }

    #[test]
    fn test_missing_train() {
        let err = Model::train("missing.ffm").unwrap_err();
        assert_eq!(err.to_string(), "No such file or directory (os error 2)".to_string());
    }

    #[test]
    fn test_missing_train_eval() {
        let err = Model::train_eval("data/train.ffm", "missing.ffm").unwrap_err();
        assert_eq!(err.to_string(), "No such file or directory (os error 2)".to_string());
    }

    #[test]
    fn test_missing_predict() {
        let model = Model::train("data/train.ffm").unwrap();
        let err = model.predict("missing.ffm").unwrap_err();
        assert_eq!(err.to_string(), "No such file or directory (os error 2)".to_string());
    }

    #[test]
    fn test_empty_line() {
        let err = Model::train("data/empty_line.ffm").unwrap_err();
        assert_eq!(err.to_string(), "FFM error (line: 2): expected line to start with int".to_string());
    }

    #[test]
    fn test_parse_line() {
        let err = Model::train("data/parse_line.ffm").unwrap_err();
        assert_eq!(err.to_string(), "FFM error (line: 1): expected line to start with int".to_string());
    }

    #[test]
    fn test_parse_node() {
        let err = Model::train("data/parse_node.ffm").unwrap_err();
        assert_eq!(err.to_string(), "FFM error (line: 1, node: 1, token: 1): expected int".to_string());
    }

    #[test]
    fn test_too_few_tokens() {
        let err = Model::train("data/few_tokens.ffm").unwrap_err();
        assert_eq!(err.to_string(), "FFM error (line: 1, node: 1, token: 3): missing token".to_string());
    }

    #[test]
    fn test_too_many_tokens() {
        let err = Model::train("data/many_tokens.ffm").unwrap_err();
        assert_eq!(err.to_string(), "FFM error (line: 1, node: 1, token: 4): too many tokens".to_string());
    }
}
