use std::error;
use std::fmt;
use std::io;

/// An error.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Line(String, usize),
    Node(String, usize, usize, usize),
}

impl error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Io(ref err) => err.fmt(f),
            Error::Line(ref msg, line) => write!(f, "FFM error (line: {}): {}", line + 1, msg),
            Error::Node(ref msg, line, node, token) => write!(f, "FFM error (line: {}, node: {}, token: {}): {}", line + 1, node + 1, token + 1, msg),
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::Io(err)
    }
}
