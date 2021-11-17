use crate::error::Error;

#[derive(Clone, Debug, Default)]
#[repr(C)]
pub struct Node {
    pub f: i32,
    pub j: i32,
    pub v: f32,
}

impl Node {
    pub fn parse(token: &str, line: usize, pos: usize) -> Result<Self, Error> {
        let mut parts = token.split(":");
        let n = Self {
            f: parts.next().ok_or_else(|| Error::Node("missing token".to_string(), line, pos, 0))?.parse().map_err(|_| Error::Node("expected int".to_string(), line, pos, 0))?,
            j: parts.next().ok_or_else(|| Error::Node("missing token".to_string(), line, pos, 1))?.parse().map_err(|_| Error::Node("expected int".to_string(), line, pos, 1))?,
            v: parts.next().ok_or_else(|| Error::Node("missing token".to_string(), line, pos, 2))?.parse().map_err(|_| Error::Node("expected float".to_string(), line, pos, 2))?,
        };

        if parts.next().is_some() {
            return Err(Error::Node("too many tokens".to_string(), line, pos, 3));
        }

        Ok(n)
    }
}
