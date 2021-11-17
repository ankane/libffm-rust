use crate::error::Error;
use crate::node::Node;
use crate::timer::Timer;
use byteorder::{NativeEndian, ReadBytesExt};
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Cursor, Read, Seek, SeekFrom, Write};
use std::mem;
use std::num::Wrapping;
use std::path::{Path, PathBuf};

const CHUNK_SIZE: usize = 10000000;

#[derive(Debug, Default)]
pub struct DiskProblemMeta {
    pub(crate) n: i32,
    pub(crate) m: i32,
    pub(crate) l: i32,
    pub(crate) num_blocks: i32,
    pub(crate) b_pos: i64,
    pub(crate) hash1: u64,
    pub(crate) hash2: u64,
}

pub struct ProblemOnDisk<W> {
    pub(crate) meta: DiskProblemMeta,
    pub(crate) y: Vec<f32>,
    pub(crate) r: Vec<f32>,
    pub(crate) p: Vec<i64>,
    pub(crate) x: Vec<Node>,
    pub(crate) b: Vec<i64>,
    pub(crate) f: W,
}

impl<W: Read + Write + Seek> ProblemOnDisk<W> {
    pub fn new(mut f: W) -> Result<Self, Error> {
        f.rewind()?;
        let meta = DiskProblemMeta::new(&mut f)?;
        f.seek(SeekFrom::Start(meta.b_pos as u64))?;
        let mut b = vec![0; meta.num_blocks as usize];
        f.read_i64_into::<NativeEndian>(&mut b)?;

        Ok(Self {
            meta,
            y: Vec::new(),
            r: Vec::new(),
            p: Vec::new(),
            x: Vec::new(),
            b: b,
            f: f,
        })
    }

    pub fn load_block(&mut self, block_index: i32) -> Result<usize, Error> {
        assert!(block_index < self.meta.num_blocks);

        self.f.seek(SeekFrom::Start(self.b[block_index as usize] as u64))?;

        let l = self.f.read_i32::<NativeEndian>()? as usize;

        self.y.resize(l, 0.0);
        self.f.read_f32_into::<NativeEndian>(&mut self.y)?;

        self.r.resize(l, 0.0);
        self.f.read_f32_into::<NativeEndian>(&mut self.r)?;

        self.p.resize(l + 1, 0);
        self.f.read_i64_into::<NativeEndian>(&mut self.p)?;

        self.x.resize(self.p[l] as usize, Node::default());
        unsafe {
            let buffer: &mut [u8] = std::slice::from_raw_parts_mut(
                self.x.as_mut_ptr().cast(),
                mem::size_of::<Node>() * self.p[l] as usize,
            );
            self.f.read_exact(buffer)?;
        }

        Ok(l)
    }
}

impl DiskProblemMeta {
    pub fn new<R: Read>(f_bin: &mut R) -> Result<Self, Error> {
        Ok(Self {
            n: f_bin.read_i32::<NativeEndian>()?,
            m: f_bin.read_i32::<NativeEndian>()?,
            l: f_bin.read_i32::<NativeEndian>()?,
            num_blocks: f_bin.read_i32::<NativeEndian>()?,
            b_pos: f_bin.read_i64::<NativeEndian>()?,
            hash1: f_bin.read_u64::<NativeEndian>()?,
            hash2: f_bin.read_u64::<NativeEndian>()?,
        })
    }

    pub fn write<W: Write + Seek>(&self, f_bin: &mut W) -> Result<(), Error> {
        f_bin.write_all(&self.n.to_ne_bytes())?;
        f_bin.write_all(&self.m.to_ne_bytes())?;
        f_bin.write_all(&self.l.to_ne_bytes())?;
        f_bin.write_all(&self.num_blocks.to_ne_bytes())?;
        f_bin.write_all(&self.b_pos.to_ne_bytes())?;
        f_bin.write_all(&self.hash1.to_ne_bytes())?;
        f_bin.write_all(&self.hash2.to_ne_bytes())?;
        Ok(())
    }
}

pub struct ProblemLoader {
    path: PathBuf,
    f_txt: File,
    quiet: bool,
}

impl ProblemLoader {
    pub fn new<P: AsRef<Path>>(path: P, quiet: bool) -> Result<Self, Error> {
        Ok(ProblemLoader {
            path: path.as_ref().to_path_buf(),
            f_txt: File::open(path)?,
            quiet,
        })
    }

    pub fn read_to_memory(&mut self) -> Result<ProblemOnDisk<Cursor<Vec<u8>>>, Error> {
        // start with 8kb
        let mut f_bin = Cursor::new(Vec::with_capacity(8192));

        let mut timer = Timer::new();

        self.log("Convert text file to binary ".to_string());
        txt2bin(&mut self.f_txt, &mut f_bin)?;
        self.logln(format!("({:.1} seconds)", timer.toc()));

        ProblemOnDisk::new(f_bin)
    }

    pub fn read_to_disk(&mut self) -> Result<ProblemOnDisk<File>, Error> {
        let mut bin_path = self.path.file_name().unwrap().to_os_string();
        bin_path.push(".bin");
        let mut f_bin = OpenOptions::new().read(true).write(true).create(true).open(&bin_path)?;

        let mut timer = Timer::new();

        self.log("First check if the text file has already been converted to binary format ".to_string());
        let same_file = check_same_txt_bin(&mut self.f_txt, &mut f_bin).unwrap_or(false);
        self.logln(format!("({:.1} seconds)", timer.toc()));

        if same_file {
            self.logln("Binary file found. Skip converting text to binary".to_string());
        } else {
            self.log("Binary file NOT found. Convert text file to binary file ".to_string());
            txt2bin(&mut self.f_txt, &mut f_bin)?;
            self.logln(format!("({:.1} seconds)", timer.toc()));
        }

        ProblemOnDisk::new(f_bin)
    }

    fn log(&self, msg: String) {
        if !self.quiet {
            print!("{}", msg);
            io::stdout().flush().unwrap();
        }
    }

    fn logln(&self, msg: String) {
        if !self.quiet {
            println!("{}", msg);
        }
    }
}

fn hashfile(f: &mut File, one_block: bool) -> Result<u64, Error> {
    let end = f.seek(SeekFrom::End(0))? as usize;
    f.rewind()?;

    let mut buffer = BufReader::with_capacity(CHUNK_SIZE, f);
    let mut magic: Wrapping<u64> = Wrapping(90359);
    let mut pos = 0;
    while pos < end {
        let next_pos = (pos + CHUNK_SIZE).min(end);
        let size = next_pos - pos;

        let mut i = 0;
        if size >= 8 {
            while i < size - 8 {
                let x = buffer.read_u64::<NativeEndian>()?;
                magic = ((magic + Wrapping(x)) * (magic + Wrapping(x + 1)) >> 1) + Wrapping(x);
                i += 8;
            }
        }
        while i < size {
            let x = buffer.read_u8()? as u64;
            magic = ((magic + Wrapping(x)) * (magic + Wrapping(x + 1)) >> 1) + Wrapping(x);
            i += 1;
        }

        pos = next_pos;
        if one_block {
            break;
        }
    }

    Ok(magic.0)
}

fn write_chunk<W: Write + Seek>(f_bin: &mut W, y: &mut Vec<f32>, r: &mut Vec<f32>, p: &mut Vec<i64>, x: &mut Vec<Node>, b: &mut Vec<i64>, meta: &mut DiskProblemMeta, p2: &mut i64) -> Result<(), Error> {
    b.push(f_bin.seek(SeekFrom::Current(0))? as i64);
    let l = y.len();
    meta.l += l as i32;

    f_bin.write_all(&(l as i32).to_ne_bytes())?;
    f_bin.write_all(&y.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>())?;
    f_bin.write_all(&r.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>())?;
    f_bin.write_all(&p.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>())?;
    f_bin.write_all(&x.iter().flat_map(|n| [n.f.to_ne_bytes(), n.j.to_ne_bytes(), n.v.to_ne_bytes()].concat()).collect::<Vec<u8>>())?;

    y.clear();
    r.clear();
    p.clear();
    p.push(0);
    x.clear();
    *p2 = 0;
    meta.num_blocks += 1;
    Ok(())
}

pub(crate) fn parse_y(token: Option<&str>, i: usize) -> Result<f32, Error> {
    if token.ok_or_else(|| Error::Line("expected line to start with int".to_string(), i))?.parse::<i32>().map_err(|_| Error::Line("expected line to start with int".to_string(), i))? > 0 {
        Ok(1.0)
    } else {
        Ok(-1.0)
    }
}

fn txt2bin<W: Write + Seek>(f_txt: &mut File, f_bin: &mut W) -> Result<(), Error> {
    let mut p2 = 0;
    let mut meta = DiskProblemMeta::default();

    let mut y: Vec<f32> = Vec::new();
    let mut r: Vec<f32> = Vec::new();
    let mut p: Vec<i64> = vec![0; 1];
    let mut x: Vec<Node> = Vec::new();
    let mut b: Vec<i64> = Vec::new();

    meta.write(f_bin)?;

    // before BufReader takes ownership
    meta.hash1 = hashfile(f_txt, true)?;
    meta.hash2 = hashfile(f_txt, false)?;

    f_txt.rewind()?;

    let reader = BufReader::new(f_txt);
    for (i, line_option) in reader.lines().enumerate() {
        let line = line_option?;
        let mut tokens = line.split(&[' ', '\t'][..]);

        // returns single token for empty line
        let y2 = parse_y(tokens.next(), i)?;

        let mut scale = 0.0;
        for (j, token) in tokens.enumerate() {
            let n = Node::parse(token, i, j)?;

            meta.m = meta.m.max(n.f + 1);
            meta.n = meta.n.max(n.j + 1);

            scale += n.v * n.v;

            x.push(n);

            p2 += 1;
        }
        scale = 1.0 / scale;

        y.push(y2);
        r.push(scale);
        p.push(p2);

        if x.len() > CHUNK_SIZE {
            write_chunk(f_bin, &mut y, &mut r, &mut p, &mut x, &mut b, &mut meta, &mut p2)?;
        }
    }
    write_chunk(f_bin, &mut y, &mut r, &mut p, &mut x, &mut b, &mut meta, &mut p2)?;

    // write a dummy empty chunk in order to know where the EOF is
    write_chunk(f_bin, &mut y, &mut r, &mut p, &mut x, &mut b, &mut meta, &mut p2)?;

    assert_eq!(meta.num_blocks as usize, b.len());

    meta.b_pos = f_bin.seek(SeekFrom::Current(0))? as i64;
    f_bin.write_all(&b.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>())?;

    f_bin.seek(SeekFrom::Start(0))?;
    meta.write(f_bin)?;

    Ok(())
}

fn check_same_txt_bin<W: Read + Write + Seek>(f_txt: &mut File, f_bin: &mut W) -> Result<bool, Error> {
    let meta = DiskProblemMeta::new(f_bin)?;
    Ok(meta.hash1 == hashfile(f_txt, true)? && meta.hash2 == hashfile(f_txt, false)?)
}
