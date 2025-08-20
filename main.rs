use std::str;
use std::cmp::min;


enum QRECLevel {
	Low,
	Medium,
	Quartile,
	High
}


#[derive(Copy, Clone)]
enum Encoding {
	Numeric,
	AlphaNumeric,
	Bytes,
	ECI
}

// ████████████████████████████████████████████████████████████████ ENCODE QR CONTENTS
struct Encoder {
	mode: Encoding
}

impl Encoder {
	const ALNUM_CHARSET: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";

	// ████████████████████████████████ CREATE ENCODER AND BEGIN ENCODING
	fn new(mode: Encoding) -> Encoder { Encoder{ mode } }
	
	fn encode_data(&self, data: &str, ver: u32, len: u32) -> Option<Vec<u8>> {
		let mut bitstring = Vec::new();
		bitstring.extend(self.mode_bits());
		bitstring.extend(&self.char_count_bits(data.len() as u32, ver));
		
		let encoded = match self.mode {
			Encoding::Numeric => self.encode_numeric(data),
			Encoding::AlphaNumeric => self.encode_alphanumeric(data),
			Encoding::Bytes => self.encode_bytes(data),
			Encoding::ECI => self.encode_ECI(data)
		};
		bitstring.extend(&encoded);
		if bitstring.len() >= len as usize { return None; }
		
		bitstring = Encoder::pad_segments(bitstring, len);
		Some(Encoder::pack_bytes(bitstring))
	}
	
	// ████████████████████████████████ ADDITIONAL INFORMATION USED IN THE ENCODING PROCESS
	fn mode_bits(&self) -> [bool; 4] {
		match self.mode {
			Encoding::Numeric => [false, false, false, true],
			Encoding::AlphaNumeric => [false, false, true, false],
			Encoding::Bytes => [false, true, false, false],
			Encoding::ECI => [false, true, true, true]
		}
	}
	
	fn char_count_bits(&self, char_count: u32, version: u32) -> Vec<bool> {
		let pos = match self.mode {
			Encoding::Numeric => 0,
			Encoding::AlphaNumeric => 1,
			Encoding::Bytes => 2,
			Encoding::ECI => 3
		};
		let num_bits = match version {
			1..10 => [10, 9, 8, 0],
			10..27 => [12, 11, 16, 0],
			27..40 => [14, 13, 16, 0],
			_ => [0; 4]
		};
		Encoder::to_binary(char_count, num_bits[pos])
	}
	
	// ████████████████████████████████ DATA ENCODING FUNCTIONS
	fn encode_numeric(&self, data: &str) -> Vec<bool> {
		let groups = Encoder::split_chunks(data, 3);
		let mut bitstring = Vec::new();
		for g in groups {
			let num_bits = match g.len() {
				3 => 10, 2 => 7, 1 => 4, _ => 0
			};
			let val = g.parse().unwrap();
			bitstring.extend(&Encoder::to_binary(val, num_bits));
		}
		bitstring
	}
	
	fn encode_alphanumeric(&self, data: &str) -> Vec<bool> {
		let groups = Encoder::split_chunks(data, 2);
		let get_idx = |search: char| Encoder::ALNUM_CHARSET.chars().position(|chr| chr == search);
		let mut bitstring = Vec::new();
		for g in groups {
			let num_bits = match g.len() {
				1 => 6, 2 => 11, _ => 0
			};
			let val = g.chars().fold(0u32, |acc, c| 45*acc + get_idx(c).unwrap() as u32);
			bitstring.extend(&Encoder::to_binary(val, num_bits));
		}
		bitstring
	}
	
	fn encode_bytes(&self, data: &str) -> Vec<bool> {
		let mut bitstring = Vec::new();
		for chr in data.chars() {
			bitstring.extend(&Encoder::to_binary(u32::from(chr), 8));
		}
		bitstring
	}
	
	fn encode_ECI(&self, data: &str) -> Vec<bool> {
		let mut bitstring = Vec::new();
		let assignval: u32 = data.parse().unwrap();
		
		if assignval < (1 << 7) {
			bitstring.extend(&Encoder::to_binary(assignval, 8));
		} else if assignval < (1 << 14) {
			bitstring.extend([true, false]);
			bitstring.extend(&Encoder::to_binary(assignval, 14));
		} else if assignval < 1_000_000 {
			bitstring.extend([true, true, false]);
			bitstring.extend(&Encoder::to_binary(assignval, 21));
		}
		bitstring
	}
	
	// ████████████████████████████████ DATA POSTPROCESSING
	fn pack_bytes(data: Vec<bool>) -> Vec<u8> {
		let mut packed = Vec::new();
		let bytes = data.chunks(8);
		for b in bytes { // convert bits to u32 in big endian format (most significant bit first)
			let num = b.iter().enumerate().fold(0, |acc, (idx, bit)| if *bit { acc + (1 << (7-idx)) }else { acc });
			packed.push(num);
		}
		packed
	}
	
	fn pad_segments(data: Vec<bool>, len: u32) -> Vec<bool> {
		let length = len as usize;
		let mut padded = data.clone();
		if padded.len() == length { return padded; }
		
		for _ in 0..min(length - padded.len(), 4) { padded.push(false); } // add terminator 0s
		if padded.len() == length { return padded; }
		
		if padded.len()%8 != 0 {
			for _ in 0..(8 - padded.len()%8) { padded.push(false); } // pad to make length a multiple of 8
		}
		if padded.len() == length { return padded; }
		
		let pad_byte1 = Encoder::to_binary(236, 8);
		let pad_byte2 = Encoder::to_binary(17, 8);
		for i in 0..(length-padded.len()) / 8 {
			if i%2 == 0 { padded.extend(&pad_byte1); }else { padded.extend(&pad_byte2); }
		}
		padded
	}
	
	// ████████████████████████████████ ENCODING UTILITY FUNCTIONS
	fn to_binary(data: u32, bits: u32) -> Vec<bool> {
		let mut bitstring = Vec::new();
		let mut rem = data;
		for i in (0..bits).rev() { // check if ith bit is 1, append true and set it to 0 if so
			if ((rem >> i) & 1) == 1 { bitstring.push(true); rem = (1 << i) ^ rem; }
			else { bitstring.push(false); }
		}
		bitstring
	}
	
	fn split_chunks(data: &str, size: usize) -> Vec<&str> {
		 data
		.as_bytes()
		.chunks(size)
		.map(str::from_utf8)
		.collect::<Result<Vec<&str>, _>>()
		.unwrap()
	}
	
	fn as_string(data: Vec<bool>) -> String {
		let bit_to_char = |c: &bool| if *c { '1' }else { '0' };
		data.iter().map(bit_to_char).fold(String::new(), |mut acc, chr| { acc.push(chr); acc })
	}
}

// ████████████████████████████████████████████████████████████████ CREATE QR CODE
struct QRCode {
	version: u8,
	size: u32,
	mask: u8,
	ec_level: QRECLevel,
	encoding: Encoding,
	modules: Vec<Vec<bool>>,
	reserved: Vec<Vec<bool>>,
}

impl QRCode {
	// ████████████████████████████████ STATIC LOOKUP TABLES
	const ECC_CODEWORDS_PER_BLOCK: [[u8; 41]; 4] = [
		[0,  7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28, 28, 28, 30, 30, 26, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
		[0, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
		[0, 13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30, 28, 30, 30, 30, 30, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
		[0, 17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28, 30, 24, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
	];

	const NUM_ERROR_CORRECTION_BLOCKS: [[u8; 41]; 4] = [
		[0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4,  4,  4,  4,  4,  6,  6,  6,  6,  7,  8,  8,  9,  9, 10, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 24, 25],
		[0, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5,  5,  8,  9,  9, 10, 10, 11, 13, 14, 16, 17, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31, 33, 35, 37, 38, 40, 43, 45, 47, 49],
		[0, 1, 1, 2, 2, 4, 4, 6, 6, 8, 8,  8, 10, 12, 16, 12, 17, 16, 18, 21, 20, 23, 23, 25, 27, 29, 34, 34, 35, 38, 40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68],
		[0, 1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25, 25, 34, 30, 32, 35, 37, 40, 42, 45, 48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81],
	];
	
	const FINDER_PATTERN: [[u8; 7]; 7] = [
		[0, 0, 0, 0, 0, 0, 0],
		[0, 1, 1, 1, 1, 1, 0],
		[0, 1, 0, 0, 0, 1, 0],
		[0, 1, 0, 0, 0, 1, 0],
		[0, 1, 0, 0, 0, 1, 0],
		[0, 1, 1, 1, 1, 1, 0],
		[0, 0, 0, 0, 0, 0, 0]
	];
	
	const ALIGNMENT_PATTERN: [[u8; 5]; 5] = [
		[0, 0, 0, 0, 0],
		[0, 1, 1, 1, 0],
		[0, 1, 0, 1, 0],
		[0, 1, 1, 1, 0],
		[0, 0, 0, 0, 0]
	];

	// ████████████████████████████████ CONSTRUCT A QR CODE WITH THE GIVEN PARAMETERS
	fn new(text: &str, ec_level: QRECLevel, encoding: Encoding) -> QRCode {
		let pos = match ec_level {
			QRECLevel::Low => 0,
			QRECLevel::Medium => 1,
			QRECLevel::Quartile => 2,
			QRECLevel::High => 3
		};
		let encoder = Encoder{ mode: encoding };
		let mut qrcode = QRCode { version: 0, size: 1, mask: 0, ec_level, encoding, modules: Vec::new(), reserved: Vec::new() };
		
		let mut data_bytes = None;
		while !data_bytes.is_some() {
			if qrcode.version > 40 { println!("data is too long to be stored in a QR code"); return qrcode; }
			qrcode.version += 1;
			let data_capacity = (qrcode.total_modules() / 8)
				- (QRCode::ECC_CODEWORDS_PER_BLOCK[pos][qrcode.version as usize] as u32 
				*QRCode::NUM_ERROR_CORRECTION_BLOCKS[pos][qrcode.version as usize] as u32);
			
			data_bytes = encoder.encode_data(&text, qrcode.version as u32, data_capacity*8);
		}
		
		println!("qrcode version: {}", qrcode.version);
		let usize_bytes: Vec<usize> = data_bytes.unwrap().into_iter().map(|x| x as usize).collect();

		let interleaved = qrcode.interleave_data(usize_bytes);
		
		qrcode.size = 4*qrcode.version as u32 + 17;
		for _ in 0..qrcode.size as usize {
			qrcode.modules.push(vec![true; qrcode.size as usize]);
			qrcode.reserved.push(vec![false; qrcode.size as usize]);
		}

		qrcode.draw_patterns();
		qrcode.draw_data(interleaved);
		qrcode.apply_mask();
		qrcode.draw_version_info();
		qrcode.draw_format_info();
		
		qrcode
	}
	
	// ████████████████████████████████ DISPLAY THE CODE ON THE TERMINAL
	fn draw(&self, qz_width: usize) {
		let width = self.size as usize;
		let sample_buffer = |x: usize, y: usize| {
			let range = qz_width..width+qz_width;
			if y >= width+2*qz_width { return false; }
			return if range.contains(&x) && range.contains(&y) { self.modules[y-qz_width][x-qz_width] }else { true };
		};
		for j in (0..(width+2*qz_width)).step_by(2) {
			for i in 0..width+2*qz_width {
				let character = match (sample_buffer(i, j), sample_buffer(i, j+1)) {
					(true, true) => '█', (true, false) => '▀', (false, true) => '▄', (false, false) => ' '
				};
				print!("{character}");
			}
			println!("");
		}
	}
	
	// ████████████████████████████████ FUNCTIONS FOR WRITING DATA TO THE MODULE BUFFER
	fn draw_data(&mut self, data: Vec<usize>) {
		let mut column = self.size as i32-1;
		let mut bitstring = data.iter().fold(Vec::new(), |mut acc, byte| { acc.extend(&Encoder::to_binary(*byte as u32, 8)); acc });
		bitstring.extend([true; 8]); // pad bits so the reader doesn't hit the end of the bitstring when self.total_modules() > bitstring.len()
		
		let mut bit_reader = bitstring.iter();
		while column > 0 {
			if column == 6 { column = 5; }
			for pos in 0..self.size as usize {
				let horizontal = column as usize;
				let vertical = if ((horizontal + 1) / 2)%2 == 0 { self.size as usize-pos-1 }else { pos };
				if !self.reserved[vertical][horizontal] { self.modules[vertical][horizontal] = !*bit_reader.next().unwrap(); }
				if !self.reserved[vertical][horizontal-1] { self.modules[vertical][horizontal-1] = !*bit_reader.next().unwrap(); }
			}
			column -= 2;
		}
	}
	
	fn apply_mask(&mut self) {
		let mask_function = match self.mask {
			0 => |w: usize, h: usize| (w+h) % 2 == 0,
			1 => |w: usize, h: usize| h % 2 == 0,
			2 => |w: usize, h: usize| w % 3 == 0,
			3 => |w: usize, h: usize| (w+h) % 3 == 0,
			4 => |w: usize, h: usize| (h/2 + w/3) % 2 == 0,
			5 => |w: usize, h: usize| ((w*h)%2 + (w*h)%3) == 0,
			6 => |w: usize, h: usize| ((w*h)%2 + (w*h)%3) % 2 == 0,
			7 => |w: usize, h: usize| ((w+h)%2 + (w*h)%3) % 2 == 0,
			_ => |w: usize, h: usize| false
		};
		for j in 0..self.size as usize {
			for i in 0..self.size as usize {
				if !self.reserved[j][i] && mask_function(i, j) { self.modules[j][i] = !self.modules[j][i]; }
		}}
	}
	
	fn draw_patterns(&mut self) {
		let size = self.size as usize;
		let as_bool = |x: u8| if x == 1 { true }else { false };
		// corner reserved areas
		for j in 0..9 {
			for i in 0..9 { self.reserved[j][i] = true; }
		}
		for j in 0..8 {
			for i in 0..9 { self.reserved[j+size-8][i] = true; }
		}
		for j in 0..9 {
			for i in 0..8 { self.reserved[j][i+size-8] = true; }
		}
		// dark module
		self.modules[size-8][8] = false;
		// finder_patterns
		for j in 0..7 {
			for i in 0..7 {
				self.modules[j][i] = as_bool(QRCode::FINDER_PATTERN[j][i]);
				self.modules[j+size-7][i] = as_bool(QRCode::FINDER_PATTERN[j][i]);
				self.modules[j][i+size-7] = as_bool(QRCode::FINDER_PATTERN[j][i]);
		}}
		// timing patterms
		for i in 0..size-14 {
			self.modules[6][i+7] = i%2 == 0;
			self.reserved[6][i+7] = true;
			self.modules[i+7][6] = i%2 == 0;
			self.reserved[i+8][6] = true;
		}
		
		// alignment patterns
		if self.version == 1 { return; }
		
		let mut pattern_coords = Vec::new();
		let intervals = (self.version as usize / 7) + 1;  // Number of gaps between alignment patterns
		let distance = size - 13;  // distance between first and last alignment pattern
		let mut step = distance / intervals;  // round spacing to nearest integer
		step += step%2;  // round step to next even number
		pattern_coords.push(4);
		
		for i in 1..intervals+1 {
		    pattern_coords.push(4 + distance - step * (intervals - i));  // start right/bottom and go left/up by step*k
		}
		for j in pattern_coords.iter() {
			for i in pattern_coords.iter() {
				// exclude alignment patterns intersecting with finder patterns
				if (*i == 4) && (*j == 4) { continue; }
				if (*i == 4) && (*j == size-9) { continue; }
				if (*i == size-9) && (*j == 4) { continue; }
				for h in 0..5 {
					for w in 0..5 {
						self.modules[h+j][w+i] = as_bool(QRCode::ALIGNMENT_PATTERN[h][w]);
						self.reserved[h+j][w+i] = true;
		}}}}
		
		// version information area
		if self.version < 7 { return; }
		for j in 0..6 {
			for i in 0..3 {
				self.reserved[j][i+size-11] = true;
				self.reserved[i+size-11][j] = true;
		}}
	}
	
	fn draw_format_info(&mut self) {
		let ec_bits = match self.ec_level {
			QRECLevel::Low => 1,
			QRECLevel::Medium => 0,
			QRECLevel::Quartile => 3,
			QRECLevel::High => 2
		};
		let format_codewords = {
			let data = ec_bits << 3 | self.mask as u32;
			let mut rem = data;
			for i in 0..10 {
				rem = (rem << 1) ^ ((rem >> 9) * 0x537);
			}
			(data << 10 | rem) ^ 0x5412
		};
		let format_bits = Encoder::to_binary(format_codewords, 15);
		println!("format bits: {}", Encoder::as_string(format_bits.clone()));
		for i in 0..7 {
			self.modules[self.size as usize-i-1][8] = !format_bits[i];
		}
		for i in 0..8 {
			self.modules[8][self.size as usize-8+i] = !format_bits[i+7];
		}
		for i in 0..6 {
			self.modules[8][i] = !format_bits[i];
			self.modules[5-i][8] = !format_bits[i+9];
		}
		self.modules[8][7] = !format_bits[6];
		self.modules[8][8] = !format_bits[7];
		self.modules[7][8] = !format_bits[8];
	}
	
	fn draw_version_info(&mut self) {
		if self.version < 7 { return; }
		
		let version_codewords = {
			let data = self.version as u32;
			let mut rem = data;
			for i in 0..12 {
				rem = (rem << 1) ^ ((rem >> 11) * 0x1F25);
			}
			data << 12 | rem
		};
		let mut version_bits = Encoder::to_binary(version_codewords, 18);
		version_bits.reverse();
		println!("version bits: {}", Encoder::as_string(version_bits.clone()));
		for i in 0..18 {
			let x = self.size as usize - 11 + i%3;
			let y = i / 3;
			self.modules[y][x] = !version_bits[i];
			self.modules[x][y] = !version_bits[i];
		}
	}
	
	// ████████████████████████████████ DATA ANALYSIS AND PROCESSING
	fn total_modules(&self) -> u32 {
		let version = self.version as u32;
		let mut capacity = (4*version + 17) * (4*version + 17);
		capacity -= 225; // (81 + 72 + 72) reserved regions in corners
		capacity -= 8*version; // timing patterns
		if version >= 2 {
			let num_align = version/7 + 2; // number of alignment patterns along each edge
			capacity -= (num_align*num_align - 3) * 25; // alignment patterns
			capacity += (num_align - 2) * 10; // avoid double counting timing and alignment pattern overlap
		}
		if version >= 7 { capacity -= 36; } // version info
		capacity
	}
	
	
	fn interleave_data(&self, data: Vec<usize>) -> Vec<usize> {
		let pos = match self.ec_level {
			QRECLevel::Low => 0,
			QRECLevel::Medium => 1,
			QRECLevel::Quartile => 2,
			QRECLevel::High => 3
		};
	
		let numblocks = QRCode::NUM_ERROR_CORRECTION_BLOCKS[pos][self.version as usize] as usize;
		let blockecclen = QRCode::ECC_CODEWORDS_PER_BLOCK[pos][self.version as usize] as usize;
		let rawcodewords = (self.total_modules() / 8) as usize;
		let numshortblocks = numblocks - rawcodewords % numblocks;
		let shortblocklen = rawcodewords / numblocks;
		
		let mut blocks = Vec::new();
		let total_short_len = (shortblocklen - blockecclen) * numshortblocks;

		let (short_group, long_group) = data.split_at(total_short_len);
		let short_blocks = short_group.chunks(shortblocklen - blockecclen);
		let long_blocks = long_group.chunks(shortblocklen - blockecclen + 1);
		
		for block in short_blocks {
			let mut dat = block.to_vec();
			dat.push(0);
			dat.extend(QRCode::ec_codewords(block.to_vec(), blockecclen).to_vec());
			blocks.push(dat);
		}
		for block in long_blocks {
			let mut dat = block.to_vec();
			dat.extend(QRCode::ec_codewords(block.to_vec(), blockecclen).to_vec());
			blocks.push(dat);
		}
		let mut interleaved = Vec::new();
		for i in 0..shortblocklen+1 {
			for j in 0..numblocks {
				if (j < numshortblocks) && (i == shortblocklen-blockecclen) { continue; }
				interleaved.push(blocks[j][i]);
		}}
		interleaved
	}
	
	// ████████████████████████████████ REED-SOLOMON ERROR CORRECTION
	fn ec_codewords(data: Vec<usize>, num_ec_codewords: usize) -> Vec<usize> {
		let mut ec_codewords = data.clone();
		let gen_poly = QRCode::gen_poly(num_ec_codewords);
		let (log, alog) = QRCode::log_alog_table();
		let pow = data.len() + num_ec_codewords - gen_poly.len() + 1;
		
		for i in 0..pow {
			if ec_codewords.len() <= num_ec_codewords { ec_codewords.push(0); }

			let mut g1: Vec<usize> = gen_poly.iter().map(|word| log[(word + alog[ec_codewords[0]]) % 255]).collect();
			g1.resize(ec_codewords.len(), 0);
			
			let mut xor_result: Vec<usize> = g1.iter().zip(ec_codewords.iter()).map(|(gp, ec)| gp ^ ec).collect();
			xor_result.remove(0);

			ec_codewords = xor_result;			
		}
		ec_codewords
	}
	
	fn gen_poly(degree: usize) -> Vec<usize> {
		let mut gen_poly = vec![0, 0];
		let (log, alog) = QRCode::log_alog_table();
		
		for i in 0..degree-1 {
			let fac_poly = vec![0, i+1];
			let mut like_terms = Vec::new();
			for i in 0..gen_poly.len()+1 { like_terms.push(Vec::new()); }
			
			for pow1 in 0..gen_poly.len() {
				for pow2 in 0..2 {
					let c = (gen_poly[pow1] + fac_poly[pow2]) % 255;
					like_terms[pow1+pow2].push(c);
			}}
			let mut product_poly = Vec::new();
			for term in like_terms.iter() {
				if term.len() == 1 { product_poly.push(term[0]); }
				else { product_poly.push(alog[log[term[0]] ^ log[term[1]]]); }	
			}
			gen_poly = product_poly;
		}
		gen_poly
	}
	
	fn log_alog_table() -> ([usize; 256], [usize; 256]) {
		let (mut log, mut alog) = ([0; 256], [0; 256]);
		let mut pow = 1;
		
		for i in 0..256 {
			log[i] = pow;
			alog[pow] = i;
			pow = (pow << 1) ^ ((pow >> 7) * 0x11D);
		}
		alog[1] = 0;
		(log, alog)
	}
}


// ████████████████████████████████████████████████████████████████ CREATE VCARD
struct VCard {
	display_string: String
}

impl VCard {
	fn new() -> VCard {
		VCard { display_string: String::new() }
	}

	fn display(&self) -> String {
		let mut vcard_string = "BEGIN:VCARD\nVERSION:4.0\n".to_string();
		vcard_string.push_str(&self.display_string);
		vcard_string.push_str("END:VCARD");
		vcard_string
	}
	
	fn add_property(&mut self, info: Vec<&str>) {
		let template = match info[0] {
			"NAME" => |info: Vec<&str>| format!("FN:{}\nN;CHARSET=utf-8:{}\n", info[1], info[1]),
			// WORK, CELL, HOME
			"PHONE" => |info: Vec<&str>| format!("TEL;type={}:{}\n", info[1], info[2]),
			// WORK, HOME
			"EMAIL" => |info: Vec<&str>| format!("EMAIL;type={}:{}\n", info[1], info[2]),
			// WORK, HOME, (street, city, state, country)
			"ADDRESS" => |info: Vec<&str>| format!("ADR;type={}:;;{};{};{};{};{}\n", info[1], info[2], info[3], info[4], info[5], info[6]),
			"NOTE" => |info: Vec<&str>| format!("NOTE:{}\n", info[1]),
			"TITLE" => |info: Vec<&str>| format!("TITLE:{}", info[1]),
			"ORG" => |info: Vec<&str>| format!("ORG:{}", info[1]),
			_ => |info: Vec<&str>| String::new()
		};
		self.display_string.push_str(&template(info));
	}
}


fn wifi_network(encryption_type: &str, SSID: &str, password: &str) -> String {
	format!("WIFI:T:{encryption_type};S:{SSID};P:{password};;")
}


fn main() {
	let mut vcard = VCard::new();
	vcard.add_property(vec!["NAME", "Jane Smith"]);
	vcard.add_property(vec!["PHONE", "CELL", "+18008135420"]);
	vcard.add_property(vec!["EMAIL", "WORK", "email@gmail.com"]);
	vcard.add_property(vec!["NOTE", "Hello World!"]);
	vcard.add_property(vec!["ADDRESS", "WORK", "1234 W East St", "Townsville", "MN", "29863", "Canada"]);
	
	//let text = wifi_network("WPA", "MyNetwork", "password123");
	let text = vcard.display();
	println!("{text}\n");
	let qrcode = QRCode::new(&text, QRECLevel::Low, Encoding::Bytes);
	qrcode.draw(3);
}

