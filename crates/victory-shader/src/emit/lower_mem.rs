//! Image and buffer instruction lowering helpers extracted from `lower.rs`.

use super::*;

impl<'a> Lower<'a> {
    pub(super) fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            6 if name == "image_load" || name == "image_get_resinfo" => (3, 2),
            7 if name == "image_load" || name == "image_get_resinfo" => (8, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // Full 32-bit address VGPR (offset, bias, z-compare are always 32-bit
        // even with `a16`).
        let addr_dword = |l: &mut Self, w: usize| -> Result<V, Why> {
            let r = *regs.get(w).ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            l.i32_of(v)
        };
        // Packed component `k` starting at VGPR `base` (with `a16`, two 16-bit
        // components per VGPR, low half first).
        let addr_packed = |l: &mut Self, base: usize, k: usize| -> Result<V, Why> {
            let w = base + if *a16 { k / 2 } else { k };
            let x = addr_dword(l, w)?;
            if !*a16 {
                return Ok(x);
            }
            let c = l.cur;
            let (sh, mask) = (l.s.m.i32(16 * (k as u32 % 2)), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        let addr_val = |l: &mut Self, k: usize| addr_packed(l, 0, k);
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // GLC returns the old value. DXIL implements subtraction by adding
            // the two's complement of the operand at the same width.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            if *d16 || *a16 {
                return Err("a 16-bit image atomic");
            }
            // Compare-exchange consumes a replacement followed by a comparand.
            let compare = code == u32::MAX;
            let wide = match (compare, *dmask) {
                (false, 1) | (true, 3) => false,
                (false, 3) | (true, 15) => true,
                _ => return Err("an image atomic data mask"),
            };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            let width = if wide { 2 } else { 1 };
            let vals = self.vgprs(inst, data, if compare { 2 * width } else { width })?;
            let (ty, component) = if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64;
                (self.s.m.int(64), 0x107) // one uint64_t in an R32G32_UINT view
            } else {
                (self.s.t.i32, 0x105)
            };
            let original = if wide {
                self.join64(vals[0], vals[1])
            } else {
                vals[0]
            };
            let value = if op == "sub" {
                let zero = self.s.m.int_const(ty, 0);
                self.s.m.bin(c, Bin::Sub, zero, original)
            } else {
                original
            };
            let h = self.handle(site, src, false, (kind | 0x1000, component))?;
            let mut args = vec![h];
            let (intrinsic, opcode) = if compare {
                let comparand = if wide {
                    self.join64(vals[2], vals[3])
                } else {
                    vals[1]
                };
                args.extend(cs);
                args.extend([comparand, value]);
                let intrinsic = if wide {
                    "atomicCompareExchange.i64"
                } else {
                    "atomicCompareExchange.i32"
                };
                (intrinsic, 79)
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                (
                    if wide {
                        "atomicBinOp.i64"
                    } else {
                        "atomicBinOp.i32"
                    },
                    78,
                )
            };
            let at = self.begin_guarded(guard);
            let r = self
                .s
                .op(self.cur, intrinsic, Attr::NoUnwind, ty, opcode, &args);
            if cache.glc {
                let old = self.end_guarded(at, &[(r, original)])[0];
                if wide {
                    let (lo, hi) = self.split64(old);
                    self.write(inst, Reg::V(reg), lo, None);
                    self.write(inst, Reg::V(reg + 1), hi, None);
                } else {
                    self.write(inst, Reg::V(reg), old, None);
                }
            } else {
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            // shortcut: stores are taken to write all four channels of a
            // float format (the console converts by the view's format). The
            // data VGPRs hold the dmask's channels in order; the others are
            // written as 0.
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let mut vals = Vec::new();
            for ch in 0..4u8 {
                if dmask >> ch & 1 == 0 {
                    vals.push(self.s.m.f32(0));
                    continue;
                }
                let k = (dmask & ((1 << ch) - 1)).count_ones() as u8;
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Samples and gathers: the DXIL operation follows from the form.
        let form = sample_form(name);
        let gather = form.is_some_and(|f| f.gather);
        let (op, sample) = match name {
            "image_get_lod" if self.s.pixel => (81, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ if form.is_some() => (0, true),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address: DXIL's dimensions; a channel DXIL leaves
            // out reads 1 (UAVs: one level).
            let h = self.handle(site, src, false, props)?;
            let mip = if uav || matches!(kind, 3 | 8) {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0)?
            };
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            let present = match kind {
                1 => [true, false, false, true],
                2 | 3 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav || !present[ch as usize] {
                    one
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        // The four texels of a gather made of samples (`GATHER_AT_LEVEL`).
        let mut gathered: Option<[V; 4]> = None;
        let r = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            let f = form.unwrap_or_default();
            if f.c && dmask.count_ones() != 1 {
                return Err("a comparison sample of several channels");
            }
            // Comparison: the sampler is a comparison sampler.
            let sh = self.handle(site, ss, true, (14 | if f.c { 0x8000 } else { 0 }, 0))?;
            // Coordinates the size divides (not the array slice).
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            // Address dwords in order (LLVM ImageDimIntrinsics.td, Mesa
            // ac_build_image_opcode): the texel offset, bias and comparison
            // reference, a dword each (an a16 bias is an f16 in the low
            // half, LLVM SIISelLowering::lowerImage); the derivatives, x
            // then y; the coordinates, then the level or level clamp.
            // Derivatives are 16-bit in the g16 opcodes, coordinates and
            // levels with a16 (GFX10.3 keeps the two apart, as LLVM's
            // lowerImage does where hasG16): two per dword, low half first,
            // each direction starting a dword.
            let whole = usize::from(f.o) + usize::from(f.b) + usize::from(f.c);
            let bias_at = (usize::from(f.o), a16.then_some(0));
            // A cube's derivatives are those of its two face coordinates
            // (the third coordinate is the face).
            let grads = match (f.d, kind) {
                (false, _) => 0,
                (true, 5) => 2,
                (true, _) => spatial,
            };
            let g16 = f.g16;
            let per_direction = if g16 { grads.div_ceil(2) } else { grads };
            let grad_at = |dir: usize, k: usize| {
                if g16 {
                    (whole + dir * per_direction + k / 2, Some(k as u32 % 2))
                } else {
                    (whole + dir * grads + k, None)
                }
            };
            let first = whole + 2 * per_direction;
            // Coordinate k; k = coords is the level or the level clamp.
            let coord_at = |k: usize| {
                if *a16 {
                    (first + k / 2, Some(k as u32 % 2))
                } else {
                    (first + k, None)
                }
            };
            let raw = |l: &mut Self, at: usize| -> Result<V, Why> {
                let r = *regs.get(at).ok_or("an image address")?;
                let v = inst.read(Reg::V(r)).ok_or("an image address")?;
                l.i32_of(v)
            };
            // A float address: a dword, or an f16 half of one.
            let float = |l: &mut Self, (at, half): (usize, Option<u32>)| -> Result<V, Why> {
                let x = raw(l, at)?;
                let Some(h) = half else {
                    return Ok(l.as_float(x));
                };
                let c = l.cur;
                let (sh, mask) = (l.s.m.i32(16 * h), l.s.m.i32(0xFFFF));
                let x = l.s.m.bin(c, Bin::LShr, x, sh);
                let x = l.s.m.bin(c, Bin::And, x, mask);
                let f32t = l.s.t.f32;
                Ok(l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]))
            };
            // A float address known at translation (an f16 only when 0).
            let known =
                |l: &mut Self, (at, half): (usize, Option<u32>)| -> Result<Option<f32>, Why> {
                    let x = raw(l, at)?;
                    Ok(l.s.m.known(x).and_then(|bits| match half {
                        None => Some(f32::from_bits(bits as u32)),
                        Some(h) => (bits >> (16 * h) & 0x7FFF == 0).then_some(0.0),
                    }))
                };
            let level = if f.l {
                known(self, coord_at(coords))?
            } else {
                f.lz.then_some(0.0)
            };
            let level_zero = level == Some(0.0);
            // The DXIL operation. SM 6.6 has no comparison at a level, with
            // a bias or with gradients (SampleCmpLevel is SM 6.7,
            // SampleCmpBias and SampleCmpGrad 6.8) and gathers level 0 only;
            // those forms are taken where a level or bias known to be 0 makes
            // them one SM 6.6 has.
            let op = if op == 81 {
                81
            } else if f.gather {
                if f.b {
                    return Err("a gather with a level bias");
                }
                if !f.l && !f.lz {
                    return Err("a gather with implicit level");
                }
                if dmask.count_ones() != 1 {
                    return Err("a gather of several channels");
                }
                if !matches!(kind, 2 | 5 | 7) {
                    return Err("a gather from a 1D or 3D image");
                }
                if level_zero {
                    if f.c { 74 } else { 73 }
                } else if f.c || kind == 5 {
                    // A comparison's function is in the sampler, and a cube's
                    // texel grid is per face.
                    return Err("a gather at a level not known to be 0");
                } else {
                    GATHER_AT_LEVEL
                }
            } else if f.c {
                if f.d {
                    return Err("a comparison sample with explicit derivatives");
                }
                if f.l && !level_zero {
                    return Err("a comparison sample at a level not known to be 0");
                }
                if f.b && known(self, bias_at)?.is_none_or(|b| b != 0.0) {
                    return Err("a comparison sample with a level bias");
                }
                if f.l || f.lz {
                    65
                } else if self.s.pixel || self.s.threads.is_some() && self.s.mesh.is_none() {
                    // Implicit derivatives: pixel and compute shaders.
                    64
                } else {
                    return Err(
                        "a comparison sample with implicit derivatives outside pixel and compute shaders",
                    );
                }
            } else if f.d {
                63
            } else if f.l || f.lz {
                62
            } else if f.b {
                61
            } else {
                60
            };
            let mut cs = vec![fu; 4];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = float(self, coord_at(k))?;
            }
            let mut cube_face = None;
            if kind == 5 {
                if *unorm || op == 81 {
                    return Err("a cube sample with texel coordinates or a level query");
                }
                if f.o {
                    return Err("a texel offset on a cube image");
                }
                cube_face = Some(cs[2]);
                let dir = self.cube_direction(cs[0], cs[1], cs[2]);
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let lod = self.s.op(
                        c,
                        "calculateLOD.f32",
                        Attr::ReadOnly,
                        f32t,
                        81,
                        &[
                            h,
                            sh,
                            cs[0],
                            if spatial > 1 { cs[1] } else { fu },
                            if spatial > 2 { cs[2] } else { fu },
                            clamped,
                        ],
                    );
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if kind == 5 { 0 } else { spatial.min(3) };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            if f.o {
                // Texel offsets: x in bits 0-5, y in 8-13, z in 16-21, signed.
                let w = raw(self, 0)?;
                let field = |w: i64, k: usize| (w as i32) << (26 - 8 * k) >> 26;
                let immediate = self
                    .s
                    .m
                    .known(w)
                    .filter(|&w| (0..offsets).all(|k| (-8..=7).contains(&field(w, k))));
                if let (Some(w), false) = (immediate, f.gather) {
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        *o = self.s.m.i32(field(w, k) as u32);
                    }
                } else {
                    // Bits at..at+5 sign-extended: shift up, then down.
                    let k26 = self.s.m.i32(26);
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        let left = self.s.m.i32(26 - 8 * k as u32);
                        let t = self.s.m.bin(c, Bin::Shl, w, left);
                        *o = self.s.m.bin(c, Bin::AShr, t, k26);
                    }
                }
                if !f.gather && immediate.is_none() {
                    // A sample's offsets are immediates in -8..7 (gathers
                    // take any, per lane). At a level known here, the
                    // coordinates move by offset / that level's size instead:
                    // the same texels, the coordinate rounded once more (the
                    // console adds the offset to the texel coordinate). A
                    // sampler whose LOD clamp or bias moves the sampled
                    // level away from the one given, or a level past the
                    // view's last, differs.
                    let Some(n) = level.filter(|n| *n >= 0.0 && n.fract() == 0.0) else {
                        return Err(
                            "a texel offset past -8..7 or per lane at a level not known here",
                        );
                    };
                    let i32t = self.s.t.i32;
                    let f32t = self.s.t.f32;
                    let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                    let n = self.s.m.i32(n as u32);
                    let d = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, n]);
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        let size = self.s.m.extract(c, d, k as u32);
                        let size = self.s.m.cast(c, Cast::UiToFp, size, f32t);
                        let texels = self.s.m.cast(c, Cast::SiToFp, *o, f32t);
                        let shift = self.s.m.bin(c, Bin::SDiv, texels, size);
                        cs[k] = self.s.m.bin(c, Bin::Add, cs[k], shift);
                        *o = zero;
                    }
                }
            }
            // The level clamp, after the coordinates.
            let clamp = if f.cl {
                float(self, coord_at(coords))?
            } else {
                fu
            };
            let mut args = vec![h, sh];
            args.extend(&cs);
            match op {
                60 | 61 if !self.s.pixel => {
                    // Outside pixel shaders the console still takes the level
                    // from each quad of lanes (lane 1 minus lane 0 across,
                    // lane 2 minus lane 0 down); DXIL before validator 1.8
                    // has no implicit derivatives there, so the quad's
                    // differences become explicit gradients. A bias scales
                    // them by 2^bias, the same level for isotropic filtering.
                    self.cross_lane(true)?;
                    let scale = if op == 61 {
                        let bias = float(self, bias_at)?;
                        Some(self.unary(21, bias))
                    } else {
                        None
                    };
                    let g = self.quad_gradients(&cs[..spatial], scale);
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                60 => {
                    args.extend(&os);
                    args.push(clamp);
                    self.s
                        .op(c, "sample.f32", Attr::ReadOnly, self.s.t.res_ret, 60, &args)
                }
                61 => {
                    let bias = float(self, bias_at)?;
                    args.extend(&os);
                    args.extend([bias, clamp]);
                    self.s.op(
                        c,
                        "sampleBias.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        61,
                        &args,
                    )
                }
                63 => {
                    // shortcut: the coarse forms (`_cd`) compute one level
                    // per quad; the host's per pixel is the same when the
                    // gradients agree across the quad.
                    let mut g = vec![fu; 6];
                    if let Some(face) = cube_face {
                        // The face coordinates' derivatives, as the
                        // direction's (`cube_gradient`).
                        for dir in 0..2 {
                            let ds = float(self, grad_at(dir, 0))?;
                            let dt = float(self, grad_at(dir, 1))?;
                            let d = self.cube_gradient(ds, dt, face);
                            g[3 * dir..3 * dir + 3].copy_from_slice(&d);
                        }
                    } else {
                        for k in 0..spatial {
                            g[k] = float(self, grad_at(0, k))?;
                            g[3 + k] = float(self, grad_at(1, k))?;
                        }
                    }
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                64 | 65 => {
                    let at = usize::from(f.o) + usize::from(f.b);
                    let reference = float(self, (at, None))?;
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(clamp);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
                GATHER_AT_LEVEL => {
                    // A gather at a level the code computes. DXIL's gathers
                    // read level 0 only, so the four texels of the footprint
                    // are sampled one by one at their centres at that level:
                    // at a texel's centre every filter returns that texel,
                    // and the sampler applies its own address mode and level
                    // range, as it does for the console's gather.
                    // shortcut: the footprint is that of the level rounded to
                    // the nearest whole one and clamped to the view's; a
                    // sampler whose LOD bias or range moves the sampled level
                    // off it reads another grid.
                    let lod = float(self, coord_at(coords))?;
                    let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
                    let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                    let d0 = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, zero]);
                    let levels = self.s.m.extract(c, d0, 3);
                    let one = self.s.m.i32(1);
                    let last = self.s.m.bin(c, Bin::Sub, levels, one);
                    let li = self.unary(26, lod);
                    let li = self.s.m.cast(c, Cast::FpToSi, li, i32t);
                    let over = self.s.m.cmp(c, pred::SGT, li, last);
                    let li = self.s.m.select(c, over, last, li);
                    let under = self.s.m.cmp(c, pred::SLT, li, zero);
                    let li = self.s.m.select(c, under, zero, li);
                    let d = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, li]);
                    let level = self.s.m.cast(c, Cast::SiToFp, li, f32t);
                    let half = self.s.m.f32(0.5f32.to_bits());
                    // The footprint's lower texel, floor(u * size - 0.5),
                    // moved by the texel offsets.
                    let mut size = [fu; 2];
                    let mut low = [fu; 2];
                    for k in 0..2 {
                        let n = self.s.m.extract(c, d, k as u32);
                        let n = self.s.m.cast(c, Cast::UiToFp, n, f32t);
                        let t = self.s.m.bin(c, Bin::Mul, cs[k], n);
                        let t = self.s.m.bin(c, Bin::Sub, t, half);
                        let t = self.unary(27, t);
                        let o = self.s.m.cast(c, Cast::SiToFp, os[k], f32t);
                        low[k] = self.s.m.bin(c, Bin::Add, t, o);
                        size[k] = n;
                    }
                    let mut none = vec![iu; 3];
                    for o in none.iter_mut().take(offsets) {
                        *o = zero;
                    }
                    let channel = dmask.trailing_zeros();
                    let mut last_r = None;
                    let mut texels = [fu; 4];
                    // The console's (and DXIL's) gather order.
                    for (n, by) in [(0.0f32, 1.0f32), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
                        .into_iter()
                        .enumerate()
                    {
                        let mut at = cs.clone();
                        for (k, step) in [by.0, by.1].into_iter().enumerate() {
                            let centre = self.s.m.f32((step + 0.5).to_bits());
                            let x = self.s.m.bin(c, Bin::Add, low[k], centre);
                            at[k] = self.s.m.bin(c, Bin::SDiv, x, size[k]);
                        }
                        let mut a = vec![h, sh];
                        a.extend(&at);
                        a.extend(&none);
                        a.push(level);
                        let r = self.s.op(
                            c,
                            "sampleLevel.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            62,
                            &a,
                        );
                        texels[n] = self.s.m.extract(c, r, channel);
                        last_r = Some(r);
                    }
                    gathered = Some(texels);
                    last_r.expect("four samples")
                }
                62 => {
                    let lod = if f.l {
                        float(self, coord_at(coords))?
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                _ => {
                    args.extend(&os[..2]);
                    args.push(self.s.m.i32(dmask.trailing_zeros()));
                    let name = if op == 74 {
                        let reference = float(self, (usize::from(f.o), None))?;
                        args.push(reference);
                        "textureGatherCmp.f32"
                    } else {
                        "textureGather.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" || matches!(kind, 3 | 8) {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            )
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if gather {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = match gathered {
                Some(texels) => texels[ch as usize],
                None => self.s.m.extract(c, r, ch),
            };
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                self.f16_nearest(x)
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel.
    // shortcut: the view is taken as a cube, not a cube array (whose face
    // operand adds 8 × the slice).
    fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        // Per face +x, -x, +y, -y, +z, -z.
        self.by_face(
            face,
            [
                [one, mone, u, u, u, nu],
                [nv, nv, one, mone, nv, nv],
                [nu, u, v, nv, one, mone],
            ],
        )
    }

    /// The derivative of a cube sample's direction for derivatives `ds` and
    /// `dt` of its face coordinates: [`Self::cube_direction`] is linear in s
    /// and t on a face (u = 2s - 3, v = 2t - 3), so its derivative is the
    /// same table with 2 ds for u, 2 dt for v and 0 for the constant axis.
    fn cube_gradient(&mut self, ds: V, dt: V, face: V) -> [V; 3] {
        let c = self.cur;
        let two = self.s.m.f32(2.0f32.to_bits());
        let u = self.s.m.bin(c, Bin::Mul, ds, two);
        let v = self.s.m.bin(c, Bin::Mul, dt, two);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let o = self.s.m.f32(0);
        self.by_face(
            face,
            [
                [o, o, u, u, u, nu],
                [nv, nv, o, o, nv, nv],
                [nu, u, v, nv, o, o],
            ],
        )
    }

    /// Each row's value for cube face `face` (a float 0-5).
    fn by_face(&mut self, face: V, table: [[V; 6]; 3]) -> [V; 3] {
        let c = self.cur;
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, f, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }

    pub(super) fn raw_dword(&mut self, h: V, coords: [V; 2]) -> V {
        let c = self.cur;
        let (mask, align) = (self.s.m.i8(1), self.s.m.i32(4));
        let r = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139,
            &[h, coords[0], coords[1], mask, align]);
        let x = self.s.m.extract(c, r, 0);
        self.as_int(x)
    }

    fn buffer_compare_exchange(&mut self, h: V, coords: [V; 3], old: V, new: V, wide: bool) -> V {
        let name = if wide { "atomicCompareExchange.i64" } else { "atomicCompareExchange.i32" };
        let ty = self.s.m.type_of(old);
        self.s.op(self.cur, name, Attr::NoUnwind, ty, 79,
            &[h, coords[0], coords[1], coords[2], old, new])
    }

    /// A CAS must run on a real lane and at an in-range address: outside
    /// those cases DXIL can return an unspecified old value.
    fn buffer_bounds(&mut self, h: V, index: Option<V>, off: V, stride: u32, size: u32) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let unused = self.s.m.undef(i32t);
        let dimensions = self.s.op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, unused]);
        let length = self.s.m.extract(c, dimensions, 0);
        if let Some(index) = index {
            let record = self.s.m.cmp(c, pred::IULT, index, length);
            let limit = self.s.m.i32(stride - (size - 1));
            let member = self.s.m.cmp(c, pred::IULT, off, limit);
            return self.s.m.bin(c, Bin::And, record, member);
        }
        let tail = self.s.m.i32(size - 1);
        let last = self.s.m.bin(c, Bin::Add, off, tail);
        let fits = self.s.m.cmp(c, pred::IULT, last, length);
        let no_wrap = self.s.m.cmp(c, pred::IUGE, last, off);
        self.s.m.bin(c, Bin::And, fits, no_wrap)
    }

    fn buffer_store_guard(&mut self, bounds: V, guard: Option<V>) -> V {
        let c = self.cur;
        let g = guard.map_or(bounds, |g| self.s.m.bin(c, Bin::And, g, bounds));
        if !self.s.pixel {
            return g;
        }
        let helper = self.s.op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        let no = self.s.m.bool(false);
        self.s.m.select(c, helper, no, g)
    }

    pub(super) fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name == "buffer_gl0_inv" || name == "buffer_gl1_inv" {
            // Console cache coherence: gl0 writes back and invalidates the
            // per-CU shader L0, gl1 invalidates GL1, so one wavefront's
            // writes become visible to others. D3D12 already guarantees UAV
            // coherence across wavefronts and dispatches, DXIL has no
            // equivalent instruction, and this project's host-visible
            // writeback comes from command-stream GpuWrite ranges rather
            // than shader cache hints. Ordering is kept by the surrounding
            // s_waitcnt/s_barrier, which the translator does lower.
            return Ok(());
        }
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err("an MTBUF store requiring a packed-format write");
        }
        if name.contains("_format_d16_hi_") {
            return Err("a typed buffer access to the high half with an untracked numeric format");
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        let atomic = name.strip_prefix("buffer_atomic_");
        let wide = atomic.is_some_and(|n| n.ends_with("_x2"));
        let atomic = atomic.map(|n| n.strip_suffix("_x2").unwrap_or(n));
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let native_atomic = match atomic {
            None => None,
            Some("add" | "sub") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap" | "inc" | "dec" | "csub") => None,
            Some("fcmpswap" | "fmin" | "fmax") => {
                // Mesa aco_insert_fp_mode.cpp, vmem_default_needs.
                return Err("a buffer floating-point atomic requiring the console's denormal mode");
            }
            Some(_) => return Err(name),
        };
        let sub_load = name.strip_prefix("buffer_load_").and_then(subword_load);
        let sub_store = match name {
            "buffer_store_byte" | "buffer_store_byte_d16_hi" => Some(8),
            "buffer_store_short" | "buffer_store_short_d16_hi" => Some(16),
            _ => None,
        };
        // d16: two 16-bit channels per VGPR, low half first (buffer and
        // tbuffer formats alike).
        let d16 = name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() || sub_load.is_some() || sub_store.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let stride = used.map_or(0, |u| u.stride);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            if store || atomic.is_some() {
                return Err("a write or atomic operation on a constant buffer");
            }
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let size = self.root_view_size(src);
            if let Some((width, signed, half)) = sub_load {
                let (byte, ok) = match size {
                    Some(size) => {
                        let (ok, a) = self.in_root_view(size, byte, width / 8);
                        (a, Some(ok))
                    }
                    None => (byte, None),
                };
                let x =
                    self.read_subword(byte, width, signed, |l, at| Ok(l.cbuffer_dword(h, at)))?;
                let x = self.zero_unless(ok, x);
                self.write_subword_result(inst, reg, x, half, guard)?;
                return Ok(());
            }
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let (a, ok) = match size {
                    Some(size) => {
                        let (ok, a) = self.in_root_view(size, a, 4);
                        (a, Some(ok))
                    }
                    None => (a, None),
                };
                let x = self.cbuffer_dword(h, a);
                let x = self.zero_unless(ok, x);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if sub_load.is_some() || sub_store.is_some() {
            let data = data.as_ref().ok_or("buffer sub-dword data")?;
            let Operand::Vector { reg, .. } = *data else {
                return Err("buffer sub-dword data");
            };
            let h = if index.is_some() {
                if stride < 4 || !stride.is_multiple_of(4) {
                    return Err("a structured sub-dword buffer access without a dword-aligned stride");
                }
                self.handle(site, src, false, (12 | u, stride))?
            } else {
                self.handle(site, src, false, (11 | u, 0))?
            };
            let root_size = self.root_view_size(src);
            let coords = |byte| index.map_or([byte, iu], |i| [i, byte]);
            if let Some((width, signed, half)) = sub_load {
                let bounds = if let Some(size) = root_size {
                    if index.is_none() {
                        Some(self.in_root_view(size, off, width / 8).0)
                    } else {
                        None
                    }
                } else if !matches!(src, Source::Built { .. }) {
                    Some(self.buffer_bounds(h, index, off, stride, width / 8))
                } else {
                    None
                };
                let at = self.begin_guarded(bounds);
                let x = self.read_subword(off, width, signed, |l, at| Ok(l.raw_dword(h, coords(at))))?;
                let zero = self.s.m.i32(0);
                let x = self.end_guarded(at, &[(x, zero)])[0];
                self.write_subword_result(inst, reg, x, half, guard)?;
            } else if let Some(width) = sub_store {
                let mut x = self.vgprs(inst, data, 1)?[0];
                if name.ends_with("_hi") {
                    let sixteen = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, sixteen);
                }
                let bounds = if let Some(size) = root_size {
                    if index.is_none() {
                        self.in_root_view(size, off, width / 8).0
                    } else {
                        self.s.m.bool(true)
                    }
                } else {
                    self.buffer_bounds(h, index, off, stride, width / 8)
                };
                let active = self.buffer_store_guard(bounds, guard);
                let at = self.begin_guarded(Some(active));
                self.write_subword(off, width, x, |l, byte, mask, bits| {
                    let [a, b] = coords(byte);
                    let zero = l.s.m.i32(0);
                    l.atomic_update(zero, |l, old| {
                        let x = l.atomic_new_value(AtomicUpdate::MaskOr, old, mask, bits);
                        l.buffer_compare_exchange(h, [a, b, iu], old, x, false)
                    });
                });
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        // Root views: the bytes accessed checked against the view's size
        // (raw accesses, and the raw loads formatted ones make).
        let (off, root_ok) = match self.root_view_size(src) {
            Some(size) if (!typed || format.is_some()) && index.is_none() => {
                let bytes = if format.is_some() {
                    16
                } else if atomic.is_some() {
                    if wide { 8 } else { 4 }
                } else {
                    4 * count
                };
                let (ok, at) = self.in_root_view(size, off, bytes);
                (at, Some(ok))
            }
            _ => (off, None),
        };
        if let Some(op) = atomic {
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            if op == "csub" && !returns {
                // LLVM BUFInstructions.td only selects the returning form
                // on GFX10.3.
                return Err("buffer_atomic_csub without the required GLC bit");
            }
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let words = if wide { 2 } else { 1 };
            let vals = self.vgprs(inst, d, if op == "cmpswap" { 2 * words } else { words })?;
            let val = if wide { self.join64(vals[0], vals[1]) } else { vals[0] };
            let ty = self.s.m.type_of(val);
            let zero = self.s.m.int_const(ty, 0);
            let (h, in_bounds, coords) = match index {
                Some(index) => {
                    if wide && !stride.is_multiple_of(8) {
                        return Err("a 64-bit buffer atomic without an eight-byte-aligned stride");
                    }
                    let (h, ok, a, b) = self.structured(
                        site,
                        src,
                        u,
                        stride,
                        4 * u32::from(words),
                        (index, off),
                    )?;
                    (h, ok, [a, b, iu])
                }
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    let (ok, safe_off) = self.bound_raw_offset(off);
                    (h, ok, [safe_off, iu, iu])
                }
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_HEAP;
            }
            let g = self.and_guard(guard, in_bounds);
            let bounded_guard = self.narrow(g, root_ok);
            let active = if (wide || matches!(op, "sub" | "inc" | "dec" | "csub"))
                && !matches!(src, Source::Built { .. })
            {
                let bounds = self.buffer_bounds(h, index, off, stride, 4 * u32::from(words));
                Some(self.buffer_store_guard(bounds, bounded_guard))
            } else {
                bounded_guard
            };
            let at = self.begin_guarded(active);
            let r = if op == "cmpswap" {
                // cmpswap: data is (new, compare).
                let compare = if wide { self.join64(vals[2], vals[3]) } else { vals[1] };
                self.buffer_compare_exchange(h, coords, compare, val, wide)
            } else if let Some(code) = native_atomic {
                let name = if wide { "atomicBinOp.i64" } else { "atomicBinOp.i32" };
                let code = self.s.m.i32(code);
                let data = if op == "sub" { self.s.m.bin(self.cur, Bin::Sub, zero, val) } else { val };
                self.s.op(self.cur, name, Attr::NoUnwind, ty, 78,
                    &[h, code, coords[0], coords[1], coords[2], data])
            } else {
                let update = match op {
                    "inc" => AtomicUpdate::Inc,
                    "dec" => AtomicUpdate::Dec,
                    "csub" => AtomicUpdate::ClampSub,
                    _ => unreachable!(),
                };
                self.atomic_update(zero, |l, old| {
                    let x = l.atomic_new_value(update, old, val, zero);
                    l.buffer_compare_exchange(h, coords, old, x, wide)
                })
            };
            if returns {
                let r = self.end_guarded(at, &[(r, zero)])[0];
                self.write_atomic_result(inst, reg, r, wide, guard);
            } else {
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            let mut args = Vec::new();
            let (op, name, in_bounds) = if typed {
                // shortcut: typed stores are taken to write channels of a float
                // format.
                if *offen || *offset != 0 || matches!(off, V::Inst(_)) {
                    return Err("a typed buffer store at an offset");
                }
                args.push(self.handle(site, src, false, (10 | u, 0x409))?);
                let idx = index.unwrap_or(self.s.m.i32(0));
                let (ok, safe_idx) = self.bound_typed_index(idx);
                args.extend([safe_idx, iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                args.resize(3 + 4, self.s.m.undef(self.s.t.f32));
                (69, "bufferStore.f32", ok)
            } else {
                let ok = match index {
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        let (h, ok, a, b) =
                            self.structured(site, src, u, stride, 4 * count, (index, off))?;
                        args.extend([h, a, b]);
                        ok
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        let (ok, safe_off) = self.bound_raw_offset(off);
                        args.extend([safe_off, iu]);
                        ok
                    }
                };
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32", ok)
            };
            args.push(self.s.m.i8((1u8 << count) - 1));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let g = self.and_guard(guard, in_bounds);
            let store_guard = self.narrow(g, root_ok);
            let at = self.begin_guarded(store_guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("an indexed MTBUF load requiring a raw view of a structured descriptor");
            }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let mut xs = self.format_load(h, off, f, count)?;
            if d16 {
                xs = self.d16_channels(f, xs)?;
            }
            for (k, x) in xs.into_iter().enumerate() {
                let x = self.zero_unless(root_ok, x);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let (r, in_bounds) = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, 0x409))?;
            let index = index.unwrap_or(self.s.m.i32(0));
            let (ok, safe_idx) = self.bound_typed_index(index);
            let r = self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, safe_idx, iu],
            );
            (r, ok)
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    let (h, ok, a, b) =
                        self.structured(site, src, u, stride, 4 * count, (index, off))?;
                    let r = self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, a, b, mask, align],
                    );
                    (r, ok)
                }
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    let (ok, safe_off) = self.bound_raw_offset(off);
                    let r = self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, safe_off, iu, mask, align],
                    );
                    (r, ok)
                }
            }
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        let zero = self.s.m.i32(0);
        if d16 {
            // Two f16 channels per VGPR (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let h = self.f16_nearest(f);
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                let x = self.s.m.select(c, in_bounds, x.expect("a channel"), zero);
                let x = self.zero_unless(root_ok, x);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            let x = self.s.m.select(c, in_bounds, x, zero);
            let x = self.zero_unless(root_ok, x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

}
