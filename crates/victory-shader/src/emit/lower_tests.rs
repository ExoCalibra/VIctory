    use super::super::Target;
    use super::{Lowered, Numeric, Parts, SubDword, buffer_format, half_shift, inline_f16, lower};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// Lowers a compute program of constants: whether it translates.
    fn lowers(words: &[u32]) -> Result<(), crate::ShaderError> {
        lowers_cs(words, false, 0).map(|_| ())
    }

    /// The DXIL of a compute program of constants.
    fn dxil(words: &[u32], wave64: bool) -> Result<Vec<u8>, crate::ShaderError> {
        lowers_cs(words, wave64, 0).map(|l| l.dxil)
    }

    fn lowers_cs(
        words: &[u32],
        wave64: bool,
        const_sgprs: u8,
    ) -> Result<super::Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let layout = if const_sgprs > 0 {
            vec![RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: u32::from(const_sgprs),
                base_register: 0,
                user_data: 0,
                slot: 0,
            }]
        } else {
            vec![]
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: u32::from(const_sgprs) << 1,
                threads: [if wave64 { 64 } else { 32 }, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        lower(&Parts {
            psv: None,
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: Target::ANY,
            console_cf: false,
            per_lane_only: false,
        })
    }

    /// Lowers `words` as a `stage` program (`wave64`: the console's wave
    /// size) with no resources or inputs.
    fn lower_program(
        words: &[u32],
        stage: HwStage,
        wave64: bool,
        io: StageIo,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage,
            wave64,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io,
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        lower(&Parts {
            psv: None,
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: Target::ANY,
            console_cf: false,
            per_lane_only: false,
        })
    }

    #[test]
    fn med3_f32_emits_dxil_isnan_instead_of_fcmp_uno() {
        // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
        let src0 = 256 + 1u32;
        let src1 = 256 + 2u32;
        let src2 = 256 + 3u32;
        let words = [
            0x35 << 26 | (0x157u32 << 16),
            src0 | (src1 << 9) | (src2 << 18),
            S_ENDPGM,
        ];
        let lowered = lowers_cs(&words, false, 0).unwrap();
        assert!(
            lowered.has_isnan_op,
            "v_med3_f32 must emit dx.op.isSpecialFloat.f32 (IsNaN)"
        );
        assert!(
            !lowered.has_fcmp_uno,
            "v_med3_f32 must not emit fcmp uno (folded away by some drivers)"
        );
    }

    #[test]
    fn lane_mask_phi_joining_zero_and_compare_lowers() {
        let cnd = 0x81 | (0x82 << 9) | (2 << 18); // v_cndmask_b32 v1, 1, 2, s2
        let words = [
            0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
            0xBF85_0002, // 4: s_cbranch_scc1 -> 16
            0xBE82_0380, // 8: s_mov_b32 s2, 0
            0xBF82_0002, // 12: s_branch -> 24
            0xD4C2_0002,
            0x0002_0080, // 16: v_cmp_eq_u32 s2, 0, v0
            0xD501_0001,
            cnd, // 24: v_cndmask_b32 v1, 1, 2, s2
            S_ENDPGM,
        ];
        lowers_cs(&words, false, 1).unwrap();
    }

    #[test]
    fn wave64_loop_mask_accumulator_from_zero_lowers() {
        let cnd = 0x81 | (0x82 << 9) | (2 << 18); // v_cndmask_b32 v1, 1, 2, s[2:3]
        let words = [
            0xBE82_0480, // 0: s_mov_b64 s[2:3], 0
            0xBE84_0380, // 4: s_mov_b32 s4, 0
            0xD501_0001,
            cnd,         // 8: v_cndmask_b32 v1, 1, 2, s[2:3]
            0x7D84_0080, // 16: v_cmp_eq_u32 vcc, 0, v0
            0x8882_026A, // 20: s_or_b64 s[2:3], vcc, s[2:3]
            0x8004_8104, // 24: s_add_u32 s4, s4, 1
            0xBF0A_8404, // 28: s_cmp_lt_u32 s4, 4
            0xBF85_FFF9, // 32: s_cbranch_scc1 -> 8
            0xD501_0004,
            cnd, // 36: v_cndmask_b32 v4, 1, 2, s[2:3]
            S_ENDPGM,
        ];
        lowers_cs(&words, true, 0).unwrap();
    }

    #[test]
    fn lane_mask_phi_with_partial_constant_enables_console_lanes() {
        let cnd = 0x81 | (0x82 << 9) | (2 << 18); // v_cndmask_b32 v1, 1, 2, s2
        let words = [
            0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
            0xBF85_0002, // 4: s_cbranch_scc1 -> 16
            0xBE82_038F, // 8: s_mov_b32 s2, 15 (partial constant mask 0xF)
            0xBF82_0002, // 12: s_branch -> 24
            0xD4C2_0002,
            0x0002_0080, // 16: v_cmp_eq_u32 s2, 0, v0
            0xD501_0001,
            cnd, // 24: v_cndmask_b32 v1, 1, 2, s2
            S_ENDPGM,
        ];
        let lowered = lowers_cs(&words, false, 1).unwrap();
        assert!(
            !lowered.dxil.is_empty(),
            "expected DXIL with WaveSize(32) and WAVE_OPS"
        );
    }

    /// Lowers a pixel program with one target: whether it translates.
    fn lowers_ps(words: &[u32]) -> Result<(), crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code,
        };
        let p = ir::build(&native.code, false, 4).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let target = crate::emit::container::SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        };
        lower(&Parts {
            psv: None,
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![target],
            prims: vec![],
            mesh_stride: 0,
            target: Target::ANY,
            console_cf: false,
            per_lane_only: false,
        })
        .map(|_| ())
    }

    /// A lane select keeps only the bits the wave has lanes for, as the
    /// hardware does: lane 35 of a wave32 is lane 3 and lane 64 of a wave64
    /// is lane 0, so the program is the one naming the low lane, and the
    /// next lane up is another.
    #[test]
    fn a_lane_select_past_the_wave_wraps_as_the_hardware_does() {
        // src1 is the lane's inline constant (128 + lane, lanes 0 to 64).
        let program = |write: u32, read: u32| {
            [
                0xBE80_0387, // s_mov_b32 s0, 7
                0xD761_0001,
                (128 + write) << 9, // v_writelane_b32 v1, s0, write
                0xD760_0002,
                (128 + read) << 9 | 0x101, // v_readlane_b32 s2, v1, read
                0xBF07_8002,               // s_cmp_lg_u32 s2, 0
                0xBF85_0001,               // s_cbranch_scc1 -> s_endpgm
                0xBE83_0380,               // s_mov_b32 s3, 0
                S_ENDPGM,
            ]
        };
        for (wave64, lane, past) in [(false, 3, 35), (true, 0, 64)] {
            let low = dxil(&program(lane, lane), wave64).unwrap();
            assert_ne!(dxil(&program(lane + 1, lane), wave64).unwrap(), low);
            assert_eq!(dxil(&program(past, lane), wave64).unwrap(), low);
            assert_eq!(dxil(&program(lane, past), wave64).unwrap(), low);
        }
    }

    /// Lowers a compute program with the root parameters `layout` (in user
    /// SGPRs from s0 on) and the used resources `used`.
    fn lower_with(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_wave(words, layout, used, false)
    }

    /// [`lower_with`] in wave64 if `wave64`.
    fn lower_wave(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                // The layout's user SGPRs (at least two), 512 bytes of LDS.
                rsrc2: 1 << 15 | sgprs.max(2) << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let res = resource::track(&p, &native.layout, native.stage)?;
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        lower(&Parts {
            psv: None,
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: Target::ANY,
            console_cf: false,
            per_lane_only: false,
        })
    }

    /// Root parameter `slot` in user SGPRs from `user_data` on.
    fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: if root == RootType::Descriptor { 2 } else { 1 },
            base_register: 0,
            user_data,
            slot,
        }
    }

    #[test]
    fn a_compressed_export_unpacks_two_f16_channels_per_dword() {
        // A compr export: each source dword packs two f16 channels, the
        // low half first, so four channels come from two VGPRs.
        let words = [
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0282, // v_mov_b32 v1, 2
            0xF800_0C0F,
            0x0000_0100, // exp mrt0, v0, v0, v1, v1 done compr
            S_ENDPGM,
        ];
        lowers_ps(&words).unwrap();
    }

    /// A table at s0 holding structured buffers of dwords.
    fn dword_buffers() -> (Vec<RootParam>, Vec<UsedResource>) {
        let root = RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        };
        let used = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        };
        (vec![root], vec![used])
    }

    #[test]
    fn a_spin_wait_on_a_table_buffer_gets_the_spin_limit() {
        // Each lane rereads its flag in a buffer loaded from the table at
        // s0 until it is no longer 0, then copies it to LDS.
        let spin = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE84_0300, // s_mov_b32 s4, s0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xBE82_0380, // s_mov_b32 s2, 0
            0xE030_6000,
            0x8002_0100, // 0x14: buffer_load_dword v1, v0, s[8:11], 0 idxen glc
            0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
            0x8802_026A, // s_or_b32 s2, vcc_lo, s2
            0x8A7E_027E, // s_andn2_b32 exec_lo, exec_lo, s2
            0xBF89_FFFA, // s_cbranch_execnz -> 0x14
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let (layout, used) = dword_buffers();
        let l = lower_with(&spin, layout, used).unwrap();
        assert_eq!(l.spin_loops, 1);
    }

    #[test]
    fn a_loop_that_writes_lds_keeps_the_loop_limit() {
        // The same loop with the LDS write inside it: it has an effect each
        // round, so it is no spin wait.
        let writes = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE84_0300, // s_mov_b32 s4, s0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xBE82_0380, // s_mov_b32 s2, 0
            0xE030_6000,
            0x8002_0100, // 0x14: buffer_load_dword v1, v0, s[8:11], 0 idxen glc
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
            0x8802_026A, // s_or_b32 s2, vcc_lo, s2
            0x8A7E_027E, // s_andn2_b32 exec_lo, exec_lo, s2
            0xBF89_FFF8, // s_cbranch_execnz -> 0x14
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            S_ENDPGM,
        ];
        let (layout, used) = dword_buffers();
        let l = lower_with(&writes, layout, used).unwrap();
        assert_eq!(l.spin_loops, 0);
    }

    #[test]
    fn scalar_code_inside_a_whole_wave_section_is_lowered() {
        // Assembled with llvm-mc (gfx1030, wave32): scalar code the
        // compiler scheduled into a whole-wave section, branched on after.
        let words = [
            0xBE84_0385, // s_mov_b32 s4, 5
            0xBE80_3DC1, // s_or_saveexec_b32 s0, -1
            0x8005_8104, // s_add_u32 s5, s4, 1
            0x7E02_0280, // v_mov_b32_e32 v1, 0
            0xBEFE_0300, // s_mov_b32 exec_lo, s0
            0xBF06_8605, // s_cmp_eq_u32 s5, 6
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE86_0380, // s_mov_b32 s6, 0
            S_ENDPGM,
        ];
        lowers(&words).unwrap();
    }

    #[test]
    fn buffer_cache_invalidations_lower_as_no_ops() {
        // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
        // host to invalidate its cached view of a buffer.
        assert!(lowers(&[0xE1C4_0000, 0, S_ENDPGM]).is_ok(), "buffer_gl0_inv");
        assert!(lowers(&[0xE1C8_0000, 0, S_ENDPGM]).is_ok(), "buffer_gl1_inv");
    }

    #[test]
    fn a_scalar_carry_chain_reaches_the_select() {
        // The high dword of a 64-bit add or subtract, then a select on the
        // carry (borrow) the high half leaves in SCC, and a branch on it
        // (only code with an effect needs its operands).
        let add = [
            0xBE80_03C1, // s_mov_b32 s0, -1
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8000_8100, // s_add_u32 s0, s0, 1
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        let sub = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8080_8100, // s_sub_u32 s0, s0, 1
            0x8281_8001, // s_subb_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        for words in [add, sub] {
            lowers(&words).unwrap();
        }
    }

    fn perm_features(selector: u32) -> u64 {
        // Synthetic compute shader:
        // v0 is the thread ID and can supply a dynamic selector.
        // Store the result in LDS to keep the permutation live.
        let words = [
            0x7E02_02FFu32,
            0x817F_34AB,
            0x7E04_02FF,
            0xFE01_56CD,
            0xD744_0003,
            257 | (258 << 9) | (selector << 18),
            0x3408_0082,
            0xD834_0000,
            0x0000_0304,
            S_ENDPGM,
        ];

        let code: Vec<u8> = words.iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [64, 1, 1],
            }),
            code,
        };

        let p = ir::build(&native.code, false, 256).unwrap();

        assert!(p.blocks.iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_perm_b32"));

        let res = resource::track(
            &p, &native.layout, native.stage
        ).unwrap();

        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();

        let emitted = lower(&Parts {
            psv: None,
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: Target::ANY,
            console_cf: false,
            per_lane_only: false,
        }).unwrap();

        let container =
            crate::Container::parse(&emitted.dxil).unwrap();

        let features =
            container.part(crate::FourCC::SFI0).unwrap();

        u64::from_le_bytes(features.data.try_into().unwrap())
    }

    #[test]
    fn perm_constant_selector_needs_no_int64() {
        assert_eq!(perm_features(128) & 0x8000, 0);
    }

    #[test]
    fn perm_dynamic_selector_needs_no_int64() {
        assert_eq!(
            perm_features(256) & 0x8000,
            0,
            "dynamic v_perm_b32 must not require INT64"
        );
    }

    #[test]
    fn a_descriptor_selected_by_scc_picks_its_heap_index() {
        // Two buffer descriptors loaded from the table at s0, one picked by
        // a root constant (s1) with s_cselect_b64 per half, and loaded
        // through by index and stored to LDS so the load is needed.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF408_0302,
            0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
            0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
            0xE030_2000,
            0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        // Both are structured buffers of dwords in the table's range.
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        lower_with(&words, layout(), used()).unwrap();
    }

    /// Tables of 2D textures at s0 and samplers at s1.
    fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let range = |kind, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
            user_data,
            slot,
        };
        (
            vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
            vec![
                range(ResourceType::Srv, 0, 0),
                range(ResourceType::Sampler, 1, 1),
            ],
        )
    }

    #[test]
    fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
        use crate::resource::tests::{
            WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
        };
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
            (WATERFALL_PICKED_WAVE32, false),
        ] {
            let (layout, used) = texture_tables();
            let l = lower_wave(words, layout, used, wave64).unwrap();
            // The loop's reads of the first lane became each lane's own
            // dwords: no wave operation is left.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    #[test]
    fn sub_dword_lds_names() {
        let s = |write, bits, signed, d16| {
            Some(SubDword {
                write,
                bits,
                signed,
                d16,
            })
        };
        assert_eq!(SubDword::of("ds_read_i8"), s(false, 8, true, None));
        assert_eq!(SubDword::of("ds_read_u16"), s(false, 16, false, None));
        assert_eq!(
            SubDword::of("ds_read_u8_d16"),
            s(false, 8, false, Some(false))
        );
        assert_eq!(
            SubDword::of("ds_read_i16_d16_hi"),
            s(false, 16, true, Some(true))
        );
        assert_eq!(SubDword::of("ds_write_b8"), s(true, 8, false, None));
        assert_eq!(
            SubDword::of("ds_write_b16_d16_hi"),
            s(true, 16, false, Some(true))
        );
        for name in [
            "ds_read_b32",
            "ds_read2_b32",
            "ds_write_b64",
            "ds_write_b8_d16",
            "ds_read_u8_d32",
            "ds_add_u32",
        ] {
            assert_eq!(SubDword::of(name), None, "{name}");
        }
    }

    #[test]
    fn private_memory_is_as_large_as_its_addresses_reach() {
        let bytes = |words: &[u32]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            super::scratch_bytes(&ir::build(&code, false, 256).unwrap())
        };
        let store = [
            0x7E02_0280, // v_mov_b32 v1, 0
            0xDC70_401C, // scratch_store_dword v1, v2, off offset:28
            0x007D_0201,
        ];
        let indexed = [
            0x3606_0087, // v_and_b32 v3, 7, v0
            0x3406_0682, // v_lshlrev_b32 v3, 2, v3
            0xDC30_4000, // scratch_load_dword v4, v3, off
            0x047D_0003,
        ];
        assert_eq!(bytes(&[&store[..], &indexed, &[S_ENDPGM]].concat()), 32);
        // An address the code does not bound (the thread id).
        let free = [0xDC30_4000, 0x047D_0000]; // scratch_load_dword v4, v0, off
        assert_eq!(
            bytes(&[&store[..], &free, &[S_ENDPGM]].concat()),
            super::SCRATCH_CAP
        );
        let far = [
            0x7E02_02FF, // v_mov_b32 v1, 0x100000
            0x0010_0000,
            0xDC70_4000, // scratch_store_dword v1, v2, off
            0x007D_0201,
            S_ENDPGM,
        ];
        assert_eq!(bytes(&far), super::SCRATCH_MAX);
    }

    #[test]
    fn generic_flat_memory_is_refused() {
        // flat_store_dword v[0:1], v2: the address may be private, shared
        // or global memory.
        let e = lowers(&[0xDC70_0000, 0x007D_0200, S_ENDPGM]).unwrap_err();
        assert!(
            matches!(
                e,
                crate::ShaderError::Unsupported {
                    what: "flat memory",
                    ..
                }
            ),
            "{e}"
        );
    }

    /// An f16 (sign, 5-bit exponent, 10-bit fraction) as an f32; normal numbers only.
    fn f16_value(h: u32) -> f32 {
        let sign = if h >> 15 != 0 { -1.0 } else { 1.0 };
        let exponent = (h >> 10 & 0x1F) as i32 - 15;
        sign * (1.0 + (h & 0x3FF) as f32 / 1024.0) * 2f32.powi(exponent)
    }

    #[test]
    fn inline_float_constants_mean_the_same_in_f16() {
        // v_fma_mix and v_pack_b32_f16 read these as f16: 1.0 must be 0x3C00,
        // not the low half (0) of the f32 bits 0x3F800000.
        assert_eq!(inline_f16(242), 0x3C00);
        for code in 240..=248u8 {
            let f32_value = f32::from_bits(crate::resource::inline(code).unwrap());
            let f16_value = f16_value(inline_f16(code));
            assert!(
                (f32_value - f16_value).abs() <= f32_value.abs() * 1e-3,
                "inline constant {code}: f32 {f32_value}, f16 {f16_value}"
            );
        }
    }

    #[test]
    fn scalar_conditional_moves_lower() {
        // Each sets SCC, moves under it into a destination that already has a
        // value, and branches on the result (only code with an effect needs
        // its operands).
        let tail = [
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let b32 = [
            0xBE82_0387, // s_mov_b32 s2, 7
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBF07_8001, // s_cmp_lg_u32 s1, 0
            0xBE82_0501, // s_cmov_b32 s2, s1
        ];
        let k32 = [
            0xBE82_0387, // s_mov_b32 s2, 7
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBF07_8001, // s_cmp_lg_u32 s1, 0
            0xB102_0005, // s_cmovk_i32 s2, 5
        ];
        let b64 = [
            0xBE82_0387, // s_mov_b32 s2, 7
            0xBE83_0387, // s_mov_b32 s3, 7
            0xBE80_0381, // s_mov_b32 s0, 1
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBF07_8000, // s_cmp_lg_u32 s0, 0
            0xBE82_0600, // s_cmov_b64 s[2:3], s[0:1]
        ];
        for head in [&b32[..], &k32[..], &b64[..]] {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers(&words).unwrap();
        }
    }

    #[test]
    fn a_wave_prefix_of_min_lowers() {
        // A whole-wave section that leaves the inclusive prefix of a min:
        // each lane the minimum of the lanes up to and including it. The
        // scan runs with `row_shr` steps inside each row, and the first
        // row's prefix is read out and folded under the second. The store
        // after the section reads the prefix, so the value cannot be left
        // missing: a wave prefix of a min has to lower.
        let words = [
            0x7E00_0280, // v_mov_b32 v0, 0
            // 32-bit scalar forms: the wave32 pass follows their masks.
            0xBE84_3DC1, // s_or_saveexec_b32 s4, -1
            0x876A_C104, // s_and_b32 s106, s4, -1
            0x0202_00C1, // v_cndmask_b32 v1, -1, v0, vcc
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:1, v1
            0xFF01_1101,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:2, v1
            0xFF01_1201,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:3, v1
            0xFF01_1301,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:4, v1
            0xFF01_1401,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:5, v1
            0xFF01_1501,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:6, v1
            0xFF01_1601,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:7, v1
            0xFF01_1701,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:8, v1
            0xFF01_1801,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:9, v1
            0xFF01_1901,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:10, v1
            0xFF01_1A01,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:11, v1
            0xFF01_1B01,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:12, v1
            0xFF01_1C01,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:13, v1
            0xFF01_1D01,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:14, v1
            0xFF01_1E01,
            0x2602_02FA, // v_min_u32 v1, v1 row_shr:15, v1
            0xFF01_1F01,
            0xD760_0006, // v_readlane_b32 s6, v1, 15
            0x0001_1F01,
            0x7E04_0206, // v_mov_b32 v2, s6
            0x2602_02FA, // v_min_u32 v1, v2 quad_perm:[0,1,2,3] row_mask:0x2, v1
            0x2F00_E402,
            0xBEFE_0304, // s_mov_b32 exec_lo, s4
            0xD834_0000, // ds_write_b32 v0, v1
            0x0000_0100,
            S_ENDPGM,
        ];
        lower_with(&words, vec![], vec![]).unwrap();
    }

    /// A wave32 graphics program's host wave may have 64 lanes, whose upper
    /// half is a second console wave: a lane mask read as a number picks the
    /// reader's half of the ballot, and `v_mbcnt_lo` reports its rank as a
    /// host lane number.
    #[test]
    fn a_wave32_mask_read_takes_its_ballot_half() {
        let words = [
            0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v2
            0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
            0xBF07_806A, // s_cmp_lg_u32 vcc_lo, 0
            0x7E08_0204, // v_mov_b32 v4, s4
            0xD765_0005,
            0x0001_006A, // v_mbcnt_lo_u32_b32 v5, vcc_lo, 0
            0xD766_0006,
            0x0002_0A6A, // v_mbcnt_hi_u32_b32 v6, vcc_lo, v5
            0xD765_0007,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v7, -1, 0
            0xBF85_0001, // s_cbranch_scc1 -> the export
            S_ENDPGM,    // no lane had its bit: done
            0xF800_180F,
            0x0706_0504, // exp mrt0, v4, v5, v6, v7 done vm
            S_ENDPGM,
        ];
        let lowered = lower_program(&words, HwStage::Ps, false, pixel_io()).unwrap();
        let f = |name| lowered.op_names.iter().position(|n| n == name);
        let (lane, ballot, bits) = (
            f("dx.op.waveGetLaneIndex").expect("no lane index for the pick"),
            f("dx.op.waveActiveBallot").expect("no ballot for the mask read"),
            f("dx.op.unaryBits.i32").expect("no bit count"),
        );
        let l = &lowered.insts;
        let lanes: Vec<usize> = l
            .iter()
            .enumerate()
            .filter(|(_, i)| **i == format!("call f{lane} [k111]"))
            .map(|(i, _)| i)
            .collect();
        assert!(!lanes.is_empty(), "no lane index for the pick");
        let i_ballot = l
            .iter()
            .position(|i| i.starts_with(&format!("call f{ballot} [k116, ")))
            .expect("no ballot");
        let (i_x, i_y) = (
            at(l, &format!("extract i{i_ballot} 0")),
            at(l, &format!("extract i{i_ballot} 1")),
        );
        // The pick: upper host lanes read the ballot's second dword.
        let i_cmp = l
            .iter()
            .position(|i| {
                i.starts_with("cmp:") && lanes.iter().any(|n| i.ends_with(&format!("i{n} k32")))
            })
            .expect("no `lane >= 32` compare");
        at(l, &format!("select i{i_cmp} i{i_y} i{i_x}"));
        // v_mbcnt_lo counts below the lane in its own half; a wave32 mask's
        // high half is empty, so v_mbcnt_hi's count is of no bits.
        assert!(
            lanes
                .iter()
                .any(|n| l.contains(&format!("bin:And i{n} k31"))),
            "no `lane & 31` in {l:?}"
        );
        at(l, &format!("call f{bits} [k31, k0]"));
        // The count is a lane number, so a second wave's rank takes its
        // 32-lane base: `v_mbcnt_lo(-1)` is the host lane index.
        let bases: Vec<usize> = l
            .iter()
            .enumerate()
            .filter(|(_i, s)| {
                s.starts_with("bin:And ")
                    && lanes.iter().any(|n| **s == format!("bin:And i{n} k32"))
            })
            .map(|(i, _)| i)
            .collect();
        assert!(!bases.is_empty(), "no `lane & 32` wave base in {l:?}");
        for base in &bases {
            assert!(
                l.iter().any(|s| {
                    s.starts_with("bin:Add ")
                        && s["bin:Add ".len()..]
                            .split(' ')
                            .any(|a| a == format!("i{base}"))
                }),
                "the wave base `i{base}` is never added in {l:?}"
            );
        }
    }

    /// The ballot-half pick is for wave32 graphics programs only: a wave64
    /// program's mask dwords are its own, and the console path (compute) is
    /// the whole host wave already, so neither reads the lane index for it.
    #[test]
    fn only_host_wave32_reads_take_a_ballot_half() {
        let words = [
            0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v2
            0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
            0x7E08_0204, // v_mov_b32 v4, s4
            0xF800_180F,
            0x0404_0404, // exp mrt0, v4, v4, v4, v4 done vm
            S_ENDPGM,
        ];
        let l64 = lower_program(&words, HwStage::Ps, true, pixel_io()).unwrap();
        assert!(!l64.op_names.iter().any(|n| n == "dx.op.waveGetLaneIndex"));
        let words = [
            0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v0
            0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
            0xBF07_8004, // s_cmp_lg_u32 s4, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE85_0380, // s_mov_b32 s5, 0
            S_ENDPGM,
        ];
        let cs = lower_program(&words, HwStage::Cs, false, compute_io()).unwrap();
        assert!(!cs.op_names.iter().any(|n| n == "dx.op.waveGetLaneIndex"));
    }

    fn pixel_io() -> StageIo {
        StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        }
    }

    fn compute_io() -> StageIo {
        StageIo::Cs(ComputeIo {
            rsrc2: 0,
            threads: [1, 1, 1],
        })
    }

    /// The index of the instruction `pat` names in `lines`.
    fn at(lines: &[String], pat: &str) -> usize {
        lines
            .iter()
            .position(|l| l == pat)
            .unwrap_or_else(|| panic!("no `{pat}` in {lines:?}"))
    }
    #[test]
    fn buffer_formats_by_number() {
        // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
        // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
        assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
        assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
        assert_eq!(
            buffer_format(50),
            Some((&[10, 10, 10, 2][..], Numeric::Unorm))
        );
        assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
        assert_eq!(
            buffer_format(77),
            Some((&[32, 32, 32, 32][..], Numeric::Float))
        );
        // Invalid, past the table.
        assert_eq!(buffer_format(0), None);
        assert_eq!(buffer_format(78), None);
    }

    /// MIMG `op vdata, v[vaddr..], s[srsrc..srsrc+7]` on a 2D image.
    const fn mimg_2d(op: u8, dmask: u8, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [
            0x3C << 26 | (op as u32 & 0x7F) << 18 | (dmask as u32) << 8 | 1 << 3,
            ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
        ]
    }

    #[test]
    fn an_image_read_as_float_and_by_an_atomic_is_annotated_twice() {
        use crate::native::{ResourceType, RootParam, RootType, UsedResource};

        // One 2D image UAV from the descriptor table at s4, loaded (four
        // float channels) and then atomically minimised in the same block.
        let mut words = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            // s_load_dwordx8 s[8:15], s[4:5], 0
            0x3D << 26 | 0x03 << 18 | 8 << 6 | 4 >> 1,
            125 << 25,
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0280, // v_mov_b32 v1, 0
        ];
        words.extend(mimg_2d(0x00, 0xF, 2, 0, 8)); // image_load v[2:5], v[0:1], s[8:15]
        words.extend(mimg_2d(0x15, 0x1, 2, 0, 8)); // image_atomic_umin v2, v[0:1], s[8:15]
        words.push(S_ENDPGM);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 4,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let lowered = lower(&Parts {
            psv: None,
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: Target::ANY,
            console_cf: false,
            per_lane_only: false,
        })
        .unwrap();

        const TEXTURE_2D_UAV: u32 = 0x1002;
        assert!(
            lowered.annotations.contains(&(TEXTURE_2D_UAV, 0x409)),
            "the load's annotation: {:x?}",
            lowered.annotations
        );
        assert!(
            lowered.annotations.contains(&(TEXTURE_2D_UAV, 0x105)),
            "the atomic's annotation: {:x?}",
            lowered.annotations
        );
    }

    #[test]
    fn packed_half_inputs_pair_unlisted_rows() {
        use super::super::container::SigElement;
        use super::half_inputs;
        use crate::native::{PsInput, StageIo, Usage};

        let sig = |row: u32, half: bool| SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half,
        };
        let ps_in = |row: u8, attr: u8, pack2: bool, row2: u8| PsInput {
            usage: Usage::Generic,
            row,
            mask: 3,
            attr,
            flat: false,
            fp16: true,
            pack2,
            row2,
        };
        let inputs = vec![
            sig(0, true),
            sig(1, true),
            sig(2, true),
            sig(3, false),
            sig(4, true),
            sig(5, true),
        ];
        let io = StageIo::Ps {
            inputs: vec![
                ps_in(0, 0, true, 0),
                ps_in(2, 1, true, 4),
                ps_in(5, 3, true, 5),
            ],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let map = half_inputs(&io, &inputs);
        assert_eq!(map.get(&0), Some(&(true, Some(1))));
        assert_eq!(map.get(&2), Some(&(true, Some(4))));
        assert_eq!(map.get(&5), Some(&(true, None)));

        let inputs2 = vec![sig(0, true), sig(2, true), sig(3, true)];
        let io2 = StageIo::Ps {
            inputs: vec![
                ps_in(0, 0, true, 0),
                ps_in(0, 9, false, 0),
                ps_in(2, 1, true, 2),
            ],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let map2 = half_inputs(&io2, &inputs2);
        assert_eq!(map2.get(&0), Some(&(true, None)));
        assert_eq!(map2.get(&2), Some(&(true, Some(3))));
    }

    #[test]
    fn cube_and_msaa_descriptor_data_lower() {
        use crate::native::{RootParam, RootType};

        for dim in [3u32, 6, 7] {
            let mimg_w0 = 0xF000_0000 | (0x0E << 18) | (1 << 8) | (dim << 3);
            let words = [
                0xBE81_0380u32,
                0xF40C_0100,
                0xFA00_0000,
                0x900C_9007,
                0xBF07_800C,
                0xBF85_0001,
                0xBE8D_0380,
                0x7E00_0280,
                mimg_w0,
                0x0001_0000,
                S_ENDPGM,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let res = resource::track(&p, &native.layout, native.stage).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            lower(&Parts {
                psv: None,
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: Target::ANY,
                console_cf: false,
                per_lane_only: false,
            })
            .unwrap();
        }
    }

    #[test]
    fn msaa_image_load_and_get_resinfo_lower() {
        use crate::native::{RootParam, RootType};

        for (dim, op) in [(6u32, 0x00u32), (6, 0x0E), (7, 0x00), (7, 0x0E)] {
            let mimg_w0 = 0xF000_0000 | (op << 18) | (0xF << 8) | (dim << 3);
            let words = [
                0xBE81_0380u32,
                0xF40C_0100,
                0xFA00_0000,
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0280,
                0x7E06_0280,
                mimg_w0,
                0x0001_0000,
                0x7D9A_0080,
                0xBF81_0000,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let res = resource::track(&p, &native.layout, native.stage).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            lower(&Parts {
                psv: None,
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: Target::ANY,
                console_cf: false,
                per_lane_only: false,
            })
            .unwrap();
        }
    }

    #[test]
    fn partial_channel_typed_buffer_store_lowers() {
        use crate::native::{RootParam, RootType};

        for op in [0x04u32, 0x05, 0x06, 0x86] {
            let mubuf_w0 = 0xE000_2000 | (op << 18);
            let words = [
                0xBE81_0380u32,
                0xF408_0100,
                0xFA00_0000,
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0280,
                0x7E06_0280,
                mubuf_w0,
                0x8001_0100,
                S_ENDPGM,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let res = resource::track(&p, &native.layout, native.stage).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            lower(&Parts {
                psv: None,
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: Target::ANY,
                console_cf: false,
                per_lane_only: false,
            })
            .unwrap();
        }
    }

    #[test]
    fn packed_buffer_formats_by_number() {
        // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
        // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
        // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
        assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
        assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
        assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
        assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
        assert_eq!(
            buffer_format(44),
            Some((&[2, 10, 10, 10][..], Numeric::Unorm))
        );
        assert_eq!(
            buffer_format(45),
            Some((&[2, 10, 10, 10][..], Numeric::Snorm))
        );
        assert_eq!(
            buffer_format(46),
            Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
        );
        assert_eq!(
            buffer_format(47),
            Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
        );
        assert_eq!(
            buffer_format(48),
            Some((&[2, 10, 10, 10][..], Numeric::Uint))
        );
        assert_eq!(
            buffer_format(49),
            Some((&[2, 10, 10, 10][..], Numeric::Sint))
        );
    }

    /// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
    /// `legacyF16ToF32` widens it.
    fn half(h: u32) -> f32 {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 if m == 0 => f32::INFINITY,
            31 => f32::NAN,
            _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
        };
        if s == 1 { -v } else { v }
    }

    #[test]
    fn small_floats_widen_as_halves() {
        // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
        // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
        let f11 = |x: u32| half(x << half_shift(11));
        let f10 = |x: u32| half(x << half_shift(10));
        assert_eq!(f11(0), 0.0);
        assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
        assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
        assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
        assert_eq!(f11(0x3C0), 1.0);
        assert_eq!(f11(0x3E0), 1.5);
        assert_eq!(f11(0x7BF), 65024.0); // largest
        assert_eq!(f11(0x7C0), f32::INFINITY);
        assert!(f11(0x7C1).is_nan());
        assert_eq!(f10(0x001), 2f32.powi(-19));
        assert_eq!(f10(0x020), 2f32.powi(-14));
        assert_eq!(f10(0x1E0), 1.0);
        assert_eq!(f10(0x3DF), 64512.0);
        assert_eq!(f10(0x3E0), f32::INFINITY);
        assert!(f10(0x3FF).is_nan());
        // A 16-bit float is a half already, sign included.
        assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
    }

#[cfg(test)]
mod select_tests {
    use super::super::super::emit;
    use crate::native::{ComputeIo, RootParam};
    use crate::{Container, FourCC, HwStage, NativeProgram, ResourceType, RootType, UsedResource};

    #[test]
    fn selects_from_independently_written_mask_words() {
        let words = [
            0xbe81_0380u32, // s1 = 0 (table address high)
            0xf408_0200,
            0xfa00_0000, // load UAV into s[8:11]
            0xbe84_03ff,
            0x8000_0000, // s4 = lane 31
            0xbe85_0381, // s5 = lane 32 in wave64
            0x7e02_028b,
            0x7e04_029d, // v1 = 11, v2 = 29
            0xd501_0003,
            0x0012_0501, // v_cndmask_b32 v3, v1, v2, s[4:5]
            0x3402_0082, // v1 = 4 * v0
            0xe070_1000,
            0x8002_0301, // store v3 to UAV at v1
            0xbf81_0000,
        ];
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        for (wave64, uses_select) in [(false, true), (true, true), (true, false)] {
            let mut words = words;
            if !uses_select {
                words[12] = 0x8002_0101; // store v1 instead of the select result
            }
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 0,
                    slot: 0,
                }],
                io: crate::native::StageIo::Cs(ComputeIo {
                    rsrc2: 2,
                    threads: [if wave64 { 64 } else { 32 }, 1, 1],
                }),
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = crate::ir::build(&native.code, wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
            let shader = emit(&c, &native, &p, &st, &io, &res, super::super::super::Target::ANY).unwrap();
            let emitted = Container::parse(&shader.dxil).unwrap();
            let features = emitted.part(FourCC::SFI0).unwrap();
            let wave_ops = u64::from_le_bytes(features.data.try_into().unwrap()) & 0x4000 != 0;
            assert_eq!(wave_ops, uses_select);
        }
    }
}
