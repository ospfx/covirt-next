#include <compiler/generic_lifter.hpp>
#include <compiler/generic_emitter.hpp>
#include <compiler/generic_vm.hpp>
#include <compiler/default_vm_enter.hpp>

#include <optional>

#include <zasm/zasm.hpp>

namespace covirt::vm {
    enum class v0_op : uint8_t {
        vm_enter, vm_exit, push_imm, push_reg, pop, read, write, add, sub, bxor, band, bor, cmp, jmp, jz, jnz, jb, jnb, jbe, jnbe, jl, jle, jnl, jnle, call, lea, execute_native, test
    };

    class v0_emitter : public generic_emitter {
    public:
    #define LAZY_EMIT(x) \
        template <typename S, typename... Tx> \
        auto& x(S encode_size, Tx&&... args) \
        { \
            emplace(opcode(v0_op::x, int(encode_size)), std::forward<Tx>(args)...); \
            return *this; \
        }

        LAZY_EMIT(push_reg);
        LAZY_EMIT(push_imm);
        LAZY_EMIT(pop);
        LAZY_EMIT(read);
        LAZY_EMIT(write);
        LAZY_EMIT(add);
        LAZY_EMIT(sub);
        LAZY_EMIT(bxor);
        LAZY_EMIT(band);
        LAZY_EMIT(bor);
        LAZY_EMIT(cmp);
        LAZY_EMIT(test);
        LAZY_EMIT(lea);
        LAZY_EMIT(call);
    #undef LAZY_EMIT
    };

    class v0_lifter : public generic_lifter {
    public:
        v0_emitter e;

        std::map<ZydisMnemonic, fn_instruction_translator_t>& get_translation_table() override
        {
            return lift_impl;
        }

        generic_emitter& get_emitter() override 
        {
            return e;
        }

        void vm_exit(uint16_t bytes_to_skip) override
        {
            e >> e.opcode(v0_op::vm_exit, 1) >> uint16_t(bytes_to_skip);
        }

        void native(uint8_t *ins_bytes, size_t length) override
        {
            e >> e.opcode(v0_op::execute_native, 1) >> uint8_t(length);
            for (int i = 0; i < length; i++) e >> ins_bytes[i];
        } 

    private:
        static constexpr uint8_t tmp_reg_idx = 14;

        void push_address(covirt::zydis_operand &operand);
        void push_operand(covirt::zydis_operand &operand, std::optional<int> override_size = {});
        void pop_operand(covirt::zydis_operand &operand, std::optional<int> override_size = {}, std::optional<covirt::zydis_operand> src = {});

        std::map<ZydisMnemonic, fn_instruction_translator_t> lift_impl = {
            {
                ZYDIS_MNEMONIC_MOV, [&](covirt::zydis_operand &dst, covirt::zydis_operand &src) {
                    push_operand(src, dst.size);
                    pop_operand(dst, {}, src);
                    return true;
                }
            },

#define LAZY_ARITH(mnemonic, op) \
            { \
                mnemonic, [&](covirt::zydis_operand &dst, covirt::zydis_operand &src) { \
                    push_operand(dst); \
                    push_operand(src, dst.size); \
                    e.op(dst.size); \
                    pop_operand(dst); \
                    return true; \
                } \
            }

            LAZY_ARITH(ZYDIS_MNEMONIC_ADD, add),
            LAZY_ARITH(ZYDIS_MNEMONIC_SUB, sub),
            LAZY_ARITH(ZYDIS_MNEMONIC_XOR, bxor),
            LAZY_ARITH(ZYDIS_MNEMONIC_AND, band),
            LAZY_ARITH(ZYDIS_MNEMONIC_OR, bor),
#undef LAZY_ARITH

            {
                ZYDIS_MNEMONIC_CMP, [&](covirt::zydis_operand &dst, covirt::zydis_operand &src) {
                    push_operand(dst);
                    push_operand(src, dst.size);
                    e.cmp(dst.size);
                    return true; \
                }
            },
            {
                // [BUG-E-FIX] ZYDIS_MNEMONIC_TEST mirrors CMP: push dst, push src,
                // vtest pops both + pushes real 2-byte flags word. Previously test
                // went native (exec_native) which pushes NO flags, yet the following
                // vjcc pops 2 bytes -> garbage branch + permanent +2 vsp drift
                // (vsp climbs past vstack top into retaddr/vtable globals).
                ZYDIS_MNEMONIC_TEST, [&](covirt::zydis_operand &dst, covirt::zydis_operand &src) {
                    push_operand(dst);
                    push_operand(src, dst.size);
                    e.test(dst.size);
                    return true; \
                }
            },

#define LAZY_JUMP(mnemonic, op) \
            { \
                mnemonic, [&](covirt::zydis_operand &dst, covirt::zydis_operand &src) { \
                    e >> e.opcode(op, 1) >> uint16_t(0); \
                    fill_in_gaps.push_back({ dst.references_bb.value(), e.get().size() - sizeof(uint16_t), sizeof(uint16_t) }); \
                    return true; \
                } \
            }

            LAZY_JUMP(ZYDIS_MNEMONIC_JMP, v0_op::jmp),
            LAZY_JUMP(ZYDIS_MNEMONIC_JZ, v0_op::jz),
            LAZY_JUMP(ZYDIS_MNEMONIC_JNZ, v0_op::jnz),
            LAZY_JUMP(ZYDIS_MNEMONIC_JB, v0_op::jb),
            LAZY_JUMP(ZYDIS_MNEMONIC_JNB, v0_op::jnb),
            LAZY_JUMP(ZYDIS_MNEMONIC_JBE, v0_op::jbe),
            LAZY_JUMP(ZYDIS_MNEMONIC_JNBE, v0_op::jnbe),
            LAZY_JUMP(ZYDIS_MNEMONIC_JL, v0_op::jl),
            LAZY_JUMP(ZYDIS_MNEMONIC_JLE, v0_op::jle),
            LAZY_JUMP(ZYDIS_MNEMONIC_JNL, v0_op::jnl),
            LAZY_JUMP(ZYDIS_MNEMONIC_JNLE, v0_op::jnle),
#undef LAZY_JUMP

            {
                ZYDIS_MNEMONIC_LEA, [&](covirt::zydis_operand &dst, covirt::zydis_operand &src) {
                    if (src.as_memory().base != ZYDIS_REGISTER_RIP)
                        return false;

                    e.lea(dst.size, int32_t(dst.references_rva.value()));
                    pop_operand(dst);
                    return true;
                }
            },
            {
                ZYDIS_MNEMONIC_CALL, [&](covirt::zydis_operand &dst, covirt::zydis_operand &src) {
                    e.call(1, int32_t(dst.references_rva.value()));
                    return true;
                }
            },
        };
    };

    class v0_vm : public generic_vm {
    public:
        void initialize(zasm::x86::Assembler &a) override;
        void finalize(zasm::x86::Assembler& a) override;

        std::map<uint8_t, fn_vm_handler_t>& get_handlers() override { return vm_impl; }
        generic_vm_enter& get_vm_enter() override { return vm_enter_emitter; }

        void set_code_size(size_t size) override { code_size = size; };
        void set_stack_size(size_t size) override { stack_size = size; };

    private:
        zasm::x86::Gp64 vip, vsp;

        size_t code_size = 0;
        size_t stack_size = 0;

        std::map<std::string, zasm::Label> global_labels = {
            {"saved_rsp", {}},
            {"_vsp", {}},
            {"_vip", {}},
            {"vstack", {}},
            {"vcode", {}},
            {"vtable", {}},
            {"retaddr", {}},
            {"venter", {}},
            {"vexit", {}},
            {"vpush_imm", {}},
            {"vpush_reg", {}},
            {"vpop", {}},
            {"vread", {}},
            {"vwrite", {}},
            {"vadd", {}},
            {"vsub", {}},
            {"vxor", {}},
            {"vand", {}},
            {"vor", {}},
            {"vcmp", {}},
            {"vtest", {}},
            {"vjmp", {}},
            {"vjz", {}},
            {"vjnz", {}},
            {"vjb", {}},
            {"vjnb", {}},
            {"vjbe", {}},
            {"vjnbe", {}},
            {"vjl", {}},
            {"vjle", {}},
            {"vjnl", {}},
            {"vjnle", {}},
            {"vcall", {}},
            {"vlea", {}},
            {"vexenative", {}},
            {"vflags", {}}
        };

        default_vm_enter vm_enter_emitter;

        // to-do: look into `embedLabelRel` instead of runtime creation? idk
        //
        template <typename T, typename... Tx>
        void create_jump_table_once(zasm::x86::Assembler& a, T table, Tx&&... entries)
        {
            auto pass = a.createLabel();

            a.cmp(zasm::x86::qword_ptr(zasm::x86::rip, table), 0);
            a.jnz(pass);

            a.lea(zasm::x86::r9, zasm::x86::qword_ptr(zasm::x86::rip, table));
            (([&](auto &&label) { 
                a.lea(zasm::x86::r10, zasm::x86::qword_ptr(zasm::x86::rip, label));
                a.mov(zasm::x86::qword_ptr(zasm::x86::r9), zasm::x86::r10);
                a.add(zasm::x86::r9, 8);
            })(entries), ...);

            a.bind(pass);
        }

        void vm_next_instruction(zasm::x86::Assembler& a, std::optional<zasm::Label> label = {});
        void jump_using_table(zasm::x86::Assembler& a, zasm::Label &paths);
        void get_size_from_opcode(zasm::x86::Assembler& a, zasm::Label &start);
        void get_vreg_address(zasm::x86::Assembler& a);
        void get_vreg_value(zasm::x86::Assembler& a);

        std::map<uint8_t, fn_vm_handler_t> vm_impl = {
            {
                uint8_t(v0_op::vm_enter), [&](zasm::x86::Assembler& a) {
                    a.bind(global_labels["venter"]);

                    a.pop(zasm::x86::r11);
                    a.mov(zasm::x86::qword_ptr(zasm::x86::rip, global_labels["retaddr"]), zasm::x86::r11);
                    a.pop(zasm::x86::r11);
                    a.mov(zasm::x86::qword_ptr(zasm::x86::rip, global_labels["saved_rsp"]), zasm::x86::rsp);

                    // to-do: save r9, r10 before?
                    // to-do: not ideal, this is repetitive

                    a.push(zasm::x86::r15); // -8
                    a.push(zasm::x86::r14); // -16
                    a.push(zasm::x86::r13); // -24
                    a.push(zasm::x86::r12); // -32
                    a.push(zasm::x86::r11); // -40
                    a.push(zasm::x86::r10); // -48
                    a.push(zasm::x86::r9); // -56
                    a.push(zasm::x86::r8); // -64
                    a.push(zasm::x86::rdi); // -72
                    a.push(zasm::x86::rsi); // -80
                    a.push(zasm::x86::rbp); // -88
                    // [BUG-G-FIX] vregs[rsp] slot must hold the TRUE guest rsp at the
                    // marker point = saved_rsp + 0x200: the stub did sub rsp,0x200
                    // before venter, but exec_native/vcall/vexit run native code at
                    // rsp = saved_rsp + 0x200 (revert_effects undoes the stub's sub).
                    // The old push of [saved_rsp] (= marker_rsp - 0x200) made every
                    // lifted rsp-relative memory access ([rsp+disp] via push_reg 4 +
                    // disp) compute marker_rsp-0x200+disp while native code computed
                    // marker_rsp+disp — 0x200 off (e.g. CeEncode's lea r14,[rsp+0x1f]
                    // native vs write0 [r4+31] lifted pointed at different slots).
                    // r9 is already saved at -56 here, safe to use as scratch.
                    a.mov(zasm::x86::r9, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["saved_rsp"]));
                    a.add(zasm::x86::r9, 0x200);
                    a.push(zasm::x86::r9); // -96  ← vregs[rsp] = marker rsp
                    a.push(zasm::x86::rbx); // -104
                    a.push(zasm::x86::rdx); // -112
                    a.push(zasm::x86::rcx); // -120
                    a.push(zasm::x86::rax); // -128
                    a.pushfq(); // -136
                    // [REGSAVE-REORDER] guest regs saved BEFORE the jump-table fill:
                    // mba/smc expansions clobber r15/r14/r13/r12/r8/rdi/rbx; saving first
                    create_jump_table_once(a, global_labels["vtable"], global_labels["venter"], global_labels["vexit"], 
                            global_labels["vpush_imm"], global_labels["vpush_reg"], global_labels["vpop"], global_labels["vread"], global_labels["vwrite"], 
                            global_labels["vadd"], global_labels["vsub"], global_labels["vxor"], global_labels["vand"], global_labels["vor"], 
                            global_labels["vcmp"], global_labels["vjmp"], global_labels["vjz"], global_labels["vjnz"], global_labels["vjb"], 
                            global_labels["vjnb"], global_labels["vjbe"], global_labels["vjnbe"], global_labels["vjl"], global_labels["vjle"],
                            global_labels["vjnl"], global_labels["vjnle"], global_labels["vcall"], global_labels["vlea"], global_labels["vexenative"], global_labels["vtest"]);


                    a.lea(vsp, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vstack"]));
                    a.add(vsp, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["_vsp"]));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::r11);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::vm_exit), [&](zasm::x86::Assembler& a) {
                    a.bind(global_labels["vexit"]);

                    a.add(vip, 1);
                    a.mov(zasm::x86::r10w, zasm::x86::word_ptr(vip));
                    a.add(zasm::x86::word_ptr(zasm::x86::rip, global_labels["retaddr"]), zasm::x86::r10w);

                    a.popfq();
                    a.pop(zasm::x86::rax);
                    a.pop(zasm::x86::rcx);
                    a.pop(zasm::x86::rdx);
                    a.pop(zasm::x86::rbx);
                    a.pop(zasm::x86::rbp); // rsp
                    a.pop(zasm::x86::rbp);
                    a.pop(zasm::x86::rsi);
                    a.pop(zasm::x86::rdi);
                    a.pop(zasm::x86::r8);
                    a.pop(zasm::x86::r9);
                    a.pop(zasm::x86::r10);
                    a.pop(zasm::x86::r11);
                    a.pop(zasm::x86::r12);
                    a.pop(zasm::x86::r13);
                    a.pop(zasm::x86::r14);
                    a.pop(zasm::x86::r15);

                    vm_enter_emitter.revert_effects(a);

                    a.jmp(zasm::x86::qword_ptr(zasm::x86::rip, global_labels["retaddr"]));
                }
            },
            {
                uint8_t(v0_op::push_imm), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto vpushz = [&]<typename T>(int size, zasm::x86::Gp v, T ptr) {
                        a.bind(labels[1 + size]);
                        a.sub(vsp, 1 << size);
                        a.mov(v, ptr(std::forward<zasm::x86::Gp64>(vip)));
                        a.mov(ptr(std::forward<zasm::x86::Gp64>(vsp)), v);
                        a.add(vip, 1 << size);
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vpush_imm"]);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    vpushz(0b00, zasm::x86::cl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    vpushz(0b01, zasm::x86::cx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    vpushz(0b10, zasm::x86::ecx, zasm::x86::dword_ptr<zasm::x86::Gp64>);
                    vpushz(0b11, zasm::x86::rcx, zasm::x86::qword_ptr<zasm::x86::Gp64>);
                    
                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                uint8_t(v0_op::push_reg), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto vpushz = [&]<typename T>(int size, zasm::x86::Gp v, T ptr) {
                        a.bind(labels[1 + size]);
                        a.sub(vsp, 1 << size);
                        a.mov(ptr(std::forward<zasm::x86::Gp64>(vsp)), v);
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vpush_reg"]);
                    get_vreg_value(a);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    vpushz(0b00, zasm::x86::dl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    vpushz(0b01, zasm::x86::dx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    vpushz(0b10, zasm::x86::edx, zasm::x86::dword_ptr<zasm::x86::Gp64>);
                    vpushz(0b11, zasm::x86::rdx, zasm::x86::qword_ptr<zasm::x86::Gp64>);
                    
                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                uint8_t(v0_op::pop), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto vpopz = [&]<typename T>(int size, zasm::x86::Gp v, T ptr) {
                        a.bind(labels[1 + size]);
                        a.mov(v, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.add(vsp, 1 << size);
                        a.mov(ptr(const_cast<zasm::x86::Gp64&&>(zasm::x86::rdx)), v);
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vpop"]);
                    get_vreg_address(a);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    vpopz(0b00, zasm::x86::cl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    vpopz(0b01, zasm::x86::cx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    // [P6-ZEXT32] 32 位写零扩展: mov ecx,[vsp] 已零扩展 rcx,
                    // 再以 qword 写 vregs 槽 -> 高 32 位清零(兼容 x86 写 r32 语义)
                    // (dword_ptr 不带显式模板参: 让 TArgs 推导为 Gp64&, 绑定 lvalue vsp;
                    //  显式 <Gp64> 会把形参变成 Gp64&&, 传 lvalue 编译错)
                    a.bind(labels[3]);
                    a.mov(zasm::x86::ecx, zasm::x86::dword_ptr(vsp));
                    a.add(vsp, 4);
                    a.mov(zasm::x86::qword_ptr(zasm::x86::rdx), zasm::x86::rcx);
                    a.jmp(labels[5]);
                    vpopz(0b11, zasm::x86::rcx, zasm::x86::qword_ptr<zasm::x86::Gp64>);
                    
                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                uint8_t(v0_op::read), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto vreadz = [&]<typename T>(int size, zasm::x86::Gp v, T ptr) {
                        a.bind(labels[1 + size]);
                        a.mov(zasm::x86::rdx, zasm::x86::qword_ptr(vsp));
                        a.add(vsp, 8 - (1 << size));
                        a.mov(v, ptr(const_cast<zasm::x86::Gp64&&>(zasm::x86::rdx)));
                        a.mov(ptr(std::forward<zasm::x86::Gp64>(vsp)), v);
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vread"]);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    vreadz(0b00, zasm::x86::cl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    vreadz(0b01, zasm::x86::cx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    vreadz(0b10, zasm::x86::ecx, zasm::x86::dword_ptr<zasm::x86::Gp64>);
                    vreadz(0b11, zasm::x86::rcx, zasm::x86::qword_ptr<zasm::x86::Gp64>);

                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                uint8_t(v0_op::write), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto vwritez = [&]<typename T>(int size, zasm::x86::Gp v, T ptr) {
                        a.bind(labels[1 + size]);
                        a.mov(zasm::x86::r10, zasm::x86::qword_ptr(vsp));
                        a.mov(ptr(const_cast<zasm::x86::Gp64&&>(zasm::x86::r10)), v);
                        a.add(vsp, 8);
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vwrite"]);
                    get_vreg_value(a);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    vwritez(0b00, zasm::x86::dl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    vwritez(0b01, zasm::x86::dx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    vwritez(0b10, zasm::x86::edx, zasm::x86::dword_ptr<zasm::x86::Gp64>);
                    vwritez(0b11, zasm::x86::rdx, zasm::x86::qword_ptr<zasm::x86::Gp64>);

                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                uint8_t(v0_op::add), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto varith = [&]<typename T>(int size, zasm::x86::Gp v0, zasm::x86::Gp v1, T ptr) {
                        a.bind(labels[1 + size]);
                        a.mov(v0, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.add(vsp, 1 << size);
                        a.mov(v1, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.add(v0, v1);
                        a.mov(ptr(std::forward<zasm::x86::Gp64>(vsp)), v0);
                        a.pushfq();
                        a.pop(v0.r64());
                        a.mov(zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"]), v0.r16());
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vadd"]);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    varith(0b00, zasm::x86::cl, zasm::x86::dl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    varith(0b01, zasm::x86::cx, zasm::x86::dx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    varith(0b10, zasm::x86::ecx, zasm::x86::edx, zasm::x86::dword_ptr<zasm::x86::Gp64>);
                    varith(0b11, zasm::x86::rcx, zasm::x86::rdx, zasm::x86::qword_ptr<zasm::x86::Gp64>);

                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                uint8_t(v0_op::sub), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto varith = [&]<typename T>(int size, zasm::x86::Gp v0, zasm::x86::Gp v1, T ptr) {
                        a.bind(labels[1 + size]);
                        a.mov(v0, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.add(vsp, 1 << size);
                        a.mov(v1, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.sub(v1, v0);
                        a.mov(ptr(std::forward<zasm::x86::Gp64>(vsp)), v1);
                        a.pushfq();
                        a.pop(v0.r64());
                        a.mov(zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"]), v0.r16());
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vsub"]);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    varith(0b00, zasm::x86::cl, zasm::x86::dl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    varith(0b01, zasm::x86::cx, zasm::x86::dx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    varith(0b10, zasm::x86::ecx, zasm::x86::edx, zasm::x86::dword_ptr<zasm::x86::Gp64>);
                    varith(0b11, zasm::x86::rcx, zasm::x86::rdx, zasm::x86::qword_ptr<zasm::x86::Gp64>);

                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                uint8_t(v0_op::bxor), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto varith = [&]<typename T>(int size, zasm::x86::Gp v0, zasm::x86::Gp v1, T ptr) {
                        a.bind(labels[1 + size]);
                        a.mov(v0, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.add(vsp, 1 << size);
                        a.mov(v1, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.xor_(v1, v0);
                        a.mov(ptr(std::forward<zasm::x86::Gp64>(vsp)), v1);
                        a.pushfq();
                        a.pop(v0.r64());
                        a.mov(zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"]), v0.r16());
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vxor"]);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    varith(0b00, zasm::x86::cl, zasm::x86::dl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    varith(0b01, zasm::x86::cx, zasm::x86::dx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    varith(0b10, zasm::x86::ecx, zasm::x86::edx, zasm::x86::dword_ptr<zasm::x86::Gp64>);
                    varith(0b11, zasm::x86::rcx, zasm::x86::rdx, zasm::x86::qword_ptr<zasm::x86::Gp64>);

                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                uint8_t(v0_op::band), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto varith = [&]<typename T>(int size, zasm::x86::Gp v0, zasm::x86::Gp v1, T ptr) {
                        a.bind(labels[1 + size]);
                        a.mov(v0, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.add(vsp, 1 << size);
                        a.mov(v1, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.and_(v1, v0);
                        a.mov(ptr(std::forward<zasm::x86::Gp64>(vsp)), v1);
                        a.pushfq();
                        a.pop(v0.r64());
                        a.mov(zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"]), v0.r16());
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vand"]);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    varith(0b00, zasm::x86::cl, zasm::x86::dl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    varith(0b01, zasm::x86::cx, zasm::x86::dx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    varith(0b10, zasm::x86::ecx, zasm::x86::edx, zasm::x86::dword_ptr<zasm::x86::Gp64>);
                    varith(0b11, zasm::x86::rcx, zasm::x86::rdx, zasm::x86::qword_ptr<zasm::x86::Gp64>);

                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                uint8_t(v0_op::bor), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto varith = [&]<typename T>(int size, zasm::x86::Gp v0, zasm::x86::Gp v1, T ptr) {
                        a.bind(labels[1 + size]);
                        a.mov(v0, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.add(vsp, 1 << size);
                        a.mov(v1, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.or_(v1, v0);
                        a.mov(ptr(std::forward<zasm::x86::Gp64>(vsp)), v1);
                        a.pushfq();
                        a.pop(v0.r64());
                        a.mov(zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"]), v0.r16());
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vor"]);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    varith(0b00, zasm::x86::cl, zasm::x86::dl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    varith(0b01, zasm::x86::cx, zasm::x86::dx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    varith(0b10, zasm::x86::ecx, zasm::x86::edx, zasm::x86::dword_ptr<zasm::x86::Gp64>);
                    varith(0b11, zasm::x86::rcx, zasm::x86::rdx, zasm::x86::qword_ptr<zasm::x86::Gp64>);

                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                uint8_t(v0_op::cmp), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto varith = [&]<typename T>(int size, zasm::x86::Gp v0, zasm::x86::Gp v1, T ptr) {
                        a.bind(labels[1 + size]);
                        a.mov(v0, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.add(vsp, 1 << size);
                        a.mov(v1, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.cmp(v1, v0);
                        a.pushfq();
                        a.pop(v0.r64());
                        a.add(vsp, 1 << size);
                        a.mov(zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"]), v0.r16());
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vcmp"]);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    varith(0b00, zasm::x86::cl, zasm::x86::dl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    varith(0b01, zasm::x86::cx, zasm::x86::dx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    varith(0b10, zasm::x86::ecx, zasm::x86::edx, zasm::x86::dword_ptr<zasm::x86::Gp64>);
                    varith(0b11, zasm::x86::rcx, zasm::x86::rdx, zasm::x86::qword_ptr<zasm::x86::Gp64>);

                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                // [BUG-E-FIX] vtest mirrors vcmp: pop 2 operands, compute flags,
                // store real flags word to vflags global for the following vjcc.
                // (test used to go native via exec_native -> no flags pushed ->
                //  vjcc popped garbage + permanent +2 vsp drift)
                uint8_t(v0_op::test), [&](zasm::x86::Assembler& a) {
                    auto labels = [&]{ std::array<zasm::Label, 6> res; for (auto&x:res) x = a.createLabel(); return res; }();

                    auto varith = [&]<typename T>(int size, zasm::x86::Gp v0, zasm::x86::Gp v1, T ptr) {
                        a.bind(labels[1 + size]);
                        a.mov(v0, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.add(vsp, 1 << size);
                        a.mov(v1, ptr(std::forward<zasm::x86::Gp64>(vsp)));
                        a.test(v1, v0);
                        a.pushfq();
                        a.pop(v0.r64());
                        a.add(vsp, 1 << size);
                        a.mov(zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"]), v0.r16());
                        a.jmp(labels[5]);
                    };

                    get_size_from_opcode(a, global_labels["vtest"]);

                    create_jump_table_once(a, labels[0], labels[1], labels[2], labels[3], labels[4]);
                    jump_using_table(a, labels[0]);

                    varith(0b00, zasm::x86::cl, zasm::x86::dl, zasm::x86::byte_ptr<zasm::x86::Gp64>);
                    varith(0b01, zasm::x86::cx, zasm::x86::dx, zasm::x86::word_ptr<zasm::x86::Gp64>);
                    varith(0b10, zasm::x86::ecx, zasm::x86::edx, zasm::x86::dword_ptr<zasm::x86::Gp64>);
                    varith(0b11, zasm::x86::rcx, zasm::x86::rdx, zasm::x86::qword_ptr<zasm::x86::Gp64>);

                    vm_next_instruction(a, labels[5]);
                }
            },
            {
                uint8_t(v0_op::jmp), [&](zasm::x86::Assembler& a) {
                    a.bind(global_labels["vjmp"]);
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rcx, zasm::x86::word_ptr(vip));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::rcx);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::jz), [&](zasm::x86::Assembler& a) {
                    auto vnext = a.createLabel(), truth = a.createLabel();

                    a.bind(global_labels["vjz"]);
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // ZF = 1?
                    a.and_(zasm::x86::rdx, 0x0040);
                    a.test(zasm::x86::rdx, zasm::x86::rdx);
                    a.jnz(truth);
                    a.add(vip, 2);
                    a.jmp(vnext);
                    
                    a.bind(truth);
                    a.movzx(zasm::x86::rcx, zasm::x86::word_ptr(vip));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::rcx);
                    a.bind(vnext);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::jnz), [&](zasm::x86::Assembler& a) {
                    auto vnext = a.createLabel(), truth = a.createLabel();

                    a.bind(global_labels["vjnz"]);
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // ZF = 0?
                    a.and_(zasm::x86::rdx, 0x0040);
                    a.test(zasm::x86::rdx, zasm::x86::rdx);
                    a.jz(truth);
                    a.add(vip, 2);
                    a.jmp(vnext);
                    
                    a.bind(truth);
                    a.movzx(zasm::x86::rcx, zasm::x86::word_ptr(vip));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::rcx);
                    a.bind(vnext);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::jb), [&](zasm::x86::Assembler& a) {
                    auto vnext = a.createLabel(), truth = a.createLabel();

                    a.bind(global_labels["vjb"]);
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // CF = 1?
                    a.and_(zasm::x86::rdx, 0x0001);
                    a.test(zasm::x86::rdx, zasm::x86::rdx);
                    a.jnz(truth);
                    a.add(vip, 2);
                    a.jmp(vnext);
                    
                    a.bind(truth);
                    a.movzx(zasm::x86::rcx, zasm::x86::word_ptr(vip));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::rcx);
                    a.bind(vnext);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::jnb), [&](zasm::x86::Assembler& a) {
                    auto vnext = a.createLabel(), truth = a.createLabel();

                    a.bind(global_labels["vjnb"]);
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // CF = 0?
                    a.and_(zasm::x86::rdx, 0x0001);
                    a.test(zasm::x86::rdx, zasm::x86::rdx);
                    a.jz(truth);
                    a.add(vip, 2);
                    a.jmp(vnext);
                    
                    a.bind(truth);
                    a.movzx(zasm::x86::rcx, zasm::x86::word_ptr(vip));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::rcx);
                    a.bind(vnext);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::jbe), [&](zasm::x86::Assembler& a) {
                    auto vnext = a.createLabel(), truth = a.createLabel();

                    a.bind(global_labels["vjbe"]);
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // CF = 1?
                    a.and_(zasm::x86::rdx, 0x0001);
                    a.test(zasm::x86::rdx, zasm::x86::rdx);
                    a.jnz(truth);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // or ZF = 1?
                    a.and_(zasm::x86::rdx, 0x0040);
                    a.test(zasm::x86::rdx, zasm::x86::rdx);
                    a.jnz(truth);
                    a.add(vip, 2);
                    a.jmp(vnext);
                    
                    a.bind(truth);
                    a.movzx(zasm::x86::rcx, zasm::x86::word_ptr(vip));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::rcx);
                    a.bind(vnext);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::jnbe), [&](zasm::x86::Assembler& a) {
                    auto vnext = a.createLabel(), truth = a.createLabel(), ntruth = a.createLabel();

                    a.bind(global_labels["vjnbe"]);
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // CF = 0?
                    a.and_(zasm::x86::rdx, 0x0001);
                    a.test(zasm::x86::rdx, zasm::x86::rdx);
                    a.jnz(ntruth);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // and ZF = 0?
                    a.and_(zasm::x86::rdx, 0x0040);
                    a.test(zasm::x86::rdx, zasm::x86::rdx);
                    a.jnz(ntruth);
                    a.jmp(truth);
                    a.bind(ntruth);
                    a.add(vip, 2);
                    a.jmp(vnext);
                    
                    a.bind(truth);
                    a.movzx(zasm::x86::rcx, zasm::x86::word_ptr(vip));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::rcx);
                    a.bind(vnext);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::jl), [&](zasm::x86::Assembler& a) {
                    auto vnext = a.createLabel(), truth = a.createLabel();

                    a.bind(global_labels["vjl"]);
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // or SF != OF?
                    a.and_(zasm::x86::rdx, 0x0880);
                    a.popcnt(zasm::x86::rdx, zasm::x86::rdx);
                    a.cmp(zasm::x86::rdx, 1);
                    a.jz(truth);
                    a.add(vip, 2);
                    a.jmp(vnext);
                    
                    a.bind(truth);
                    a.movzx(zasm::x86::rcx, zasm::x86::word_ptr(vip));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::rcx);
                    a.bind(vnext);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::jle), [&](zasm::x86::Assembler& a) {
                    auto vnext = a.createLabel(), truth = a.createLabel();

                    a.bind(global_labels["vjle"]);
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // ZF = 1?
                    a.and_(zasm::x86::rdx, 0x0040);
                    a.test(zasm::x86::rdx, zasm::x86::rdx);
                    a.jnz(truth);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // or SF != OF?
                    a.and_(zasm::x86::rdx, 0x0880);
                    a.popcnt(zasm::x86::rdx, zasm::x86::rdx);
                    a.cmp(zasm::x86::rdx, 1);
                    a.jz(truth);
                    a.add(vip, 2);
                    a.jmp(vnext);
                    
                    a.bind(truth);
                    a.movzx(zasm::x86::rcx, zasm::x86::word_ptr(vip));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::rcx);
                    a.bind(vnext);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::jnl), [&](zasm::x86::Assembler& a) {
                    auto vnext = a.createLabel(), truth = a.createLabel();

                    a.bind(global_labels["vjnl"]);
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // or SF == OF?
                    a.and_(zasm::x86::rdx, 0x0880);
                    a.popcnt(zasm::x86::rdx, zasm::x86::rdx);
                    a.cmp(zasm::x86::rdx, 1);
                    a.jnz(truth);
                    a.add(vip, 2);
                    a.jmp(vnext);
                    
                    a.bind(truth);
                    a.movzx(zasm::x86::rcx, zasm::x86::word_ptr(vip));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::rcx);
                    a.bind(vnext);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::jnle), [&](zasm::x86::Assembler& a) {
                    auto vnext = a.createLabel(), truth = a.createLabel(), ntruth = a.createLabel();

                    a.bind(global_labels["vjnle"]);
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // ZF = 0?
                    a.and_(zasm::x86::rdx, 0x0040);
                    a.test(zasm::x86::rdx, zasm::x86::rdx);
                    a.jnz(ntruth);
                    a.movzx(zasm::x86::rdx, zasm::x86::word_ptr(zasm::x86::rip, global_labels["vflags"])); // and SF == OF?
                    a.and_(zasm::x86::rdx, 0x0880);
                    a.popcnt(zasm::x86::rdx, zasm::x86::rdx);
                    a.cmp(zasm::x86::rdx, 1);
                    a.jz(ntruth);
                    a.jmp(truth);
                    a.bind(ntruth);
                    a.add(vip, 2);
                    a.jmp(vnext);
                    
                    a.bind(truth);
                    a.movzx(zasm::x86::rcx, zasm::x86::word_ptr(vip));
                    a.lea(vip, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vcode"]));
                    a.add(vip, zasm::x86::rcx);
                    a.bind(vnext);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::call), [&](zasm::x86::Assembler& a) {
                    a.bind(global_labels["vcall"]);
                    a.add(vip, 1);
                    a.mov(zasm::x86::r11, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["retaddr"]));
                    a.movsxd(zasm::x86::r9, zasm::x86::dword_ptr(vip));
                    a.add(zasm::x86::r11, zasm::x86::r9);

                    // push original global_labels["retaddr"] to global_labels["vstack"], incase we vmenter somewhere else
                    a.sub(vsp, 8);
                    a.mov(zasm::x86::r9, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["retaddr"]));
                    a.mov(zasm::x86::qword_ptr(vsp), zasm::x86::r9);

                    // push current vip to global_labels["vstack"], in case we vmenter somewhere else
                    a.add(vip, 4);
                    a.sub(vsp, 8);
                    a.mov(zasm::x86::qword_ptr(vsp), vip);

                    // set vsp offset in case we vmenter somewhere else
                    a.mov(zasm::x86::qword_ptr(zasm::x86::rip, global_labels["_vsp"]), vsp);
                    a.lea(zasm::x86::r9, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vstack"]));
                    a.sub(zasm::x86::qword_ptr(zasm::x86::rip, global_labels["_vsp"]), zasm::x86::r9);

                    // vmexit proc
                    a.popfq();
                    a.pop(zasm::x86::rax);
                    a.pop(zasm::x86::rcx);
                    a.pop(zasm::x86::rdx);
                    a.pop(zasm::x86::rbx);
                    a.pop(zasm::x86::rbp); // rsp
                    a.pop(zasm::x86::rbp);
                    a.pop(zasm::x86::rsi);
                    a.pop(zasm::x86::rdi);
                    a.pop(zasm::x86::r8);
                    a.pop(zasm::x86::r9);
                    a.pop(zasm::x86::r10);
                    // [VCALL-R11-INTENT] r11 must NOT be restored here: it holds the
                    // native call target (r11=[retaddr]+disp computed above) consumed
                    // by `call r11` AFTER this pop sequence. Restoring r11 here would
                    // clobber the target with guest r11 -> call garbage (observed:
                    // guest r11=0x320 -> call 0x320 -> SIGSEGV). Guest r11 is
                    // deliberately sacrificed (caller-saved; native callee may clobber
                    // it anyway). vexit's pop r11;pop r12 stays correct.
                    a.pop(zasm::x86::r12);
                    a.pop(zasm::x86::r12);
                    a.pop(zasm::x86::r13);
                    a.pop(zasm::x86::r14);
                    a.pop(zasm::x86::r15);

                    vm_enter_emitter.revert_effects(a);

                    // [BUG-M v2] No pre-call stash: rsp stays at E through the call;
                    // re-entry restores vsp from the _vsp global (exec_native-style).
                    // (The old [NEST-FIX] stash clobbered guest r10 with the _vsp offset
                    // and re-pushed it into vregs[r10] — Md5Compress keeps b in r10
                    // across Rol32 vcalls, so r10 became 0x736=vsp offset instead of
                    // b=0xefcdab89, corrupting the MD5 rotate.)
                    a.call(zasm::x86::r11);

                    // ===== vcall re-entry: restore VM state + re-save guest regs =====
                    // (BUG-M v2 rewrite) The old [NEST-FIX] stash clobbered guest r10
                    // with the _vsp offset and re-pushed it into vregs[r10]; Md5Compress
                    // keeps b in r10 across Rol32 vcalls -> r10 became 0x736 (vsp offset)
                    // instead of b=0xefcdab89, corrupting the MD5 rotate. exec_native
                    // already restores vsp from the _vsp GLOBAL with no stash and passes;
                    // nested VM regions save/restore _vsp themselves via venter/vexit.
                    // Caller's `sub rsp,0x8` above balanced the ABI; rsp is at E here.
                    vm_enter_emitter.assemble_effects(a); // rsp = saved_rsp (E-0x200)

                    // [NEST-FIX] re-anchor saved_rsp: a nested VM region inside the
                    // callee stomped the global; our save area base is rsp right now.
                    a.mov(zasm::x86::qword_ptr(zasm::x86::rip, global_labels["saved_rsp"]), zasm::x86::rsp);

                    a.push(zasm::x86::r15); // -8
                    a.push(zasm::x86::r14); // -16
                    a.push(zasm::x86::r13); // -24
                    a.push(zasm::x86::r12); // -32
                    // [VCALL-R11-PRESERVE] r11 slot must hold GUEST r11: r11 held the call
                    // target ([retaddr]+disp) and was consumed by `call r11`; callee may
                    // further clobber it (caller-saved). Guest r11 at [saved_rsp-40].
                    // rsp == saved_rsp-32 now (r15..r12 pushed) => [rsp-8] == [saved_rsp-40].
                    a.mov(zasm::x86::r11, zasm::x86::qword_ptr(zasm::x86::rsp, -8));
                    a.push(zasm::x86::r11); // -40  ← guest r11
                    // [BUG-M-FIX] r10 slot must hold GUEST r10 (b in Md5Compress): host r10
                    // may have been clobbered by callee. Guest r10 at [saved_rsp-48] = [rsp-8].
                    a.mov(zasm::x86::r10, zasm::x86::qword_ptr(zasm::x86::rsp, -8));
                    a.push(zasm::x86::r10); // -48  ← guest r10
                    a.push(zasm::x86::r9); // -56
                    a.push(zasm::x86::r8); // -64
                    a.push(zasm::x86::rdi); // -72
                    a.push(zasm::x86::rsi); // -80
                    a.push(zasm::x86::rbp); // -88
                    // [BUG-G-FIX] vregs[rsp] slot holds marker-point guest rsp = E:
                    // revert_effects above returned rsp to saved_rsp = E-0x200; native
                    // code runs at E. Store saved_rsp+0x200 (=E) in the rsp slot.
                    a.mov(zasm::x86::r9, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["saved_rsp"]));
                    a.add(zasm::x86::r9, 0x200);
                    a.push(zasm::x86::r9); // -96  ← vregs[rsp] = E
                    a.push(zasm::x86::rbx); // -104
                    a.push(zasm::x86::rdx); // -112
                    a.push(zasm::x86::rcx); // -120
                    a.push(zasm::x86::rax); // -128
                    a.pushfq(); // -136

                    // restore vsp from the _vsp global (exec_native-style)
                    a.lea(vsp, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vstack"]));
                    a.add(vsp, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["_vsp"]));

                    a.mov(vip, zasm::x86::qword_ptr(vsp));
                    a.add(vsp, 8);
                    a.mov(zasm::x86::r11, zasm::x86::qword_ptr(vsp));
                    a.mov(zasm::x86::qword_ptr(zasm::x86::rip, global_labels["retaddr"]), zasm::x86::r11);
                    // [P7-VCALL-BALANCE] pop the retaddr slot: vcall pushed 16 bytes
                    // (retaddr@[vsp-8] + vip@[vsp-16]) but the re-entry only popped the vip
                    // slot, leaving vsp 8 below its pre-call level FOREVER -> every vcall
                    // permanently drained 8 bytes of the vstack; the selftest's hundreds of
                    // calls underflowed it (_vsp=-11, vip=0xffffff, SIGSEGV at opcode fetch).
                    // exec_native was already balanced (push vip, pop vip); this makes vcall
                    // balanced too (push 16, pop 16).
                    a.add(vsp, 8);

                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::lea), [&](zasm::x86::Assembler& a) {
                    a.bind(global_labels["vlea"]);
                    a.add(vip, 1);
                    a.movsxd(zasm::x86::ecx, zasm::x86::dword_ptr(vip));
                    a.add(vip, 4);
                    a.add(zasm::x86::rcx, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["retaddr"]));
                    a.sub(vsp, 8);
                    a.mov(zasm::x86::qword_ptr(vsp), zasm::x86::rcx);
                    vm_next_instruction(a);
                }
            },
            {
                uint8_t(v0_op::execute_native), [&](zasm::x86::Assembler& a) {
                    a.bind(global_labels["vexenative"]);
                    
                    auto loop = a.createLabel();
                    auto native_code_section = a.createLabel();
                    auto done = a.createLabel();
                    
                    a.add(vip, 1);
                    a.movzx(zasm::x86::rcx, zasm::x86::byte_ptr(vip));
                    a.lea(zasm::x86::rdx, zasm::x86::qword_ptr(zasm::x86::rip, native_code_section));

                    a.add(vip, 1);

                    // to-do: use rep movsb? (bc currently mba is hardcoded to use rdi)
                    //
                    a.bind(loop);
                    a.test(zasm::x86::rcx, zasm::x86::rcx);
                    a.jz(done);
                    a.mov(zasm::x86::r9b, zasm::x86::byte_ptr(vip));
                    a.mov(zasm::x86::byte_ptr(zasm::x86::rdx), zasm::x86::r9b);
                    a.add(zasm::x86::rdx, 1);
                    a.add(vip, 1);
                    a.sub(zasm::x86::rcx, 1);
                    a.jmp(loop);
                    a.bind(done);

                    // push current vip to global_labels["vstack"], in case we vmenter somewhere else
                    a.sub(vsp, 8);
                    a.mov(zasm::x86::qword_ptr(vsp), vip);

                    // set vsp offset in case we vmenter somewhere else
                    a.mov(zasm::x86::qword_ptr(zasm::x86::rip, global_labels["_vsp"]), vsp);
                    a.lea(zasm::x86::r9, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vstack"]));
                    a.sub(zasm::x86::qword_ptr(zasm::x86::rip, global_labels["_vsp"]), zasm::x86::r9);

                    a.popfq();
                    a.pop(zasm::x86::rax);
                    a.pop(zasm::x86::rcx);
                    a.pop(zasm::x86::rdx);
                    a.pop(zasm::x86::rbx);
                    a.pop(zasm::x86::rbp); // rsp
                    a.pop(zasm::x86::rbp);
                    a.pop(zasm::x86::rsi);
                    a.pop(zasm::x86::rdi);
                    a.pop(zasm::x86::r8);
                    a.pop(zasm::x86::r9);
                    a.pop(zasm::x86::r10);
                    // [BUG-J-FIX] restore r11 slot into r11 (was popped into r12 and
                    // overwritten -> guest r11 lost after every vcall/exec_native).
                    // vexit had it right (pop r11; pop r12); vcall/exec_native
                    // mirrored it wrong. Md5Compress keeps kMd5T base in r11 across
                    // exec_native[lea eax] -> base garbage -> kMd5T[i] wrong -> MD5 FAIL.
                    a.pop(zasm::x86::r11);
                    a.pop(zasm::x86::r12);
                    a.pop(zasm::x86::r13);
                    a.pop(zasm::x86::r14);
                    a.pop(zasm::x86::r15);

                    vm_enter_emitter.revert_effects(a);

                    a.bind(native_code_section);
                    for (int i = 0; i < 16; i++)
                        a.nop();

                    vm_enter_emitter.assemble_effects(a);

                    a.push(zasm::x86::r15); // -8
                    a.push(zasm::x86::r14); // -16
                    a.push(zasm::x86::r13); // -24
                    a.push(zasm::x86::r12); // -32
                    a.push(zasm::x86::r11); // -40
                    a.push(zasm::x86::r10); // -48
                    a.push(zasm::x86::r9); // -56
                    a.push(zasm::x86::r8); // -64
                    a.push(zasm::x86::rdi); // -72
                    a.push(zasm::x86::rsi); // -80
                    a.push(zasm::x86::rbp); // -88
                    // [BUG-G-FIX] vregs[rsp] slot must hold the TRUE guest rsp at the
                    // marker point = saved_rsp + 0x200: the stub did sub rsp,0x200
                    // before venter, but exec_native/vcall/vexit run native code at
                    // rsp = saved_rsp + 0x200 (revert_effects undoes the stub's sub).
                    // The old push of [saved_rsp] (= marker_rsp - 0x200) made every
                    // lifted rsp-relative memory access ([rsp+disp] via push_reg 4 +
                    // disp) compute marker_rsp-0x200+disp while native code computed
                    // marker_rsp+disp — 0x200 off (e.g. CeEncode's lea r14,[rsp+0x1f]
                    // native vs write0 [r4+31] lifted pointed at different slots).
                    // r9 is already saved at -56 here, safe to use as scratch.
                    a.mov(zasm::x86::r9, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["saved_rsp"]));
                    a.add(zasm::x86::r9, 0x200);
                    a.push(zasm::x86::r9); // -96  ← vregs[rsp] = marker rsp
                    a.push(zasm::x86::rbx); // -104
                    a.push(zasm::x86::rdx); // -112
                    a.push(zasm::x86::rcx); // -120
                    a.push(zasm::x86::rax); // -128
                    a.pushfq(); // -136

                    a.lea(vsp, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vstack"]));
                    a.add(vsp, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["_vsp"]));

                    a.mov(vip, zasm::x86::qword_ptr(vsp));
                    a.add(vsp, 8);

                    a.lea(zasm::x86::rdx, zasm::x86::qword_ptr(zasm::x86::rip, native_code_section));
                    a.mov(zasm::x86::dword_ptr(zasm::x86::rdx), 0x90909090);
                    a.mov(zasm::x86::dword_ptr(zasm::x86::rdx, 4), 0x90909090);
                    a.mov(zasm::x86::dword_ptr(zasm::x86::rdx, 8), 0x90909090);
                    a.mov(zasm::x86::dword_ptr(zasm::x86::rdx, 12), 0x90909090);

                    vm_next_instruction(a);
                }
            }
        };
    };

    namespace debug {
        void dump_v0(lift_result &result);
    }
}
