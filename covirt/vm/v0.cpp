#include "v0.hpp"

#include <stack>
#include <utils/log.hpp>

void covirt::vm::v0_lifter::push_address(covirt::zydis_operand &operand)
{
    // [BUG-I-FIX] SIB addressing: push base + index*scale + disp.
    // Old code pushed only base + disp, silently DROPPING the index register and
    // scale — every `[rbp+rdx*4+disp]`-style access read [base+disp] instead of
    // [base + index*scale + disp]. Md5Compress indexes kMd5S[i]/kMd5T[i]/m[g] via
    // rbp/r11/rsp + rdx*4 -> all loads hit element 0 -> wrong MD5 -> MgSign FAIL.
    // vcode has no shl/mul; emulate index*scale by adding index `scale` times.
    // Zydis mem.scale stores the raw multiplier (1,2,4,8) — verified on artifact:
    // [rbp+rdx*4] produced 16 adds with `1<<scale` (scale field==4) => wrong;
    // loop exactly `scale` times gives 4 adds. scale<=8 => <=8 adds.
    e.push_reg(8, uint8_t(operand.register_index()))
     .push_imm(8, uint64_t(operand.as_memory().disp.value))
     .add(8);
    if (operand.as_memory().index != ZYDIS_REGISTER_NONE) {
        const unsigned scale = operand.as_memory().scale;
        for (unsigned k = 0; k < scale; ++k) {
            e.push_reg(8, uint8_t(operand.register_index(true)))
             .add(8);
        }
    }
}

void covirt::vm::v0_lifter::push_operand(covirt::zydis_operand &operand, std::optional<int> override_size)
{
    int size = override_size.value_or(operand.size);
    
    if (operand.is_register())
        e.push_reg(size, uint8_t(operand.register_index()));
    else if (operand.is_immediate())
        e.push_imm(size, e.cast(operand.immediate(), size));
    else /* if (operand.is_memory()) */ {
        push_address(operand);
        e.read(size);
    }
}

void covirt::vm::v0_lifter::pop_operand(covirt::zydis_operand &operand, std::optional<int> override_size, std::optional<covirt::zydis_operand> src)
{
    int size = override_size.value_or(operand.size);
    auto src_operand = src.value_or(covirt::zydis_operand());

    if (operand.is_register())
        e.pop(size, uint8_t(operand.register_index()));
    else /* if (operand.is_memory()) */ {
        if (src_operand.is_empty() || src_operand.is_immediate()) {
            e.pop(size, uint8_t(tmp_reg_idx));
            push_address(operand);
            e.write(size, uint8_t(tmp_reg_idx));
        }
        else {
            push_address(operand);
            e.write(size, uint8_t(src_operand.register_index()));
        }
    }
}

void covirt::vm::v0_vm::initialize(zasm::x86::Assembler &a)
{
    for (auto& [name, label] : global_labels)
        label = a.createLabel(name.c_str());

    vip = zasm::x86::rax;
    vsp = zasm::x86::rsi;
    
    a.section(".text");
}

void covirt::vm::v0_vm::finalize(zasm::x86::Assembler& a) 
{
    a.section(".data", zasm::Section::Attribs::Data);

    a.bind(global_labels["vcode"]); a.db(0, code_size);
    a.bind(global_labels["saved_rsp"]); a.dq(0);
    a.bind(global_labels["_vsp"]); a.dq(stack_size);
    a.bind(global_labels["_vip"]); a.dq(0);
    // [VSTACK-GUARD] 在 _vip 与 vstack 之间插入保护区:
    // 布局为 vcode | saved_rsp | _vsp | _vip | vstack | retaddr | vflags | vtable,
    // 而操作数栈(vsp)从 vstack 顶部向下增长 —— 一旦出现压/弹不配平(历史上有过
    // vcall 每次泄漏 8 字节的实例, 修复后 PE 侧仍有残留路径), 越界写会直接命中
    // _vip/_vsp/saved_rsp: 表现为 _vsp 变成垃圾值(实测 225/ -11), 随后
    // "add vsp, [_vsp]" 得到野指针, 下一次寄存器保存区访问即 0xC0000005,
    // 且只在某个平台/某段字节码上复现, 极难定位。
    // 加 4KB 保护区后: 小幅漂移只落在填充区, 关键全局不再被破坏; 漂移更大时
    // 会在保护区中留下痕迹便于诊断。
    a.bind(global_labels["vstack_guard"]); a.db(0, 4096);
    a.bind(global_labels["vstack"]); a.db(0, stack_size);
    a.bind(global_labels["retaddr"]); a.dq(0);
    a.bind(global_labels["vflags"]); a.dq(0);

    a.bind(global_labels["vtable"]);
    // [VTABLE-SIZE-FIX] 跳转表按 opcode 索引: vm_next_instruction 里
    //   movzx rcx, byte [vip]; and cl, 0b00111111   → 索引 0..63
    // 而 venter 的 create_jump_table_once 会写入 28 个入口(224 字节)。
    // 原实现只分配 8 项(64 字节): 填表时越界写 160 字节, 覆盖紧随其后的
    // 数据/代码。PE 布局下这些被踩的字节随后被使用, 表现为
    // "第一次进入虚拟化区域正常, 第二次进入即 0xC0000005 访问违例"
    // (Linux 布局恰好把越界区落在填充区, 所以一直没暴露)。
    // 必须按 6 位 opcode 全空间分配 64 项 = 512 字节。
    a.dq(0, 64);
}

void covirt::vm::v0_vm::vm_next_instruction(zasm::x86::Assembler& a, std::optional<zasm::Label> label) 
{
    if (label.has_value())
        a.bind(label.value());

    a.movzx(zasm::x86::rcx, zasm::x86::byte_ptr(vip));
    a.and_(zasm::x86::cl, 0b00111111);
    a.lea(zasm::x86::r9, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["vtable"]));
    a.jmp(zasm::x86::qword_ptr(zasm::x86::r9, zasm::x86::rcx, 8));
}

void covirt::vm::v0_vm::jump_using_table(zasm::x86::Assembler& a, zasm::Label &paths)
{
    a.lea(zasm::x86::r9, zasm::x86::qword_ptr(zasm::x86::rip, paths));
    a.jmp(zasm::x86::qword_ptr(zasm::x86::r9, zasm::x86::rcx, 8));

    a.bind(paths);
    a.dq(0, 4);
}

void covirt::vm::v0_vm::get_size_from_opcode(zasm::x86::Assembler& a, zasm::Label &start)
{
    a.bind(start);
    a.movzx(zasm::x86::rcx, zasm::x86::byte_ptr(vip));
    a.shr(zasm::x86::cl, 6);
    a.add(vip, 1);
}

void covirt::vm::v0_vm::get_vreg_address(zasm::x86::Assembler& a)
{
    a.mov(zasm::x86::r9, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["saved_rsp"]));
    a.sub(zasm::x86::r9, 16*8);
    a.movzx(zasm::x86::r10, zasm::x86::byte_ptr(vip));
    a.lea(zasm::x86::rdx, zasm::x86::qword_ptr(zasm::x86::r9, zasm::x86::r10, 8));
    a.add(vip, 1);
}

void covirt::vm::v0_vm::get_vreg_value(zasm::x86::Assembler& a)
{
    a.mov(zasm::x86::r9, zasm::x86::qword_ptr(zasm::x86::rip, global_labels["saved_rsp"]));
    a.sub(zasm::x86::r9, 16*8);
    a.movzx(zasm::x86::r10, zasm::x86::byte_ptr(vip));
    a.mov(zasm::x86::rdx, zasm::x86::qword_ptr(zasm::x86::r9, zasm::x86::r10, 8));
    a.add(vip, 1);
}

// to-do: ugly code, refactor plz, this was rush job
//
void covirt::vm::debug::dump_v0(lift_result &result)
{
    auto bytes = result.bytes;
    auto equivs = result.dump_index_table;

    static char suffix[] = { 'b', 'w', 'd', 'q' };
    int x = 0;
    int r = 0;

    std::stack<std::string> expression_stack;

    std::println("");
    std::println("| off | idx | lifted from                          | vm instruction             | expression");
    std::println("|-----|-----|--------------------------------------|----------------------------|---------------------------");

    for (int i = 0; i < bytes.size();) {
        uint8_t opcode = bytes[i] & 0b00111111;
        uint8_t size = 1 << (bytes[i] >> 6);

        std::print("| {:>12} | {:>12} | {:<36} | ", out::red(i), out::red(x), equivs.contains(x) ? equivs[x] : "");
        x++;

        switch (opcode) {
        using enum v0_op;
        case int(vm_exit):
            std::println("{:<26} | goto {} + {}", "vmexit", out::purple("retaddr"), out::value(*(uint16_t*)(&bytes[i + 1])));
            i += 2;
            break;
        case int(push_imm):
            std::print("push{} ", suffix[bytes[i] >> 6]);
            switch (size) {
            case 1: std::println("{:<29} | ", out::value_hex(*(uint8_t*)(&bytes[i + 1]))); expression_stack.push(out::value(*(int8_t*)(&bytes[i + 1]))); break;
            case 2: std::println("{:<29} | ", out::value_hex(*(uint16_t*)(&bytes[i + 1]))); expression_stack.push(out::value(*(int16_t*)(&bytes[i + 1]))); break;
            case 4: std::println("{:<29} | ", out::value_hex(*(uint32_t*)(&bytes[i + 1]))); expression_stack.push(out::value(*(int32_t*)(&bytes[i + 1]))); break;
            case 8: std::println("{:<29} | ", out::value_hex(*(uint64_t*)(&bytes[i + 1]))); expression_stack.push(out::value(*(int64_t*)(&bytes[i + 1]))); break;
            }
            i += size;
            break;
        case int(push_reg):
            std::println("{:<35} | ", std::format("push{} {}", suffix[bytes[i] >> 6], out::green(std::format("v{}", bytes[i + 1]))));
            if (bytes[i + 1] == 5) expression_stack.push(out::green(std::format("vbp", bytes[i + 1])));
            else expression_stack.push(out::green(std::format("v{}", bytes[i + 1])));
            i++;
            break;
        case int(pop):
            std::print("{:<35} | ", std::format("pop{} {}", suffix[bytes[i] >> 6], out::green(std::format("v{}", bytes[i + 1]))));
            {
                auto a = expression_stack.top(); expression_stack.pop();
                std::println("{} = ({}){}", out::green(std::format("v{}", bytes[i + 1])), out::yellow(std::format("u{}", size * 8)), a); 
            }
            i++;
            break;
        case int(read):
            std::print("{:<26} | ", std::format("read{}", suffix[bytes[i] >> 6]));
            {
                auto a = expression_stack.top(); expression_stack.pop();
                std::println("{} = *({}*)({})", out::purple(std::format("t{}", r)), out::yellow(std::format("u{}", size * 8)), a); 
            }
            expression_stack.push(out::purple(std::format("t{}", r++)));
            break;
        case int(write):
            std::print("{:<35} | ", std::format("write{} {}", suffix[bytes[i] >> 6], out::green(std::format("v{}", bytes[i + 1]))));
            {
                auto a = expression_stack.top(); expression_stack.pop();
                std::println("*({}*)({}) = {}", out::yellow(std::format("u{}", size * 8)), a, out::green(std::format("v{}", bytes[i + 1]))); 
            }
            i++;
            break;
        case int(add):
            std::print("{:<26} | ", std::format("add{}", suffix[bytes[i] >> 6]));
            { 
                auto a = expression_stack.top(); expression_stack.pop();
                auto b = expression_stack.top(); expression_stack.pop();
                std::println("{} = {} + {}", out::purple(std::format("t{}", r)), b, a); 
            }
            expression_stack.push(out::purple(std::format("t{}", r++)));
            break;
        case int(sub):
            std::print("{:<26} | ", std::format("sub{}", suffix[bytes[i] >> 6]));
            { 
                auto a = expression_stack.top(); expression_stack.pop();
                auto b = expression_stack.top(); expression_stack.pop();
                std::println("{} = {} - {}", out::purple(std::format("t{}", r)), b, a); 
            }
            expression_stack.push(out::purple(std::format("t{}", r++)));
            break;
        case int(bxor):
            std::print("{:<26} | ", std::format("xor{}", suffix[bytes[i] >> 6]));
            { 
                auto a = expression_stack.top(); expression_stack.pop();
                auto b = expression_stack.top(); expression_stack.pop();
                std::println("{} = {} ^ {}", out::purple(std::format("t{}", r)), b, a); 
            }
            expression_stack.push(out::purple(std::format("t{}", r++)));
            break;
        case int(cmp):
            std::println("{:<26} | ", std::format("cmp{}", suffix[bytes[i] >> 6]));
            expression_stack.push(out::purple(std::format("flags", r++)));
            break;
        case int(test):
            std::println("{:<26} | ", std::format("test{}", suffix[bytes[i] >> 6]));
            expression_stack.push(out::purple(std::format("flags", r++)));
            break;
        case int(jz) ... int(jnle):
            {
                auto a = expression_stack.top(); expression_stack.pop();
                std::println("{:<26} | using {} goto {}", "jcc", a, out::red(*(uint16_t*)(&bytes[i + 1])));
                i += 2;
            }
            break;
        case int(jmp):
            {
                std::println("{:<26} | goto {}", "jmp", out::red(*(uint16_t*)(&bytes[i + 1])));
                i += 2;
            }
            break;
        case int(call):
            {
                std::println("{:<26} | goto {}", "call", out::red(*(uint16_t*)(&bytes[i + 1])));
                i += 4;
            }
            break;
        case int(lea):
            {
                std::println("{:<26} | goto {}", "lea", out::red(*(uint16_t*)(&bytes[i + 1])));
                i += 4;
            }
            break;
        case int(execute_native):
            {
                std::println("{:<26} | ", "exe_native");
                i += bytes[i + 1] + 1;
            }
            break;
        default:
            std::println("{:<35} | ", std::format("(bad:{:x})", bytes[i]));
            break;
        }

        i++;
    }

    std::println("");
}
