#include "disasm.hpp"

#include <utils/log.hpp>
#include <print>

covirt::zydis_operand::zydis_operand(ZydisDecodedOperand &op)
{
    assign(op);
}

void covirt::zydis_operand::assign(ZydisDecodedOperand &op)
{
    size = op.size / 8;
    switch (op.type) {
    case ZYDIS_OPERAND_TYPE_REGISTER: value = op.reg; break;
    case ZYDIS_OPERAND_TYPE_IMMEDIATE: value = op.imm; break;
    case ZYDIS_OPERAND_TYPE_MEMORY: value = op.mem; break;
    default: value = std::monostate{}; break;
    }
}

int covirt::zydis_operand::register_index(bool use_index)
{
    auto reg = ZYDIS_REGISTER_NONE;

    if (is_register()) reg = as_register().value;
    if (is_memory()) reg = !use_index ? as_memory().base : as_memory().index;
    
    out::assertion(reg != ZYDIS_REGISTER_NONE, "zydis_operand isn't a register");

    switch (reg) {
    // [BUG-H-FIX] Zydis GPR8 enum order is AL,CL,DL,BL,AH,CH,DH,BH,SPL,BPL,SIL,DIL,
    // R8B..R15B — AH..BH sit between BL and SPL, so `reg - ZYDIS_REGISTER_AL` maps
    // SPL/BPL/SIL/DIL/R8B..R15B onto the WRONG vreg slots (SIL->10 instead of 6, etc).
    // 8-bit regs must map to the physical 64-bit GPR index (AL=rax=0, SPL=rsp=4,
    // BPL=rbp=5, SIL=rsi=6, DIL=rdi=7, R8B=8..R15B=15). AH/CH/DH/BH have no separate
    // vreg slot; keep legacy `reg-AL` mapping (4/5/6/7) for compatibility.
    case ZYDIS_REGISTER_AL: return 0;
    case ZYDIS_REGISTER_CL: return 1;
    case ZYDIS_REGISTER_DL: return 2;
    case ZYDIS_REGISTER_BL: return 3;
    case ZYDIS_REGISTER_SPL: return 4;
    case ZYDIS_REGISTER_BPL: return 5;
    case ZYDIS_REGISTER_SIL: return 6;
    case ZYDIS_REGISTER_DIL: return 7;
    case ZYDIS_REGISTER_R8B: return 8;
    case ZYDIS_REGISTER_R9B: return 9;
    case ZYDIS_REGISTER_R10B: return 10;
    case ZYDIS_REGISTER_R11B: return 11;
    case ZYDIS_REGISTER_R12B: return 12;
    case ZYDIS_REGISTER_R13B: return 13;
    case ZYDIS_REGISTER_R14B: return 14;
    case ZYDIS_REGISTER_R15B: return 15;
    case ZYDIS_REGISTER_AH ... ZYDIS_REGISTER_BH: return reg - ZYDIS_REGISTER_AL;
    case ZYDIS_REGISTER_AX ... ZYDIS_REGISTER_R15W: return reg - ZYDIS_REGISTER_AX;
    case ZYDIS_REGISTER_EAX ... ZYDIS_REGISTER_R15D: return reg - ZYDIS_REGISTER_EAX;
    case ZYDIS_REGISTER_RAX ... ZYDIS_REGISTER_R15: return reg - ZYDIS_REGISTER_RAX;
    default: return -1;
    }
}

void covirt::disasm(std::span<const uint8_t> content, uint64_t base_address, std::function<void(uint64_t, ZydisDisassembledInstruction)> callback)
{
    ZydisDisassembledInstruction ins;
    size_t offset = 0;

    while (ZYAN_SUCCESS(ZydisDisassembleIntel(
        ZYDIS_MACHINE_MODE_LONG_64,
        base_address,
        &content[0] + offset,
        content.size() - offset,
        &ins
    ))) {
        callback(base_address, ins);
        
        base_address += ins.info.length;
        offset += ins.info.length;
    }
}
