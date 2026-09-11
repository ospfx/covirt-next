#include "binary.hpp"
#include <utils/log.hpp>

covirt::binary::binary(const std::string &file) : generic(LIEF::Parser::parse(file)), out_path(file + suffix) 
{
    switch (generic->format()) {
    case LIEF::Binary::FORMATS::ELF:
        specific = dynamic_cast<LIEF::ELF::Binary*>(generic.get());
        break;
    case LIEF::Binary::FORMATS::PE:
        specific = dynamic_cast<LIEF::PE::Binary*>(generic.get());
        break;
    default:
        out::assertion(false, "unknown file format");
    }
}

void covirt::binary::add_section(const std::string &string, std::vector<uint8_t> content, bool ex, bool wr)
{
    using enum lief_elf_flags_t;
    using enum lief_pe_flags_t;

    std::visit([&](auto &&x){
        using T = std::decay_t<decltype(x)>;

        if constexpr (std::is_same_v<T, LIEF::ELF::Binary*>) {
            LIEF::ELF::Section new_section(string.c_str());
            new_section.flags(uint64_t(ex ? EXECINSTR : NONE) | uint64_t(wr ? WRITE : NONE));
            new_section.content(content);
            x->add(new_section);
        }
        else if constexpr (std::is_same_v<T, LIEF::PE::Binary*>) {
            LIEF::PE::Section new_section(string.c_str());
            new_section.characteristics(uint64_t(ex ? MEM_EXECUTE : lief_pe_flags_t(0)) | uint64_t(wr ? MEM_WRITE : lief_pe_flags_t(0)));
            new_section.content(content);
            x->add_section(new_section);
        }
    }, specific);

    update();
}

lief_sections_iterator_t covirt::binary::sections()
{
    return std::visit([](auto&& x) { return dynamic_cast<LIEF::Binary*>(x)->sections(); }, specific);
}

bool covirt::binary::is_section_executable(lief_section &section)
{
    return std::visit([&](auto&& x) { 
        using T = std::decay_t<decltype(x)>;

        if constexpr (std::is_same_v<T, LIEF::ELF::Binary*>) 
            return as_elf(section)->has(lief_elf_flags_t::EXECINSTR);
        else if constexpr (std::is_same_v<T, LIEF::PE::Binary*>) 
            return as_pe(section)->has_characteristic(lief_pe_flags_t::MEM_EXECUTE);
    }, specific);
}

uint64_t covirt::binary::imagebase()
{
    return std::visit([](auto&& x) { return x->imagebase(); }, specific);
}

lief_section *covirt::binary::get_section(const std::string &name)
{
    return std::visit([&](auto&& x) {
        auto sections = x->sections();
        auto it = std::find_if(sections.begin(), sections.end(), [&](lief_section &section) { return section.name() == name.c_str(); });

        // 未找到时返回 nullptr(由调用方断言), 不能解引用 end()(UB -> SIGSEGV)
        if (it == sections.end()) return static_cast<lief_section *>(nullptr);
        return dynamic_cast<lief_section*>(&(*it));
    }, specific);
}

lief_section *covirt::binary::get_section(uint64_t address)
{
    return std::visit([&](auto&& x) {
        using T = std::decay_t<decltype(x)>;

        // [PE-VA-FIX] 调用方传入的是绝对 VA(marker VA = imagebase + RVA), 而
        // LIEF PE::Section::virtual_address() 返回的是 RVA, 必须补 imagebase
        // 才能落在同一地址空间; ELF 的 Section::virtual_address() 已含
        // imagebase, 不能重复加。
        // 修复前(无 bias) PE 永远匹配不到 -> 解引用 end() -> SIGSEGV, 表现为
        // Windows PE 虚拟化从未成功(产物里没有 .covirt0 段)。
        uint64_t bias = 0;
        if constexpr (std::is_same_v<T, LIEF::PE::Binary*>)
            bias = x->imagebase();

        auto sections = x->sections();
        auto it = std::find_if(sections.begin(), sections.end(), [&](lief_section &section) {
            auto va_start = bias + section.virtual_address();
            return address >= va_start && address < va_start + section.size();
        });

        if (it == sections.end()) return static_cast<lief_section *>(nullptr);
        return dynamic_cast<lief_section*>(&(*it));
    }, specific);
}

void covirt::binary::update()
{
    std::visit([this](auto&& x) { x->write(out_path); }, specific);
}

void covirt::binary::write_vm_entries(std::vector<covirt::subroutine> &routines, covirt::generic_vm_enter &vm_enter)
{
    auto vm_section = get_section(".covirt0");
    out::assertion(vm_section != nullptr, "vm section (.covirt0) not found");
    auto section_of_block = get_section(routines[0].start_va);
    out::assertion(section_of_block != nullptr, "section containing first protected block not found");
    auto base = is_elf() ? section_of_block->virtual_address() : (imagebase() + section_of_block->virtual_address());

    std::vector<uint8_t> content;
    auto cc = section_of_block->content();
    content.assign(cc.begin(), cc.end());

    for (auto & routine : routines) {
        for (uintptr_t i = routine.start_va; i < routine.end_va; i++)
            content[i - base] = covirt::rand<uint8_t>();

        auto offset = routine.start_va - base - __covirt_vm_stub_length;

        vm_enter.set_call_offset(base, is_elf() ? vm_section->virtual_address() : (imagebase() + vm_section->virtual_address()), offset);
        vm_enter.set_vm_bytecode_offset(routine.offset_into_lift);

        auto vm_enter_bytes = vm_enter.get_bytes();

        std::memcpy(&content[offset], vm_enter_bytes.get(), vm_enter.get_length());
        section_of_block->content(content);
    }

    section_of_block->content(content);
}

void covirt::binary::write_vm_bytecode(std::vector<uint8_t> &lifted_bytes, std::vector<uint8_t> &vm_section_bytes, size_t data_start, size_t vcode_size)
{
    auto vm_section = get_section(".covirt0");
    out::assertion(vm_section != nullptr, "vm section (.covirt0) not found");
    std::memcpy(&vm_section_bytes[data_start], &lifted_bytes[0], lifted_bytes.size());
    for (int i = 0; i < vcode_size - lifted_bytes.size(); i++)
        // [VCODE-TRAP] 填充改为可辨认的陷阱 opcode(见 v0.hpp kTrapOpcode):
        // 原实现用随机字节, 导致"越过字节码末尾"时行为随构建而变(有时恰好能跑完,
        // 有时立刻崩), 缺陷不可复现。填成陷阱后, 越界执行会稳定地产生 SIGILL。
        vm_section_bytes[data_start + lifted_bytes.size() + i] = 63;
    vm_section->content(vm_section_bytes);
    update();
}