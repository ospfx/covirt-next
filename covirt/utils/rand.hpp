#pragma once

#include <cstdint>
#include <random>
#include <ranges>

namespace covirt {
    template <typename T>
    inline T rand()
    {
        // [DETERMINISM-FIX] 固定种子: 原实现用 std::random_device 播种, 导致
        // 每次构建的填充字节/随机取值都不同 —— 同一份源码两次构建得到的虚拟化产物
        // 行为不一致(实测: 同一 pin 一次 selftest 全绿, 一次在步骤1 崩),
        // 既让缺陷不可复现、也让"修复是否有效"无法判定。
        // 该随机值用于填充 vcode 缓冲区尾部等位置, 一旦 VM 越过 lift 出的字节码末尾
        // 就会把填充字节当指令执行 —— 固定种子后行为可复现, 便于定位真正原因。
        static std::mt19937 generator(0xC0FFEEu);

        if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
            static std::uniform_int_distribution<T> distribution(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
            return distribution(generator);
        }
        else {
            return T(rand<int>() & 0xff);
        }
    }
}
