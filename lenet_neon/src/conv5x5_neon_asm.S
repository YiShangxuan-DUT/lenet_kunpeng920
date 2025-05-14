        .text
        .global  kern25_fma4
        .type    kern25_fma4, %function
/* ---------------------------------------------------------------- *
 *  5×5 SAME 卷积：一次计算连续 4 像素
 *  参数：x0 in, x1 ker, x2 out, w3 ic, w4 iw, w5 ih, v0.s[0] bias
 *  只用 caller-saved 浮点寄存器；通用寄存器 x20/x21/x22 用但**先保存**
 * ---------------------------------------------------------------- */

kern25_fma4:
        /* ---- 保存被我们要用到的 callee-saved 通用寄存器 ---- */
        stp     x20, x21, [sp, #-16]!
        stp     x22, x23, [sp, #-16]!   // x23 仅占位保证 16 字节对齐

        cbz     w3,  ret_zero

        lsl     x8,  x4, #2             // row_stride = iw * 4
        mov     x20, x0                 // in_base
        mov     x21, x1                 // ker_base
        mov     w22, w3                 // ic counter

        dup     v16.4s, v0.s[0]         // 累加器 v16 <- bias

0:      /* ========= per-IC loop ========= */
        mov     x9,  x20                // in_ptr
        mov     x10, x21                // ker_ptr
        mov     w11, #5                 // kh = 5

1:      /* ---------- 5 kernel rows ---------- */
        ld1     {v0.4s}, [x9]           // r0[0..3]
        add     x12, x9, #16
        ld1     {v1.4s}, [x12]          // r0[4..7]

        ext     v2.16b, v0.16b, v1.16b, #4
        ext     v3.16b, v0.16b, v1.16b, #8
        ext     v4.16b, v0.16b, v1.16b, #12

        ld1r    {v17.4s}, [x10], #4
        fmla    v16.4s, v0.4s, v17.4s   // k00
        ld1r    {v17.4s}, [x10], #4
        fmla    v16.4s, v2.4s, v17.4s   // k01
        ld1r    {v17.4s}, [x10], #4
        fmla    v16.4s, v3.4s, v17.4s   // k02
        ld1r    {v17.4s}, [x10], #4
        fmla    v16.4s, v4.4s, v17.4s   // k03
        ld1r    {v17.4s}, [x10], #4
        fmla    v16.4s, v1.4s, v17.4s   // k04

        add     x9,  x9,  x8            // 下一行
        subs    w11, w11, #1
        b.ne    1b

        /* ---- 下一 IC ---- */
        lsl     x12, x4, #2
        mul     x12, x12, x5            // bytes per IC
        add     x20, x20, x12
        mov     x21, x10
        subs    w22, w22, #1
        b.ne    0b

        /* 写回 4 像素 */
        st1     {v16.4s}, [x2]

ret_zero:
        /* ---- 恢复寄存器 & 返回 ---- */
        ldp     x22, x23, [sp], #16
        ldp     x20, x21, [sp], #16
        ret
        .size   kern25_fma4, .-kern25_fma4
