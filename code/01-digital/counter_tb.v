// counter_tb.v —— 模 10（BCD）计数器 + testbench
// 编译运行：iverilog -o counter_sim counter_tb.v && vvp counter_sim
// 预期输出：25 拍计数轨迹 + "PASS: 25 拍后 q=5，模 10 循环正确"

module counter_mod10(
    input        clk,
    input        rst_n,    // 异步复位，低有效
    input        en,
    output reg [3:0] q,
    output       carry     // 计到 9 时进位输出
);
    assign carry = en & (q == 4'd9);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            q <= 4'd0;
        else if (en)
            q <= (q == 4'd9) ? 4'd0 : q + 4'd1;
    end
endmodule

module counter_tb;
    reg clk, rst_n, en;
    wire [3:0] q;
    wire carry;
    integer i;

    counter_mod10 dut(.clk(clk), .rst_n(rst_n), .en(en),
                      .q(q), .carry(carry));

    always #5 clk = ~clk;              // 周期 10 个时间单位

    initial begin
        clk = 0; rst_n = 0; en = 0;
        #12 rst_n = 1; en = 1;         // 释放复位后开始计数
        for (i = 0; i < 25; i = i + 1) begin
            @(posedge clk); #1;
            $display("cycle=%0d q=%0d carry=%b", i, q, carry);
        end
        if (q == 4'd5)
            $display("PASS: 25 拍后 q=5，模 10 循环正确");
        else
            $display("FAIL: q=%0d", q);
        $finish;
    end
endmodule
