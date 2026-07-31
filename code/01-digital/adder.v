// adder.v —— 4 位行波进位加法器 + testbench
// 编译运行：iverilog -o adder_sim adder.v && vvp adder_sim
// 预期输出：PASS: 全部 256 组加法测试通过

module full_adder(input a, b, cin, output sum, cout);
    assign sum  = a ^ b ^ cin;
    assign cout = (a & b) | ((a ^ b) & cin);
endmodule

module adder4(input [3:0] a, b, input cin,
              output [3:0] sum, output cout);
    wire [2:0] c;
    full_adder fa0(a[0], b[0], cin,  sum[0], c[0]);
    full_adder fa1(a[1], b[1], c[0], sum[1], c[1]);
    full_adder fa2(a[2], b[2], c[1], sum[2], c[2]);
    full_adder fa3(a[3], b[3], c[2], sum[3], cout);
endmodule

module adder4_tb;
    reg  [3:0] a, b;
    reg        cin;
    wire [3:0] sum;
    wire       cout;
    integer i, j, errors;

    adder4 dut(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));

    initial begin
        errors = 0;
        cin = 0;
        for (i = 0; i < 16; i = i + 1)
            for (j = 0; j < 16; j = j + 1) begin
                a = i; b = j; #10;
                if ({cout, sum} !== i + j) begin
                    errors = errors + 1;
                    $display("FAIL: %0d + %0d = %0d (期望 %0d)",
                             i, j, {cout, sum}, i + j);
                end
            end
        if (errors == 0)
            $display("PASS: 全部 256 组加法测试通过");
        $finish;
    end
endmodule
