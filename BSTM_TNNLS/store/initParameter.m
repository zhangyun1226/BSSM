%% column 相关
columnNum = 512; %% 柱的数量
activeNum = 10;   %% 激活柱的数量
neuronNum = 64;   %% 每个柱的神经元的个数
rate_in = 20;     %% 空间层输入的频率
sysinput = 1000;  %% 空间层输入神经元数量
%% LIF
Tmax = 10;                %周期长度
T_step = 1;               %时间步长
Theta = 1;
Tau_m = 5;               
Tau_s = 1.25;             
Tau_a = 5;               
Eta = Tau_m/Tau_s;
Vnorm = (Eta^(Eta/(Eta-1)))/(Eta-1);
periodicTime = 0:T_step:Tmax; %模拟时间
testPeriodicTime = periodicTime;