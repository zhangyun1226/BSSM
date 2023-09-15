columnNum = 512;   %柱的数量列表
activeNum = 10;
neuronNum = 32;
segmentConnect = 7;                       %每个segment和激活神经元相连的个数
maxSegmentConnect = 4*segmentConnect;     %每个segments最多记忆4个上文，且不要共用
maxSim = 0.5;
minSegmentCon = ceil(segmentConnect*maxSim);
%% STDP para
tau = 25;               %% STDP的参数
%% LIF
Theta = 1;
Tau_m = 10;
Tau_s = 2.5;
Tau_a = 5;
Eta = Tau_m/Tau_s;
Vnorm = (Eta^(Eta/(Eta-1)))/(Eta-1);

T = 100;
dt = 1;
nt = round(T/dt);
%% theta  oscillation
fDG = 10;
through = 1;
peak = T;
period = T;
%% choose data
ca3Ivalue = -0.1*Vnorm;