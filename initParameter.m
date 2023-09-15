columnNum = 512;   %���������б�
activeNum = 10;
neuronNum = 32;
segmentConnect = 7;                       %ÿ��segment�ͼ�����Ԫ�����ĸ���
maxSegmentConnect = 4*segmentConnect;     %ÿ��segments������4�����ģ��Ҳ�Ҫ����
maxSim = 0.5;
minSegmentCon = ceil(segmentConnect*maxSim);
%% STDP para
tau = 25;               %% STDP�Ĳ���
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