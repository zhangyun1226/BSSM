%% column ���
columnNum = 512; %% ��������
activeNum = 10;   %% ������������
neuronNum = 64;   %% ÿ��������Ԫ�ĸ���
rate_in = 20;     %% �ռ�������Ƶ��
sysinput = 1000;  %% �ռ��������Ԫ����
%% LIF
Tmax = 10;                %���ڳ���
T_step = 1;               %ʱ�䲽��
Theta = 1;
Tau_m = 5;               
Tau_s = 1.25;             
Tau_a = 5;               
Eta = Tau_m/Tau_s;
Vnorm = (Eta^(Eta/(Eta-1)))/(Eta-1);
periodicTime = 0:T_step:Tmax; %ģ��ʱ��
testPeriodicTime = periodicTime;