% 1 I; 2 have; 3 a; 4 monkey; 5 my; 6 is; 7 very; 8 small; 9 it; 10 lovely;
% 11 likes; 12 to; 13 sit; 14 on; 15 head;
% �����ı��ĵ�
% ͳ�ƾ��Ӹ���M�����ظ����ʸ���N
M = 4;
N = 15;
dT = 20;
T = 8*dT;              % ÿ�����Ӳ�����8������
eta = zeros(N,1);       % ÿ�����ʵĵ�����
delta = zeros(N, N);    % ÿ�Ե���֮���ͻ��Ч��
sentences = zeros(N, M); 
sentences(:,1) = [0; 1*dT; 2*dT; 3*dT; inf(11,1)];
sentences(:,2) = [inf; inf; inf; 1*dT; 0; 2*dT; 3*dT; 4*dT; inf(7,1)];
sentences(:,3) = [inf(5,1); 1*dT; 2*dT; inf; 0; 3*dT; inf(5,1)];
sentences(:,4) = [inf(4,1); 5*dT; inf; inf; inf; 0; inf; 1*dT; 2*dT; 3*dT; 4*dT; 6*dT];
w = zeros(N, N);        % ÿ�Ե���֮���ͻ��ǿ��
for i = 1: 4
    to = sentences(:,i);
%     ANAKG;
    STDP;
end
%%
% subplot(121)
% [x,y] = meshgrid(1:1:15);
% surf(x,y,w),xlabel('x'),ylabel('y'),zlabel('z')
% subplot(122)
% [x,y] = meshgrid(1:1:15);
% surf(x,y,w),xlabel('x'),ylabel('y'),zlabel('z')