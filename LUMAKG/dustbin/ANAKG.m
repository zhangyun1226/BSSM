temp = to'-to;
temp(temp<=0) = inf;
temp(isnan(temp)) = inf;
tC = 20;
tR = 300;
theta = 1;
tau = 4;
temp_delta = (1./(1+(temp-tC)./(theta*tR))).^tau;
delta = delta + temp_delta;     % ÿ����һ�����ӣ��ʶԼ����ڴ�ǰ�Ļ������ۼ�

temp_eta = zeros(size(to));
temp_eta(to~=inf) = 1;
eta = eta + temp_eta;       % ÿ�����ʳ��ֵĴ����ۼ�

w = theta * (eta .* delta) ./ (eta .* delta + eta.^2 - delta.^2 );  % Ȩ�ز��ۼƣ����¼���
w(isnan(w)) = 0;