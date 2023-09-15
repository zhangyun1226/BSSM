function [flag, locs] = issubset(a, b)
% �ж�����a�Ƿ�����b��ĳһ�У��У��������ڣ��򷵻����У��У��������򷵻ؿ�
flag = 0;
locs = [];
[m1, n1] = size(a);
[m2, n2] = size(b);
if n1 == 1 && m1 == m2  % aΪ����������������b��ͬ
    p = repmat(a, 1, n2) - b;
    locs = find(all(p == 0, 1) == 1);
    if ~isempty(loc)
        flag = 1;
    end 
elseif m1 == 1 && n1 == n2  % aΪ����������������b��ͬ
    p = repmat(a, m2, 1) - b;
    locs = find(all(p == 0, 2) == 1);
    if ~isempty(locs)
        flag = 1;
    end 
else
    fprintf('Dimension mismatch');
end
end
