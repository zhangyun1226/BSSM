function [flag, locs] = issubset(a, b)
% 判断向量a是否属于b的某一行（列），若属于，则返回其行（列）数；否则返回空
flag = 0;
locs = [];
[m1, n1] = size(a);
[m2, n2] = size(b);
if n1 == 1 && m1 == m2  % a为列向量，且行数与b相同
    p = repmat(a, 1, n2) - b;
    locs = find(all(p == 0, 1) == 1);
    if ~isempty(loc)
        flag = 1;
    end 
elseif m1 == 1 && n1 == n2  % a为行向量，且列数与b相同
    p = repmat(a, m2, 1) - b;
    locs = find(all(p == 0, 2) == 1);
    if ~isempty(locs)
        flag = 1;
    end 
else
    fprintf('Dimension mismatch');
end
end
