
clear
clc
fid=fopen('cbt_10_words.txt');  %���ı��ļ�
idx=0;
inputLablesCell = {};
sequenceCell = {};
text = cell(1000, 1);
num = 0;
while ~feof(fid)
    str = fgetl(fid);      % ��ȡһ��, str���ַ���
    s=regexp(str,'\s+');   % �ҳ�str�еĿո�, �Կո���Ϊ�ָ����ݵ��ַ�
    s = [0,s];             % ��һ�����ʵĿո�Ϊ0
    num = num+1;
    if num >1000
        break;
    end
    tempStr = {};
    for i=1:length(s)      % ���þ�����е���ȫ��������
        %% 1.1 ��ȡ��i���ʵ��ַ���
        if i~=length(s)
            currentStr = str(s(i)+1:s(i+1)-1);
        else
            currentStr = str(s(i)+1:end);
        end
        tempStr{1,i} = currentStr;
    end
    text{num} = tempStr;
end