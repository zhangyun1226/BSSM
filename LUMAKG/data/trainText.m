
clear
clc
fid=fopen('cbt_10_words.txt');  %打开文本文件
idx=0;
inputLablesCell = {};
sequenceCell = {};
text = cell(1000, 1);
num = 0;
while ~feof(fid)
    str = fgetl(fid);      % 读取一行, str是字符串
    s=regexp(str,'\s+');   % 找出str中的空格, 以空格作为分割数据的字符
    s = [0,s];             % 第一个单词的空格为0
    num = num+1;
    if num >1000
        break;
    end
    tempStr = {};
    for i=1:length(s)      % 将该句的所有单词全部存下来
        %% 1.1 获取第i单词的字符串
        if i~=length(s)
            currentStr = str(s(i)+1:s(i+1)-1);
        else
            currentStr = str(s(i)+1:end);
        end
        tempStr{1,i} = currentStr;
    end
    text{num} = tempStr;
end