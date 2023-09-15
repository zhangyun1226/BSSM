clear;
hold off
load figure2.mat; %% change
y1 = v_background;
y2 = v_context;
y3 = depolar;
y4 = v_inhibit;
y5 = v;

len = 6;
dt = 0.1;
simtime = 1:1:1000;
gcf1 = figure(1);
set(gcf1,'color','w');
%% pic A
subplot(5,1,1);
% y1 = -sin(simtime);
plot(simtime,y1);
ylabel('Oscillation'); 
xlabel('Time');
Ts = (0:len-1)*200;
xticks(sort(Ts));
for i = 1:len
    hold on
    plot([(i-1)*200 (i-1)*200],[-1,1],'k--');
    tickx{i} = int2str((i-1)*2);
    tickx{i} = [tickx{i},'¦Ð'];
end
set(gca,'XTickLabel',tickx);
xlim([-10, 1010]);
% plot([ti(1),ti(1)],[1,y1(ti(1))])
hold off
%% pic B
subplot(5,1,2);
theta = .5;
% y2 = rand(length(simtime),1);%% change!!!!
x2 = simtime;
plot(x2,y2(4:5,:));
ylabel('Dendrite Potential'); 
xlabel('Time');
hold on 
plot([0 200*len],[theta,theta],'r--');
Ty2 = [0,theta];
yticks(Ty2);
ticky = [{'0'},{'¦È'}];
xticks(sort(Ts));
set(gca,'YTickLabel',ticky);
set(gca,'XTickLabel',tickx);
ylim([-.1,max(y2(:)) + .1]);
xlim([-10, 1010]);
hold off
% xlim([4*100,6*100]);
%% pic C
subplot(5,1,3);
fireTime = 2.5*2*pi; %%change
% y3 = 0.5*ones(length(simtime),1);
% y3index = round(fireTime/dt);
% y3(1:y3index) = 0;
plot(simtime,y3(4:5,:));
ylabel('depolarization'); 
xlabel('Time');
xticks(sort(Ts));
set(gca,'XTickLabel',tickx);
xlim([-10, 1010]);
%% pic D
subplot(5,1,4);
% x4 = 0:dt:2*pi;
% y4 = exp(-x4); %% change
x4 = simtime;
plot(x4,y4(4:5,:));
ylabel('inhibition'); 
xlabel('Time');
xticks(sort(Ts));
set(gca,'XTickLabel',tickx);
xlim([-10, 1010]);
ylim([-2, .5]);
hold off
%% pic E
subplot(5,1,5);
theta = 1.0;
x5 = simtime;
% y5 = rand(length(x5),1);%% change!!!!
plot(x5,y5(4:5,:));
ylabel('Membrane potential'); 
xlabel('Time');
hold on 
plot([0 2*100*len],[theta,theta],'r--');
Ty2 = [0,theta];
yticks(Ty2);
ticky = [{'0'},{'¦È'}];
xticks(sort(Ts));
set(gca,'YTickLabel',ticky);
set(gca,'XTickLabel',tickx);
xlim([-10,1010]); 
% legend('A','B','C','D','E','location','north','Orientation','horizontal')