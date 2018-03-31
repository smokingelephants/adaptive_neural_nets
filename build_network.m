function [ Qnet,lnc] = create_nw_mdp_caller_2(epsilon,samples_i,...
    samples_t,tsamples_i,tsamples_t,last_update,initq,...
    update_sarsa,batchupdate)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% close all
global f;
global layers;
global layers_node_count;
global layerwts;
global inpwts;
global outwts;
global biaswts;
global lbiaswt;
global train_x;
global train_t;
global create_test_x;
global create_test_t;
global current_x;
global current_create_test_x;
global epsilon2;
global bin_size2;
global Q_value2;
global Q_value2_batch;
global explore2;
global store_correct_score;
global call_counter;
global net2;
global prev_layer_acc;
global current_net;
global current_y;
global abstraction_bonus_cutoff;
global oldwts
global showcr;
global createpower
global createpid
global inclay
global createstatecounter
% global test_x;
% global test_t;
% create_test_x=test_x;
% create_test_t=test_t;
store_correct_score=0;
abstraction_bonus_cutoff=0;
%parameter adjust
% epsilon=.9;
% initq=1;
% last_update=0;
% % addpath(genpath('..\dataset_funcs'));
% [samples_i,samples_t,tsamples_i,tsamples_t]=getgen_data(10);%getgen_data(3)
% acc=100-testnet(samples_i,samples_t,tsamples_i,tsamples_t);
% update_sarsa=1;
%parameter adjust
inp_epsilon=epsilon;
% disp(epsilon);
% numgendata=1;
% addpath(genpath('..'));
% acc=-1;
% load 'testdat.mat'
% while(acc==-1)
% rng('shuffle');
% num_classes=randi(10,1,numgendata)+1;
% points_per_class=ceil(randi(4,1,numgendata)*100);
% num_dimensions=randi(5,1,numgendata)+1;
% % points_per_class=200;
% % num_dimensions=2;
% % num_classes=12;
% for i=1:numgendata
% %      load 'testdat.mat'
% %     [a,b,c,d]=generate_class_data(num_classes(i),...
% %         points_per_class(i),num_dimensions(i));
% [a,b,c,d]=circle_data;
%      d_store{i}.x=a;
%      d_store{i}.t=b;
%      d_store{i}.test_x=c;
%      d_store{i}.test_t=d;
%      d_store{i}.acc=testnet(a,b,c,d);
% %      d_store{i}.acc=100;
% end
% acc=100;
% % for i=1:size(d_store,2)
% %     inp_x=d_store{i}.x;
% %     inp_t=d_store{i}.t;
% %     t_x=d_store{i}.test_x;
% %     t_t=d_store{i}.test_t;
% % end
% % subplot(1,2,2)
% % scatter(t_x(1,:),t_x(2,:),[],vec2ind(t_t))
% % drawnow
% % acc=testnet(inp_x,inp_t,t_x,t_t);
% % acc=0;
% end
% disp('ppossible accuracy');
% disp(d_store{1}.acc);
% last_update=0;
% initq=1;
% update_sarsa=1;
net2 = network(2,2,[1;1],[0 1;1 0],[0 0;1 0],[0 1]);
epsilon2=inp_epsilon;
bin_size2=3;
if(initq==1)
    rng('shuffle');
    Q_value2=rand(ceil(100/bin_size2),ceil(100/bin_size2),3);
    createstatecounter=zeros(ceil(100/bin_size2),ceil(100/bin_size2));
    disp('init2 hi this createlord80100')
%     Q_value2(:,:,1)=1.01;
    showcr=1;
    createpower=0;
    inclay=[0,0,0,0];
    Q_value2_batch=zeros(ceil(100/bin_size2),ceil(100/bin_size2),3);
    global createpolicy
    load('createpid80100.mat')
    createpolicy=createpid;
end
if(last_update==1)
    %update
    %need Q2ns
    %need Q2a
    %need Q1ns
    sarsa(state,action,next_state,next_action,reward);
end
if(batchupdate==1)
disp('doing batch')
    dobatch();
end
layers_node_count=0;
explore2=0;
% train_x=inp_x;
% train_t=inp_t;
% create_test_x=t_x;
% create_test_t=t_t;
% % [x,t]=generate_class_data(9,50,12);
% % [x,t]=winequalitywhite_dataset;
% % tempnet=feedforwardnet([6 4]);
% % tempnet=train(tempnet,x,t);
% % tempy=tempnet(x);
% % perform(tempnet,t,tempy)
% % plotconfusion(t,tempy);
% current_x=train_x;
% current_create_test_x=create_test_x;
% layers=[];
% layerwts=[];
% inpwts=[];
% outwts=[];
% biaswts=[];
% lbiaswt=[];
max_episodes=1;
% first_s=add_node();
max_epsilon=epsilon2;
running_avg=0;
dataset=1;
msc=0;
% add_node();
% add_node();
% add_node();
% add_node();
% add_node();
% add_node();
% increase_layer();
% add_node();
% add_node();
% add_node();
% % increase_layer();
% % add_node();
% % add_node();
% % add_node();
% create_final_nw();
call_counter=1;
global min_err
min_err=2;
for episode=1:max_episodes
    showcr=showcr+1;
%     epsilon2=max_epsilon*(max_episodes-episode)/max_episodes;
% %     state=first_s;
% epsilon2=epsilon2*.999;
%     dataset=mod(episode,numgendata)+1;
%     [Xtrain Ytrain Xval Yval Xtest Ytest modelos card_obtained dep_obtained]=ml_generator(10, 400, 600, 1000,...
%     50, 2.5, 0.4, 0.5, 100, 500, 0.80, 0.15, 3, 0.5);
    inp_x=samples_i;
    inp_t=samples_t;
    t_x=tsamples_i;
    t_t=tsamples_t;
    train_x=inp_x;
    train_t=inp_t;
    create_test_x=t_x;
    create_test_t=t_t;
    current_x=train_x;
    current_create_test_x=create_test_x;
    layers=[];
    layerwts=[];
    inpwts=[];
    outwts=[];
    biaswts=[];
    lbiaswt=[];
    abstraction_bonus_cutoff=ceil(100/bin_size2);
    oldwts=0;
%     if(episode>=max_episodes-14)
%         epsilon2=0;
%     end
    goal_reached=0;
    count=0;
    explore2=0;
    same_count=1;
    quit_now=0;
    global numdim
    current_net=0;
    current_x=train_x;
    numdim=max([size(train_x,1),size(create_test_t,1)]);
    current_create_test_x=create_test_x;
    layers=[];
    layerwts=[];
    inpwts=[];
    outwts=[];
    biaswts=[];
    lbiaswt=[];
    layers_node_count=0;
    prev_layer_acc=ceil(100/bin_size2);
%     state=add_first_node();
% add_node(size(train_x,1),6);
% net=feedforwardnet([]);
% net.layers{1}.transferFcn='tansig';
% net.trainParam.showWindow=false;
% net=train(net,train_x,train_t);
% y=net(create_test_x);
% c_mat=confusionmat(vec2ind(create_test_t),vec2ind(y))';
% temp=get_current_state(c_mat);
prev_layer_acc=ceil(100/bin_size2);%temp(1);
state=add_node(6,6);
% state=[ceil(100/bin_size2),ceil(100/bin_size2)];%ceil(100/bin_size2)
    action=get_action(state);
    totalrew=0;
    totaladdvalue=0;
    while(goal_reached==0)
         createstatecounter(state(1),state(2))...
             =createstatecounter(state(1),state(2))+1;
        [next_state,reward,goal_reached]=mdp(state,action,quit_now);
        next_action=get_action(next_state);
        if(update_sarsa==1)
        addvalue=sarsa(state,action,next_state,next_action,reward);
        totaladdvalue=totaladdvalue+abs(addvalue);
        end
        if(state(1)==next_state(1))
            same_count=same_count+1;
            if(same_count>msc)
                msc=same_count;
            end
        else
            same_count=1; %change
        end
        state=next_state;
        action=next_action;
        count=count+1;
%         disp(sum(sum_correct>0))
    if(same_count>=5 || layers_node_count(end)>=100 ||...
            size(layers_node_count,2)>=20)
        quit_now=1;
    end
    totalrew=totalrew+reward;
    end
    if(reward<0)
        r=100+reward;
    else
        r=reward;
    end
%     running_avg(dataset)=(((episode-1)*running_avg(dataset))+r)/episode;
%     running_avg(dataset)=(running_avg(dataset)+r)/2;
acc_value=next_state(1)*bin_size2;
% running_avg(dataset)=running_avg(dataset)+.1*(acc_value-running_avg(dataset));
% running_avg=running_avg+.1*(r-running_avg);
%     if(mod(episode,5)==0)
%         fprintf('\n');
%         disp('epis,correct,steps,total,avg');
%     end
    createpower=createpower+.2*((102-acc_value)-createpower);
    correct_score=ceil(state*bin_size2);
%     fprintf('\n');
%     disp([episode,correct_score,count,sum(layers_node_count),ceil(running_avg(dataset))])
%     disp(explore/count);
%     fprintf(2,['explore: ',num2str(explore/(count+1)),' \n'])
    
%     fprintf(2,[num2str(layers_node_count),' \n'])
%     figure(1);

%    figure(1);set(gcf,'Visible', 'off'); 
%    subplot(2,2,1)
    store_correct_score(episode)=acc_value;
    call_counter=call_counter+1;
%     fprintf('\n')
%     disp([call_counter,dataset,round(acc_value),round(acc),explore2/(count+1),totalrew]) %,explore2/(count+1) d_store{dataset}.acc
%     disp(layers_node_count)
    
%    if(size(store_correct_score,2)>=100)
%        plot(store_correct_score(end-99:end),'r');
%        temp=mean(store_correct_score(end-99:end));
%    else
%        plot(store_correct_score,'r');
%        temp=mean(store_correct_score);
%    end
%    hold on;
%    plot(2,temp,'x');
%    hold off;
%    drawnow
    if(mod(call_counter,50)==0 || call_counter<=5)
        
%         Qnet=create_final_nw();
%         save('Qnet.mat','Qnet');
%         curr_policy=getpolicy();
%         save('createnw.mat','store_correct_score','Qnet','curr_policy');
%         save('chknew.mat','store_correct_score');
%           plot(store_correct_score)
%           drawnow
%         showmenetwork;
%         message=0;
%         send_text_message2('task running network',message,1)
%         plot(curr_policy)
%         drawnow
%         subplot(1,2,1)
%         scatter(t_x(1,:),t_x(2,:),[],vec2ind(current_y))
%         drawnow
    end
    
    if(mod(showcr,40)==0 || showcr==2)
        fprintf('<d>');
        [~,createpid]=max(Q_value2,[],3);
%         sizeQ=size(Q_value2,1);
%         x1 = 1:sizeQ; 
%         x2 = 1:sizeQ;
%         [X1,X2] = meshgrid(x1,x2);
%         figure(1)
%         subplot(1,3,2)
%         gscatter(X1(:),X2(:),reshape(pid,1,sizeQ*sizeQ))
%         subplot(1,3,3)
%         bar([0 createpower 100],.4)
% %         legend(1:3,'a','i','n')
% %         legend('I_K','U_L','jk');
%         drawnow
    end
end
global beststructure;
% disp('******');
% fprintf([num2str(ceil(totaladdvalue)) '\n' ...
    fprintf([num2str(layers_node_count) '(' ...
        num2str(beststructure.layers_node_count) ')\n'])
% disp('******');
Qnet=create_final_nw();

lnc=beststructure.layers_node_count;
% save('Qnet.mat','Qnet');
end


function action=get_action(state)
global epsilon2;
global Q_value2;
global explore2
global createpolicy
r=rand;
%chk if action>3
% if(r>epsilon2)
%     %exploit
%     [~,action]=max(Q_value2(state(1),state(2),:));
% %     [~,action]=max(Q_value2(state,:));
% else
%     %explore
%     explore2=explore2+1;
%     action=randi(2);
% end
% action=1;
action=createpolicy(state(1),state(2));
end


function [next_state,reward,goal_reached]=mdp(current_state,action,quit_now)
global bin_size2;
global prev_layer_acc;
global layers_node_count;
global numdim
global current_net;
global current_y;
global create_test_t;
global abstraction_bonus_cutoff;
global inclay
goal_count=100/bin_size2;
reward=0;
goal_reached=0;
y=zeros(size(create_test_t));
% action=3;
% add_node2;
% increase_layer2();
% add_node2();
% increase_layer2;
% add_node2();
% add_node2();
% increase_layer2;
% add_node2();
% add_node2();
% add_node2;
% action=1;
% action=input('action:');
% while(size(action,2)==0)
%     action=input('action:');
% end
% action=1;
disp(action)
if(action==1)%add best
%     fprintf('n');

        numnew=ceil(numdim/4);
        if(numnew<2)
            numnew=2;
        end
        if(numnew>4)
            numnew=4;
        end
        [next_state]=add_node(numnew,6);

% goal_reached=1;
elseif(action==2)%remove worst
%     fprintf('L');
%     if(layers_node_count(1)~=0)
        prev_layer_acc=current_state(1);
        num_nodes=min(layers_node_count(end),size(create_test_t,1));
        [next_state,y]=increase_layer(num_nodes);

elseif(action==3)%train
    fprintf('<t>');
%    next_state=trainmynet(100);
        next_state=current_state;

    quit_now=1;
end
% disp(layers_node_count)
% disp([action next_state goal_count])

reward=-(next_state(1)-current_state(1))*bin_size2;
% if(next_state(1)==current_state(1))
%     fprintf('s');
% elseif(next_state(1)<current_state(1))
%     fprintf('x');
% end
    
if(action==2)
    if(reward<0)
        inclay(1)=inclay(1)+1;
    end
    inclay(2)=inclay(2)+1;
elseif(action==1)
    if(reward<0)
        inclay(3)=inclay(3)+1;
    end
    inclay(4)=inclay(4)+1;
end
% curr_acc=(ceil(100/bin_size2)-current_state(1));
% if(action==2 && current_state(1)<current_state(2))
%     if(reward<0)
%         reward=1;
%     end
%     abstraction_bonus_cutoff=current_state(1);
%     next_state(2)=abstraction_bonus_cutoff;
% end
% if(action==2 && current_state(1)<current_state(2))
% %     curr_acc=(ceil(100/bin_size2)-current_state(1));
%     val=current_state(2)-current_state(1);
%     reward=reward+(2*val*bin_size2);
%     abstraction_bonus_cutoff=current_state(1);
% %     next_state(2)=abstraction_bonus_cutoff;
%     fprintf('b');
% end
% rewardtemp=0;
% reward=reward*10;
% if(action==1)
%     reward=rewardtemp-10;
% elseif(action==2)
%     reward=rewardtemp+10;
% end
% reward=reward/layers_node_count(end);
% if(reward<0)
%     reward=reward*.2;
% end
%if next_state is goal state, get reward
if(mod(sum(inclay([2,4])),400)==0 && action~=3)
    fprintf(['\ninclay: ' num2str(inclay) '\n']);
    disp(layers_node_count)
end
if(next_state(1)==1 || quit_now==1)
%     reward=get_reward(next_state,goal_count);
%     reward=reward+(-next_state(1)*bin_size2);
%reward=-(next_state(1)-ceil(100/bin_size2))*bin_size2;
current_y=y;
    goal_reached=1;
end

end

function [next_state]=add_node2()
global layers_node_count;
global current_net;
%add one to last layer in layers node count
layers_node_count(end)=layers_node_count(end)+1;
%create final network with extra node
if(size(layers_node_count,2)==1)
    type=1;
else
    type=2;
end
net=create_final_nw2(type);
%train for some epochs
%get state

current_net=net;
next_state=trainmynet(10);
end

function [next_state]=increase_layer2()
global current_net;
global layers_node_count;
layers_node_count(end+1)=1;
net=create_final_nw2(3);
current_net=net;
next_state=trainmynet(10);
end

function [next_state]=trainmynet(num_epochs)
global current_net;
global current_x;
global train_t;
global current_create_test_x;
global create_test_t;
current_net.trainParam.epochs=1000;
current_net.trainParam.showWindow=false;
current_net.inputs{1}.processFcns={};
current_net.outputs{2}.processFcns={};
current_net=configure(current_net,current_x,train_t);
current_net=train(current_net,current_x,train_t);
y=current_net(current_create_test_x);
perform(current_net,create_test_t,y)
c_mat=confusionmat(vec2ind(create_test_t),vec2ind(y))';
next_state=get_current_state(c_mat);
end

function [next_state]=add_first_node()
global current_net;
global layers_node_count;
layers_node_count=1;
current_net=patternnet(layers_node_count);
next_state=trainmynet(3);
end

function [next_state,y]=add_node(num_nodes,maxfail)
% disp('adding node')

global current_x;
global train_t;
global current_create_test_x;
% global create_test_x;
global create_test_t;
global net2;
global layerwts;
global inpwts;
global outwts;
global lbiaswt;
global biaswts;
%add node at topmost layer
global layers_node_count;
global layers;
global oldwts;
% fprintf(['a',num2str(sum(layers_node_count))]);
layer_id=size(layers_node_count,2);
%get current activation outputs to be used as input
layer_out=zeros(1,size(current_x,2));
layer_test_out=zeros(1,size(create_test_t,2));
for i=1:layers_node_count(end)
    layer_out=[layer_out;layers{layer_id}(i).activation_out];
    layer_test_out=[layer_test_out;layers{layer_id}(i).activation_test_out];
end
layer_out=layer_out(2:end,:); 
layer_test_out=layer_test_out(2:end,:); 
%create network
input1_size=size(layer_out,1);
input2_size=size(current_x,1);
net = network(2,2,[1;1],[0 1;1 0],[0 0;1 0],[0 1]);
%net.trainFcn='trainlm';
net.layers{1}.size=1;
net.layers{1}.transferFcn='tansig';

% net=net2;
% netf=feedforwardnet(1);
net.inputs{2}.size=input2_size;
net.layers{1}.size=num_nodes; %one new node
net.trainFcn='trainlm';
net.layers{1}.transferFcn='tansig';
net.layers{2}.transferFcn='tansig';
% 
net.layerWeights{2,1}.learnFcn='learngdm';
net.performFcn = 'mse';
net.inputs{1}.processFcns={};%'mapminmax'
% net.inputs{2}.processFcns={};
net.inputs{1}.size=input1_size;%otherwise mapminmax sets it to 0
net.outputs{1}.processFcns={};%'mapminmax'
net.biases{1,1}.learnFcn='learngdm';
net.biases{2,1}.learnFcn='learngdm';
net.layers{2,1}.initFcn='initnw';
net.layers{1,1}.initFcn='initnw';
net.adaptFcn='adaptwb';
net.inputWeights{1,1}.learnFcn='learngdm';
net.divideFcn='dividerand';
net.biases{1,1}.learnFcn='learngdm';
net.biases{2,1}.learnFcn='learngdm';
net.outputs{1,2}.processFcns={};%'mapminmax'
net.layerWeights{2,1}.learnFcn='learngdm';
% %first feature out then x
train_x=[layer_out;current_x];
create_test_x=[layer_test_out;current_create_test_x];
% %set init functions
net.layers{1}.initFcn='initnw';
net.layers{2}.initFcn='initnw';
net.initFcn='initlay';
net.divideFcn='dividerand';
net.trainParam.max_fail=6;
net.trainParam.showWindow=true;
% net.trainParam.epochs=6000;
% % net=train(net,train_x,t);
% % net=init(net);
% net.trainParam.epochs=1;
% % net=configure(net,train_x,train_t);
net.layers{2,1}.size=size(train_t,1);
% net.IW
% net=init(net);
% net.IW{1,2}=rand(size(net.IW{1,2})).*1e-5;
% p=size(net.IW{2,1},2);
% o=size(train_t,1);
% if(p~=0)
% net.IW{2,1}=rand(o,p).^9;
% end
% net=revert(net);
% net=init(net);
% view(net)
% snet=net;
% isconfigured(snet)
% net=configure(net,'inputs',current_x,2);
% net=configure(net,'inputs',layer_out,1);
% net=configure(net,'outputs',train_t,1);

net.IW{1,2}=randn(size(net.IW{1,2}));
net.LW{2,1}=randn(size(net.LW{2,1})).*1e-1;
% net.IW{2,1}=outwts;
p=size(net.IW{2,1},2);
o=size(train_t,1);
if(p~=0)
net.IW{2,1}=outwts;%rand(o,p).^9;
net.b{2,1}=lbiaswt;
end

% net.LW{1,2}=rand(size(net.LW{1,2})).^9;
% net=configure(net,[layer_out;current_x],train_t);
% view(net)
temp_int_inp=[train_x,create_test_x];
r1=size(train_x,2);
r2=size(create_test_x,2);
r3=size(temp_int_inp,2);
net.divideFcn='divideind';
% 
% [net.divideParam.trainInd,net.divideParam.valInd,...
%     net.divideParam.testInd] = divideind(r3,1:r1,r1+1:r1+ceil(.9*r2),...
%     r1+1+ceil(.9*r2):r3);
% [net.divideParam.trainInd,net.divideParam.valInd,...
%     net.divideParam.testInd] = divideind(r3,1:ceil(.4*r3),...
%     ceil(.4*r3)+1:r3,...
%     r3-10:r3);
% [net.divideParam.trainInd,net.divideParam.valInd,...
%     net.divideParam.testInd] = dividerand(r3,.4,.59,.01);
net.derivFcn='defaultderiv';
% net=train(net,temp_int_inp,[train_t,create_test_t]);
% train_x=[layer_out;current_x];
% create_test_x=[layer_test_out;current_create_test_x];
savedwts=[];
if(p~=0)
savedwts=[outwts,lbiaswt];
end
% savedwts=100*ones(size(savedwts));
mynet=nwbpropcreate2(current_x,train_t,layer_out,...
    current_create_test_x,create_test_t,layer_test_out,...
    num_nodes,savedwts);
% train_x,train_t,X2,val_x,val_t,VX2,numnodes,initwts)
%train harder
%     oldacc=-1;
%     acc=0;
%     countval=0;
%     while(acc>=oldacc  && countval<10)
% %         disp([oldacc acc])
%     if(acc==oldacc)
%         countval=countval+1;
%     else
%         countval=0;
%     end
%     oldacc=acc;
%     oldnet=net;
%     net=train(net,train_x,train_t);
%     y=net(create_test_x);
%     cmat=confusionmat(vec2ind(create_test_t),vec2ind(y));
% 	acc=(sum(diag(cmat))/size(create_test_t,2));
%     end
%     net=oldnet;


%train harder

% netf=configure(netf,train_x,train_t);
% netf.trainParam.showWindow=true;
% netf=train(netf,train_x,train_t);
% y=net(train_x); %there
% if(size(train_x,1)~=size(create_test_x,1))
%     breakhere=1;
% end
% view(net)

% yf=netf(create_test_x);
% net.inputs{2}.range=netf.inputs{1}.range;
numoldnodes=size(layer_out,1);
temp=mynet{2};
bout=temp(:,end);
dualweights=temp(:,1:end-1);
temp2=mynet{1};
temp2=temp2(1:end-1,:);
borl=temp2(:,end);
newnodeinpweights=temp2(:,1:end-1);
oldnodeoutweights=dualweights(:,end-numoldnodes+1:end);
newnodeoutweights=dualweights(:,1:end-numoldnodes);
net.IW{1,2}=newnodeinpweights;
net.IW{2,1}=oldnodeoutweights;
net.LW{2,1}=newnodeoutweights;
net.b{1}=borl;
net.b{2}=bout;
y=net(create_test_x);
% y=net(train_x);
% perform(net,t,y)
% view(net)
% 
% net.layers{1}.size=1;
% net.layers{1}.transferFcn='tansig';
% net.layers{2}.transferFcn='tansig';
% net.trainFcn='trainscg';
% net.layers{2,1}.initFcn='initnw';
% net.layers{1,1}.initFcn='initnw';
% net.inputs{1}.processFcns={'mapminmax'};
% net.adaptFcn='adaptwb';
% net.inputWeights{1,1}.learnFcn='learngdm';
% net.divideFcn='dividerand';
% net.biases{1,1}.learnFcn='learngdm';
% net.biases{2,1}.learnFcn='learngdm';
% net.outputs{1,2}.processFcns={'mapminmax'};
% net.layerWeights{2,1}.learnFcn='learngdm';
% net=configure(net,x,t);


% if(cond==1)
% plotconfusion(t,y)
% end
% c_mat=confusionmat(vec2ind(train_t),vec2ind(y))'; %there
% subplot(1,3,1)
% 
% scatter(current_create_test_x(1,:),current_create_test_x(2,:),[],vec2ind(create_test_t))
% subplot(1,3,2)
% scatter(current_create_test_x(1,:),current_create_test_x(2,:),[],vec2ind(y))
% drawnow
% perform(net,create_test_t,y)
c_mat=confusionmat(vec2ind(create_test_t),vec2ind(y))';
% c_matf=confusionmat(vec2ind(create_test_t),vec2ind(yf))';
% c_mat=confusionmat(vec2ind(train_t),vec2ind(y))';
% disp(diag(c_mat)'./sum(c_mat))
%store new activations
% net.LW
% net.IW
% view(net)
if(layer_id==1)
    inpwts(layers_node_count(end)+1:layers_node_count(end)+num_nodes,:)=net.IW{1,2};
else
    layerwts{layer_id,layer_id-1}(layers_node_count(end)+1:layers_node_count(end)+num_nodes,:)=net.IW{1,2};
end
outwts=[net.IW{2,1},net.LW{2,1}];
biaswts{layer_id,1}(layers_node_count(end)+1:layers_node_count(end)+num_nodes,1)=net.b{1,1};
lbiaswt=net.b{end,1};
net.outputConnect=[1 0];
tempactout=net(train_x);%,'useGPU','yes'


tempacttestout=net(create_test_x);
for i=1:size(tempactout,1)
layers{layer_id}(layers_node_count(end)+1).activation_out=tempactout(i,:);
layers{layer_id}(layers_node_count(end)+1).activation_test_out=...
    tempacttestout(i,:);
layers_node_count(end)=layers_node_count(end)+1;
end
% layers{layer_id}(layers_node_count(end)+1).IW=net.IW{1,2};
% layers{layer_id}(layers_node_count(end)+1).bi=net.b{1,1};


[next_state,curr_err]=get_current_state(c_mat);
global min_err
global beststructure
% global layers;
% global layers_node_count;
% global inpwts;
% global layerwts;
% global outwts;
% global biaswts;
% global lbiaswt;
% global train_x;
% global train_t;
if(curr_err<min_err)
    %save
    beststructure.layers=layers;
    beststructure.layers_node_count=layers_node_count;
    beststructure.inpwts=inpwts;
    beststructure.layerwts=layerwts;
    beststructure.outwts=outwts;
    beststructure.biaswts=biaswts;
    beststructure.lbiaswt=lbiaswt;
    min_err=curr_err;
end
% net3=patternnet(layers_node_count);
% net3=configure(net3,current_x,train_t);
% net3.trainParam.showWindow=false;
% net3=train(net3,current_x,train_t);
% y3=net3(current_create_test_x);
% c_mat3=confusionmat(vec2ind(create_test_t),vec2ind(y3))';
% next_state3=get_current_state(c_mat3);
% disp([next_state(1) next_state3(1)])
% net.outputConnect=[0 1];
% perform(net,create_test_t,y)
end


function [next_state,y]=increase_layer(num_nodes)
% disp('increasing layer')
% fprintf('<i>');
global current_x;
global current_create_test_x;
global train_t;
%add node at topmost layer
global layers_node_count;
global layers;
layer_id=size(layers_node_count,2);
for i=1:layers_node_count(end)
    layer_out(i,:)=layers{layer_id}(i).activation_out;
    layer_test_out(i,:)=layers{layer_id}(i).activation_test_out;
end
% i=1:layers_node_count(end);
% t_out=layers{layer_id}(1:3).activation_out;
current_x=layer_out;
current_create_test_x=layer_test_out;
layers_node_count(end+1)=0;
[next_state,y]=add_node(num_nodes,12);
% for i=1:num_nodes-1
%     add_node();
% end
% [next_state,y]=add_node();
end


function reward=get_reward(current_count,goal_count)
global bin_size2;
% global layers_node_count;
current_in_100=current_count*bin_size2;
if(current_count==goal_count)
    reward=100;
else
    reward=-(100-current_in_100);
%     reward=current_in_100;
end
% reward=current_in_100;
% reward=reward/sum(layers_node_count);
end

function addvalue=sarsa(state,action,next_state,next_action,reward)
global Q_value2;
global Q_value2_batch
alpha=.1;
gamma=.9;
Q_state=Q_value2(state(1),state(2),action);
Q_next_state=Q_value2(next_state(1),next_state(2),next_action);
% Q_state=Q_value2(state,action);
% Q_next_state=Q_value2(next_state,next_action);
addvalue=(alpha*(reward+(gamma*Q_next_state)-Q_state));
% Q_value2_batch(state(1),state(2),action)=Q_value2_batch(state(1),state(2),...
%     action)+addvalue;
Q_value2(state(1),state(2),action)=Q_state+addvalue;
end
function dobatch()
% global Q_value2;
% global Q_value2_batch
% global bin_size2
% Q_value2=Q_value2+Q_value2_batch;
% Q_value2_batch=zeros(ceil(100/bin_size2),ceil(100/bin_size2),2);
end
function sarsa2(state,action,next_state,next_action,reward)
global Q_value2;
alpha=.1;
gamma=.99;
Q_state=gauss_sum(state,action);
% Q_state=Q_value(state(1),state(2),action);
% Q_next_state=Q_value(next_state(1),next_state(2),next_action);
Q_next_state=gauss_sum(next_state,next_action);
value=(alpha*(reward+(gamma*Q_next_state)-Q_state));
% Q_value(state(1),state(2),action)=value;
update_Q_gauss(state,action,value);
end

function value=gauss_sum(state,action)
global Q_value2;
F=getfilter(state);
plane_Q=Q_value2(:,:,action);
value=sum(sum(plane_Q.*F));
% disp([value Q_value(state(1),state(2),action)]);
end

function update_Q_gauss(state,action,value)
global Q_value2;
F=getfilter(state);
% Q_value2(:,:,action)=value.*F;
Q_value2(:,:,action)=Q_value2(:,:,action)+(value.*F);


end

function F=getfilter(state)
global Q_value2;
mu = state;
Sigma = [2.5 .1; .1 2.5];
sizeQ=size(Q_value2,1);
x1 = 1:sizeQ; 
x2 = 1:sizeQ;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
end

function value=percent_bin_it(input_value)
global bin_size2;
value=ceil(input_value/bin_size2);
if(value==0)
    value=1;
end
end

function [next_state,unique_sum_correct]=get_current_state(c_mat)
global current_create_test_x;
global prev_layer_acc;
global abstraction_bonus_cutoff;
goal_count=size(current_create_test_x,2);
goal_count=sum(sum(c_mat));
sum_correct=sum(diag(c_mat));
h=goal_count-sum_correct;
% unique_sum_correct=sum_correct/goal_count;
unique_sum_correct=h/goal_count;
% unique_sum_correct=1-unique_sum_correct;
next_state=[percent_bin_it(unique_sum_correct*100),prev_layer_acc];%,abstraction_bonus_cutoff
end

function net=create_final_nw2(type)
global layers;
global layers_node_count;
global inpwts;
global layerwts;
global outwts;
global biaswts;
global lbiaswt;
global train_x;
global train_t;
global current_net;
layer_id=size(layers_node_count,2);
%type 1 add node to input layer
%type 2 add node to other layer (layerwts only 2nd,3rd last columns need be
%changed)
%type 3 add node to increased layer
inpwts=current_net.IW;
layerwts=current_net.LW;
biaswts=current_net.b(1:end-1,:);
lbiaswt=current_net.b{end,1};
% biaswts(end,:)=[];

net=patternnet(layers_node_count);
net.inputs{1}.processFcns={};
net.outputs{1,end}.processFcns={};
net=configure(net,train_x,train_t);


% outind=size(layers_node_count,2)+1;
% layerwts{outind,outind-1}=outwts;
% layerwts{outind,outind}=[];

%changed
if(type==1)
    inpwts{1,1}(end+1,:)=net.IW{1,1}(end,:);
    layerwts{2,1}(:,end+1)=net.LW{2,1}(:,end);
    biaswts{1,1}(end+1,:)=net.b{1,1}(end,:);
elseif(type==2)
    tempwts=layerwts;
    [r,c]=size(net.LW);
    tempwts{r,c-1}=[tempwts{r,c-1},net.LW{r,c-1}(:,end)]; %for second last column
    tempwts{r-1,c-2}=[tempwts{r-1,c-2};net.LW{r-1,c-2}(end,:)]; %for third last column
    layerwts=tempwts;
    %changed
    biaswts{end,1}=[biaswts{end,1};net.b{end-1,1}(end)]; %changed end-1 to end to incorporate in same if
    lbiaswt=net.b{end,1};
%     biaswts{end,1}=net.b{end-1,1};
    %changed
elseif(type==3)
    tempwts=layerwts;
    [r,c]=size(net.LW);
    tempwts{r,c-1}=net.LW{r,c-1}; %for second last column
    tempwts{r-1,c-2}=net.LW{r-1,c-2}; %for third last column
    layerwts=tempwts;
    layerwts{r,c}=[];
    biaswts{end+1,1}=net.b{end-1,1};
    lbiaswt=net.b{end,1};
end
%changed
net.IW{1,1}=inpwts{1,1};
net.LW=layerwts;
biaswts{end+1,1}=lbiaswt;

net.b=biaswts;
end


function net=create_final_nw()
global beststructure
layers=beststructure.layers;
layers_node_count=beststructure.layers_node_count;
inpwts=beststructure.inpwts;
layerwts=beststructure.layerwts;
outwts=beststructure.outwts;
biaswts=beststructure.biaswts;
lbiaswt=beststructure.lbiaswt;
global train_x;
global train_t;
net=patternnet(layers_node_count);
% net2.inputs{1}.processFcns={'mapminmax'};
% net2.outputs{1,2}.processFcns={'mapminmax'};
% net2=configure(net2,x,t);
% net2.IW{1,1}=inpwts;
% 
% net = network(1,2,[1;1],[1;0],[0 0;1 0],[0 1]);
% net.layers{1}.size=1;
% net.layers{1}.transferFcn='tansig';
% net.layers{2}.transferFcn='tansig';
% net.trainFcn='trainscg';
% net.layers{2,1}.initFcn='initnw';
% net.layers{1,1}.initFcn='initnw';
net.inputs{1}.processFcns={};
% net.adaptFcn='adaptwb';
% net.inputWeights{1,1}.learnFcn='learngdm';
% net.divideFcn='dividerand';
% net.biases{1,1}.learnFcn='learngdm';
% net.biases{2,1}.learnFcn='learngdm';
net.outputs{1,end}.processFcns={};
% net.layerWeights{2,1}.learnFcn='learngdm';
net=configure(net,train_x,train_t);

net.IW{1,1}=inpwts;
outind=size(layers_node_count,2)+1;
layerwts{outind,outind-1}=outwts;
layerwts{outind,outind}=[];
net.LW=layerwts;
biaswts{end+1,1}=lbiaswt;
net.b=biaswts;

% net2.LW=layerwts;
% net2.b=biaswts;

% fiddiff(net,net2)
y=net(train_x);
% net.outputConnect=[1 0];
% f=[layers{1}(1).activation_out;tansig([net.IW{1,1},net.b{1,1}]*[x;ones(1,size(x,2))]);net(x)];
% plotconfusion(t,y);
end

function fiddiff(net,net2)

n = struct(net); n2 = struct(net2);
for fn=fieldnames(n)';
  if(~isequaln(n.(fn{1}),n2.(fn{1})))
    fprintf('fields %s differ\n', fn{1});
  end
end

end

function [unique_sum_correct]=testnet(x,t,test_x,test_t)
% mynet=patternnet(1);
% mynet.trainFcn='trainlm';
% mynet.trainParam.showWindow=true;
% mynet=configure(mynet,x,t);
% mynet=train(mynet,x,t);
% y=mynet(test_x);
% % subplot(1,3,3)
% % scatter(test_x(1,:),test_x(2,:),[],vec2ind(y))
% 
% c_mat=confusionmat(vec2ind(test_t),vec2ind(y))';
% goal_count=size(test_x,2);
% sum_correct=sum(diag(c_mat));
% unique_sum_correct=100*sum_correct/goal_count;
mdl=ClassificationKNN.fit(x',vec2ind(t)');
y=predict(mdl,test_x');
tempy=full(ind2vec(y'));
cmat=confusionmat(vec2ind(test_t),y)';
unique_sum_correct=100*sum(diag(cmat))/size(test_t,2);
% unique_sum_correct=0;
end


function curr_policy=getpolicy()
global Q_value2;
cv=Q_value2(:,:,1)-Q_value2(:,:,2);
cv=abs(cv);
cv=cv/max(max(cv));
cv=imresize(cv,.01);
imshow(cv)
[~,curr_policy]=max(Q_value2,[],3);
end
