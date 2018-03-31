function [ output_args ] = caller_multilayer( input_args )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% global f;
global curr_id;
global best_id;
global worst_id;
global final_ids;
global Q_value;
global epsilon;
global node_arr;
global sum_correct;
global bin_size;
global explore;
global x;
global t;
global tx;
global tt;
global vx;
global vt;
global num_new;
global current_layer;
global node_per_layer;
global test_x;
global test_t;
global current_test_x;
global current_test_t;
global train_sum_correct;
global initq;
global store_correct_score;
global call_counter;
global d_store
global oldwts24;
global createpid;
global typepid;
global createpower;
global createpowerdat
global netpresent
global redo_x;
global redo_t;
global batchupdate
global showoldacc;%useless
global createstatecounter
showoldacc=[];
netpresent=0;
store_correct_score=1;
call_counter=1;
initq=1;
epsilon=0.9;
bin_size=6;
Q_value=rand(ceil(100/bin_size),ceil(100/bin_size),3);
statecounter=zeros(ceil(100/bin_size),ceil(100/bin_size));
disp('init1 selectlordtest')
% Q_value(:,:,3)=1.001;
% Q_value(:,1:4,3)=.002;
current_layer=1;
node_per_layer=0;
curr_id=1;
best_id=0;
worst_id=0;
addpath(genpath('..'));
% [x,t]=simpleclass_dataset;%simpleclass_dataset
% [x,t]=circle_data();
% f=xlsread('train.csv');
% [x,t]=winequalitywhite_dataset();
dataset=1;
num_new=1;
% [x,t,test_x,test_t]=abalone_dataset();
% [d_store{2}.x,d_store{2}.t,d_store{2}.test_x,d_store{2}.test_t]=getsimpleclass_data();
% [d_store{1}.x,d_store{1}.t,d_store{1}.test_x,d_store{1}.test_t]=getthyroid_data();
% [d_store{2}.x,d_store{2}.t,d_store{2}.test_x,d_store{2}.test_t]=circle_data();
% [d_store{1}.x,d_store{1}.t,d_store{1}.test_x,d_store{1}.test_t]=poker_dataset();


train_sum_correct=zeros(1,size(x,2));
final_ids=[];
node_arr=zeros(1,0);

% calculate_sum_correct();

% numgendata=50;
% num_classes=randi(10,1,numgendata)+2;
% points_per_class=ceil((randi(1000,1,numgendata)+7000)./num_classes);
% num_dimensions=randi(10,1,numgendata)+1;
% for i=1:numgendata
%     [a,b,c,d,accsvm,accnet]=generate_class_data(num_classes(i),...
%         points_per_class(i),num_dimensions(i));
% %      [a,b,c,d]=getgen_data(i);
%      d_store{i}.x=a;
%      d_store{i}.t=b;
%      d_store{i}.test_x=c;
%      d_store{i}.test_t=d;
%      d_store{i}.accsvm=accsvm;
%      d_store{i}.accnet=accnet;
%      disp([accsvm accnet])
% end
% load 'fullset.mat'
% for i=1:1000
% a=d_store{i}.x;
% b=d_store{i}.t;
% c=d_store{i}.test_x;
% d=d_store{i}.test_t;
% [accsvm,accnet]=getacc(a,b,c,d);
% d_store{i}.accsvm=accsvm;
% disp([i accsvm d_store{i}.accnet]);
% end
% 
% save('fullset.mat','d_store');
% [d_store,avgacc,numgendata]=gettargettedset(-1,30);
load('newset26.mat')
which('newset26.mat')
numgendata=80;
avgacc=.4;
% [d_store1,avgacc,numgendata]=gettargettedset(.3,0);
% [d_store2,avgacc,numgendata]=gettargettedset(1,1);
% d_store=[d_store1,d_store2];
%for i=1:numgendata
%    d_store{i}.x=d_store{i}.x(:,1:500);
%    d_store{i}.t=d_store{i}.t(:,1:500);
%    d_store{i}.test_x=d_store{i}.test_x(:,1:125);
%    d_store{i}.test_t=d_store{i}.test_t(:,1:125);
%end
%d_store=d_store(randperm(size(d_store,2)));
numgendata=size(d_store,2);
disp('data used')
disp([numgendata avgacc]);

% save('dstore83_new.mat','d_store');
%knntest;
global selectpolicy
load('selectpidfixed.mat')
selectpolicy=selectpid;
for i=1:size(d_store,2)
targetaccs(i)=d_store{i}.accnet;
end
% timetest()
% testdata();
num_datasets=size(d_store,2);
start=152;
scramble=start:start+49;

first_s=[1,ceil(100/bin_size)];
msc=0;
max_episodes=6000;

reward=0;
r=0;
running_avg=-1*ones(1,num_datasets);
meanoverlap=-1*ones(1,num_datasets);
max_epsilon=epsilon;
% num_datasets=ceil(.85*numgendata);
num_datasets=ceil(1*numgendata);
disp(['training size:' num2str(num_datasets)])
disp('epis,dataset,correct,reward,used/total,avg,overlap');
global update_sarsa;
global old_actual_sum;
global maxcreate;
global saved_net;
global addcallcount;
global addfailcount;
global indclassaccs
global oldtiewts
addcallcount=0;
addfailcount=0;
update_sarsa=1;
    for i=1:size(d_store,2)
        targetaccarr(i,:)=[d_store{i}.accsvm];
    end
    
    acctable=num2cell(zeros(1,203));
for episode=1:max_episodes
%     update_sarsa=0;

   epsilon=max_epsilon*(max_episodes-episode)/max_episodes;
    if(episode>max_episodes-200)
        epsilon=0.000;
        num_datasets=numgendata;
        update_sarsa=0;
    end
    
    explore=0;
    curr_id=1;
    best_id=0;
    worst_id=0;
    current_layer=1;
    node_per_layer=0;
    final_ids=[];
    node_arr=zeros(1,0);
    createpowerdat=-1;
    maxcreate=0;
    redo_x=[];
    redo_t=[];
    saved_net=0;
% 
%     if(mod(episode,4)==0)
%         [x,t]=abalone_dataset;%simpleclass_dataset
%         dataset=1;
%     elseif(mod(episode,4)==1)
%         [x,t]=thyroid_dataset;%iris_dataset;
%         dataset=2;
%     elseif(mod(episode,4)==2)
%         [x,t]=circle_data();%thyroid_dataset;
%         dataset=3;
%     elseif(mod(episode,4)==3)
%         [x,t]=winequalitywhite_dataset();
%         dataset=4;
%     end
% if(episode<num_datasets+1)
%     dataset=mod(episode,num_datasets)+1;
% else
%     dataset=randi(num_datasets);
% end
% dataset=mod(episode,num_datasets)+1;
dataset=scramble(mod(episode,50)+1);
% dataset=10;
%     x=dataset_id{dataset}.x;
%     t=dataset_id{dataset}.t;
% dataset=3;
% if(size(d_store{dataset}.x,2)>200)
%     y=randsample(size(d_store{dataset}.x,2),200);
% else
%     y=1:size(d_store{dataset}.x,2);
% end
    x=d_store{dataset}.x;
    t=d_store{dataset}.t;
    tx=d_store{dataset}.test_x;
    tt=d_store{dataset}.test_t;
    vx=d_store{dataset}.val_x;
    vt=d_store{dataset}.val_t;
    test_x=d_store{dataset}.test_x;
    test_t=d_store{dataset}.test_t;
    [current_test_x,current_test_t]=gettestsample(.2,1);
    train_sum_correct=zeros(1,size(x,2));
    sum_correct=zeros(1,size(current_test_x,2));
%     calculate_sum_correct();
    old_actual_sum=sum_correct>0;
    [samples_i,samples_t]=getsample(0,.05);
    [tsamples_i,tsamples_t]=gettestsample(.2,1);
%     node_arr=createnew(samples_i,samples_t,tsamples_i,tsamples_t,num_new);
%     oldwts24=node_arr.network.LW(end,end-1);
%     oldwts24{1}
%     oldwts24
    final_ids=[];
    reevaluate(.2,0,0);%for best id
    state=first_s;
    action=get_action(state);
    goal_reached=0;
    count=0;
    same_count=1;
    quit_now=0;
    totalrew=0;
    sarsaupdate=0;
    indclassaccs=0;
    if(mod(episode,num_datasets)==0)%num_datasets
        batchupdate=1;
    else
        batchupdate=0;
    end
%     mdp(state,1,quit_now);
%     figure(1)
%     imshow(imresize(statecounter>0,20));
%     drawnow
% action=1
oldtiewts=[];
maxstate=-1;
    while(goal_reached==0)      
%         action=1;
        statecounter(state(1),state(2))=statecounter(state(1),state(2))+1;
        [next_state,reward,goal_reached]=mdp(state,action,quit_now);
%         goal_reached=1;
        next_action=get_action(next_state);
        if(update_sarsa==1)
        sarsaupdate=sarsaupdate+sarsa(state,action,next_state,next_action,reward);
        if(action==1)
            workmode=1;
            totalacc=Q_value(next_state(1),next_state(2),next_action);
            select_network_type(epsilon,0,0,0,...
                0,0,0,0,workmode,...
                0,totalacc)
        end
        end
%         sum(state==next_state)==size(next_state,2)
        if(state(1)==next_state(1))%state(1)==next_state(1)
            same_count=same_count+1;
            if(same_count>msc)
                msc=same_count;
            end
        else
            same_count=1;%change
        end
        state=next_state;
        
        [~,testacc]=reevaluate(1,1,0,2);
	if(testacc>maxstate)
		maxstate=testacc;
	end
%          disp([action next_state(1)])
        action=next_action;
        count=count+1;
       
%         disp(sum(sum_correct>0))
    if(same_count>=5)%change
        goal_reached=1;
    end
%     if(size(final_ids,2)>=40)
%         goal_reached=1;
%     end
    totalrew=totalrew+reward;
%     [~,tempact_acc]=reevaluate(1,1,0,2);
%     disp(tempact_acc)
%     goal_reached=1;
    end
    nws=[];
    for i=1:size(final_ids,2)
        temp=node_arr(i).layers_node_count;
        nws(i,1:size(temp,2))=temp;
    end
    networkschosen{dataset}=nws;
    nwcount(dataset)=size(final_ids,2);
    if(episode>1)
        wtstatus(episode,:)=wtstatus(episode-1,:);
    end
    wtstatus(episode,dataset)=state(2);
    sarsalist(episode)=sarsaupdate;
    if(totalrew>(next_state(1)*bin_size))
        breakhere=1;
        disp('reward be higher')
    end
    create_acc_dataset(dataset)=createpowerdat;
    if(reward<0)
        r=next_state(1)*bin_size;
    else
        r=next_state(1)*bin_size;
    end
%     running_avg(dataset)=(((episode-1)*running_avg(dataset))+r)/episode;
%     running_avg(dataset)=(running_avg(dataset)+r)/2;
if(running_avg(dataset)<=2)
    running_avg(dataset)=r;
else
%     running_avg(dataset)=running_avg(dataset)+.2*(r-running_avg(dataset));
    running_avg(dataset)=r;
end
fprintf('\n')
    if(mod(episode,5)==0)
        disp('epis,dataset,correct,reward,used/total,avg,overlap');
    end
    temp=0;
%     if(size(final_ids,2)>0)
%         temp=node_arr(final_ids(1)).layer_id;
%     end
%     correct_score=ceil(100*sum(sum_correct>0)/size(test_x,2));
% if(size(final_ids,2)==0)
%     fprintf('E')
% mdp(state,1,quit_now);
% end
[correct_score,act_acc]=reevaluate(1,1,0,2);%ceil(next_state*bin_size);
running_avg(dataset)=floor(act_acc*100);
acctable{dataset}(end+1)=(act_acc*100);
if(meanoverlap(dataset)~=-1)
    meanoverlap(dataset)=(meanoverlap(dataset))+.5*(correct_score(2)-meanoverlap(dataset));
else
    meanoverlap(dataset)=correct_score(2);
end
% disp([episode dataset])
    disp([episode,dataset,floor(act_acc*100),ceil(totalrew),size(final_ids,2),size(node_arr,2),ceil(running_avg(dataset)),correct_score(2),floor(100*maxcreate),maxstate])%create_acc_dataset(dataset)
%     disp([explore/count epsilon]);
    fprintf(2,[num2str(explore/count),' \n'])
    node_per_layer=[node_per_layer(2:end),size(final_ids,2)];
%     disp(node_per_layer);
    fprintf(2,[num2str(node_per_layer),' \n'])
    accvals=[d_store{dataset}.accsvm d_store{dataset}.accnet].*100;
%     [data.accnet,data.acclda,data.accknn,data.accsvm];
    fprintf('net\t\t\t  lda\t\t\tknn\t\t\t  svm\n')
    fprintf(2,[num2str(d_store{dataset}.accsvm .* 100),' \n'])
    fprintf(2,['addfails:',num2str(addfailcount/(addcallcount+.00001)),' ',num2str(addfailcount),' ',num2str(addcallcount),' \n']);
    store_correct_score2(episode,:)=running_avg(:);
    store_sum_overlap(episode)=sum(meanoverlap(1:num_datasets));
%     disp([(d_store{dataset}.accarr(end)) episode dataset])
%     f=figure('Visible','off');
 %   f=figure(1);set(gcf,'Visible', 'off'); 
 %   subplot(2,2,[3 4])
 %   plot(store_correct_score2);
 %   subplot(2,2,2)
 %   plot(store_sum_overlap)
 %   drawnow
    
%     if(mod(episode,100)==0)
% %         save('mystore.mat','store_correct_score2','store_sum_overlap');
%  %       saveas(f,'123','jpg')
%  %       message=num2str([episode 41831 running_avg]);
%  %       send_text_message('task running',message,1)
%     end
    if(mod(episode,10)==0 || episode==1 ||episode==max_episodes)
%         [~,selectpid]=max(Q_value,[],3);
        save('acctablelordtestwostopbig4.mat','acctable');
        save('mystore83_newlordtestwostopbig4.mat','store_correct_score2',...
            'selectpid','createpid','typepid','createpower',...
            'targetaccs','sarsalist','num_datasets','nwcount',...
            'wtstatus','statecounter','networkschosen','createstatecounter');
%         fprintf('<w>');
%         networkschosen;
%         currentacc=store_correct_score2(end,1:num_datasets)';
%         accdiffs=currentacc-floor(targetaccarr(1:num_datasets).*100);
%         meanerror=mean(abs((accdiffs(accdiffs<0))));
%         message=num2str(meanerror);
%         send_text_message(num2str(episode),message,0)
%         showstatus
%         sizeQ=size(Q_value,1);
%         x1 = 1:sizeQ; 
%         x2 = 1:sizeQ;
%         [X1,X2] = meshgrid(x1,x2);
%         figure(1)
%         subplot(1,3,1)
%         gscatter(X1(:),X2(:),reshape(pid,1,sizeQ*sizeQ))
%         drawnow
    end
%     [g,gid]=max(Q_value(1:20,1:20,:));
end
%saveas(f,'123','fig')
%message=num2str(running_avg);
disp(msc)
%send_text_message('task complete',message,0)

% [samples_i,samples_t]=getsample();
% node_arr=createnew(samples_i,samples_t,num_new);
% final_ids=1:num_new;
% recalculate_num_correct();
% sum_correct=calculate_sum_correct();
%  sum(sum_correct>0)
% [samples_i,samples_t]=getsample();
% node_arr=createnew(samples_i,samples_t,num_new);
% 
% for i=1:size(node_arr,2)
%     disp(node_arr(i).count_correct)
% end
for i=1:numgendata
    disp([d_store{i}.accsvm d_store{i}.accnet store_correct_score2(end,i)]);
end
% save('dstore.mat','d_store');
end

function action=get_action(state)
global selectpolicy
global epsilon;
global Q_value;
global explore
r=rand;
%chk if action>3
% if(r>epsilon)
%     %exploit
%     [~,action]=max(Q_value(state(1),state(2),:));
% else
%     %explore
%     explore=explore+1;
%     action=randi(3);
% end
action=selectpolicy(state(1),state(2));
% action=1;
end

function [samples_i,samples_t]=getsample(include_redo,amt)
global x;
global t;
global tx;
global tt;
global d_store
global saved_net;
global netpresent
global final_ids
global node_arr
global redo_x;
global redo_t;
% global sum_correct;
global train_sum_correct;

totalset=[x;t]';
total_tset=[tx;tt]';
total_size=size(x,2);
sample_amt=amt;
sample_size=ceil(size(totalset,1)*sample_amt);
% sample_size=amt;
% include_redo=0;
redo_amt=.5;
if(include_redo==1)
    if(size(redo_x,2)<redo_amt*sample_size)
        sample_size=sample_size-size(redo_x,2);
    else
        redo_size=ceil(redo_amt*sample_size);
        sample_size=sample_size-redo_size;
        redo_x=redo_x(:,1:redo_size);
        redo_t=redo_t(:,1:redo_size);
    end
end
    
% if(size(totalset,1)>100)
% sample_size=100;
% else
%     sample_size=size(totalset,1);
% end
% 
% % sum_correct=calculate_sum_correct();
% max_sum_correct=max(train_sum_correct);
% if(max_sum_correct==0)
%     sample_prob=ones(1,total_size)./total_size;
% else
%     sp=((max_sum_correct+1)-train_sum_correct);
% %     sp=1./(sum_correct+.01);
%     sample_prob=sp/sum(sp);
% end
% sample_prob=ones(1,total_size)./total_size;
% [samples,inds]=datasample(totalset,sample_size,'Weights',sample_prob','Replace',false);

% minperf=2;
[samples,inds]=datasample(totalset,sample_size,'Replace',false);
samples_i=samples(:,1:size(x,1))';
samples_t=samples(:,size(x,1)+1:end)';
if(include_redo==1)
    samples_i=[samples_i redo_x];
    samples_t=[samples_t redo_t];
end
samples_i=mapminmax(samples_i);
samples_t=mapminmax(samples_t);
% disp('qa');

% disp('qw')
% sample_size=ceil(size(total_tset,1)*1);
% [samples_test,inds]=datasample(total_tset,sample_size,'Replace',false);
% tsamples_i=samples_test(:,1:size(tx,1))';
% tsamples_t=samples_test(:,size(tx,1)+1:end)';

train_sum_correct(inds)=train_sum_correct(inds)+1;
% ind=vec2ind(t);
% [~,indw]=sort(ind);
% plot(cumsum(sample_prob(indw)));
% drawnow
% ind=vec2ind(samples_t);
% disp([sum(ind==1),sum(ind==2),sum(ind==3)])
% fprintf('<trs>')
end

function [tsamples_i,tsamples_t]=gettestsample(sample_size,mode)
global tx;
global tt;
global vx;
global vt;

if(mode==1)%validation
    ux=vx;
    ut=vt;
elseif(mode==2)%final test
    ux=tx;
    ut=tt;
end
total_tset=[ux;ut]';
% sample_size=.1;
sample_size=ceil(size(total_tset,1)*sample_size);
[samples_test,inds]=datasample(total_tset,sample_size,...
    'Replace',false);
tsamples_i=samples_test(:,1:size(ux,1))';
tsamples_t=samples_test(:,size(ux,1)+1:end)';
% fprintf('<tes>')
tsamples_i=mapminmax(tsamples_i);
tsamples_t=mapminmax(tsamples_t);
end

function node_arr=createnew(samples_i,samples_t,tsamples_i,tsamples_t,num_features)
global curr_id;
global node_arr;
global epsilon;
global initq
global update_sarsa;
global current_test_x
global current_test_t
global createpowerdat;
global indclassaccs;
global sum_correct;
global batchupdate
% [net,layers_node_count]=create_nw_mdp_caller_2(epsilon,samples_i,...
%     samples_t,tsamples_i,tsamples_t,0,initq,update_sarsa);
workmode=2;
totalacc=sum(sum_correct)/size(sum_correct,2);
[tsamples_i,tsamples_t]=gettestsample(.4,1);%size,mode
% [current_test_x,current_test_t]=gettestsample(1,1);
total_i=[samples_i,tsamples_i];
total_t=[samples_t,tsamples_t];
numsamples=size(total_i,2);
ids=randperm(numsamples);
total_i=total_i(:,ids);
total_t=total_t(:,ids);
r1=floor(.7*numsamples);
samples_i=total_i(:,1:r1);
samples_t=total_t(:,1:r1);
tsamples_i=total_i(:,r1+1:end);
tsamples_t=total_t(:,r1+1:end);
[net,layers_node_count,nettype]=select_network_type(epsilon,samples_i,...
    samples_t,tsamples_i,tsamples_t,0,initq,update_sarsa,workmode,...
    indclassaccs,totalacc,batchupdate);
% fprintf(num2str(layers_node_count(1)))
initq=0;
node_arr(curr_id).nettype=nettype;
node_arr(curr_id).id=curr_id;
[node_arr(curr_id).correct_out,node_arr(curr_id).train_out]=get_correct_out_node2(net,nettype);
t=node_arr(curr_id).correct_out;
acc=sum(t)/size(t,2);
% fprintf([' ' num2str(acc) ' structure:' num2str(layers_node_count)])
% global inclay
% inclay
node_arr(curr_id).network=net;
node_arr(curr_id).layers_node_count=layers_node_count;
node_arr(curr_id).hasinit=0;
curr_id=curr_id+num_features;
% recalculate_num_correct();
if(createpowerdat==-1)
    createpowerdat=acc;
else
createpowerdat=createpowerdat+.2*(acc-createpowerdat);
end
if(batchupdate==1)
    batchupdate=0;
end
recalculate_num_correct();
end

function [correct_out,train_out]=get_correct_out_node2(thisnet,nettype)
global current_test_x;
global current_test_t;
global x;
ly=thisnet(current_test_x);
train_out=thisnet(x);
if(nettype==0)
correctbb=(vec2ind(current_test_t)==vec2ind(ly));
else
    temp1=current_test_t(nettype,:);
    temp=[temp1;1-temp1];
    correctbb=(vec2ind(temp)==vec2ind(ly));
end
correct_out=correctbb;
end

%get output from input to hidden layer
%train hidden to output layer
%create final network

function correct_out=get_correct_out_node(node)
global x;
global t;
net=feedforwardnet(1);
net.inputs{1}.processFcns={'mapminmax'};
net.outputs{1,2}.processFcns={'mapminmax'};
net=configure(net,x,t);
net.trainParam.showWindow=0; 
net.layers{2}.transferFcn='tansig';
net.IW{1,1}=node.IW;
net.b{1,1}=node.bi;
net.LW{2,1}=node.LW;
net.b{2,1}=node.bo;
ly=net(x)>.9;
correctbb=sum(ly==t)==size(t,1);
correct_out=correctbb2;
% if(max(correctbb)>0)
% showme(correctbb)
% end
% showme(correctbb,1)
% net.outputConnect=[1 0];
% ly=net(x)>.5;
% showme(ly,2)
end

function ret=showme(correctbb,type)
global x;
global t;
figure(2);
disp(sum(correctbb)/size(correctbb,2))
if(type==1)
    subplot(1,2,1);
    c=(correctbb.*vec2ind(t))+1;
    unique(c)
else
    subplot(1,2,2);
    c=correctbb+1;
end
scatter(x(1,:),x(2,:),[],c);
end


function recalculate_num_correct()
global final_ids;
global node_arr;
global sum_correct;
global test_x;
global current_test_x;
global maxcreate
%calculate sum with finalids
% sum_correct=calculate_sum_correct();
%for finalids
    %subtract from sum and calculate
    
%update correct out of every network

for i=1:size(final_ids,2)
    correct_out_node=node_arr(final_ids(i)).correct_out;
    sum_correct_without_node=sum_correct-correct_out_node;
    temp=sum(correct_out_node.*(sum_correct_without_node==0));
    node_arr(final_ids(i)).count_correct=temp;
end

%for others
    %calculate
    numreqd=sum(sum_correct==0);
for i=1:size(node_arr,2)
    if(sum(i==final_ids)==0)
        correct_out_node=node_arr(i).correct_out;
        node_arr(i).accuracy=sum(correct_out_node)/size(sum_correct,2);
        temp=sum(correct_out_node.*(sum_correct==0));
        node_arr(i).count_correct=temp;
        node_arr(i).addcriteria=node_arr(i).accuracy*...
            temp/sum(sum_correct==0);
        if(node_arr(i).addcriteria>1)
            disp('more correct than required')
        end
    end
end

% for i=1:size(node_arr,2)
%     correct=node_arr(i).correct_out;
%     correctpercent=sum(correct)/size(correct,2);
%     if(maxcreate<correctpercent)
%         maxcreate=correctpercent;
%     end
% end

end

function [max_value,max_id]=find_best()
global final_ids;
global node_arr;
%find max count_correct id in candidate list
max_value=0;
max_id=0;
for i=1:size(node_arr,2)
    if(sum(i==final_ids)==0)
%         value=node_arr(i).count_correct;
        value=sum(node_arr(i).addcriteria);
        if(value>=max_value)
            max_value=value;
            max_id=i;
        end
    end
end

end

function [min_value,min_id]=find_worst() %fix
global final_ids;
global node_arr;
%find min count_correct id in final list
min_value=0;
min_id=0;
if(size(final_ids,2)~=0)
    min_value=node_arr(final_ids(1)).remove_criteria;
    min_id=final_ids(1);
    for i=1:size(final_ids,2)
        value=node_arr(final_ids(i)).remove_criteria;
        if(value<=min_value)
            min_value=value;
            min_id=final_ids(i);
        end

    end
end

end

function [next_state,reward,goal_reached]=mdp(current_state,action,quit_now)
global bin_size;
global node_arr;
goal_count=100/bin_size;
global sum_correct;
reward=0;
global final_ids;
global addcallcount;
global addfailcount;
global current_test_t;
% disp('tqw')
% action=input('act:');
% next_state=add_best_node();
% action=3;
if(action==1)%add best
    fprintf('c')
%     create_new_and_add(current_state);
    create_new_and_add(current_state);
    failflag=0;
    reward=0;
    fprintf('a');
    [next_state,failflag]=add_best_node();
    tempreward=(next_state(1)-current_state(1))*bin_size;
    
    if(failflag==0 && size(final_ids,2)>1)
        addcallcount=addcallcount+1;
        if(tempreward<0)
            addfailcount=addfailcount+1;
            fprintf(['x' num2str(-next_state(1)+current_state(1))]);
        elseif(tempreward==0)
            fprintf('s')
        end
%         breakhere=1;
    end
        
elseif(action==2)%remove worst
    fprintf('r');
%     next_state=current_state;
    [next_state,failflag]=remove_worst_node();
%     next_state=current_state;
%     [next_state,~]=reevaluate(.1,1,0);
    
%     failflag=0;
elseif(action==3)%create new and add best
    fprintf('Q')
 %  quit_now=1;
    reward=0;
%     next_state=current_state;
    [next_state,~]=reevaluate(.2,1,1,1);
    failflag=0;
% % elseif(action==4)
% %     if(size(final_ids,2)>2)
% %         next_state=increase_layer_and_create_new_and_add();
% %         reward=0;
% %     else
% %         next_state=current_state;
% %     end
end
% if(failflag==1)
%     next_state=current_state;
% end
% disp([action next_state goal_count])
goal_reached=0;
chkthis=0;
if(next_state(1)==goal_count)
    for recheck=1:3
        [next_state,~]=reevaluate(.2,1,0,1);
        chkthis=1;
        if(next_state(1)~=goal_count)
            break;
        end
    end
end
% reward=(.8*(next_state(1)-current_state(1)))+(.2*(current_state(2)-next_state(2)));
% reward=reward*10;
% reward=(next_state(1)-current_state(1));
reward=sum((next_state-current_state).*([1 0]))*bin_size;


% if(action~=2)
%     reward=(next_state(1)-current_state(1));
% else
%     reward=0;
% end
% disp(reward)
%if next_state is goal state, get reward
if(action==1 && failflag==0 && reward<1)
    breakhere=1;
end
% if((action==3 && reward~=0) || (failflag==1 && reward~=0))
%     breakhere=1;
%     disp('problem')
% end

if(action==3)
    reward=reward-1;
end
% reward=0;
if(next_state(1)==goal_count || quit_now==1) %next_state(1)==goal_count || 
%     reward=get_reward(next_state,goal_count);
%     reevaluate(1,1,0);
% reward=next_state(1)*bin_size;
%     if(quit_now==1)
%         reward=reward-5;
%     end
%     reward=100;
    if(size(final_ids,2)==0)
        reward=-100;
    end
    goal_reached=1;
end

end

function [next_state,failflag]=add_best_node()
global best_id;
global final_ids;
global node_arr
if(best_id~=0)
final_ids=[final_ids,best_id];
failflag=0;
else
    fprintf('f');
    failflag=1;
end
b=1-failflag;
[next_state,~]=reevaluate(.2,b,b,1);
end

function [next_state,failflag]=remove_worst_node()
global worst_id;
global final_ids;
global node_arr
if(worst_id~=0 && size(final_ids,2)>1)
final_ids=final_ids(final_ids~=worst_id);
node_arr(worst_id).remove_criteria=[];
node_arr(worst_id).hasinit=0;
failflag=0;
else
    fprintf('f');
    failflag=1;
end
b=1-failflag;
[next_state,~]=reevaluate(.2,b,b,1);
end

function [next_state]=create_new_and_add(current_state)
global num_new;
global best_id;
global final_ids;
global node_arr
global current_test_x;
global current_test_t;
global maxcreate;
% global node_arr;
% global sum_correct;
[samples_i,samples_t]=getsample(1,.3); %mode, size
% [tsamples_i,tsamples_t]=gettestsample();
createnew(samples_i,samples_t,current_test_x,current_test_t,num_new);
% final_ids=[final_ids,size(node_arr,2)];%add created 
[~,best_id]=find_best();%because best id changes but worst
                            %etc will be handled by reevaluate 
                            %called by add_best_node
                           
% next_state=add_best_node();
correct=node_arr(end).correct_out;
correctpercent=sum(correct)/size(correct,2);
if(maxcreate<correctpercent)
    maxcreate=correctpercent;
end
% [next_state,~]=reevaluate(.2,0,0,1);
end

function [next_state]=increase_layer_and_create_new_and_add()
%set x
% global x;
% global test_x;
% global curr_id;
% global final_ids;
% global node_arr;
% global current_layer;
% global sum_correct;
% global node_per_layer;
% global best_id;
% global worst_id;
% global t;
% store_best_id=best_id;
% store_worst_id=worst_id;
% store_current_layer=current_layer;
% store_x=x;
% store_test_x=test_x;
% store_node_per_layer=node_per_layer;
% store_final_ids=final_ids;
% store_node_arr=node_arr;
% store_curr_id=curr_id;
% store_sum_correct=sum_correct;
% 
% current_layer=current_layer+1;
% mynet=feedforwardnet(size(final_ids,2));
% mynet.inputs{1}.processFcns={'mapminmax'};
% mynet.outputs{1,2}.processFcns={'mapminmax'};
% mynet=configure(mynet,x,t);
% for i=1:size(final_ids,2)
%     mynet.IW{1,1}(i,:)=node_arr(final_ids(i)).IW;
%     mynet.b{1,1}(i)=node_arr(final_ids(i)).bi;
% end
% mynet.outputConnect=[1 0];
% x=mynet(x,'useGPU','yes');
% test_x=mynet(test_x,'useGPU','yes');
%    %create new training set
% %create new test set
% node_per_layer=[node_per_layer,size(final_ids,2)];
% final_ids=[];
% node_arr=zeros(1,0);
% curr_id=1;
% sum_correct=calculate_sum_correct();
% next_state=create_new_and_add();
% 
% if(size(final_ids,2)==0)
%     %revert
%     current_layer=store_current_layer;
%     x=store_x;
%     test_x=store_test_x;
%     node_per_layer=store_node_per_layer;
%     final_ids=store_final_ids;
%     node_arr=store_node_arr;
%     curr_id=store_curr_id;
%     sum_correct=store_sum_correct;
%     sum_correct=calculate_sum_correct();
%     best_id=store_best_id;
%     worst_id=store_worst_id;
%     next_state=reevaluate;
% end

end

function calculate_sum_correct()
global final_ids;
global node_arr;
global current_test_x;
global sum_correct;
% k=sum_correct;
% sum_correct=zeros(1,size(current_test_x,2));
for i=1:size(node_arr,2)
    node_arr(i).correct_out=get_correct_out_node2(node_arr(i).network,node_arr(i).nettype);
end

% for i=1:size(final_ids,2)
%     sum_correct=sum_correct+node_arr(final_ids(i)).correct_out;
% end
% if(sum(k>0)>sum(sum_correct>0))
%     breakhere=1;
% end
% recalculate_num_correct();
end

function reward=get_reward(current_count,goal_count)
global bin_size;
current_in_100=current_count*bin_size;
if(current_count==goal_count)
    reward=100;
else
    reward=-(100-current_in_100);
%     reward=current_in_100;
end
end

function addval=sarsa(state,action,next_state,next_action,reward)
global Q_value;
alpha=.1;
gamma=.99;
%Q_state=gauss_sum(state,action);
Q_state=Q_value(state(1),state(2),action);
Q_next_state=Q_value(next_state(1),next_state(2),next_action);
% Q_next_state=gauss_sum(next_state,next_action);
addval=(alpha*(reward+(gamma*Q_next_state)-Q_state));
value=Q_state+addval;
Q_value(state(1),state(2),action)=value;
% update_Q_gauss(state,action,value);
addval=abs(addval);
end

function value=gauss_sum(state,action)
global Q_value;
F=getfilter(state);
plane_Q=Q_value(:,:,action);
value=sum(sum(plane_Q.*F));
% disp([value Q_value(state(1),state(2),action)]);
end

function update_Q_gauss(state,action,value)
global Q_value;
F=getfilter(state);
Q_value(:,:,action)=Q_value(:,:,action)+(value.*F);


end

function F=getfilter(state)
global Q_value;
mu = state;
Sigma = [.3 .05; .05 .3];
sizeQ=size(Q_value,1);
x1 = 1:sizeQ; 
x2 = 1:sizeQ;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
end

% function next_state=reevaluate()
% global best_id;
% global worst_id;
% global sum_correct;
% global test_x;
% global test_t;
% global final_ids;
% global node_arr;
% global t;
% global x;
% 
% if(size(final_ids,2)>0)
% finalnet=feedforwardnet(size(final_ids,2));
% finalnet.inputs{1}.processFcns={'mapminmax'};
% finalnet.outputs{1,2}.processFcns={'mapminmax'};
% finalnet=configure(finalnet,x,t);
% 
% for i=1:size(final_ids,2)
%     hidden_out(i,:)=node_arr(final_ids(i)).train_out;
%     finalnet.IW{1,1}(i,:)=node_arr(final_ids(i)).IW;
%     finalnet.b{1,1}(i)=node_arr(final_ids(i)).bi;
% end
% 
% mynet=patternnet([]);
% mynet.inputs{1}.processFcns={'mapminmax'};
% mynet.outputs{1,1}.processFcns={'mapminmax'};
% mynet.trainParam.showWindow=0;
% mynet.trainParam.max_fail=100;
% mynet=train(mynet,hidden_out,t,'useGPU','yes');
% % perform(mynet,samples_t,hidden_out)
% finalnet.LW{2,1}=mynet.IW{1,1};
% ly=finalnet(test_x,'useGPU','yes');
% sum_correct=(vec2ind(test_t)==vec2ind(ly));
% bre=1;
% else
%     sum_correct=zeros(1,size(test_t,2));
% end
% 
% goal_count=size(sum_correct,2);
% unique_sum_correct=sum(sum_correct>0)/goal_count;
% recalculate_num_correct();
% [~,best_id]=find_best();
% [~,worst_id]=find_worst();
% next_state=percent_bin_it(unique_sum_correct*100);
% end

function [next_state,unique_sum_correct]=reevaluate(sample_size,bsample,btrain,mode)
global best_id;
global worst_id;
global final_ids
global node_arr
global tr_acc
global sum_correct;
global current_test_x;
global current_test_t;
global old_actual_sum;
global maxcreate;
if(bsample==1)
    if(mode==1)%val
        [current_test_x,current_test_t]=gettestsample(1,mode);
    else%test
        [current_test_x,current_test_t]=gettestsample(1,mode);
    end
end
% load 'testing.mat';
goal_count=size(current_test_x,2);
% calculate_sum_correct();
[actual_sum_correct,tr_acc]=createopnw2(btrain);
% [actual_sum_correct,tr_acc]=createopnw_equalvoting(btrain);
sum_correct=actual_sum_correct;
% if(btrain==1)
% actual_sum_correct=createopnw2(btrain);
% old_actual_sum=actual_sum_correct;
% else
%     actual_sum_correct=old_actual_sum;
% end
% actual_sum_correct=sum_correct>0;
unique_sum_correct=sum(actual_sum_correct)/goal_count;
unique_sum_correct_old=sum(sum_correct>0)/goal_count;
overlap_sum_correct=sum(sum_correct>1)/goal_count;
% overlap_sum_correct=0;
% if(size(node_arr,2)>0)
% if(size(sum_correct,2)~=size(node_arr(end).correct_out,2))
%     breakhere=1;
% end
% end
calculate_sum_correct();%get output of nws for current input test set
recalculate_num_correct();
[~,best_id]=find_best();
[~,worst_id]=find_worst();
% if(best_id>0)
% % addpower=node_arr(best_id).count_correct./(.1+goal_count-sum(actual_sum_correct));
% addpower=sum(node_arr(best_id).correct_out)./goal_count;
% % addpower=addpower/maxcreate;
% else
%     addpower=0;
% end
if(worst_id>0)
% addpower=node_arr(best_id).count_correct./(.1+goal_count-sum(actual_sum_correct));
for i=1:size(final_ids,2)
r_criteria(i)=node_arr(final_ids(i)).remove_criteria;
end
max_r_crit=max(r_criteria);
min_r_crit=min(r_criteria);
removepower=min_r_crit/max_r_crit;
% removepower=sum(node_arr(best_id).correct_out)./goal_count;
% addpower=addpower/maxcreate;
else
    removepower=1;
end
% disp([unique_sum_correct,unique_sum_correct_old])
% disp(unique_sum_correct)
next_state=[percent_bin_it(unique_sum_correct*100),percent_bin_it(removepower*100)];
end

function actual_sum_correct=createopnw(casetype)
global final_ids;
global node_arr;
global current_test_x;
global current_test_t;
global sum_correct;
global oldwts24;
intermediate_inp=zeros(1,size(current_test_t,2));
if(size(final_ids,2)==1)
    oldwts24=node_arr(final_ids(1)).network.LW(end,end-1);
    actual_sum_correct=sum_correct>0;
elseif(size(final_ids,2)>1)
%     disp('used')
    for i=1:size(final_ids,2)
        node_arr(final_ids(i)).network.outputConnect(end-1:end)=[1 0];
        nw_out=node_arr(final_ids(i)).network(current_test_x);
        intermediate_inp=[intermediate_inp;nw_out];
        h(i,:)=vec2ind(nw_out);
        node_arr(final_ids(i)).network.outputConnect(end-1:end)=[0 1];
    end
    intermediate_inp=intermediate_inp(2:end,:);
    net=patternnet([]);
% %     net.trainFcn='trainlm';
    net.trainParam.showWindow=false;
    net.inputs{1}.processFcns={'mapminmax'};
    net.outputs{1}.processFcns={'mapminmax'};
    net=configure(net,intermediate_inp,current_test_t);
    
    if(casetype~=2 && casetype~=4)
    net.IW{1}=rand(size(net.IW{1})).^(9);
    s=size(oldwts24{1},2);
    if(size(net.IW{1},2)<s)
        disp('yup it happened')
        send_text_message('HEYYYYY!!!!',[],404)
    end
    net.IW{1}(:,1:s)=oldwts24{1};
    end
    if(casetype==4)
        net.IW=oldwts24;
    elseif(casetype==2)
        oldwts24=net.IW;
    end
    net=train(net,intermediate_inp,current_test_t);
    
    if(casetype~=2 && casetype~=4)
    oldwts24=net.IW;
    end
    
    y=net(intermediate_inp);
%     if(size(h,1)>1)
%         prediction=mode(h);
%     else
%         prediction=h;
%     end
    actual_sum_correct=(vec2ind(current_test_t)==vec2ind(y));
    cmat=confusionmat(vec2ind(current_test_t),vec2ind(y));
	acc=(sum(diag(cmat))/size(current_test_t,2));
%     createopnw()
else
    actual_sum_correct=sum_correct>0;
end

end

function votingrights=getvotewts(forvote,samples_t)
total=zeros(size(samples_t));
for i=1:size(forvote,2)
    total=total+forvote{i};
end
totalaccuracy=sum(vec2ind(total)==vec2ind(samples_t))/size(samples_t,2);
for i=1:size(forvote,2)
    temptotal=total-forvote{i};
    tempaccuracy(i)=sum(vec2ind(temptotal)==vec2ind(samples_t))/size(samples_t,2);
end
tempaccuracy=1-tempaccuracy;
votingrights=tempaccuracy./sum(tempaccuracy);
votingrights=ones(size(votingrights));
end

function [actual_sum_correct,tr_acc]=createopnw_equalvoting(btrain)
global final_ids;
global node_arr;
global current_test_x;
global current_test_t;
global sum_correct;
global oldwts24;
global saved_net;
global oldacc;
global indclassaccs;
global lastnet42
global netpresent
global redo_x;
global redo_t;

tr_acc=0;
if(size(final_ids,2)>0)     
    [samples_i,samples_t]=getsample(0,.2);
    intermediate_outp=zeros(size(samples_t));
    for i=1:size(final_ids,2)
        nw_out=node_arr(final_ids(i)).network(samples_i);
        intermediate_outp=[intermediate_outp+nw_out];
        forvote{i}=nw_out;
    end
    votingrights=getvotewts(forvote,samples_t);
    valintermediate_outpv=zeros(size(current_test_t));
    valintermediate_outpw=zeros(size(current_test_t));
    for i=1:size(final_ids,2)
        nw_out=node_arr(final_ids(i)).network(current_test_x);
        valintermediate_outpv=[valintermediate_outpv+...
            (votingrights(i).*nw_out)];
        valintermediate_outpw=[valintermediate_outpw+...
            (1.*nw_out)];
    end
    updatemeanwtsvote(votingrights);
    y=valintermediate_outpv;
    actual_sum_correct=(vec2ind(current_test_t)==vec2ind(y));
    cmat=confusionmat(vec2ind(current_test_t),vec2ind(y));
	accv=(sum(diag(cmat))/size(current_test_t,2));
    y=valintermediate_outpw;
    actual_sum_correct=(vec2ind(current_test_t)==vec2ind(y));
    cmat=confusionmat(vec2ind(current_test_t),vec2ind(y));
	accw=(sum(diag(cmat))/size(current_test_t,2));
    %disp([accv accw])
    
    y=intermediate_outp;
    trcmat=confusionmat(vec2ind(samples_t),vec2ind(y));
    tr_acc=(sum(diag(trcmat))/size(samples_t,2));
    idx=(vec2ind(y)==vec2ind(samples_t))==0;
    redo_x=samples_i(:,idx);
    redo_t=samples_t(:,idx);
    
    indclassaccs=0;
    indclassaccs=diag(cmat)'./sum(cmat');
    netpresent=1;

elseif(size(final_ids,2)==0)
    actual_sum_correct=zeros(1,size(current_test_x,2));
    indclassaccs=zeros(1,size(current_test_t,1));
end



end



function [actual_sum_correct,tr_acc]=createopnw2(btrain)
global final_ids;
global node_arr;
global current_test_x;
global current_test_t;
global sum_correct;
global oldwts24;
global saved_net;
global oldacc;
% global tr_acc
global indclassaccs;
global lastnet42
global netpresent
global redo_x;
global redo_t;
global oldtiewts
% if(size(final_ids,2)==1)
%     oldwts24=node_arr(final_ids(1)).network.LW(end,end-1);
%     actual_sum_correct=sum_correct>0;
tr_acc=0;
if(size(final_ids,2)>0)
%     disp('used')
[samples_i,samples_t]=getsample(0,1);
intermediate_inp=zeros(1,size(samples_i,2));
    for i=1:size(final_ids,2)
        node_arr(final_ids(i)).network.outputConnect(end-1:end)=[1 0];
        nw_out=node_arr(final_ids(i)).network(samples_i);
        intermediate_inp=[intermediate_inp;nw_out];
        h(i,:)=vec2ind(nw_out);
        node_arr(final_ids(i)).network.outputConnect(end-1:end)=[0 1];
    end
    intermediate_inp=intermediate_inp(2:end,:);
        
    valintermediate_inp=zeros(1,size(current_test_x,2));
    for i=1:size(final_ids,2)
        node_arr(final_ids(i)).network.outputConnect(end-1:end)=[1 0];
        nw_out=node_arr(final_ids(i)).network(current_test_x);
        valintermediate_inp=[valintermediate_inp;nw_out];
%         h(i,:)=vec2ind(nw_out);
        node_arr(final_ids(i)).network.outputConnect(end-1:end)=[0 1];
    end
    valintermediate_inp=valintermediate_inp(2:end,:);
    
    if(btrain==1)
    %create
    net=patternnet([]);
    net.trainFcn='trainscg';
%    net.layers{1}.transferFcn='purelin';
    net.trainParam.showWindow=false;
%     net.inputs{1}.processFcns={'mapminmax'};
    net.inputs{1}.processFcns={};
    net.outputs{1}.processFcns={};
    net=configure(net,intermediate_inp,samples_t);
%     net.IW{1}=rand(size(net.IW{1})).*1e-5;
%      net.IW{1}=net.IW{1}.*1e-1;
%     net.trainParam.min_grad=1e-19;
    net.trainParam.max_fail=6;
%    net.trainParam.epochs=4;
temp_int_inp=[intermediate_inp,valintermediate_inp];
r1=size(intermediate_inp,2);
r2=size(valintermediate_inp,2);
r3=size(temp_int_inp,2);
net.divideFcn='divideind';

[net.divideParam.trainInd,net.divideParam.valInd,...
    net.divideParam.testInd] = divideind(r3,1:r1,r1+1:r3-1,r3-1:r3);
%     oldacc=-1;
%     acc=0;
%     countval=0;
%     while(acc>=oldacc && countval<4)
% %         disp([oldacc acc])
%     if(acc==oldacc)
%         countval=countval+1;
%     else
%         countval=0;
%     end
%     oldacc=acc;
%     oldnet=net;
%     net=train(net,intermediate_inp,samples_t);
%     y=net(valintermediate_inp);
%     cmat=confusionmat(vec2ind(current_test_t),vec2ind(y));
% 	acc=(sum(diag(cmat))/size(current_test_t,2));
%     end
%     net=oldnet;
decay=.9;
initweights=zeros(size(samples_t,1),1);
for f=1:size(final_ids,2)
    lastlayersize=node_arr(final_ids(f)).layers_node_count(end);
    if(node_arr(final_ids(f)).hasinit==1)
%         randwts=1 - 2.*rand(size(node_arr(final_ids(f)).lastweights));
    initweights=[initweights,...
        (decay.*node_arr(final_ids(f)).lastweights)];
    else
        initweights=[initweights,...
            1 - 2.*rand(size(samples_t,1),lastlayersize)];
    end
end
initweights=initweights(:,2:end);
if(size(oldtiewts,2)==0)
    initweights=[initweights,...
            1 - 2.*rand(size(samples_t,1),1)];
else
    initweights=[initweights,...
        decay.*oldtiewts(:,end)];
end
total_i=[intermediate_inp,valintermediate_inp];
total_t=[samples_t,current_test_t];
numsamples=size(total_i,2);
ids=randperm(numsamples);
total_i=total_i(:,ids);
total_t=total_t(:,ids);
r1=floor(.5*numsamples);
tempsamples_i=total_i(:,1:r1);
tempsamples_t=total_t(:,1:r1);
tsamples_i=total_i(:,r1+1:end);
tsamples_t=total_t(:,r1+1:end);


    weights=nwbpropinit(tempsamples_i,tempsamples_t,...
        tsamples_i,tsamples_t,initweights);
    oldtiewts=weights{1};
    net.IW{1}=weights{1}(:,1:size(net.IW{1},2));
    net.b{1}=weights{1}(:,end);
%     net=train(net,temp_int_inp,[samples_t,current_test_t]);
    y=net(valintermediate_inp);
    cmat=confusionmat(vec2ind(current_test_t),vec2ind(y));
	oldacc=(sum(diag(cmat))/size(current_test_t,2));
    
%     [net2,layers_node_count,nettype]=typecaller(.1,intermediate_inp,...
%     samples_t,valintermediate_inp,current_test_t,0,0,0,2,...
%     indclassaccs,0,0);
%     y2=net2(valintermediate_inp);
%     cmat2=confusionmat(vec2ind(current_test_t),vec2ind(y2));
% 	acc2=(sum(diag(cmat2))/size(current_test_t,2));
%     global showoldacc
%     global inclay
% 
% 
%     showoldacc(end+1,:)=[oldacc];
%         disp([oldacc inclay(end-1)/inclay(end)...
%         max(showoldacc)-min(showoldacc)])
%     plot(showoldacc)
%     drawnow
%     net=net2;
    y=net(intermediate_inp);
    trcmat=confusionmat(vec2ind(samples_t),vec2ind(y));
    tr_acc=(sum(diag(trcmat))/size(samples_t,2));
    idx=(vec2ind(y)==vec2ind(samples_t))==0;
    redo_x=samples_i(:,idx);
    redo_t=samples_t(:,idx);
    saved_net=net;
    %create end
    else
        net=saved_net;
    end
    

    updatemeanwts(net);
    y=net(valintermediate_inp);
    actual_sum_correct=(vec2ind(current_test_t)==vec2ind(y));
    cmat=confusionmat(vec2ind(current_test_t),vec2ind(y));
	acc=(sum(diag(cmat))/size(current_test_t,2));
    indclassaccs=0;
    indclassaccs=diag(cmat)'./sum(cmat');
    lastnet42=net;
    netpresent=1;
%     if(btrain==1)
%         saved_net=net;
%         oldacc=acc;
%     elseif(acc>old_acc)
%         saved_net=net;
%         oldacc=acc;
%     else
%         net=saved_net;
%         acc=oldacc;
%     end
        
        
    
%     sum_correct=actual_sum_correct;
elseif(size(final_ids,2)==0)
    actual_sum_correct=zeros(1,size(current_test_x,2));
    indclassaccs=zeros(1,size(current_test_t,1));
end

end

function updatemeanwts(net)
global final_ids;
global node_arr;
global curr_net_meanwts;

mean_netwt=mean(abs(net.IW{1}));
start=1;
curr_net_meanwts=0;
for i=1:size(final_ids,2)
    node_count=node_arr(final_ids(i)).layers_node_count(end);
%	disp([size(mean_netwt,2) start start+node_count-1]);
    temp=mean(mean_netwt(start:start+node_count-1));
    node_arr(final_ids(i)).remove_criteria=temp;%temp
    node_arr(final_ids(i)).lastweights=...
        net.IW{1}(:,start:start+node_count-1);
    start=start+node_count;
    node_arr(final_ids(i)).hasinit=1;
end

end

function updatemeanwtsvote(votingrights)
global final_ids;
global node_arr;
global curr_net_meanwts;

% mean_netwt=mean(abs(net.IW{1}));
start=1;
curr_net_meanwts=0;
for i=1:size(final_ids,2)
%     node_count=node_arr(final_ids(i)).layers_node_count(end);
%	disp([size(mean_netwt,2) start start+node_count-1]);
%     temp=mean(mean_netwt(start:start+node_count-1));
    node_arr(final_ids(i)).remove_criteria=votingrights(i);%temp
%     start=start+node_count;
end

end

function value=percent_bin_it(input_value)
global bin_size;
value=ceil(input_value/bin_size);
if(value==0)
    value=1;
end
end



function performance=create_final_network()
global node_arr;
global final_ids;
global x;
global t;
net=feedforwardnet(size(final_ids,2));
net.trainFcn='traingd';
% net.trainFcn = 'trainscg';
net.trainParam.showWindow=0; 
net.layers{2}.transferFcn='tansig';
% net.biasConnect=[1;0];

net.trainParam.epochs=10;
net=configure(net,x,t);
% net=train(net,x,t);

net.IW{1,1}=get_input_weights();
net.LW{2,1}=get_layer_weights();
net.b=get_bias_weights();


y=net(x);
perform(net,t,y)
ly=net(x)>.9;
correctbb=sum(ly==t)==size(t,1);
sum(correctbb)
breakhere=1;
end

function testdata()
global d_store;


for i=1:size(d_store,2)
    x=d_store{i}.x;
    t=d_store{i}.t;
    test_x=d_store{i}.test_x;
    test_t=d_store{i}.test_t;
    acc(i)=testnet(x,t,test_x,test_t);
end

end

function [unique_sum_correct]=testnet(x,t,test_x,test_t)
mynet=feedforwardnet(10);
mynet.trainParam.showWindow=false;
mynet=configure(mynet,x,t);
mynet=train(mynet,x,t);
y=mynet(test_x);

c_mat=confusionmat(vec2ind(test_t),vec2ind(y))';
goal_count=size(test_x,2);
sum_correct=sum(diag(c_mat));
unique_sum_correct=100*sum_correct/goal_count;
end




