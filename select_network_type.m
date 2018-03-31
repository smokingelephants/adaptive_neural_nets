function [net,layers_node_count,nettype]=typecaller(epsilon,samples_i,samples_t,tsamples_i,tsamples_t,last_update,initq,update_sarsa,workmode,indclassaccs,totalacc,batchupdate)
global oldsa;
global typeepsilon;
global Q_valuetype;
global epsilontype;
global bin_sizetype
global minclassid
global typecallcount;
global typepid;
% [net,layers_node_count]=create_nw_mdp_caller_2(epsilon,samples_i,...
%     samples_t,tsamples_i,tsamples_t,0,initq,update_sarsa);
epsilontype=epsilon;
bin_sizetype=20;
if(initq==1)
    Q_valuetype=zeros(ceil(100/bin_sizetype),ceil(100/bin_sizetype),2);
    Q_valuetype(:,:,1)=.001;
    typecallcount=0;
%    load('2.mat')
    global typepid_saved;
    global typepolicy
    load('typepida.mat');
    typepolicy=typepid;
%    typepid_saved=typepid;
    disp('typelordtest80100')
end
if(workmode==1)
    %updatesarsa
    qnsna=totalacc;
    updatetypesarsa(qnsna);
elseif(workmode==2)
    typecallcount=typecallcount+1;
    [minacc,minclassid]=min(indclassaccs);
    temps=max(indclassaccs);
    temps=minacc/(temps+.0001);
    state=[percent_bin_it(totalacc*100) percent_bin_it(temps*100)];
    action=getaction(state);
    [net,layers_node_count,nettype]=mdp(state,action,epsilon,samples_i,...
    samples_t,tsamples_i,tsamples_t,0,initq,update_sarsa,batchupdate);
    oldsa.state=state;
    oldsa.action=action;    
end
if(mod(typecallcount,30)==0 || typecallcount==1)
    [~,typepid]=max(Q_valuetype,[],3);
end
end

function action=getaction(state)
global typeepsilon
global Q_valuetype;
global typepolicy
r=rand;
global typepid_saved
%chk if action>3
% if(r>typeepsilon)
%     %exploit
%     [~,action]=max(Q_valuetype(state(1),state(2),:));
% else
%     %explore
%     action=randi(2);
% end
%action=typepid_saved(state(1),state(2));
action=1;
%action=typepolicy(state(1),state(2));
end

function [net,layers_node_count,nettype]=mdp(state,action,epsilon,samples_i,...
    samples_t,tsamples_i,tsamples_t,last_update,initq,...
    update_sarsa,batchupdate)
global minclassid;
if(action==1)%generic
    %simple call
     fprintf('g')
    [net,layers_node_count]=build_network(epsilon,samples_i,...
    samples_t,tsamples_i,tsamples_t,0,initq,update_sarsa,batchupdate);
    nettype=0;
elseif(action==2)
    %specific
    fprintf(num2str(minclassid))
    mod_samples_t=samples_t(minclassid,:);
    mod_tsamples_t=tsamples_t(minclassid,:);
    mod_samples_t=[mod_samples_t;1-mod_samples_t];
    mod_tsamples_t=[mod_tsamples_t;1-mod_tsamples_t];
    [net,layers_node_count]=build_network(epsilon,samples_i,...
    mod_samples_t,tsamples_i,mod_tsamples_t,0,initq,update_sarsa,batchupdate);
    nettype=minclassid;
end



end

function updatetypesarsa(qnsna)
global Q_valuetype;
global oldsa;
alpha=.1;
gamma=.99;
reward=0;
state=oldsa.state;
action=oldsa.action;
Q_state=Q_valuetype(state(1),state(2),action);
Q_next_state=qnsna;
addval=(alpha*(reward+(gamma*Q_next_state)-Q_state));
value=Q_state+addval;
Q_valuetype(state(1),state(2),action)=value;
end


function value=percent_bin_it(input_value)
global bin_sizetype;
value=ceil(input_value/bin_sizetype);
if(value==0)
    value=1;
end
end













