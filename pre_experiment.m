
%% env define
mdl = 'Test_model';
open_system(mdl)
agentblk = [mdl '/RL Agent'];
%% observation info
observationInfo = rlNumericSpec([3 1]);
observationInfo.Name = 'observations';
observationInfo.Description = 'information on tracking error and ego height';

%% Action Info 
%actionInfo = rlNumericSpec([1 1]);
actionInfo = rlNumericSpec([1 1],'LowerLimit',-3,'UpperLimit',3);
actionInfo.Name = 'elevator deflection';

%% Enviroment define
env = rlSimulinkEnv(mdl,agentblk,observationInfo,actionInfo);
Ts = 0.1;
Tf = 10;
rng(0)

%% critic define
L = 10; % number of neurons in hidden layers
statePath = [
    featureInputLayer(3,'Normalization','none','Name','observation')
    fullyConnectedLayer(L,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(L,'Name','fc2')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
    fullyConnectedLayer(L,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')];

actionPath = [
    featureInputLayer(1,'Normalization','none','Name','action')
    fullyConnectedLayer(L, 'Name', 'fc5')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = connectLayers(criticNetwork,'fc5','add/in2');
criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
critic = rlQValueRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
%% Actor define
% G = 48
% actorNetwork = [
%     featureInputLayer(1,'Normalization','none','Name','observation')
%     fullyConnectedLayer(G,'Name','fc1')
%     reluLayer('Name','relu1')
%     fullyConnectedLayer(G,'Name','fc2')
%     reluLayer('Name','relu2')
%     fullyConnectedLayer(G,'Name','fc3')
%     reluLayer('Name','relu3')
%     fullyConnectedLayer(1,'Name','fc4')
%     tanhLayer('Name','tanh1')
%     scalingLayer('Name','ActorScaling1','Scale',1,'Bias',0)];
% 
% % actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
% actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
% 
% actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,...
%   'Observation',{'observation'},'Action',{'ActorScaling1'},actorOptions);

actorNetwork = [
%     featureInputLayer(1,'Normalization','none','Name','state')
    imageInputLayer([3 1 1],'Normalization','none','Name','state')
%     fullyConnectedLayer(3,'Name','action','BiasLearnRateFactor',0,'Bias',0)
    fullyConnectedLayer(1,'Name','action','BiasLearnRateFactor',0,'Bias',0)];

%actorOpts =
%rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);%OK
actorOpts = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);


actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,'Observation',{'state'},'Action',{'action'},actorOpts);

%% Agent define

agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',64);

agentOpts.NoiseOptions.Variance = 0.3;
%agentOpts.NoiseOptions.VarianceDecayRate = 1e-6;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-6;
agentOpts.SaveExperienceBufferWithAgent = true;
agentOpts.ResetExperienceBufferBeforeTraining = false;
agent = rlDDPGAgent(actor,critic,agentOpts);

%% Training specifications
device = 'gpu';

maxepisodes = 10000;

maxsteps = Tf/Ts;
% 
trainingOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeReward',...
    'StopTrainingValue',-0);%-50 OK
% % 
% trainingOpts = rlTrainingOptions(...
%     'MaxEpisodes',maxepisodes,...
%     'MaxStepsPerEpisode',maxsteps,...
%     'Verbose',false,...
%     'Plots','training-progress',...
%     'StopTrainingCriteria','EpisodeReward',...
%     'StopTrainingValue',-5500,...
%     'SaveAgentCriteria',"EpisodeCount",...
% %    'SaveAgentValue',1,...
% %    'SaveAgentDirectory', "Agents");
trainingStats = train(agent,env,trainingOpts);
% save('test_agent_1_with_slope_climb','agent')
% save('test_agent_2_with_slope_climb','agent')
% % 
% size(ss(Alin,Blin,Clin,Dlin))
% sys = ss(Alin,Blin,Clin,Dlin)
% systf = tf(sys)
% % % % 
% [a b] = ss2tf(Alin,Blin,Clin,Dlin)
% % % 
% % H = tf([13],[1 1]);
% % bode(H)
% %save('test_agent_1_success','agent')
% %% plot the learning progress
% % for i = 1:5000
% %     ai = cell(1,200)
% %     ai = agent.ExperienceBuffer.Memory(1:200*i);
% % end
% Ep_reward = zeros(1,5000)
% % 
% % for i = 1:5000
% %     for n = 200*(i-1)+1:200*iealphag
% %         Ep_reward(i)= Ep_reward(i)+ agent.ExperienceBuffer.Memory{1,n}{1,3}
% %     end
% % end
% % reward_x_clear = [1:257]
% reward_x = [1:5000]
% Ep_reward_clear = Ep_reward (1:5000)
% Ep_reward_smooth = smoothdata(Ep_reward_clear)
% figure()
% set(gca,'FontSize',18)
% plot(reward_x,Ep_reward_clear,'LineWidth',3)
% hold on 
% plot(reward_x,Ep_reward_smooth,'LineWidth',2.1)
% xlabel('Episodes')
% ylabel('The cumulative reward recieved at each episode')
% legend('cumulative reward','average cumulative reward')
% title('Flat learning progress')
% %axis([0 200 -3e5 0])
% hold off
% grid on
% 
% 
% a = agent.ExperienceBuffer.Memory(200*4999+1:200*5000);
% 
% 
% figure()
% set(gca,'FontSize',18)
% plot(out.tout,out.reference,out.tout,out.alti1,'LineWidth',2.1)
% xlabel('Timesteps [s]')
% ylabel('Altitude [m]')
% legend('reference trejactory','tracking performance by three loop controllers')
% title('the tracking performance of three loop controllers method')
% %axis([0 200 -3e5 0])
% hold off
% grid on
% print -deps epsFig
% 
% 
% figure()
% set(gca,'FontSize',18)
% plot(out.tout,out.reference,out.tout,out.alti2,'LineWidth',2.1)
% xlabel('Timesteps [s]')
% ylabel('Altitude [m]')
% legend('reference trejactory','tracking performance by two loopcontrollers')
% title('the tracking performance of two loop controllers method')
% %axis([0 200 -3e5 0])
% hold off
% grid on
% 
% 
% figure()
% set(gca,'FontSize',18)
% plot(out.tout,out.reference,out.tout,out.alti3,'LineWidth',2.1)
% xlabel('Timesteps [s]')
% ylabel('Altitude [m]')
% legend('reference trejactory','tracking performance by flat learning agent')
% title('the tracking performance of flat learning agent')
% %axis([0 200 -3e5 0])
% hold off
% grid on
% 
% 
% 
% %a = agent.ExperienceBuffer.Memory(1:200);
% b = agent.ExperienceBuffer.Memory(498*200+1:499*200);
% c = agent.ExperienceBuffer.Memory(152601:152800);
% % d = agent.ExperienceBuffer.Memory(720200:720401);
% p1 = zeros(1,200);
% for i = 1:200
%     p1(i) = b{1,i}{1,1}{1,1}(1,1);
% end
% 
% 
% v1 = zeros(1,200);
% for i = 1:200
%     v1(i) = b{1,i}{1,1}{1,1}(2,1);
% end
% 
% 
% a1 = zeros(1,200);
% for i = 1:200
%     a1(i) = b{1,i}{1,2}{1,1};
% end
% 
% r1 = zeros(1,200);
% for i = 1:200
%     r1(i) = b{1,i}{1,3};
% end
% 
% y = [1:1:200]
% figure()                
% set(gca,'FontSize',18)
% plot(y,p1,'LineWidth',3)
% %legend('position information in episode two [m]','velocity information in episode two [m/s]' )
% xlabel('time steps [s]')
% ylabel('tracking error [m]')
% title('Tracking error at each step in episode 499')
% grid on
% % 
% figure()                
% set(gca,'FontSize',18)
% plot(y,a1,'LineWidth',3)
% xlabel('steps')
% ylabel('The action at each step [rad/s]')
% title('Action taken at each step in episode 499')
% grid on
% 
% figure()                
% set(gca,'FontSize',18)
% plot(y,r1,'LineWidth',3)
% xlabel('time steps [s]')
% ylabel('The reward recieved at each step')
% title('Reward receieved at each step in episode 499')
% grid on
% % 
% figure()
% set(gca,'FontSize',18)
% plot3(v1,p1,a1,'LineWidth',3)
% xlabel('The position information [m]')
% ylabel('The velocity information [m/s]')
% zlabel('The policy taken at each states in episode 2')
% grid on
% 
% figure()
% set(gca,'FontSize',18)
% plot3(v1,p1,r1,'LineWidth',3)
% xlabel('The position information [m]')
% ylabel('The velocity information [m/s]')
% zlabel('The reward recieved at each states in episode 2')
% grid on
% 
% figure()                
% set(gca,'FontSize',18)
% plot(y,p1,'LineWidth',3)
% %legend('position information in episode two [m]','velocity information in episode two [m/s]' )
% xlabel('time steps [s]')
% ylabel('tracking error [m]')
% title('Tracking error at each step in episode 5000')
% grid on
% 
%     
% figure()                
% set(gca,'FontSize',18)
% plot(y,a1,'LineWidth',3)
% xlabel('steps')
% ylabel('The action at each step [rad/s]')
% title('Action taken at each step in episode 1')
% grid on
% 
% 
% % % 
% xts1 = [-100:0.1:100]
% ts1 = timeseries(xts1)
