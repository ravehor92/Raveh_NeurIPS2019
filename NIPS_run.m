% main.m - Define meta parameters and run the algorithm with a simple maze, producing a graph similar to 
%          that appearing in the paper. 
% k - number of samples for each state-action
% Episodes - number of Episodes
% Num_of_Agents - Number of concurrent agents
% times - number of repeats of the same experiment
% eps_a - epsilon_a defined in the algorithm. 
% eps_b - exploration bonus 
% gamma - discount factor
% R_max - value of the reward at the top right corner of the maze. All other state-actions have zero reward. 
% Bell_iterations - maximal number of Bellman iterations. 
% Episode_Length - Number of time steps within an episode. 
% maze_size - the size of a simple maze. [M,N] means M rows and N columns. 
% km - number of groups to divide into for value iteration, as defined in the algorithm.
% weight_estimation_flag - 1 if we wish to estimate the optimal weights (if 1, then the value of w is ignored)
% var_L - variance of the agent-learner communication noise 
% var_A - vector of variances of the agent-agent communication noise
% w - the weight of the sample received from the learner. if  weight_estimation_flag=1, this value is ignored. 

% Defining parameters 
clear all; close all; clc; 
k = 9 ;Episodes = 50 ; Num_of_Agents = 4 ; times = 50 ;
eps_a = 10^(-7) ; eps_b = 0.1 ; gamma = 0.98 ; 
R_max = 1 ; Bell_iterations = 30 ; 
Episode_Length= 50 ; maze_size = [5,5] ; km=3; weight_estimation_flag = 1;   
var_L = 0.1; var_A = [0.1,0.4,1] ; w = 1; 
w_opt = (var_A+var_L)./(var_A+Num_of_Agents*var_L) ; % Optimal value of w.

% DELETE THESE
noise_type = 'gaussian' ; 
maze_type = 1 ;
 rand_a_flag = 0 ;
 noise_var = 0 ;
 est_type = 'mean' ;
 m=0 ;
%

% Initializing vectors for saving results. 
R_mean_est = zeros(Num_of_Agents,Episodes,length(var_A)) ; 
R_std_est =  zeros(Num_of_Agents,Episodes,length(var_A)) ; 

R_mean_no_comm = zeros(Num_of_Agents,Episodes,length(var_A)) ; 
R_Std_no_comm =  zeros(Num_of_Agents,Episodes,length(var_A)) ;

R_mean_uniform_avg = zeros(Num_of_Agents,Episodes,length(var_A)) ; 
R_Std_uniform_avg =  zeros(Num_of_Agents,Episodes,length(var_A)) ; 


for i=1:length(var_A)
	[R_mean_est(:,:,i), R_std_est(:,:,i),R_Median_1s(:,:,i),R_1s(:,:,i,:),R_Learner_mean_est(:,:,i), R_Learner_Std_est(:,:,i),L_Median_1s(:,:,i), L_1s(:,:,i,:),x, y,Q]=Noisy_PAC(Episodes,Num_of_Agents,times,eps_a,eps_b,gamma,R_max,...
	                                                          k,Bell_iterations,Episode_Length,maze_size,  km,...
															  var_L,var_A(i),weight_estimation_flag, 1) ;
															  
    [R_mean_no_comm(:,:,i), R_Std_no_comm(:,:,i),R_Median_2s(:,:,i),R_2s(:,:,i,:),R_Learner_mean_no_comm(:,:,i), R_Learner_Std_no_comm(:,:,i),L_Median_2s(:,:,i), L_2s(:,:,i,:),x, y,Q]=Noisy_PAC(Episodes,Num_of_Agents,times,eps_a,eps_b,gamma,R_max,...
	                                                          k,Bell_iterations,Episode_Length,maze_size,  km,...
															  var_L,var_A(i),0, 1) ;
															  
    [R_mean_uniform_avg(:,:,i), R_Std_uniform_avg(:,:,i),R_Median_3s(:,:,i),R_3s(:,:,i,:),L_mean_3s(:,:,i), L_Std_3s(:,:,i),L_Median_3s(:,:,i), L_3s(:,:,i,:),x, y,Q]=Noisy_PAC(Episodes,Num_of_Agents,times,eps_a,eps_b,gamma,R_max,...
	                                                          k,Bell_iterations,Episode_Length,maze_size,maze_type,  km,...
															  var_L,var_A(i),0, 1/Num_of_Agents) ;
    [R_mean_opt(:,:,i), R_Std_opt(:,:,i),R_Median_4s(:,:,i),R_4s(:,:,i,:),L_mean_4s(:,:,i), L_Std_4s(:,:,i),L_Median_4s(:,:,i), L_4s(:,:,i,:), x, y,Q]=Noisy_PAC(Episodes,Num_of_Agents,times,eps_a,eps_b,gamma,R_max,...
	                                                          k,Bell_iterations,Episode_Length,maze_size, km,...
															  var_L,var_A(i),0, w_opt(i)) ;													  
end
											

%%	
Name = '' ;
var_L = 0.1; var_A = 0 ; w = 1; 


var_A = [0.1,0.4,1] ;										
Name = '' ; 
figure; 
subplot(1,3,1)
errorbar(1:size(R_mean_est,2), mean(R_mean_est(:,:,1),1),mean(R_std_est(:,:,1),1),'r','LineWidth',2); hold on; errorbar(1:size(R_mean_est,2), mean(R_mean_no_comm(:,:,1),1),mean(R_Std_no_comm(:,:,1),1),'b','LineWidth',2);
errorbar(1:size(R_mean_est,2), mean(R_mean_uniform_avg(:,:,1),1),mean(R_Std_uniform_avg(:,:,1),1),'g','LineWidth',2);errorbar(1:size(R_mean_est,2), mean(R_mean_opt(:,:,1),1),mean(R_Std_opt(:,:,1),1),'black','LineWidth',2);
plot(1:size(R_mean_est,2), mean(R_Learner_mean_est(:,:,1),1),'--r','LineWidth',2); hold on; plot(1:size(R_mean_est,2), mean(R_Learner_mean_no_comm(:,:,1),1),'--b','LineWidth',2);
plot(1:size(R_mean_est,2), mean(L_mean_3s(:,:,1),1),'--g','LineWidth',2);plot(1:size(R_mean_est,2), mean(L_mean_4s(:,:,1),1),'--black','LineWidth',2);
title(['\sigma^2_A/\sigma^2_Q = ',num2str(var_A(1)/var_L),', Agent averaged Accumulated Reward']) ; 
xlabel('Episode'); ylabel('Accumulated reward');

subplot(1,3,2)
errorbar(1:size(R_mean_est,2), mean(R_mean_est(:,:,2),1),mean(R_std_est(:,:,2),1),'r','LineWidth',2); hold on; errorbar(1:size(R_mean_est,2), mean(R_mean_no_comm(:,:,2),1),mean(R_Std_no_comm(:,:,2),1),'b','LineWidth',2);
errorbar(1:size(R_mean_est,2), mean(R_mean_uniform_avg(:,:,2),1),mean(R_Std_uniform_avg(:,:,2),1),'g','LineWidth',2);errorbar(1:size(R_mean_est,2), mean(R_mean_opt(:,:,2),1),mean(R_Std_opt(:,:,2),1),'black','LineWidth',2);
plot(1:size(R_mean_est,2), mean(R_Learner_mean_est(:,:,2),1),'--r','LineWidth',2); hold on; plot(1:size(R_mean_est,2), mean(R_Learner_mean_no_comm(:,:,2),1),'--b','LineWidth',2);
plot(1:size(R_mean_est,2), mean(L_mean_3s(:,:,2),1),'--g','LineWidth',2);plot(1:size(R_mean_est,2), mean(L_mean_4s(:,:,2),1),'--black','LineWidth',2);
title(['\sigma^2_A/\sigma^2_Q = ',num2str(var_A(2)/var_L),', Agent averaged Accumulated Reward']) ; 
xlabel('Episode'); ylabel('Accumulated reward');

subplot(1,3,3)
errorbar(1:size(R_mean_est,2), mean(R_mean_est(:,:,3),1),mean(R_std_est(:,:,3),1),'r','LineWidth',2); hold on; errorbar(1:size(R_mean_est,2), mean(R_mean_no_comm(:,:,3),1),mean(R_Std_no_comm(:,:,3),1),'b','LineWidth',2);
errorbar(1:size(R_mean_est,2), mean(R_mean_uniform_avg(:,:,3),1),mean(R_Std_uniform_avg(:,:,3),1),'g','LineWidth',2);errorbar(1:size(R_mean_est,2), mean(R_mean_opt(:,:,3),1),mean(R_Std_opt(:,:,3),1),'black','LineWidth',2);
plot(1:size(R_mean_est,2), mean(R_Learner_mean_est(:,:,3),1),'--r','LineWidth',2); hold on; plot(1:size(R_mean_est,2), mean(R_Learner_mean_no_comm(:,:,3),1),'--b','LineWidth',2);
plot(1:size(R_mean_est,2), mean(L_mean_3s(:,:,3),1),'--g','LineWidth',2);plot(1:size(R_mean_est,2), mean(L_mean_4s(:,:,3),1),'--black','LineWidth',2);
title(['\sigma^2_A/\sigma^2_Q = ',num2str(var_A(3)/var_L),', Agent averaged Accumulated Reward']) ; 
xlabel('Episode'); ylabel('Accumulated reward');
legend('estimated','w=1','w=0.25','optimal','estimatedL','w=1L','w=0.25L','optimalL');

%%%%%%%%%%%%%%%%%% Plotting 

figure; 
subplot(1,3,1)
errorbar(1:size(R_Median_1s,2), mean(R_Median_1s(:,:,1),1),mean(R_std_est(:,:,1),1),'r','LineWidth',2); hold on; errorbar(1:size(R_Median_1s,2), mean(R_Median_2s(:,:,1),1),mean(R_Std_no_comm(:,:,1),1),'b','LineWidth',2);
errorbar(1:size(R_Median_1s,2), mean(R_Median_3s(:,:,1),1),mean(R_Std_uniform_avg(:,:,1),1),'g','LineWidth',2);errorbar(1:size(R_Median_1s,2), mean(R_Median_4s(:,:,1),1),mean(R_Std_opt(:,:,1),1),'black','LineWidth',2);
plot(1:size(R_Median_1s,2), mean(L_Median_1s(:,:,1),1),'--r','LineWidth',2); hold on; plot(1:size(R_Median_1s,2), mean(L_Median_2s(:,:,1),1),'--b','LineWidth',2);
plot(1:size(R_Median_1s,2), mean(L_Median_3s(:,:,1),1),'--g','LineWidth',2);plot(1:size(R_Median_1s,2), mean(L_Median_4s(:,:,1),1),'--black','LineWidth',2);
title(['\sigma^2_A/\sigma^2_Q = ',num2str(var_A(1)/var_L),', Agent median Accumulated Reward']) ; 
xlabel('Episode'); ylabel('Accumulated reward');

subplot(1,3,2)
errorbar(1:size(R_Median_1s,2), mean(R_Median_1s(:,:,2),1),mean(R_std_est(:,:,2),1),'r','LineWidth',2); hold on; errorbar(1:size(R_Median_1s,2), mean(R_Median_2s(:,:,2),1),mean(R_Std_no_comm(:,:,2),1),'b','LineWidth',2);
errorbar(1:size(R_Median_1s,2), mean(R_Median_3s(:,:,2),1),mean(R_Std_uniform_avg(:,:,2),1),'g','LineWidth',2);errorbar(1:size(R_Median_1s,2), mean(R_Median_4s(:,:,2),1),mean(R_Std_opt(:,:,2),1),'black','LineWidth',2);
plot(1:size(R_Median_1s,2), mean(L_Median_1s(:,:,2),1),'--r','LineWidth',2); hold on; plot(1:size(R_Median_1s,2), mean(L_Median_2s(:,:,2),1),'--b','LineWidth',2);
plot(1:size(R_Median_1s,2), mean(L_Median_3s(:,:,2),1),'--g','LineWidth',2);plot(1:size(R_Median_1s,2), mean(L_Median_4s(:,:,2),1),'--black','LineWidth',2);
title(['\sigma^2_A/\sigma^2_Q = ',num2str(var_A(2)/var_L),', Agent median Accumulated Reward']) ; 
xlabel('Episode'); ylabel('Accumulated reward');

subplot(1,3,3)
errorbar(1:size(R_Median_1s,2), mean(R_Median_1s(:,:,3),1),mean(R_std_est(:,:,3),1),'r','LineWidth',2); hold on; errorbar(1:size(R_Median_1s,2), mean(R_Median_2s(:,:,3),1),mean(R_Std_no_comm(:,:,3),1),'b','LineWidth',2);
errorbar(1:size(R_Median_1s,2), mean(R_Median_3s(:,:,3),1),mean(R_Std_uniform_avg(:,:,3),1),'g','LineWidth',2);errorbar(1:size(R_Median_1s,2), mean(R_Median_4s(:,:,3),1),mean(R_Std_opt(:,:,3),1),'black','LineWidth',2);
plot(1:size(R_Median_1s,2), mean(L_Median_1s(:,:,3),1),'--r','LineWidth',2); hold on; plot(1:size(R_Median_1s,2), mean(L_Median_2s(:,:,3),1),'--b','LineWidth',2);
plot(1:size(R_Median_1s,2), mean(L_Median_3s(:,:,3),1),'--g','LineWidth',2);plot(1:size(R_Median_1s,2), mean(L_Median_4s(:,:,3),1),'--black','LineWidth',2);
title(['\sigma^2_A/\sigma^2_Q = ',num2str(var_A(3)/var_L),', Agent median Accumulated Reward']) ; 
xlabel('Episode'); ylabel('Accumulated reward');
legend('estimated','w=1','w=0.25','optimal','estimatedL','w=1L','w=0.25L','optimalL');