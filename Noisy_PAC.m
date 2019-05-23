% Noisy_PAC.m - A function that runs the algorithm in a maze. 
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
% var_A - variance of the agent-agent communication noise
% w - the weight of the sample received from the learner. if  weight_estimation_flag=1, this value is ignored.

% Average_Accumulated_Reward - a vector of size [Num_of_Agents,Episode] containing the averaged accumulated reward per agent 
%                              for each episode, averaged over repeated simulations. 
% Std_Accumulated_Reward - the standard deviation of the above averaging. 
% Average_Accumulated_Reward_Learner - a vector of size [1,Episode] containing the averaged accumulated reward for an 
%                                     imaginary agent having the learners' Q-function as each time step. 
% Std_Accumulated_Reward_Learner - the standard deviation of the above averaging. 

function [Average_Accumulated_Reward,Std_Accumulated_Reward, Average_Accumulated_Reward_Learner, ...
         Std_Accumulated_Reward_Learner]=...
		 Noisy_PAC(Episodes,Num_of_Agents,times,eps_a,eps_b,gamma,R_max,k,Bell_iterations,...
	     Episode_Length,maze_size, km,var_L,var_A, weight_estimation_flag, w)
 
Average_Accumulated_Reward = zeros(Num_of_Agents, Episodes,times) ;
Average_Accumulated_Reward_Learner = zeros(1, Episodes,times) ;
var_estimate_AA = 0 ; 
var_estimate_LA = 0 ; 
w_else = (1-w)/(Num_of_Agents-1)  ;

% parallel runs  
for tt=1:times
	Qmax = R_max/(1-0.98) ;
	A = 4 ; % Number of actions
	S =  maze_size(1)*maze_size(2) ; % Number of states for a square maze
	U = zeros(S,A,2,k) ; % approximation set for a square maze. Rows - states,Columns - actions, 3rd dimension - (r,s') 4th dimension - the samples 
	num_of_samples = zeros(S,A) ; % Number of samples for each state-action 	
	
	Ka = km ; 
	while Ka(end) < k % creating the active samples ggroup
		Ka = [Ka, Ka(end)*2] ; 
	end
	if Ka(end) > k 
		Ka(end) = k ; 
	end

	% Initialization 
	states = randi(S,1,Num_of_Agents); % Randomizing initial positions, using vector representation of the maze. 
	Q = Qmax*ones(S,A) ;  % Initiated as the optimistic maximum.
	
	% Noisy Q function and variance estimation - Initialization
	Q_L = repmat(Q,1,1,Num_of_Agents) + sqrt(var_L)*randn(S,A,Num_of_Agents) ; % approximation set sent by the learner to the various agents
	Q_noisy = zeros(S,A,Num_of_Agents); % Noisy approximation set sent between agents
	step_counter = 1 ; % Counts the number of times VI_flag has been set to 1
	var_estimate_LA = zeros(1,Num_of_Agents) ; % Estimation of the agent-learner communication variance \sigma^{2}_^{L}
	var_estimate_AA = zeros(1,Num_of_Agents) ; % Estimation of the agent-agent communication variance \sigma^{2}_^{L}+\sigma^{2}_^{A}
	
	for ii = 1:Num_of_Agents % Preparing the noisy Q values and estimation for the first step
		Q_L_temp = Q_L ; Q_L_temp(:,:,ii) = [] ; 
		Q_A = Q_L_temp + sqrt(var_A)*randn(S,A,Num_of_Agents-1) ;
		if weight_estimation_flag == 0 % Optimal weight
			Q_noisy(:,:,ii) = w*Q_L(:,:,ii) + w_else*sum(Q_A,3) ;
			Q_noisy(:,:,ii) = max(0,min(Qmax,Q_noisy(:,:,ii))) ;				
		else % Estimated weight
			% for the first step we use a simple average
			w = 1/Num_of_Agents ; 
			w_else = (1-w)/(Num_of_Agents-1)  ;
			Q_noisy(:,:,ii) = w*Q_L(:,:,ii) + w_else*sum(Q_A,3) ;
			
			% And then estimate the variance
			var_estimate_LA_temp = sum(sum(((Q_L(:,:,ii)-Q_noisy(:,:,ii)).^2)))/(S*A-1) ;
			var_estimate_LA(1,ii) = var_estimate_LA(1,ii) + (var_estimate_LA_temp-var_estimate_LA(1,ii))/step_counter ;
			
			var_estimate_AA_temp = sum(sum(sum(((Q_A-repmat(Q_noisy(:,:,ii),1,1,Num_of_Agents-1)).^2))))/(S*A*(Num_of_Agents-1)-1) ;
			var_estimate_AA(1,ii) = var_estimate_AA(1,ii) + (var_estimate_AA_temp-var_estimate_AA(1,ii))/step_counter ;
			

			Q_noisy(:,:,ii) = max(0,min(Qmax,Q_noisy(:,:,ii))) ;
		end				
	end
	for Episode = 1:Episodes
		disp(['=========================']);
		disp(['Times number ',num2str(tt),' of ',num2str(times),', Episode Number',num2str(Episode)]);
		disp(['=========================']);
		disp(['']);
		states = randi(S,1,Num_of_Agents); % randomly initializing locations for each agent at the start of each episode
		stateL = randi(S,1,1); % the state of the imaginary agent having the learner Q-function 
		for Step = 1:Episode_Length
			VI_flag = 0 ; % Do value iteration if the number of active samples for a state-action has risen. 
			% imaginary learner agent moves 
			[~,a_chosen] = max(Q(stateL,:)) ;
			s_mat = zeros(1,2) ; 
			[s_mat(1),s_mat(2)] =ind2sub([maze_size(1),maze_size(2)],stateL);
			[s_new,R] = move(s_mat, a_chosen, maze_size,R_max) ; 
			s_new = sub2ind([maze_size(1),maze_size(2)],s_new(1),s_new(2)) ;
			stateL = s_new ; 
			Average_Accumulated_Reward_Learner(1,Episode,tt) = Average_Accumulated_Reward_Learner(1,Episode,tt) + R; 
			% Each agent moves once and sends its sample to the learner
			for Agent = 1:Num_of_Agents 
				[~,a_chosen] = max(Q_noisy(states(Agent),:,Agent)) ;
				s_mat = zeros(1,2) ; 
				[s_mat(1),s_mat(2)] =ind2sub([maze_size(1),maze_size(2)],states(Agent));
				[s_new,R] = move(s_mat, a_chosen, maze_size,R_max) ; 
				s_new = sub2ind([maze_size(1),maze_size(2)],s_new(1),s_new(2)) ;
				if num_of_samples(states(Agent),a_chosen) < k 
					num_of_samples(states(Agent),a_chosen) = num_of_samples(states(Agent),a_chosen) + 1 ;
					U(states(Agent), a_chosen, :, num_of_samples(states(Agent),a_chosen)) = [s_new,R] ; 
					VI_flag = 1;  
				end
				states(Agent) = s_new ; 
				Average_Accumulated_Reward(Agent,Episode,tt) = Average_Accumulated_Reward(Agent,Episode,tt) + R;  
			end
			if VI_flag==1 % Value iteration and producing the new noisy Q-function
				Q = VI(Bell_iterations,S,A, num_of_samples, Ka, Qmax, Q, U, gamma, eps_a, eps_b, km) ; 
				Q_L = repmat(Q,1,1,Num_of_Agents) + sqrt(var_L)*randn(S,A,Num_of_Agents) ; 
				Q_noisy = zeros(S,A,Num_of_Agents); 
				step_counter = step_counter + 1 ;			
				for ii = 1:Num_of_Agents
					Q_L_temp = Q_L ; Q_L_temp(:,:,ii) = [] ; 
					Q_A = Q_L_temp + sqrt(var_A)*randn(S,A,Num_of_Agents-1) ;
					if weight_estimation_flag == 0 % Optimal weights
						Q_noisy(:,:,ii) = w*Q_L(:,:,ii) + w_else*sum(Q_A,3) ;
						Q_noisy(:,:,ii) = max(0,min(Qmax,Q_noisy(:,:,ii))) ;				
					else % Estimated weights
						
						w = var_estimate_AA(1,ii)/((Num_of_Agents-1)*var_estimate_LA(1,ii) + var_estimate_AA(1,ii)) ;
						w_else = (1-w)/(Num_of_Agents-1)  ;
						Q_noisy(:,:,ii) = w*Q_L(:,:,ii) + w_else*sum(Q_A,3) ;
						
						var_estimate_LA_temp = sum(sum(((Q_L(:,:,ii)-Q_noisy(:,:,ii)).^2)))/(S*A-1) ;
						var_estimate_LA(1,ii) = var_estimate_LA(1,ii) + (var_estimate_LA_temp-var_estimate_LA(1,ii))/step_counter ;
						
						var_estimate_AA_temp = sum(sum(sum(((Q_A-repmat(Q_noisy(:,:,ii),1,1,Num_of_Agents-1)).^2))))/(S*A*(Num_of_Agents-1)-1) ;
						var_estimate_AA(1,ii) = var_estimate_AA(1,ii) + (var_estimate_AA_temp-var_estimate_AA(1,ii))/step_counter ;
						
						Q_noisy(:,:,ii) = max(0,min(Qmax,Q_noisy(:,:,ii))) ;
					end
				end
			end % VI		
		end % Steps
	end % Episodes
end		% times	

% Calculating the averages		
Std_Accumulated_Reward = std(Average_Accumulated_Reward,1,3) ;
Median_Accumulated_Reward = median(Average_Accumulated_Reward,3) ; 
Average_Accumulated_Reward = mean(Average_Accumulated_Reward,3) ; 

Std_Accumulated_Reward_Learner = std(Average_Accumulated_Reward_Learner,1,3) ;
Median_Accumulated_Reward_Learner = median(Average_Accumulated_Reward_Learner,3) ; 
Average_Accumulated_Reward_Learner = mean(Average_Accumulated_Reward_Learner,3) ;


end








